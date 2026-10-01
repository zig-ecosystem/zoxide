const cuda = @import("cuda");
const api = @import("examples_abi");
const gen = cuda.gen;
const wg = cuda.wgmma;

fn cpAsyncWaitGroup(comptime num: u32) void {
    gen.async_copy.cp_async_wait_group(num);
}

/// HGEMM bf16 on the wgmma3 pipeline: `hgemm_wgmma3`'s exact shape (3-stage
/// cp.async pipeline, A in registers via ldmatrix.x4, RS-form wgmma tiled
/// m64n16k16 x8 over N=128) with bf16 inputs. Requires sm_90a.
///
/// bf16 and f16 are both 16-bit, so *every* byte-level mechanic is identical:
/// the [64][16] A tile (32 B row pitch), the core-matrix-packed B
/// (offset(k, nb) = nb*256 + k*16, lbo 128, sbo 256, Major.mn), the 64-bit
/// descriptor bit layout, the ALayout_64x16 fragment order, the accumulator
/// mapping. The only differences are the mma mnemonic suffix
/// (`.f32.bf16.bf16`, via the sibling helper `wg.mmaAsyncM64N16K16RsBf16` —
/// the f16 helper is untouched so wgmma3's evidence stays valid) and the host
/// data, which is bf16 bit patterns in u16 storage (Zig has no bf16 type).
///
/// What that means for risk: nothing in the pipeline, addressing or hazard
/// handling is new — those are all proven on the f16 path. The novel content
/// is exactly one instruction suffix, so this kernel is the lowest-risk entry
/// in the new-shape family; the honest residual risks are ptxas acceptance of
/// the bf16 RS form and the host-side bf16 packing, both covered by the
/// exact-match bench check on first GPU run.
///
/// See `hgemm_wgmma3` for the pipeline/hazard rationale; it is not restated
/// here so the two files cannot drift apart in prose.

pub const tile_m = 64;
pub const tile_n = 128;
pub const k_slice = 16;
pub const threads = 128;
pub const stages = 3;
pub const a_bufs = 2; // A fragment register sets, matching wgmma in-flight depth
pub const n_tiles = tile_n / 16;

const a_bytes = tile_m * k_slice * 2; // 2 KB, plain [64][16] bf16
const b_bytes = k_slice * tile_n * 2; // 4 KB, core-matrix packed bf16

var as_mem: [stages][a_bytes]u8 align(128) addrspace(.shared) = undefined;
var bs_mem: [stages][b_bytes]u8 align(128) addrspace(.shared) = undefined;

fn issueTileLoad(buf: u32, a: [*]const u16, b: [*]const u16, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A: plain row-major, 64 rows x 32 B = 128 chunks of 16 B, one per thread.
    {
        const m = tid / 2;
        const kb = tid % 2;
        const src = a + (@as(usize, block_row + m) * n + k0 + kb * 8);
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][m * 32 + kb * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B: core-matrix packed, 16 k-rows x 16 n-blocks = 256 chunks, two per thread.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 16;
        const nb = chunk % 16;
        const src = b + (@as(usize, k0 + k) * n + block_col + nb * 8);
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][nb * 256 + k * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    gen.async_copy.cp_async_commit_group();
}

/// One `ldmatrix.x4` gives this warp its whole A fragment for the 64x16 tile.
fn loadAFrag(buf: u32, warp: u32, lane: u32) [4]u32 {
    const matrix = lane / 8;
    const row = lane % 8;
    const grow = warp * 16 + row + (matrix % 2) * 8;
    const koff = (matrix / 2) * 8;
    const addr: [*]addrspace(.shared) u8 = @ptrCast(&as_mem[buf][grow * 32 + koff * 2]);
    const r = gen.matrix.ldmatrix_m8n8_x4_b16(addr);
    return .{ @bitCast(r.f0), @bitCast(r.f1), @bitCast(r.f2), @bitCast(r.f3) };
}

/// One K-stage: load this warp's A fragment, then 8 wgmma stepping the B
/// descriptor 16 columns at a time, committed as one group.
///
/// `frag` is passed as a pointer to a *named* local rather than indexed out of
/// an array: a runtime array index forces the fragment to local memory (it cost
/// 20 `st.local` per iteration when written that way), which would be worse
/// than the shared-memory re-reads this kernel exists to remove. `inline` so
/// the pointer is scalarised away.
inline fn issueStage(
    sbuf: u32,
    acc: *[n_tiles]wg.Acc64x16,
    frag: *[4]u32,
    warp: u32,
    lane: u32,
    scale_d: bool,
) void {
    frag.* = loadAFrag(sbuf, warp, lane);
    // Publishes the fragment registers just written, and the accumulators, to
    // the async unit.
    wg.fence();
    const b_base = wg.smemAddr(&bs_mem[sbuf][0]);
    inline for (0..n_tiles) |t| {
        const desc_b = wg.descriptor(b_base + @as(u32, t) * 512, 128, 256, .none);
        wg.mmaAsyncM64N16K16RsBf16(&acc[t], frag.*, desc_b, scale_d, .mn);
    }
    wg.commitGroup();
}

pub fn hgemmWgmmaBf16(a: [*]const u16, b: [*]const u16, c: [*]f32, n: u32) callconv(.kernel) void {
    // Declared contract, emitted as .maxntid — same 128-thread shape as
    // hgemm_bf16, whose abi entry this kernel shares.
    cuda.launchBounds(api.hgemm_bf16.launch_bounds);
    const tid = cuda.threadIdx().x;
    const warp = tid / 32;
    const lane = cuda.laneId();

    const block_row = cuda.blockIdx().y * tile_m;
    const block_col = cuda.blockIdx().x * tile_n;

    var acc: [n_tiles]wg.Acc64x16 = @splat(.{});
    // Two named fragment sets, alternating by stage parity, so a stage never
    // overwrites fragments the previous stage's in-flight wgmma is reading.
    // Named rather than an indexed array — see issueStage.
    var afrag_even: [4]u32 = @splat(0);
    var afrag_odd: [4]u32 = @splat(0);

    const ktiles = n / k_slice;

    issueTileLoad(0, a, b, n, block_row, block_col, 0, tid);
    if (ktiles > 1) {
        issueTileLoad(1, a, b, n, block_row, block_col, k_slice, tid);
    }

    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        if (kt + 2 <= ktiles) cpAsyncWaitGroup(1) else cpAsyncWaitGroup(0);
        cuda.syncThreads();

        const sbuf = kt % stages;
        // Block-uniform branch; keeps the fragment sets in registers.
        if (kt % a_bufs == 0) {
            issueStage(sbuf, &acc, &afrag_even, warp, lane, kt != 0);
        } else {
            issueStage(sbuf, &acc, &afrag_odd, warp, lane, kt != 0);
        }

        wg.waitGroup(1);
        cuda.syncThreads();

        if (kt + 2 < ktiles) {
            issueTileLoad((kt + 2) % stages, a, b, n, block_row, block_col, (kt + 2) * k_slice, tid);
        }

        // Keep both fragment sets live across the back edge. Without this the
        // register allocator sees stage kt's fragment die at its last wgmma and
        // may reuse those registers for stage kt+1 — collapsing the double
        // buffer the code above is relying on, while an outstanding wgmma is
        // still reading them.
        wg.fenceFragment(&afrag_even);
        wg.fenceFragment(&afrag_odd);
    }

    wg.waitGroup(0);

    inline for (0..n_tiles) |t| {
        inline for (0..8) |i| {
            const p = wg.Acc64x16.coord(warp, lane, i);
            const row = block_row + p.m;
            const col = block_col + t * 16 + p.n;
            c[@as(usize, row) * n + col] = acc[t].get(i);
        }
    }
}

comptime {
    // Drift from the signature `zoxide bench` launches through is a compile
    // error here rather than a silently mis-packed argument list.
    cuda.abi.assertMatches(api.hgemm_bf16, @TypeOf(hgemmWgmmaBf16));
    _ = cuda.Keep(.{&hgemmWgmmaBf16}).__zoxide_keep_kernels;
}
