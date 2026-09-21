const cuda = @import("cuda");
const gen = cuda.gen;
const wg = cuda.wgmma;

fn cpAsyncWaitGroup(comptime num: u32) void {
    gen.async_copy.cp_async_wait_group(num);
}

/// HGEMM v5: `hgemm_wgmma2` with the A operand moved from shared memory into
/// registers (`wgmma` RS form). Requires sm_90a.
///
/// ## Why
///
/// v4 reached 58.3% of FP16 peak, and experiment had eliminated the pipeline,
/// global traffic and warpgroup concurrency as causes of the remaining 41.7pp,
/// leaving the n16 shape. The reasoning attached to that was partly wrong: I
/// claimed the operand-traffic penalty could only be fixed by a wider N per
/// instruction, which the 15-output asm limit blocks. There is a third way.
///
/// Covering a 128-wide tile with 8 `m64n16k16` in the all-shared form makes
/// every one of them re-read the whole A tile:
///
///   all-shared n16   8 * (A 2048 + B 512)  = 20480 B/stage   12.8 flops/B
///   A in registers   A 2048 + 8 * (B 512)  =  6144 B/stage   42.7 flops/B
///   one n128 (SS)    A 2048 + B 4096       =  6144 B/stage   42.7 flops/B
///
/// Loading A once into registers gives exactly what a single `m64n128k16` would
/// have read — the full operand-traffic efficiency of the wide shape, reachable
/// today, because the accumulator is still 8 registers and stays inside the
/// 15-output limit. If throughput does not move, the n16 penalty is intrinsic
/// per-instruction tensor-core efficiency rather than operand starvation, and
/// only a wider N (hence the compiler patch) can help. Either way this settles
/// which it is, and unlike the patch its result is shippable.
///
/// ## Shared memory layout
///
/// A no longer feeds a descriptor, so it does not need core-matrix packing and
/// is stored plainly as [64][16] f16 (32 B row pitch) — the layout `ldmatrix`
/// wants. One `ldmatrix.x4` per warp yields that warp's whole 4-register
/// fragment, because CUTLASS's `ALayout_64x16` is the `mma.sync m16n8k16`
/// A-fragment layout applied to each warp's own 16 rows:
///   a[0] = A[m][k0,k0+1]      a[1] = A[m+8][k0,k0+1]
///   a[2] = A[m][k0+8,k0+9]    a[3] = A[m+8][k0+8,k0+9]
/// with m = warp*16 + lane/4 and k0 = (lane%4)*2. This is the same addressing
/// `hgemm_mma2` uses, reused here.
///
/// B still goes through a descriptor and stays core-matrix packed:
///   offset(k, nb) = nb*256 + k*16,  lbo 128, sbo 256, Major.mn
///
/// ## Two hazards this has that v4 did not
///
/// An in-flight wgmma reads its A registers *after* issuing, so they cannot be
/// overwritten while the group is outstanding. `waitGroup(1)` deliberately
/// leaves the previous stage running, so the A fragments are double-buffered in
/// registers: stage `kt` uses set `kt % 2`, and set `kt % 2` is not rewritten
/// until stage `kt+2`, by which point the wait at stage `kt+1` has retired
/// stage `kt`. Eight extra registers to keep the overlap.
///
/// `wgmma.fence` must also separate the `ldmatrix` writes from the wgmma that
/// reads them, which is why the fence sits after the fragment load rather than
/// at the top of the stage.

pub const tile_m = 64;
pub const tile_n = 128;
pub const k_slice = 16;
pub const threads = 128;
pub const stages = 3;
pub const a_bufs = 2; // A fragment register sets, matching wgmma in-flight depth
pub const n_tiles = tile_n / 16;

const a_bytes = tile_m * k_slice * 2; // 2 KB, plain [64][16] f16
const b_bytes = k_slice * tile_n * 2; // 4 KB, core-matrix packed

var as_mem: [stages][a_bytes]u8 align(128) addrspace(.shared) = undefined;
var bs_mem: [stages][b_bytes]u8 align(128) addrspace(.shared) = undefined;

fn issueTileLoad(buf: u32, a: [*]const u8, b: [*]const u8, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A: plain row-major, 64 rows x 32 B = 128 chunks of 16 B, one per thread.
    {
        const m = tid / 2;
        const kb = tid % 2;
        const src = a + (@as(usize, block_row + m) * n + k0 + kb * 8) * 2;
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][m * 32 + kb * 16]), @addrSpaceCast(src));
    }
    // B: core-matrix packed, 16 k-rows x 16 n-blocks = 256 chunks, two per thread.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 16;
        const nb = chunk % 16;
        const src = b + (@as(usize, k0 + k) * n + block_col + nb * 8) * 2;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][nb * 256 + k * 16]), @addrSpaceCast(src));
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
        wg.mmaAsyncM64N16K16Rs(&acc[t], frag.*, desc_b, scale_d, .mn);
    }
    wg.commitGroup();
}

pub fn hgemmWgmma3(a: [*]const u8, b: [*]const u8, c: [*]f32, n: u32) callconv(.kernel) void {
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
    _ = cuda.Keep(.{&hgemmWgmma3}).__zoxide_keep_kernels;
}
