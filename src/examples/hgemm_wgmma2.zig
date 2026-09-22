const cuda = @import("cuda");
const api = @import("examples_abi");
const gen = cuda.gen;
const wg = cuda.wgmma;

// cp.async.wait_group takes an LLVM immarg; the generated wrapper's runtime
// i32 parameter cannot select. Wrap with a comptime parameter.
fn cpAsyncWaitGroup(comptime num: u32) void {
    gen.async_copy.cp_async_wait_group(num);
}

/// HGEMM v4: `hgemm_wgmma` with a 3-stage pipeline so the warpgroup MMA can
/// overlap the next tile's global loads. Requires sm_90a.
///
/// ## Measured outcome: no gain
///
/// This variant was credited with +4.1pp when first measured, and that was a
/// misattribution. It changed two things relative to v3 at once: the pipeline
/// depth (intended) and the replacement of a `cur: u1` buffer toggle with
/// `kt % stages` (incidental). The toggle had been costing a 1-byte local-memory
/// store every iteration. With that removed from v3 as well, v3 reaches the same
/// 58.3% of peak on H20 with two stages, and the pipeline itself measures
/// **-0.0pp**.
///
/// So draining the tensor core with `wgmma.wait_group 0` every stage — which the
/// commentary below treats as obvious waste — costs nothing measurable here. It
/// is kept as a worked example of the technique and because deeper pipelining may
/// matter at other tile shapes, but it is not the faster kernel: it uses 6 KB
/// more shared memory than v3 for the same throughput. Prefer v3 over v4, and
/// `hgemm_wgmma3` over both.
///
/// Same shape as v3 — 64x128 block tile, one warpgroup, K-slice 16, 8
/// `m64n16k16` wgmma per K-stage — so the two are a clean A/B on the pipeline
/// alone.
///
/// ## What v3 got wrong
///
/// v3 ended every K-stage with `wgmma.wait_group 0`, draining the tensor core
/// before moving on. With only two buffers it had no choice: the buffer about
/// to be refilled is the one the previous stage's wgmma was reading, and wgmma
/// reads shared memory *asynchronously*, so reusing it early corrupts the
/// operand. Draining made that safe and made the async instruction behave like
/// a synchronous one — a sports car held in first gear.
///
/// Three buffers break the dependency. Stage `kt` computes out of buffer
/// `kt % 3`, and `wgmma.wait_group 1` retires stage `kt-1` while stage `kt` is
/// still in flight. That frees buffer `(kt-1) % 3`, which is the same slot as
/// `(kt+2) % 3` — exactly the one stage `kt+2` wants. So the load for `kt+2`
/// can be issued while `kt` is still running on the tensor core.
///
/// Group accounting, both branches block-uniform:
///   cp.async — stages 0 and 1 are issued in the prologue, stage `kt+2` at the
///     end of iteration `kt`, so `min(kt+2, ktiles)` groups have been issued
///     when iteration `kt` begins and `min(kt+2, ktiles) - (kt+1)` of them are
///     for later stages. That is 1 until the tail, then 0.
///   wgmma — one group per stage, `wait_group 1` keeps the newest in flight.
///     The epilogue needs `wait_group 0` before any accumulator register is
///     read by a normal instruction.
///
/// Accumulating into the same registers across stages without waiting is the
/// intended usage: the hardware orders wgmma-to-wgmma accumulator dependencies
/// itself. What is *not* allowed is reading those registers from a non-wgmma
/// instruction before the wait, hence the epilogue drain. `wgmma.fence` is
/// re-issued per stage, matching what CUTLASS's mainloop does.
///
/// ## Why no swizzle
///
/// Still `Swizzle.none`, and deliberately. The descriptor swizzle modes exist
/// to keep accesses bank-conflict-free when a tile keeps a wide row pitch (the
/// layout TMA produces). Our tiles are core-matrix packed instead: one core
/// matrix is 128 contiguous bytes, and shared memory is 32 banks x 4 B = 128
/// bytes per bank cycle, so a core-matrix read already sweeps all 32 banks
/// exactly once. There is no conflict for a swizzle to remove.
///
/// ## What this does not fix
///
/// Global traffic. Per block this reads 64x4096 of A plus 4096x128 of B = 1.5
/// MB, and at 2048 blocks that is 3.07 GB, which at v3's 1.711 ms is 1.79 TB/s
/// against H20's ~4 TB/s HBM. L2 absorbs much of it (the 32 blocks in a row
/// share A rows, the 64 in a column share B columns) but M=64 is a narrow
/// reuse window. Widening to M=128 with two warpgroups cuts total traffic to
/// ~2 GB; that is the next lever and is kept out of this change so the
/// pipeline can be measured on its own.

pub const tile_m = 64;
pub const tile_n = 128;
pub const k_slice = 16;
pub const threads = 128;
pub const stages = 3;
pub const n_tiles = tile_n / 16; // wgmma ops per K-stage

const a_bytes = tile_m * k_slice * 2; // 2 KB per stage
const b_bytes = k_slice * tile_n * 2; // 4 KB per stage

// 18 KB total, well inside H20's 228 KB per SM, so the extra stage does not
// cost occupancy here.
var as_mem: [stages][a_bytes]u8 align(128) addrspace(.shared) = undefined;
var bs_mem: [stages][b_bytes]u8 align(128) addrspace(.shared) = undefined;

/// Core-matrix packed destinations, identical to v3:
///   A (64m x 16k, K-major):  offset(m, kb) = (m/8)*256 + kb*128 + (m%8)*16
///   B (16k x 128n, N-major): offset(k, nb) = nb*256 + k*16
fn issueTileLoad(buf: u32, a: [*]const f16, b: [*]const f16, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A: 64 rows x 2 k-blocks = 128 chunks of 16 B, one per thread.
    {
        const m = tid / 2;
        const kb = tid % 2;
        const src = a + (@as(usize, block_row + m) * n + k0 + kb * 8);
        const dst = (m / 8) * 256 + kb * 128 + (m % 8) * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][dst]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B: 16 k-rows x 16 n-blocks = 256 chunks, two per thread.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 16;
        const nb = chunk % 16;
        const src = b + (@as(usize, k0 + k) * n + block_col + nb * 8);
        const dst = nb * 256 + k * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][dst]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    gen.async_copy.cp_async_commit_group();
}

/// Issue one K-stage: 8 `m64n16k16` wgmma stepping the B descriptor 16 columns
/// at a time, committed as a single group. Does not wait.
fn issueStage(buf: u32, acc: *[n_tiles]wg.Acc64x16, scale_d: bool) void {
    const desc_a = wg.descriptor(wg.smemAddr(&as_mem[buf][0]), 128, 256, .none);
    const b_base = wg.smemAddr(&bs_mem[buf][0]);
    wg.fence();
    inline for (0..n_tiles) |c| {
        const desc_b = wg.descriptor(b_base + @as(u32, c) * 512, 128, 256, .none);
        wg.mmaAsyncM64N16K16(&acc[c], desc_a, desc_b, scale_d, .k, .mn);
    }
    wg.commitGroup();
}

pub fn hgemmWgmma2(a: [*]const f16, b: [*]const f16, c: [*]f32, n: u32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const warp = tid / 32;
    const lane = cuda.laneId();

    const block_row = cuda.blockIdx().y * tile_m;
    const block_col = cuda.blockIdx().x * tile_n;

    var acc: [n_tiles]wg.Acc64x16 = @splat(.{});

    const ktiles = n / k_slice;

    // Prologue: prime the first two stages.
    issueTileLoad(0, a, b, n, block_row, block_col, 0, tid);
    if (ktiles > 1) {
        issueTileLoad(1, a, b, n, block_row, block_col, k_slice, tid);
    }

    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        // Groups still outstanding for stages after this one: 1, or 0 at the
        // tail once no further loads have been issued. Block-uniform.
        if (kt + 2 <= ktiles) cpAsyncWaitGroup(1) else cpAsyncWaitGroup(0);
        cuda.syncThreads();

        // scale_d = false on the first stage overwrites the accumulator, so no
        // separate zeroing pass is needed.
        issueStage(kt % stages, &acc, kt != 0);

        // Retire stage kt-1 while kt keeps running. This is the whole point of
        // the third buffer.
        wg.waitGroup(1);
        cuda.syncThreads();

        if (kt + 2 < ktiles) {
            issueTileLoad((kt + 2) % stages, a, b, n, block_row, block_col, (kt + 2) * k_slice, tid);
        }
    }

    // Accumulators are about to be read by normal instructions.
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
    cuda.abi.assertMatches(api.hgemm, @TypeOf(hgemmWgmma2));
    _ = cuda.Keep(.{&hgemmWgmma2}).__zoxide_keep_kernels;
}
