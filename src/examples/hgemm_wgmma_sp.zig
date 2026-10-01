const cuda = @import("cuda");
const api = @import("examples_abi");
const gen = cuda.gen;
const wg = cuda.wgmma;

// cp.async.wait_group takes an LLVM immarg; the generated wrapper's runtime
// i32 parameter cannot select. Wrap with a comptime parameter.
fn cpAsyncWaitGroup(comptime num: u32) void {
    gen.async_copy.cp_async_wait_group(num);
}

/// Sparse HGEMM on the warpgroup path: 2:4-structured-sparse f16 GEMM via
/// `wgmma.mma_async.sp.sync.aligned.m64n16k32.f32.f16.f16`. Requires sm_90a.
///
/// Modeled on `hgemm_wgmma` (the SS, 2-stage pipeline). The RS pipeline of
/// wgmma3 does not apply: sparse A comes through a shared-memory descriptor.
///
/// ## What the sparse form changes (and what it does not)
///
/// Shape: sparse f16 wgmma is m64nNk**32** — k16 sparse exists only for tf32.
/// So one issue covers dense-k 32 (16 kept), and the K-slice is 32.
///
/// A: stored pruned, M x K/2 = 64 x 16 kept f16 per stage — exactly the byte
/// shape of the dense v1 A tile, so v1's core-matrix packing and descriptor
/// (offset(m,kb) = (m/8)*256 + kb*128 + (m%8)*16, lbo 128, sbo 256, Major.k)
/// carry over unchanged.
///
/// B: dense, k32 x 128 per stage = 8 KB, so the packing widens to
/// offset(k, nb) = nb*512 + k*16 with sbo 512 (lbo 128 unchanged). The n16
/// sub-tile for wgmma #c is nb in {2c, 2c+1}, a contiguous 1 KB at
/// b_base + c*1024.
///
/// Metadata: one u32 per row per stage (8 4-wide chunks x 4 bits), loaded
/// from shared into registers before the issue. The thread mapping is the
/// warpgroup-wide Figure-175 mapping documented on `wg.mmaSpAsyncM64N16K32`:
/// lane 4g+t of warp w supplies rows w*16+g and w*16+g+8, t==0 the low
/// k-chunks and t==1 the high. This reading — from a figure image, not text —
/// is the least-certain part of this kernel and leads the next-pod-run
/// suspect list; the exact-match bench check falsifies it hard.
///
/// Pipeline: v1's. cp.async double buffering, wgmma group committed per stage
/// and drained with wait_group 0. Deeper pipelining is the same trade as the
/// dense line (v2/v3); not the point of this variant.

pub const tile_m = 64;
pub const tile_n = 128;
pub const k_slice = 32; // dense k per stage; pruned A carries 16 per row
pub const threads = 128;
pub const n_tiles = tile_n / 16; // wgmma ops per K-stage

const a_bytes = tile_m * k_slice; // 2 KB: 64 rows x 16 kept f16
const b_bytes = k_slice * tile_n * 2; // 8 KB
const meta_bytes = tile_m * 4; // 256 B, one u32 per row

// 128-byte alignment: descriptors address shared memory in 16-byte units, and
// core matrices must not straddle the granularity the swizzle-none layout
// assumes.
var as_mem: [2][a_bytes]u8 align(128) addrspace(.shared) = undefined;
var bs_mem: [2][b_bytes]u8 align(128) addrspace(.shared) = undefined;
var ms_mem: [2][meta_bytes]u8 align(128) addrspace(.shared) = undefined;

fn issueTileLoad(buf: usize, a: [*]const f16, b: [*]const f16, meta: [*]const u32, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A pruned: 64 rows x 16 kept f16 = 128 chunks of 16 B; one per thread.
    // A row holds n/2 kept f16; k0 dense = k0/2 kept. Same packing as the
    // dense v1 A tile: core-matrix (m/8, kb) at (m/8)*256 + kb*128 + (m%8)*16.
    {
        const m = tid / 2;
        const kb = tid % 2;
        const src = a + (@as(usize, block_row + m) * (n / 2) + k0 / 2 + kb * 8);
        const dst = (m / 8) * 256 + kb * 128 + (m % 8) * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][dst]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // Metadata: 64 rows x 4B = 256B = 16 chunks; threads 0..15 load 16B
    // (4 rows' words) each. A row holds n/32 metadata words; k0 dense =
    // k0/32 words.
    if (tid < 16) {
        const src = meta + (@as(usize, block_row + tid * 4) * (n / 32) + k0 / 32);
        gen.async_copy.cp_async_cg_16(@ptrCast(&ms_mem[buf][tid * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B dense: 32 k-rows x 16 n-blocks = 512 chunks; four per thread.
    // Packing: nb*512 + k*16 (each n-block is 32 k-rows of 16 B).
    inline for (0..4) |i| {
        const chunk = tid * 4 + i;
        const k = chunk / 16;
        const nb = chunk % 16;
        const src = b + (@as(usize, k0 + k) * n + block_col) + nb * 8;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][nb * 512 + k * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    gen.async_copy.cp_async_commit_group();
}

/// The lane's metadata word for this stage, per the Figure-175 mapping.
fn loadMeta(buf: usize, warp: u32, lane: u32) u32 {
    const ptr: [*]addrspace(.shared) const u32 = @ptrCast(@alignCast(&ms_mem[buf]));
    const g = lane / 4;
    const t = lane % 4;
    const row0 = warp * 16 + g;
    const row1 = row0 + 8;
    const lo_t: u1 = @intCast(t & 1); // t==0 -> low k-chunks, t==1 -> high
    const w0 = ptr[row0];
    const w1 = ptr[row1];
    const h0: u32 = if (lo_t == 0) w0 & 0xFFFF else w0 >> 16;
    const h1: u32 = if (lo_t == 0) w1 & 0xFFFF else w1 >> 16;
    return h0 | (h1 << 16);
}

fn computeStage(comptime buf: usize, acc: *[n_tiles]wg.Acc64x16, warp: u32, lane: u32, scale_d: bool) void {
    const desc_a = wg.descriptor(wg.smemAddr(&as_mem[buf][0]), 128, 256, .none);
    const b_base = wg.smemAddr(&bs_mem[buf][0]);
    const meta = loadMeta(buf, warp, lane);
    wg.fence();
    inline for (0..n_tiles) |c| {
        const desc_b = wg.descriptor(b_base + @as(u32, c) * 1024, 128, 512, .none);
        wg.mmaSpAsyncM64N16K32(&acc[c], desc_a, desc_b, meta, scale_d, .k, .mn);
    }
    wg.commitGroup();
}

pub fn hgemmWgmmaSp(a: [*]const f16, b: [*]const f16, meta: [*]const u32, c: [*]f32, n: u32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const warp = tid / 32;
    const lane = cuda.laneId();

    const block_row = cuda.blockIdx().y * tile_m;
    const block_col = cuda.blockIdx().x * tile_n;

    var acc: [n_tiles]wg.Acc64x16 = @splat(.{});

    const ktiles = n / k_slice;

    issueTileLoad(0, a, b, meta, n, block_row, block_col, 0, tid);

    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        const has_next = kt + 1 < ktiles; // block-uniform
        if (has_next) {
            issueTileLoad((kt + 1) % 2, a, b, meta, n, block_row, block_col, (kt + 1) * k_slice, tid);
        }
        if (has_next) cpAsyncWaitGroup(1) else cpAsyncWaitGroup(0);
        cuda.syncThreads();
        if (kt % 2 == 0) computeStage(0, &acc, warp, lane, kt != 0) else computeStage(1, &acc, warp, lane, kt != 0);
        // v1 drains every stage: the buffer about to be refilled is the one
        // the previous stage's wgmma may still be reading.
        wg.waitGroup(0);
        cuda.syncThreads();
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
    cuda.abi.assertMatches(api.hgemm_wgmma_sp, @TypeOf(hgemmWgmmaSp));
    _ = cuda.Keep(.{&hgemmWgmmaSp}).__zoxide_keep_kernels;
}
