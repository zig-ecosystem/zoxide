const cuda = @import("cuda");
const api = @import("examples_abi");
const gen = cuda.gen;
const wg = cuda.wgmma;

// cp.async.wait_group takes an LLVM immarg; the generated wrapper's runtime
// i32 parameter cannot select. Wrap with a comptime parameter.
fn cpAsyncWaitGroup(comptime num: u32) void {
    gen.async_copy.cp_async_wait_group(num);
}

/// HGEMM v3: Hopper warpgroup MMA (`wgmma.mma_async`) with both operands read
/// asynchronously from shared memory. Requires sm_90a.
///
/// One warpgroup (128 threads) per block, block tile M=64 x N=128, K=16 per
/// stage. Each K-stage issues 8 `m64n16k16` wgmma ops, stepping the B
/// descriptor 16 columns at a time.
///
/// Why n16 and not n128: `m64nNk16` needs N/2 accumulator registers per
/// thread, all of which must be inline-asm output operands, and Zig's AstGen
/// rejects more than 15 outputs per `asm` expression. n16 (8 regs) fits; n32
/// (16 regs) is one over. See src/wgmma.zig and
/// docs/upstream-asm-output-limit.md. Consequence: A is re-read from shared
/// memory 8x per K-stage instead of once, so this trades away part of the
/// bandwidth win that wide-N wgmma exists to capture.
///
/// ## Shared memory layout
///
/// wgmma reads operands as 8x8 *core matrices* of 128 contiguous bytes (8 rows
/// x 16 B), so tiles are packed as a sequence of core matrices rather than as
/// plain 2-D arrays.
///
/// A tile (64 m x 16 k, K-major, `Major.k`, trans_a=0), 2048 B:
///   offset(m, kb) = (m/8)*256 + kb*128 + (m%8)*16       kb = k/8
///   leading_byte_offset = 128 (k-block stride)
///   stride_byte_offset  = 256 (m-block stride)
///
/// B tile (16 k x 128 n, N-major, `Major.mn`, trans_b=1), 4096 B:
///   offset(k, nb) = nb*256 + k*16                        nb = n/8
///   leading_byte_offset = 128 (stride between k halves)
///   stride_byte_offset  = 256 (n-block stride)
/// The n16 sub-tile for wgmma #c covers nb in {2c, 2c+1}, i.e. a contiguous
/// 512 B region at b_base + c*512, with the same two offsets.
///
/// ## Pipeline
///
/// cp.async double buffering: the prologue issues stage 0, each iteration
/// issues stage kt+1 into the other buffer, then waits for the current one.
/// `wgmma.fence` publishes the accumulator registers to the async unit, the 8
/// wgmma ops are committed as one group, and `wait_group 0` drains it before
/// the buffer is reused. Draining every stage is what makes the shared-memory
/// reuse safe; overlapping wgmma with the next stage's loads needs a third
/// buffer and is left for v4.
///
/// Registers: 8 wgmma tiles x 8 f32 = 64 accumulators, plus descriptors and
/// addresses — well under the 255 cap.
///
/// Requires n % 128 == 0 for the N tiling and 16 B-aligned global loads.

pub const tile_m = 64;
pub const tile_n = 128;
pub const k_slice = 16;
pub const threads = 128;
pub const n_tiles = tile_n / 16; // wgmma ops per K-stage

const a_bytes = tile_m * k_slice * 2; // 2 KB
const b_bytes = k_slice * tile_n * 2; // 4 KB

// 128-byte alignment: descriptors address shared memory in 16-byte units, and
// core matrices must not straddle the granularity the swizzle-none layout
// assumes.
var as_mem: [2][a_bytes]u8 align(128) addrspace(.shared) = undefined;
var bs_mem: [2][b_bytes]u8 align(128) addrspace(.shared) = undefined;

fn issueTileLoad(buf: usize, a: [*]const f16, b: [*]const f16, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A: 64 rows x 2 k-blocks = 128 chunks of 16 B; one per thread.
    {
        const m = tid / 2;
        const kb = tid % 2;
        const src = a + (@as(usize, block_row + m) * n + k0 + kb * 8);
        const dst = (m / 8) * 256 + kb * 128 + (m % 8) * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][dst]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B: 16 k-rows x 16 n-blocks = 256 chunks; two per thread.
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

fn computeStage(comptime buf: usize, acc: *[n_tiles]wg.Acc64x16, scale_d: bool) void {
    const desc_a = wg.descriptor(wg.smemAddr(&as_mem[buf][0]), 128, 256, .none);
    const b_base = wg.smemAddr(&bs_mem[buf][0]);
    wg.fence();
    inline for (0..n_tiles) |c| {
        const desc_b = wg.descriptor(b_base + @as(u32, c) * 512, 128, 256, .none);
        wg.mmaAsyncM64N16K16(&acc[c], desc_a, desc_b, scale_d, .k, .mn);
    }
    wg.commitGroup();
    wg.waitGroup(0);
}

pub fn hgemmWgmma(a: [*]const f16, b: [*]const f16, c: [*]f32, n: u32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const warp = tid / 32;
    const lane = cuda.laneId();

    const block_row = cuda.blockIdx().y * tile_m;
    const block_col = cuda.blockIdx().x * tile_n;

    var acc: [n_tiles]wg.Acc64x16 = @splat(.{});

    const ktiles = n / k_slice;
    issueTileLoad(0, a, b, n, block_row, block_col, 0, tid);

    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        const has_next = kt + 1 < ktiles; // block-uniform
        if (has_next) {
            issueTileLoad((kt + 1) % 2, a, b, n, block_row, block_col, (kt + 1) * k_slice, tid);
        }
        if (has_next) cpAsyncWaitGroup(1) else cpAsyncWaitGroup(0);
        cuda.syncThreads();
        // scale_d = false on the first stage: overwrite instead of accumulate,
        // which is why `acc` never needs an explicit zeroing pass.
        const scale_d = kt != 0;
        // Buffer parity is derived from kt rather than carried in a mutable
        // `cur: u1`. The toggle made LLVM keep it in a 1-byte __local_depot and
        // store to it every iteration — reported by the driver as 8 B/thread
        // spilled. Deriving it strength-reduces to a rotating counter with no
        // local memory at all, the same shape hgemm_wgmma2/3 use.
        if (kt % 2 == 0) computeStage(0, &acc, scale_d) else computeStage(1, &acc, scale_d);
        cuda.syncThreads();
    }

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
    cuda.abi.assertMatches(api.hgemm, @TypeOf(hgemmWgmma));
    _ = cuda.Keep(.{&hgemmWgmma}).__zoxide_keep_kernels;
}
