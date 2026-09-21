const cuda = @import("cuda");
const wg = cuda.wgmma;

/// Self-checking `wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16` smoke
/// test. Requires sm_90a.
///
/// The risky part of wgmma is not the instruction but the operand layout: the
/// tensor core reads shared memory as 8x8 *core matrices* of 128 contiguous
/// bytes, addressed through a 64-bit descriptor. This kernel exercises exactly
/// that, in isolation from any GEMM tiling:
///
///   1. one warpgroup fills a 64x16 A tile and a 16x16 B tile in shared memory
///      using the core-matrix packing that `hgemm_wgmma` also uses,
///   2. issues a single wgmma,
///   3. recomputes every accumulator element from the closed-form inputs and
///      writes its own mismatch count to `out[tid]`.
///
/// `out` is all zeros iff the descriptor encoding, the shared-memory packing
/// and the accumulator register→(m,n) mapping are all correct. Operand values
/// are small integers, so both the f16 inputs and the f32 sums are exact and a
/// mismatch means a real layout bug, not rounding.

const a_bytes = 64 * 16 * 2; // 2 KB, 8 m-blocks x 2 k-blocks of 128 B
const b_bytes = 16 * 16 * 2; // 512 B, 2 n-blocks of 256 B

var as_mem: [a_bytes]u8 align(128) addrspace(.shared) = undefined;
var bs_mem: [b_bytes]u8 align(128) addrspace(.shared) = undefined;

inline fn aVal(m: u32, k: u32) f32 {
    return @floatFromInt((m % 7) + (k % 5));
}

inline fn bVal(k: u32, n: u32) f32 {
    return @floatFromInt(((k % 3) + 1) * ((n % 2) + 1));
}

pub fn wgmmaSmoke(out: [*]u32, mismatch_dump: [*]f32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const warp = tid / 32;
    const lane = cuda.laneId();

    const as: [*]addrspace(.shared) f16 = @ptrCast(&as_mem[0]);
    const bs: [*]addrspace(.shared) f16 = @ptrCast(&bs_mem[0]);

    // A tile, K-major: offset(m, kb) = (m/8)*256 + kb*128 + (m%8)*16 bytes.
    // 128 chunks of 8 f16, one per thread.
    {
        const m = tid / 2;
        const kb = tid % 2;
        const base = ((m / 8) * 256 + kb * 128 + (m % 8) * 16) / 2;
        for (0..8) |j| {
            const k = kb * 8 + @as(u32, @intCast(j));
            as[base + j] = @floatCast(aVal(m, k));
        }
    }
    // B tile, N-major: offset(k, nb) = nb*256 + k*16 bytes.
    // 32 chunks of 8 f16; threads 32..127 idle here.
    if (tid < 32) {
        const k = tid / 2;
        const nb = tid % 2;
        const base = (nb * 256 + k * 16) / 2;
        for (0..8) |j| {
            const n = nb * 8 + @as(u32, @intCast(j));
            bs[base + j] = @floatCast(bVal(k, n));
        }
    }
    cuda.syncThreads();

    var acc: wg.Acc64x16 = .{};
    const desc_a = wg.descriptor(wg.smemAddr(&as_mem[0]), 128, 256, .none);
    const desc_b = wg.descriptor(wg.smemAddr(&bs_mem[0]), 128, 256, .none);
    wg.fence();
    // scale_d = false: overwrite the accumulator, so the zero-init above is
    // not load-bearing and a stale-register bug would show up.
    wg.mmaAsyncM64N16K16(&acc, desc_a, desc_b, false, .k, .mn);
    wg.commitGroup();
    wg.waitGroup(0);

    var bad: u32 = 0;
    inline for (0..8) |i| {
        const p = wg.Acc64x16.coord(warp, lane, i);
        var expect: f32 = 0;
        for (0..16) |k| {
            expect += aVal(p.m, @intCast(k)) * bVal(@intCast(k), p.n);
        }
        const got = acc.get(i);
        if (got != expect) {
            bad += 1;
            mismatch_dump[tid * 16 + i * 2 + 0] = got;
            mismatch_dump[tid * 16 + i * 2 + 1] = expect;
        }
    }
    out[tid] = bad;
}

comptime {
    _ = cuda.Keep(.{&wgmmaSmoke}).__zoxide_keep_kernels;
}
