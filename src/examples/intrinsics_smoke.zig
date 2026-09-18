const cuda = @import("cuda");
const gen = cuda.gen;

var smem: [16]u8 addrspace(.shared) = undefined;

/// Smoke test for generated NVVM bindings (src/cuda/gen/intrinsics.zig):
/// exercises vote, shuffle, match, redux, math, cp.async and sreg wrappers.
/// out layout per thread block (block = 32 threads):
///   out[0] = ballot of (x > 0)
///   out[1] = any_sync(x > 1)
///   out[2] = shfl_down_sync(y, 1)
///   out[3] = match_any_sync(y)
///   out[4] = redux_sync_add(y)
///   out[5] = fma_rn(x, 2, 1)
///   out[6] = sqrt_rn(x)
///   out[7] = smem[0] after cp.async of global[0..16]
///   out[8] = block_idx_x
///   out[9] = laneid via sreg
pub fn intrinsicsSmoke(x: f32, y: i32, g: [*]addrspace(.global) const u8, out: [*]i32) callconv(.kernel) void {
    out[0] = @bitCast(gen.warp.ballot_sync(@bitCast(@as(u32, 0xffffffff)), x > 0));
    out[1] = @intFromBool(gen.warp.any_sync(@bitCast(@as(u32, 0xffffffff)), x > 1));
    out[2] = gen.warp.shuffle_down_sync(@bitCast(@as(u32, 0xffffffff)), y, 1, 31);
    out[3] = gen.warp.match_any_sync(@bitCast(@as(u32, 0xffffffff)), y);
    out[4] = gen.warp.redux_sync_add(y, @bitCast(@as(u32, 0xffffffff)));
    const fa = gen.float.fma_rn_f32(x, 2.0, 1.0);
    const fb = gen.float.sqrt_rn_f32(x);
    out[5] = @intFromFloat(fa);
    out[6] = @intFromFloat(fb);

    if (cuda.threadIdx().x == 0) {
        gen.async_copy.cp_async_ca_16(&smem, g);
        gen.async_copy.cp_async_commit_group();
        gen.async_copy.cp_async_wait_all();
    }
    cuda.syncThreads();
    out[7] = smem[0];

    out[8] = gen.sreg.block_idx_x();
    out[9] = gen.sreg.thread_idx_x();
}

comptime {
    _ = cuda.Keep(.{&intrinsicsSmoke}).__zoxide_keep_kernels;
}
