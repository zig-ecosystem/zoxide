const cuda = @import("cuda");

var block_count: [1]u32 addrspace(.shared) = undefined;

/// Every thread bumps a global counter, a per-block shared counter, and a
/// histogram bin chosen from its own thread id — exercising atomicAdd on
/// global memory (u32, f32) and on shared memory.
pub fn atomicCounter(
    global_count: *u32,
    float_sum: *f32,
    histogram: [*]u32,
) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;

    if (tid == 0) block_count[0] = 0;
    cuda.syncThreads();

    _ = cuda.atomicAdd(u32, global_count, 1);
    _ = cuda.atomicAdd(f32, float_sum, 1.0);
    _ = cuda.atomicAdd(u32, &histogram[tid % 16], 1);
    _ = cuda.atomicAdd(u32, &block_count[0], 1);
    cuda.syncThreads();

    if (tid == 0) {
        // Fold the shared per-block count into histogram bin 0.
        _ = cuda.atomicAdd(u32, &histogram[0], block_count[0]);
    }
}

comptime {
    _ = cuda.Keep(.{&atomicCounter}).__zoxide_keep_kernels;
}
