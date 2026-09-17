const cuda = @import("cuda");

const block_size = 256;

var tile: [block_size]f32 addrspace(.shared) = undefined;

/// Reverse an array within each block using shared memory.
/// Launch with block_size threads per block; in/out hold
/// gridDim.x * block_size elements.
pub fn sharedReverse(in: [*]const f32, out: [*]f32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const base = cuda.blockIdx().x * block_size;

    tile[tid] = in[base + tid];
    cuda.syncThreads();
    out[base + tid] = tile[block_size - 1 - tid];
}

comptime {
    _ = cuda.Keep(.{&sharedReverse}).__zoxide_keep_kernels;
}
