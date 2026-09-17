const cuda = @import("cuda");

const block_size = 256;
const num_warps = block_size / cuda.warp_size;

var warp_sums: [num_warps]f32 addrspace(.shared) = undefined;

/// Block-wide sum reduction: warp shuffle first, then shared memory to
/// merge across warps. out[blockIdx.x] = sum of in[block] elements.
pub fn warpReduce(in: [*]const f32, out: [*]f32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const base = cuda.blockIdx().x * block_size;

    var sum = in[base + tid];

    // Intra-warp reduction via shuffle-down (16, 8, 4, 2, 1).
    var offset: u32 = cuda.warp_size / 2;
    while (offset > 0) : (offset >>= 1) {
        sum += cuda.shflDownSync(f32, 0xffffffff, sum, offset);
    }

    // Lane 0 of each warp publishes its partial sum.
    if (cuda.laneId() == 0) {
        warp_sums[cuda.warpId()] = sum;
    }
    cuda.syncThreads();

    // First warp reduces the per-warp partials.
    if (cuda.warpId() == 0) {
        sum = if (tid < num_warps) warp_sums[tid] else 0.0;
        offset = num_warps / 2;
        while (offset > 0) : (offset >>= 1) {
            sum += cuda.shflDownSync(f32, 0xffffffff, sum, offset);
        }
        if (tid == 0) {
            out[cuda.blockIdx().x] = sum;
        }
    }
}

comptime {
    _ = cuda.Keep(.{&warpReduce}).__zoxide_keep_kernels;
}
