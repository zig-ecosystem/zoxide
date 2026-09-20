const cuda = @import("cuda");

pub const tile = 32;

var as_buf: [tile * tile]f32 addrspace(.shared) = undefined;
var bs_buf: [tile * tile]f32 addrspace(.shared) = undefined;

/// Tiled SGEMM: 32x32 shared-memory tiles, one 32x32 thread block computes
/// one 32x32 C tile. C = A * B, all N x N row-major f32.
///
/// Convergence rules respected: the tile loop bound (ktiles) is uniform
/// across the block; bounds handling is branchless (predicated select to 0);
/// both syncThreads() calls are on the universal path inside the loop.
pub fn sgemmTiled(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const tx = cuda.threadIdx().x; // 0..31
    const ty = cuda.threadIdx().y; // 0..31
    const bx = cuda.blockIdx().x;
    const by = cuda.blockIdx().y;

    const row = by * tile + ty; // global row of C
    const col = bx * tile + tx; // global col of C

    const ktiles = (n + tile - 1) / tile;
    var acc: f32 = 0;

    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        const a_col = kt * tile + tx;
        const b_row = kt * tile + ty;
        // Zero-fill out-of-range tile elements so the inner product loop
        // stays unconditional (and barrier-safe).
        as_buf[ty * tile + tx] = if (row < n and a_col < n) a[row * n + a_col] else 0;
        bs_buf[ty * tile + tx] = if (b_row < n and col < n) b[b_row * n + col] else 0;
        cuda.syncThreads();

        var k: u32 = 0;
        while (k < tile) : (k += 1) {
            acc += as_buf[ty * tile + k] * bs_buf[k * tile + tx];
        }
        cuda.syncThreads();
    }

    if (row < n and col < n) {
        c[row * n + col] = acc;
    }
}

comptime {
    _ = cuda.Keep(.{&sgemmTiled}).__zoxide_keep_kernels;
}
