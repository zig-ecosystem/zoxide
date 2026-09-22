const cuda = @import("cuda");
const api = @import("examples_abi");

/// Register-blocked SGEMM: C = A * B, N x N row-major f32.
///
/// Layout: block tile 128x128, K-slice 8. 256 threads per block arranged as
/// a 16x16 grid; thread (ty, tx) accumulates the 8x8 sub-block
///   rows [ty*8, ty*8+8), cols [tx*8, tx*8+8)
/// of the block tile, in a 64-element register accumulator.
///
/// Shared tiles: as[k][r] = A[blockRow + r][k0 + k]  (r in 0..128, k in 0..8)
///               bs[k][c] = B[k0 + k][blockCol + c]
/// i.e. both stored row-major-by-k so the inner loop reads columns of A and
/// rows of B contiguously. Each of the 256 threads loads 4 A elements and
/// 4 B elements per K-slice: flat index f = tid*4 + i, A: r = f / 8, k = f
/// % 8; B: k = f / 128, c = f % 128.
///
/// Barrier discipline: tile loop bound ktiles is block-uniform; loads use
/// branchless zero-fill; both syncThreads() are on the universal path.
/// Accumulation uses @mulAdd, which lowers to fma.rn.f32 (verified).

pub const block_tile = 128;
pub const k_slice = 8;
pub const thread_rows = 8;
pub const thread_cols = 8;
pub const threads_x = 16; // blockDim.x
pub const threads_y = 16; // blockDim.y

var as_buf: [k_slice * block_tile]f32 addrspace(.shared) = undefined;
var bs_buf: [k_slice * block_tile]f32 addrspace(.shared) = undefined;

pub fn sgemmReg(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const tx = cuda.threadIdx().x; // 0..15
    const ty = cuda.threadIdx().y; // 0..15
    const tid = ty * threads_x + tx; // 0..255

    const block_row = cuda.blockIdx().y * block_tile;
    const block_col = cuda.blockIdx().x * block_tile;

    var acc: [thread_rows][thread_cols]f32 = @splat(@splat(0));

    const ktiles = (n + k_slice - 1) / k_slice;
    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        const k0 = kt * k_slice;
        // Load A slice (128x8) and B slice (8x128); zero-fill out of range.
        var i: u32 = 0;
        while (i < 4) : (i += 1) {
            const f = tid * 4 + i; // 0..1024
            // A: r = f / 8, k = f % 8
            const ar = block_row + f / k_slice;
            const ak = k0 + f % k_slice;
            as_buf[(f % k_slice) * block_tile + (f / k_slice)] =
                if (ar < n and ak < n) a[ar * n + ak] else 0;
            // B: k = f / 128, c = f % 128
            const bk = k0 + f / block_tile;
            const bc = block_col + f % block_tile;
            bs_buf[(f / block_tile) * block_tile + (f % block_tile)] =
                if (bk < n and bc < n) b[bk * n + bc] else 0;
        }
        cuda.syncThreads();

        var k: u32 = 0;
        while (k < k_slice) : (k += 1) {
            var a_frag: [thread_rows]f32 = undefined;
            var b_frag: [thread_cols]f32 = undefined;
            inline for (0..thread_rows) |jj| {
                a_frag[jj] = as_buf[k * block_tile + ty * thread_rows + jj];
            }
            inline for (0..thread_cols) |ii| {
                b_frag[ii] = bs_buf[k * block_tile + tx * thread_cols + ii];
            }
            inline for (0..thread_rows) |jj| {
                inline for (0..thread_cols) |ii| {
                    acc[jj][ii] = @mulAdd(f32, a_frag[jj], b_frag[ii], acc[jj][ii]);
                }
            }
        }
        cuda.syncThreads();
    }

    inline for (0..thread_rows) |jj| {
        const gr = block_row + ty * thread_rows + jj;
        inline for (0..thread_cols) |ii| {
            const gc = block_col + tx * thread_cols + ii;
            if (gr < n and gc < n) {
                c[gr * n + gc] = acc[jj][ii];
            }
        }
    }
}

comptime {
    // Drift from the signature `zoxide bench` launches through is a compile
    // error here rather than a silently mis-packed argument list.
    cuda.abi.assertMatches(api.sgemm, @TypeOf(sgemmReg));
    _ = cuda.Keep(.{&sgemmReg}).__zoxide_keep_kernels;
}
