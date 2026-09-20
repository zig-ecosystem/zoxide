const cuda = @import("cuda");

/// Register-blocked SGEMM with bank-conflict-free B fragment reads.
/// Same tiling as sgemm_opt: 128x128 block tile, K-slice 16, 256 threads
/// (16x16), 8x8 register accumulators, 128-bit vectorized loads.
///
/// Bank-conflict analysis (bs_buf is [k][128] f32, bank = float_index % 32,
/// one warp = ty in {2w,2w+1} x tx 0..15):
///
/// BEFORE (sgemm_reg/opt): thread tx reads its 8 columns contiguously,
///   b_frag[ii] = bs[k*128 + tx*8 + ii].
///   Per ii step, the 16 tx values hit float indices {tx*8} = stride 32B,
///   banks (8*tx + ii) % 32 ∈ {0,8,16,24} + ii — only 4 distinct banks,
///   i.e. 4-way conflict (2-way after the hardware's v4 phasing).
///   An XOR swizzle keyed by the row k CANNOT fix this: all 16 threads read
///   the same row k, so a row-uniform permutation permutes nothing per read.
///   The conflict is intra-row and needs a different fragment->column map.
///
/// AFTER (this kernel): each thread's 8 columns are two float4 chunks at
///   colOf(tx, ii) = tx*4 + (ii & 3) + (ii >> 2) * 64
///   i.e. {4tx..4tx+3} ∪ {64+4tx..64+4tx+3}. Storage layout is unchanged
///   (identity), so vectorized stores still work. Per float4 phase the
///   hardware services 8 threads; their chunks cover float indices
///   4tx+j (tx 0..7, j 0..3) = banks 0..31 all distinct → conflict-free.
///
/// A fragment reads stay broadcast (all threads in a warp share the same
/// (ty,jj) address) and need no change.

pub const block_tile = 128;
pub const k_slice = 16;

var as_buf: [block_tile * k_slice]f32 addrspace(.shared) = undefined; // [r][k]
var bs_buf: [k_slice * block_tile]f32 addrspace(.shared) = undefined; // [k][c]

/// Logical C-column of fragment element ii for thread tx (see header).
inline fn colOf(tx: u32, ii: u32) u32 {
    return tx * 4 + (ii & 3) + ((ii >> 2) << 6);
}

fn loadTile(
    a: [*]const f32,
    b: [*]const f32,
    n: u32,
    block_row: u32,
    block_col: u32,
    k0: u32,
    tid: u32,
) void {
    const vec_ok = n % 4 == 0;
    var i: u32 = 0;
    while (i < 2) : (i += 1) {
        {
            const f4 = tid * 2 + i;
            const r = f4 / (k_slice / 4);
            const q = f4 % (k_slice / 4);
            const gr = block_row + r;
            const gk = k0 + q * 4;
            const dst = r * k_slice + q * 4;
            if (vec_ok and gr < n and gk + 4 <= n) {
                const src: *align(16) const @Vector(4, f32) = @ptrCast(@alignCast(a + gr * n + gk));
                const dstp: *align(16) addrspace(.shared) @Vector(4, f32) = @ptrCast(@alignCast(&as_buf[dst]));
                dstp.* = src.*;
            } else {
                var j: u32 = 0;
                while (j < 4) : (j += 1) {
                    as_buf[dst + j] = if (gr < n and gk + j < n) a[gr * n + gk + j] else 0;
                }
            }
        }
        {
            const f4 = tid * 2 + i;
            const k = f4 / (block_tile / 4);
            const c4 = f4 % (block_tile / 4);
            const gk = k0 + k;
            const gc = block_col + c4 * 4;
            const dst = k * block_tile + c4 * 4;
            if (vec_ok and gk < n and gc + 4 <= n) {
                const src: *align(16) const @Vector(4, f32) = @ptrCast(@alignCast(b + gk * n + gc));
                const dstp: *align(16) addrspace(.shared) @Vector(4, f32) = @ptrCast(@alignCast(&bs_buf[dst]));
                dstp.* = src.*;
            } else {
                var j: u32 = 0;
                while (j < 4) : (j += 1) {
                    bs_buf[dst + j] = if (gk < n and gc + j < n) b[gk * n + gc + j] else 0;
                }
            }
        }
    }
}

pub fn sgemmSwz(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const tx = cuda.threadIdx().x; // 0..15
    const ty = cuda.threadIdx().y; // 0..15
    const tid = ty * 16 + tx;

    const block_row = cuda.blockIdx().y * block_tile;
    const block_col = cuda.blockIdx().x * block_tile;

    var acc: [8][8]f32 = @splat(@splat(0));

    const ktiles = (n + k_slice - 1) / k_slice;
    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        loadTile(a, b, n, block_row, block_col, kt * k_slice, tid);
        cuda.syncThreads();

        var k: u32 = 0;
        while (k < k_slice) : (k += 1) {
            var a_frag: [8]f32 = undefined;
            var b_frag: [8]f32 = undefined;
            inline for (0..8) |jj| {
                a_frag[jj] = as_buf[(ty * 8 + jj) * k_slice + k];
            }
            // Two float4 chunks at tx*4 and 64+tx*4 (conflict-free phases).
            inline for (0..2) |g| {
                const src: *align(16) addrspace(.shared) const @Vector(4, f32) =
                    @ptrCast(@alignCast(&bs_buf[k * block_tile + tx * 4 + g * 64]));
                const v = src.*;
                inline for (0..4) |j| b_frag[g * 4 + j] = v[j];
            }
            inline for (0..8) |jj| {
                inline for (0..8) |ii| {
                    acc[jj][ii] = @mulAdd(f32, a_frag[jj], b_frag[ii], acc[jj][ii]);
                }
            }
        }
        cuda.syncThreads();
    }

    inline for (0..8) |jj| {
        const gr = block_row + ty * 8 + jj;
        if (gr < n) {
            inline for (0..8) |ii| {
                const gc = block_col + colOf(tx, ii);
                if (gc < n) {
                    c[gr * n + gc] = acc[jj][ii];
                }
            }
        }
    }
}

comptime {
    _ = cuda.Keep(.{&sgemmSwz}).__zoxide_keep_kernels;
}
