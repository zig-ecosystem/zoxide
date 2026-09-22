const cuda = @import("cuda");
const api = @import("examples_abi");

/// Register-blocked SGEMM, optimization round 2: vectorized 128-bit global
/// loads + K-slice 16 (vs sgemm_reg: scalar loads, K-slice 8).
///
/// Layout: block tile 128x128, K-slice 16, 256 threads (16x16), each thread
/// accumulates an 8x8 sub-block in registers (see sgemm_reg.zig).
///
/// Shared tiles (both chosen so vector stores land on consecutive addresses):
///   as[r][k]  A tile, row-major 128x16: as_buf[r * 16 + k]
///   bs[k][c]  B tile, k-major 16x128:   bs_buf[k * 128 + c]
///
/// Loads: A slice 128x16 = 2048 f32 = 512 float4, so each of the 256 threads
/// loads 2 float4 for A and 2 for B. A float4 covers 4 consecutive k of one
/// row r; B float4 covers 4 consecutive c of one k row. A full float4 load
/// requires 16-byte alignment and 4 in-range columns: global offset
/// row*n + k0 + q*4 is 16B-aligned iff n % 4 == 0 (k0 % 16 == 0, q*4 f32).
/// When that fails (n % 4 != 0, or the row/tail is out of range) the thread
/// falls back to 4 guarded scalar loads. No barrier sits inside either path,
/// so divergence here is safe.
///
/// Bank conflicts: A reads are k-uniform per warp (broadcast). B reads use
/// stride tx*8 floats across the warp → up to 4-way conflicts; accepted for
/// this round (the alternative is padding or float4 fragment loads, noted in
/// README as follow-up).

pub const block_tile = 128;
pub const k_slice = 16;
pub const threads = 256; // 16 x 16

var as_buf: [block_tile * k_slice]f32 addrspace(.shared) = undefined; // [r][k]
var bs_buf: [k_slice * block_tile]f32 addrspace(.shared) = undefined; // [k][c]

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
        // A: 512 float4, f4 index = tid*2 + i; row r = f4 / 4, q = f4 % 4.
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
        // B: 512 float4; k = f4 / 32, c4 = f4 % 32 (32 float4 per 128-wide row).
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

pub fn sgemmOpt(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const tx = cuda.threadIdx().x; // 0..15
    const ty = cuda.threadIdx().y; // 0..15
    const tid = ty * 16 + tx; // 0..255

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
            inline for (0..2) |g| {
                // b_frag as two float4 shared loads (aligned: tx*8 + g*4 f32).
                const src: *align(16) addrspace(.shared) const @Vector(4, f32) =
                    @ptrCast(@alignCast(&bs_buf[k * block_tile + tx * 8 + g * 4]));
                const v = src.*;
                b_frag[g * 4 + 0] = v[0];
                b_frag[g * 4 + 1] = v[1];
                b_frag[g * 4 + 2] = v[2];
                b_frag[g * 4 + 3] = v[3];
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
        inline for (0..8) |ii| {
            const gc = block_col + tx * 8 + ii;
            if (gr < n and gc < n) {
                c[gr * n + gc] = acc[jj][ii];
            }
        }
    }
}

comptime {
    // Drift from the signature `zoxide bench` launches through is a compile
    // error here rather than a silently mis-packed argument list.
    cuda.abi.assertMatches(api.sgemm, @TypeOf(sgemmOpt));
    _ = cuda.Keep(.{&sgemmOpt}).__zoxide_keep_kernels;
}
