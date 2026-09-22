const cuda = @import("cuda");
const api = @import("examples_abi");
const asmgen = cuda.asm_gen;

/// HGEMM via tensor-core mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.
/// C = A * B: A f16 (M x K), B f16 (K x N), C f32 (M x N), square N x N.
/// Requires N % 16 == 0 (enforced by the bench harness).
///
/// Structure: block tile 64x64, 128 threads = 4 warps in a 2x2 grid.
/// Each warp computes a 32x32 sub-tile as a 2x4 grid of mma tiles (16x8).
/// K advances in slices of 16.
///
/// Fragment mapping (PTX ISA, m16n8k16, row.col, f16 in / f32 out).
/// lane = g*4 + t  (g = lane/4 is the "group", t = lane%4 the "thread in group"):
///   A fragment (16x16 f16 = 4 u32 regs, 2 f16 each, A row-major [r][k]):
///     a0,a1 = A[g    ][t*2 .. t*2+2)      -> u32 as_u32[g][t]
///     a2,a3 = A[g + 8][t*2 .. t*2+2)      -> u32 as_u32[g+8][t]
///     a4,a5 = A[g    ][t*2+8 .. t*2+10)   -> u32 as_u32[g][t+4]
///     a6,a7 = A[g + 8][t*2+8 .. t*2+10)   -> u32 as_u32[g+8][t+4]
///   B fragment (16x8 f16 = 2 u32 regs, B col-major pairs along k):
///     b0,b1 = B[t*2][c], B[t*2+1][c]      (c = g) -> packed u32
///     b2,b3 = B[t*2+8][c], B[t*2+9][c]
///     (we store B transposed in shared as [c][k] so each pair is one u32
///      at bs_u32[c][t] and bs_u32[c][t+4])
///   C/D fragment (16x8 f32 = 4 regs):
///     d0,d1 = D[g][t*2], D[g][t*2+1]
///     d2,d3 = D[g+8][t*2], D[g+8][t*2+1]
///
/// Shared per K-slice: A tile 64x16 f16 (as u32[64][8]), B tile 16x64 f16
/// transposed (as u32[64][8]). Loads: 512 u32 each over 128 threads = 4
/// u32/thread, 4B-aligned (N % 16 == 0). Both bar.sync on the uniform path.

pub const block_tile = 64;
pub const k_slice = 16;

var as_buf: [block_tile][k_slice / 2]u32 addrspace(.shared) = undefined; // [row][k-pair]
var bs_buf: [block_tile][k_slice / 2]u32 addrspace(.shared) = undefined; // [col][k-pair] (B transposed)

fn loadTiles(a: [*]const f16, b: [*]const f16, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A: 64 rows x 16 k = 512 u32 (2 f16). tid covers 4 u32.
    inline for (0..4) |i| {
        const idx = tid * 4 + i; // 0..511
        const r = idx / (k_slice / 2);
        const kp = idx % (k_slice / 2);
        const src: *align(4) const u32 = @ptrCast(@alignCast(a + (@as(usize, block_row + r) * n + k0 + kp * 2)));
        as_buf[r][kp] = src.*;
    }
    // B: 16 k x 64 cols, stored transposed [c][k]. 512 u32: idx -> c, kp.
    inline for (0..4) |i| {
        const idx = tid * 4 + i;
        const c = idx / (k_slice / 2);
        const kp = idx % (k_slice / 2);
        // two f16: B[k0 + kp*2][block_col + c], B[k0 + kp*2 + 1][block_col + c]
        // Packed as raw bit patterns, which is what the mma fragment wants; the
        // f16 pointer type is only about agreeing with the host on element size.
        const lo: u16 = @bitCast(b[@as(usize, k0 + kp * 2) * n + block_col + c]);
        const hi: u16 = @bitCast(b[@as(usize, k0 + kp * 2 + 1) * n + block_col + c]);
        bs_buf[c][kp] = @as(u32, lo) | (@as(u32, hi) << 16);
    }
}

pub fn hgemmMma(a: [*]const f16, b: [*]const f16, c: [*]f32, n: u32) callconv(.kernel) void {
    const warp = cuda.threadIdx().x / 32; // 0..3
    const lane = cuda.laneId();
    const wy = warp / 2; // 0..1
    const wx = warp % 2; // 0..1
    const g = lane / 4;
    const t = lane % 4;

    const block_row = cuda.blockIdx().y * block_tile;
    const block_col = cuda.blockIdx().x * block_tile;

    // 2x4 grid of mma tiles per warp; 4 f32 accumulators each.
    var acc: [2][4][4]f32 = @splat(@splat(@splat(0)));

    const ktiles = n / k_slice; // n % 16 == 0
    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        loadTiles(a, b, n, block_row, block_col, kt * k_slice, cuda.threadIdx().x);
        cuda.syncThreads();

        inline for (0..2) |mi| {
            inline for (0..4) |ni| {
                const r0 = wy * 32 + mi * 16;
                const c0 = wx * 32 + ni * 8;
                const a0 = as_buf[r0 + g][t];
                const a1 = as_buf[r0 + g + 8][t];
                const a2 = as_buf[r0 + g][t + 4];
                const a3 = as_buf[r0 + g + 8][t + 4];
                const b0 = bs_buf[c0 + g][t];
                const b1 = bs_buf[c0 + g][t + 4];
                const d = asmgen.matrix.mma_m16n8k16_f32_f16(
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                    a0, a1, a2, a3, b0, b1,
                );
                acc[mi][ni][0] = d.f0;
                acc[mi][ni][1] = d.f1;
                acc[mi][ni][2] = d.f2;
                acc[mi][ni][3] = d.f3;
            }
        }
        cuda.syncThreads();
    }

    inline for (0..2) |mi| {
        inline for (0..4) |ni| {
            const r0 = block_row + wy * 32 + mi * 16;
            const c0 = block_col + wx * 32 + ni * 8;
            c[@as(usize, r0 + g) * n + c0 + t * 2] = acc[mi][ni][0];
            c[@as(usize, r0 + g) * n + c0 + t * 2 + 1] = acc[mi][ni][1];
            c[@as(usize, r0 + g + 8) * n + c0 + t * 2] = acc[mi][ni][2];
            c[@as(usize, r0 + g + 8) * n + c0 + t * 2 + 1] = acc[mi][ni][3];
        }
    }
}

comptime {
    // Drift from the signature `zoxide bench` launches through is a compile
    // error here rather than a silently mis-packed argument list.
    cuda.abi.assertMatches(api.hgemm, @TypeOf(hgemmMma));
    _ = cuda.Keep(.{&hgemmMma}).__zoxide_keep_kernels;
}
