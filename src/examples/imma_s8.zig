const cuda = @import("cuda");
const api = @import("examples_abi");
const gen = cuda.gen;
const asmgen = cuda.asm_gen;

// cp.async.wait_group takes an immediate operand in LLVM (immarg); the
// generated wrapper's runtime i32 parameter cannot select. Wrap with a
// comptime parameter.
fn cpAsyncWaitGroup(comptime num: u32) void {
    gen.async_copy.cp_async_wait_group(num);
}

/// IMMA s8: int8 tensor-core GEMM, hgemm_bf16's shape with
/// mma.sync.m16n8k32.row.col.s32.s8.s8.s32 — s8 inputs, s32 accumulators.
///
/// K doubles to 32 per mma (4 packed s8 per .b32 operand register), so the
/// K-slice is 32 and the A tile is [128][32] s8 and the B tile [32][128] s8 —
/// 4 KB each, the same shared budget as the f16/bf16 variants.
///
/// Fragment loading is asymmetric, and the asymmetry is intrinsic to the
/// instruction, not an oversight:
///
///   A fragment (m16 x k32): lane l = 4*g + t wants row g, k bytes
///   4t..4t+3 — contiguous in the row-major [m][k] tile, exactly what
///   ldmatrix.x4 distributes (each lane gets one .b32 = 4 consecutive bytes
///   of a matrix row). Addressing is hgemm_bf16's verbatim: pitch 32B, the
///   four 8x8-b16 matrices are (m {0,8}) x (k {0,16}).
///
///   B fragment (k32 x n8): lane l wants column g, k bytes 4t..4t+3 —
///   four bytes a full row-pitch apart in the row-major [k][n] tile.
///   ldmatrix distributes 4 *consecutive* bytes per lane, and .trans swaps
///   which axis is consecutive but still hands each lane only 2 k-rows per
///   8x8 matrix (the f16 k16 layout), never the 4 contiguous k an s8 B
///   register needs. CUTLASS gets around this by permuting B at store time
///   into a crosswise layout; cp.async 16B chunks copy contiguous global
///   bytes, so that permutation is not available here. B fragments are
///   therefore gathered a byte at a time and packed by hand — correct first;
///   the permuted-store optimisation is future work and its absence is visible
///   in the PTX: 128 `ld.shared.b8` and no `ldmatrix.*.trans`, where
///   hgemm_bf16 has `ldmatrix.sync.aligned.m8n8.x2.trans`. That is 16 scalar
///   shared loads per warp per mma against bf16's one matrix load, so this
///   variant is expected to sit well below the int8 peak until B is permuted;
///   the measurement is the point, not the ratio.
///
/// cp.async pipeline: prologue issues tile 0; each iteration issues tile
/// kt+1 into the other buffer, waits until only the in-flight group remains
/// (cp.async.wait_group 1), syncs, computes, syncs. Both bar.sync calls are
/// on the universal path; the `has_next` guard is block-uniform.

pub const block_tile = 128;
pub const k_slice = 32;
pub const threads = 128;

const a_bytes = block_tile * k_slice; // 4 KB
const b_bytes = k_slice * block_tile; // 4 KB

var as_mem: [2][a_bytes]u8 addrspace(.shared) = undefined;
var bs_mem: [2][b_bytes]u8 addrspace(.shared) = undefined;

fn issueTileLoad(buf: usize, a: [*]const i8, b: [*]const i8, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A: 128 rows x 32B = 256 16B chunks; thread tid covers chunks tid*2, tid*2+1.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const r = chunk / 2;
        const half = chunk % 2;
        // Element offsets, not bytes: one 16 B cp.async chunk is 16 s8.
        const src = a + (@as(usize, block_row + r) * n + k0) + half * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][r * 32 + half * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B: 32 k-rows x 128B = 256 chunks.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 8;
        const part = chunk % 8;
        const src = b + (@as(usize, k0 + k) * n + block_col) + part * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][k * 128 + part * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    gen.async_copy.cp_async_commit_group();
}

fn loadAFrag(buf: usize, r0: u32, lane: u32) [4]u32 {
    const matrix = lane / 8;
    const row = lane % 8;
    const grow = r0 + row + (matrix % 2) * 8;
    const koff = (matrix / 2) * 8; // *2 bytes = {0, 16} B along the 32 B row
    const addr: [*]addrspace(.shared) u8 = @ptrCast(&as_mem[buf][grow * 32 + koff * 2]);
    const r = gen.matrix.ldmatrix_m8n8_x4_b16(addr);
    return .{ @bitCast(r.f0), @bitCast(r.f1), @bitCast(r.f2), @bitCast(r.f3) };
}

fn loadBFrag(buf: usize, c0: u32, lane: u32) [2]u32 {
    const g = lane / 4;
    const t = lane % 4;
    const col = c0 + g;
    var b0: u32 = 0;
    var b1: u32 = 0;
    inline for (0..4) |j| {
        // Column g, k bytes {4t+j} and {16+4t+j}; little-endian s8 packing.
        b0 |= @as(u32, bs_mem[buf][(4 * t + j) * 128 + col]) << (8 * j);
        b1 |= @as(u32, bs_mem[buf][(16 + 4 * t + j) * 128 + col]) << (8 * j);
    }
    return .{ b0, b1 };
}

fn computeTile(comptime buf: usize, acc: *[4][8][4]u32, wy: u32, wx: u32, lane: u32) void {
    // Load all A fragments (4) and B fragments (8) for this warp once.
    var af: [4][4]u32 = undefined;
    var bf: [8][2]u32 = undefined;
    inline for (0..4) |mi| {
        af[mi] = loadAFrag(buf, wy * 64 + @as(u32, mi * 16), lane);
    }
    inline for (0..8) |ni| {
        bf[ni] = loadBFrag(buf, wx * 64 + @as(u32, ni * 8), lane);
    }
    inline for (0..4) |mi| {
        inline for (0..8) |ni| {
            const d = asmgen.matrix.mma_m16n8k32_s32_s8(
                acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                af[mi][0],      af[mi][1],      af[mi][2],      af[mi][3],
                bf[ni][0],      bf[ni][1],
            );
            acc[mi][ni][0] = d.f0;
            acc[mi][ni][1] = d.f1;
            acc[mi][ni][2] = d.f2;
            acc[mi][ni][3] = d.f3;
        }
    }
}

pub fn immaS8(a: [*]const i8, b: [*]const i8, c: [*]i32, n: u32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const warp = tid / 32;
    const lane = cuda.laneId();
    const wy = warp / 2;
    const wx = warp % 2;
    const g = lane / 4;
    const t = lane % 4;

    const block_row = cuda.blockIdx().y * block_tile;
    const block_col = cuda.blockIdx().x * block_tile;

    var acc: [4][8][4]u32 = @splat(@splat(@splat(0)));

    const ktiles = n / k_slice;
    issueTileLoad(0, a, b, n, block_row, block_col, 0, tid);

    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        const has_next = kt + 1 < ktiles; // block-uniform
        if (has_next) {
            issueTileLoad((kt + 1) % 2, a, b, n, block_row, block_col, (kt + 1) * k_slice, tid);
        }
        if (has_next) cpAsyncWaitGroup(1) else cpAsyncWaitGroup(0);
        cuda.syncThreads();
        // Buffer parity is derived from kt rather than carried in a mutable
        // `cur: u1`; see hgemm_mma2 for why the toggle spilled.
        if (kt % 2 == 0) computeTile(0, &acc, wy, wx, lane) else computeTile(1, &acc, wy, wx, lane);
        cuda.syncThreads();
    }

    inline for (0..4) |mi| {
        inline for (0..8) |ni| {
            const r0 = block_row + wy * 64 + mi * 16;
            const c0 = block_col + wx * 64 + ni * 8;
            // s32 accumulators travel as u32 bit patterns; the store type is
            // where the sign comes back.
            c[@as(usize, r0 + g) * n + c0 + t * 2] = @bitCast(acc[mi][ni][0]);
            c[@as(usize, r0 + g) * n + c0 + t * 2 + 1] = @bitCast(acc[mi][ni][1]);
            c[@as(usize, r0 + g + 8) * n + c0 + t * 2] = @bitCast(acc[mi][ni][2]);
            c[@as(usize, r0 + g + 8) * n + c0 + t * 2 + 1] = @bitCast(acc[mi][ni][3]);
        }
    }
}

comptime {
    // Drift from the signature `zoxide bench` launches through is a compile
    // error here rather than a silently mis-packed argument list.
    cuda.abi.assertMatches(api.imma_s8, @TypeOf(immaS8));
    _ = cuda.Keep(.{&immaS8}).__zoxide_keep_kernels;
}
