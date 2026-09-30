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

/// IMMA s4: int4 tensor-core GEMM, imma_s8's shape with
/// mma.sync.m16n8k64.row.col.s32.s4.s4.s32 — s4 inputs, s32 accumulators.
///
/// s4 has no byte type: values travel two per u8 (low nibble = even k; the
/// convention is spelled once in examples_abi.packS4 and the kernel never
/// repacks — cp.async, ldmatrix and the mma all treat the data as opaque
/// bytes/words). K doubles again to 64 per mma (8 packed s4 per .b32 operand
/// register), so the A tile is [128][64] s4 = 32 B per row and the B tile is
/// [64][128] s4 = 64 B per row — 4 KB each, the same shared budget as s8.
///
/// Fragment loading keeps imma_s8's asymmetry, and for A it works out even
/// better:
///
///   A fragment (m16 x k64): lane l = 4*g + t wants row g, k values
///   8t..8t+7 — 4 *consecutive packed bytes* of the row-major [m][k] tile,
///   exactly what ldmatrix.x4 distributes. The addressing is imma_s8's
///   verbatim, pitch 32B: the four 8x8-b16 matrices are (m {0,8}) x
///   (byte-k {0,16}) = value-k {0,32}.
///
///   B fragment (k64 x n8): lane l wants column g, k values 8t..8t+7 —
///   eight values a full row-pitch apart, and half of them in the *other*
///   nibble of their byte. ldmatrix's distribution (4 consecutive bytes per
///   lane, or 2 rows per 8x8 matrix with .trans) cannot produce 8
///   k-contiguous values of one column, the same wall imma_s8's B path hit.
///   B is therefore gathered a byte at a time and the wanted nibble shifted
///   into place — 8 ld.shared.b8 per register, 128 per warp per mma stage.
///   The permuted-store fix is the same future work as for s8, and its
///   absence is equally visible in the PTX (no ldmatrix.*.trans).
///
/// cp.async pipeline: prologue issues tile 0; each iteration issues tile
/// kt+1 into the other buffer, waits until only the in-flight group remains
/// (cp.async.wait_group 1), syncs, computes, syncs. Both bar.sync calls are
/// on the universal path; the `has_next` guard is block-uniform.

pub const block_tile = 128;
pub const k_slice = 64;
pub const threads = 128;

const a_bytes = block_tile * k_slice / 2; // 4 KB
const b_bytes = k_slice * block_tile / 2; // 4 KB

var as_mem: [2][a_bytes]u8 addrspace(.shared) = undefined;
var bs_mem: [2][b_bytes]u8 addrspace(.shared) = undefined;

fn issueTileLoad(buf: usize, a: [*]const u8, b: [*]const u8, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    const row_bytes = n / 2; // two s4 per byte; n % 128 == 0 is bench-enforced
    const k0_bytes = k0 / 2;
    // A: 128 rows x 32B = 256 16B chunks; thread tid covers chunks tid*2, tid*2+1.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const r = chunk / 2;
        const half = chunk % 2;
        const src = a + (@as(usize, block_row + r) * row_bytes + k0_bytes) + half * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][r * 32 + half * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B: 64 k-rows x 64B = 256 chunks.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 4;
        const part = chunk % 4;
        const src = b + (@as(usize, k0 + k) * row_bytes + block_col / 2) + part * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][k * 64 + part * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    gen.async_copy.cp_async_commit_group();
}

fn loadAFrag(buf: usize, r0: u32, lane: u32) [4]u32 {
    const matrix = lane / 8;
    const row = lane % 8;
    const grow = r0 + row + (matrix % 2) * 8;
    const koff = (matrix / 2) * 8; // *2 bytes = {0, 16} B = value-k {0, 32}
    const addr: [*]addrspace(.shared) u8 = @ptrCast(&as_mem[buf][grow * 32 + koff * 2]);
    const r = gen.matrix.ldmatrix_m8n8_x4_b16(addr);
    return .{ @bitCast(r.f0), @bitCast(r.f1), @bitCast(r.f2), @bitCast(r.f3) };
}

fn loadBFrag(buf: usize, c0: u32, lane: u32) [2]u32 {
    const g = lane / 4;
    const t = lane % 4;
    // Column g lives in byte c0/2 + g/2, nibble selected by g's parity.
    const byte_col = c0 / 2 + g / 2;
    const nib_shift: u3 = @intCast((g % 2) * 4);
    var b0: u32 = 0;
    var b1: u32 = 0;
    inline for (0..8) |j| {
        // Column g, k values {8t+j} and {32+8t+j}; value j goes to bits 4j.
        b0 |= @as(u32, (bs_mem[buf][(8 * t + j) * 64 + byte_col] >> nib_shift) & 0xF) << (4 * j);
        b1 |= @as(u32, (bs_mem[buf][(32 + 8 * t + j) * 64 + byte_col] >> nib_shift) & 0xF) << (4 * j);
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
            const d = asmgen.matrix.mma_m16n8k64_s32_s4(
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

pub fn immaS4(a: [*]const u8, b: [*]const u8, c: [*]i32, n: u32) callconv(.kernel) void {
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
    cuda.abi.assertMatches(api.imma_s4, @TypeOf(immaS4));
    _ = cuda.Keep(.{&immaS4}).__zoxide_keep_kernels;
}
