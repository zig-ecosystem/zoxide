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

/// HGEMM v2: tensor-core mma.sync.m16n8k16 + ldmatrix fragment loads +
/// cp.async double-buffered tile pipeline.
///
/// Block tile 128x128, K-slice 16, 128 threads = 4 warps (2x2), warp tile
/// 64x64 = 4(m) x 8(n) mma tiles of 16x8. Requires N % 16 == 0 (bench
/// harness enforces); global loads are 16B-aligned because N % 16 == 0 makes
/// every row start 32B-aligned and k0 is a multiple of 16 f16 = 32B.
///
/// Fragment loading with ldmatrix (PTX ISA: lane i supplies a row address):
///   A tile in shared: [128][16] f16, row pitch 32B. A 16x16 fragment =
///   four 8x8 matrices m0..m3 = (rows r0+{0,8}) x (k {0,8}); lane i addresses
///   matrix i/8 row i%8:
///     addr = base + (r0 + i%8 + (i/8 % 2)*8)*32 + (i/16)*16
///   One ldmatrix.x4 yields a0..a3 for one 16x16 tile.
///   B tile in shared: [16][128] f16 (k rows, 256B pitch). The mma B operand
///   wants column-pair fragments, i.e. a transposed 8x8 read: ldmatrix.x2
///   with .trans on the k-major tile; lanes 0..15 address matrix i/8 row i%8:
///     addr = base + ((i/8)*8 + i%8)*256 + c0*2
///
/// cp.async pipeline: prologue issues tile 0; each iteration issues tile
/// kt+1 into the other buffer, waits until only the in-flight group remains
/// (cp.async.wait_group 1), syncs, computes, syncs. Both bar.sync calls are
/// on the universal path; the `has_next` guard is block-uniform.
///
/// Register budget: acc 4x8x4 = 128 f32 + A frags 16 u32 + B frags 16 u32 +
/// addresses/loop ~30 → ~190, under the 255 cap.

pub const block_tile = 128;
pub const k_slice = 16;
pub const threads = 128;

const a_bytes = block_tile * k_slice * 2; // 4 KB
const b_bytes = k_slice * block_tile * 2; // 4 KB

var as_mem: [2][a_bytes]u8 addrspace(.shared) = undefined;
var bs_mem: [2][b_bytes]u8 addrspace(.shared) = undefined;

fn issueTileLoad(buf: usize, a: [*]const f16, b: [*]const f16, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A: 128 rows x 32B = 256 16B chunks; thread tid covers chunks tid*2, tid*2+1.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const r = chunk / 2;
        const half = chunk % 2;
        // Element offsets, not bytes: one 16 B cp.async chunk is 8 f16.
        const src = a + (@as(usize, block_row + r) * n + k0) + half * 8;
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][r * 32 + half * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B: 16 k-rows x 256B = 256 chunks.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 16;
        const part = chunk % 16;
        const src = b + (@as(usize, k0 + k) * n + block_col) + part * 8;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][k * 256 + part * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    gen.async_copy.cp_async_commit_group();
}

fn loadAFrag(buf: usize, r0: u32, lane: u32) [4]u32 {
    const matrix = lane / 8;
    const row = lane % 8;
    const grow = r0 + row + (matrix % 2) * 8;
    const koff = (matrix / 2) * 8;
    const addr: [*]addrspace(.shared) u8 = @ptrCast(&as_mem[buf][grow * 32 + koff * 2]);
    const r = gen.matrix.ldmatrix_m8n8_x4_b16(addr);
    return .{ @bitCast(r.f0), @bitCast(r.f1), @bitCast(r.f2), @bitCast(r.f3) };
}

fn loadBFrag(buf: usize, c0: u32, lane: u32) [2]u32 {
    const matrix = lane / 8; // 0,1 (lanes 16-31 addresses ignored for x2)
    const row = lane % 8;
    const addr: [*]addrspace(.shared) u8 = @ptrCast(&bs_mem[buf][(matrix * 8 + row) * 256 + c0 * 2]);
    const r = gen.matrix.ldmatrix_m8n8_x2_trans_b16(addr);
    return .{ @bitCast(r.f0), @bitCast(r.f1) };
}

fn computeTile(comptime buf: usize, acc: *[4][8][4]f32, wy: u32, wx: u32, lane: u32) void {
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
            const d = asmgen.matrix.mma_m16n8k16_f32_f16(
                acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                af[mi][0],     af[mi][1],     af[mi][2],     af[mi][3],
                bf[ni][0],     bf[ni][1],
            );
            acc[mi][ni][0] = d.f0;
            acc[mi][ni][1] = d.f1;
            acc[mi][ni][2] = d.f2;
            acc[mi][ni][3] = d.f3;
        }
    }
}

pub fn hgemmMma2(a: [*]const f16, b: [*]const f16, c: [*]f32, n: u32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const warp = tid / 32;
    const lane = cuda.laneId();
    const wy = warp / 2;
    const wx = warp % 2;
    const g = lane / 4;
    const t = lane % 4;

    const block_row = cuda.blockIdx().y * block_tile;
    const block_col = cuda.blockIdx().x * block_tile;

    var acc: [4][8][4]f32 = @splat(@splat(@splat(0)));

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
        // `cur: u1`. The toggle made LLVM keep it in a 1-byte __local_depot and
        // store to it every iteration — reported by the driver as 8 B/thread
        // spilled. Deriving it strength-reduces to a rotating counter with no
        // local memory at all, the same shape hgemm_wgmma2/3 use.
        if (kt % 2 == 0) computeTile(0, &acc, wy, wx, lane) else computeTile(1, &acc, wy, wx, lane);
        cuda.syncThreads();
    }

    inline for (0..4) |mi| {
        inline for (0..8) |ni| {
            const r0 = block_row + wy * 64 + mi * 16;
            const c0 = block_col + wx * 64 + ni * 8;
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
    cuda.abi.assertMatches(api.hgemm, @TypeOf(hgemmMma2));
    _ = cuda.Keep(.{&hgemmMma2}).__zoxide_keep_kernels;
}
