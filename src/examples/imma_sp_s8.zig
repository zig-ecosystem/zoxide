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

/// Sparse IMMA s8: 2:4-structured-sparse int8 GEMM on
/// mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 —
/// hgemm_sp's sparsity contract crossed with imma_s8's s8 fragment handling.
///
/// Fragment reading (PTX ISA 9.7.16.6.2.5, recorded so the exact-match bench
/// can falsify it on first GPU run):
///
///   A (sparse, k32): 2 .b32 regs per lane. Lane l = 4*g + t covers the
///   8-wide dense chunk k = 8t..8t+7; a0 = the 4 kept values of row g's
///   chunk, a1 = row g+8's, packed low-to-high in chunk order. The pruned
///   row is 16 s8 = 16 B, so a lane's 4 bytes are consecutive —
///   ldmatrix.x2 over the [16][16B] tile distributes exactly this (same
///   byte shape as hgemm_sp's A path).
///
///   B (dense): identical to dense imma_s8 — b0 = (k 4t..4t+3, col g),
///   b1 = (k 16+4t..16+4t+3, col g) — and equally unable to use ldmatrix,
///   so the manual 4-byte gather comes over unchanged.
///
///   Metadata: one u32 per *row* per mma (16 2-bit index pairs, k-chunk j
///   owning nibble [4j+3:4j], low 2 bits = first kept index). The m16n8k32
///   selector names a thread PAIR, not a single thread: selector 0 = lanes
///   t==0 and t==1 contribute. Read as: t==0 carries row g's word, t==1
///   row g+8's. This assignment is the least-certain part of the reading;
///   the host packs both candidate rows and the exact check fails loudly
///   if it is swapped.
///
/// Pipeline: hgemm_sp's — 128x128 tile, dense k-slice 32, cp.async
/// double-buffered (A pruned 2 KB + meta 512 B + B dense 4 KB per stage),
/// both bar.sync on the universal path.

pub const block_tile = 128;
pub const k_slice = 32; // dense k per stage; pruned A carries 16 per row
pub const threads = 128;

const a_bytes = block_tile * k_slice / 2; // 2 KB
const b_bytes = k_slice * block_tile; // 4 KB
const meta_bytes = block_tile * 4; // 512 B, one u32 per row

var as_mem: [2][a_bytes]u8 addrspace(.shared) = undefined;
var bs_mem: [2][b_bytes]u8 addrspace(.shared) = undefined;
var ms_mem: [2][meta_bytes]u8 addrspace(.shared) = undefined;

fn issueTileLoad(buf: usize, a: [*]const i8, b: [*]const i8, meta: [*]const u32, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A pruned: 128 rows x 16B = 128 chunks, one per thread. A row holds
    // n/2 kept s8; k0 dense = k0/2 kept.
    {
        const src = a + (@as(usize, block_row + tid) * (n / 2) + k0 / 2);
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][tid * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // Metadata: 128 rows x 4B = 512B = 32 chunks; threads 0..31 load 16B
    // (4 rows' words) each. A row holds n/32 metadata words; k0 dense =
    // k0/32 words.
    if (tid < 32) {
        const src = meta + (@as(usize, block_row + tid * 4) * (n / 32) + k0 / 32);
        gen.async_copy.cp_async_cg_16(@ptrCast(&ms_mem[buf][tid * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B dense: 32 k-rows x 128B = 256 chunks, imma_s8 verbatim.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 8;
        const part = chunk % 8;
        const src = b + (@as(usize, k0 + k) * n + block_col) + part * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][k * 128 + part * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    gen.async_copy.cp_async_commit_group();
}

fn loadAFrag(buf: usize, r0: u32, lane: u32) [2]u32 {
    const matrix = lane / 8; // 0,1 (lanes 16-31 addresses ignored for x2)
    const row = lane % 8;
    const grow = r0 + row + matrix * 8;
    const addr: [*]addrspace(.shared) u8 = @ptrCast(&as_mem[buf][grow * 16]);
    const r = gen.matrix.ldmatrix_m8n8_x2_b16(addr);
    return .{ @bitCast(r.f0), @bitCast(r.f1) };
}

fn loadMeta(buf: usize, r0: u32, g: u32, t: u32) u32 {
    // Selector 0: lane t==0 supplies row g's word, t==1 row g+8's. Lanes
    // t==2,3 load the same rows; the hardware ignores them.
    const row = r0 + g + (t & 1) * 8;
    const ptr: [*]addrspace(.shared) const u32 = @ptrCast(@alignCast(&ms_mem[buf]));
    return ptr[row];
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
    // Load all A fragments (4), metadata words (4) and B fragments (8) for
    // this warp once.
    var af: [4][2]u32 = undefined;
    var mf: [4]u32 = undefined;
    var bf: [8][2]u32 = undefined;
    inline for (0..4) |mi| {
        af[mi] = loadAFrag(buf, wy * 64 + @as(u32, mi * 16), lane);
        mf[mi] = loadMeta(buf, wy * 64 + @as(u32, mi * 16), lane / 4, lane % 4);
    }
    inline for (0..8) |ni| {
        bf[ni] = loadBFrag(buf, wx * 64 + @as(u32, ni * 8), lane);
    }
    inline for (0..4) |mi| {
        inline for (0..8) |ni| {
            const d = asmgen.matrix.mma_sp_ordered_metadata_m16n8k32_s32_s8(
                acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                af[mi][0],      af[mi][1],
                bf[ni][0],      bf[ni][1],
                mf[mi],
                0, // sparsity selector: lanes t==0 and t==1 contribute
            );
            acc[mi][ni][0] = d.f0;
            acc[mi][ni][1] = d.f1;
            acc[mi][ni][2] = d.f2;
            acc[mi][ni][3] = d.f3;
        }
    }
}

/// `a` is the 2:4-pruned A (n/2 kept s8 per row), `meta` one u32 per row per
/// 32 dense k (n/32 words per row), `b` dense s8, `c` s32. `n` is the dense
/// element count per row.
pub fn immaSpS8(a: [*]const i8, b: [*]const i8, meta: [*]const u32, c: [*]i32, n: u32) callconv(.kernel) void {
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
    issueTileLoad(0, a, b, meta, n, block_row, block_col, 0, tid);

    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        const has_next = kt + 1 < ktiles; // block-uniform
        if (has_next) {
            issueTileLoad((kt + 1) % 2, a, b, meta, n, block_row, block_col, (kt + 1) * k_slice, tid);
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
    cuda.abi.assertMatches(api.imma_sp_s8, @TypeOf(immaSpS8));
    _ = cuda.Keep(.{&immaSpS8}).__zoxide_keep_kernels;
}
