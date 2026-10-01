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

/// Sparse IMMA s4: structured-sparse int4 GEMM on
/// mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 —
/// imma_sp_s8's shape with the int4 sparsity rule, which is NOT plain 2:4.
///
/// Fragment reading (PTX ISA 9.7.16.6.2.7 + the integer sparsity section,
/// recorded so the exact-match bench can falsify it on first GPU run):
///
///   Sparsity granularity is 4:8 and *pair-clustered*: within each 8-wide
///   dense k chunk, the two-wide sub-chunks are all-zero or all-nonzero, and
///   exactly two of the four sub-chunks survive. Metadata per chunk is two
///   2-bit indices naming the surviving sub-chunks (not per-element indices
///   like the s8 shape). Kept pairs stay pairs, so under the packS4
///   convention (low nibble = even index) each surviving pair is exactly one
///   byte of the pruned row — the prune is byte-granular, no nibble surgery.
///
///   A (sparse, k64): 2 .b32 regs per lane. Lane l = 4*g + t covers the
///   16-wide dense span k = 16t..16t+15 (two 8-wide chunks); a0 = row g's 8
///   kept values of that span, a1 = row g+8's, packed low-to-high in chunk
///   order. The pruned row is 32 s4 = 16 B, and chunk j's kept bytes sit at
///   2j..2j+1, so lane t wants bytes 4t..4t+3 — consecutive, exactly what
///   ldmatrix.x2 distributes over the [16][16B] tile (same byte shape as
///   imma_sp_s8's A path).
///
///   B (dense): identical to dense imma_s4 — b0 = (k 8t..8t+7, col g),
///   b1 = (k 32+8t..32+8t+7, col g) — manual byte+nibble gather, unchanged.
///
///   Metadata: one u32 per *row* per mma (8 chunks x 4 bits; chunk j owns
///   nibble [4j+3:4j], low 2 bits = first surviving sub-chunk, high 2 =
///   second). The m16n8k64-s4 selector names a thread PAIR (selector 0 =
///   lanes t==0 and t==1), read as t==0 carries row g's word and t==1 row
///   g+8's — the same least-certain assignment as imma_sp_s8, listed on the
///   next-pod-run suspect list.
///
/// Pipeline: imma_sp_s8's — 128x128 tile, dense k-slice 64, cp.async
/// double-buffered (A pruned 2 KB + meta 512 B + B dense 4 KB per stage),
/// both bar.sync on the universal path.

pub const block_tile = 128;
pub const k_slice = 64; // dense k per stage; pruned A carries 32 per row
pub const threads = 128;

const a_bytes = block_tile * k_slice / 4; // 2 KB (2:4 kept, 2 per byte)
const b_bytes = k_slice * block_tile / 2; // 4 KB
const meta_bytes = block_tile * 4; // 512 B, one u32 per row

var as_mem: [2][a_bytes]u8 addrspace(.shared) = undefined;
var bs_mem: [2][b_bytes]u8 addrspace(.shared) = undefined;
var ms_mem: [2][meta_bytes]u8 addrspace(.shared) = undefined;

fn issueTileLoad(buf: usize, a: [*]const u8, b: [*]const u8, meta: [*]const u32, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A pruned: 128 rows x 16B = 128 chunks, one per thread. A row holds
    // n/4 packed bytes; k0 dense = k0/4 bytes.
    {
        const src = a + (@as(usize, block_row + tid) * (n / 4) + k0 / 4);
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][tid * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // Metadata: 128 rows x 4B = 512B = 32 chunks; threads 0..31 load 16B
    // (4 rows' words) each. A row holds n/64 metadata words.
    if (tid < 32) {
        const src = meta + (@as(usize, block_row + tid * 4) * (n / 64) + k0 / 64);
        gen.async_copy.cp_async_cg_16(@ptrCast(&ms_mem[buf][tid * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B dense: 64 k-rows x 64B = 256 chunks, imma_s4 verbatim.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 4;
        const part = chunk % 4;
        const src = b + (@as(usize, k0 + k) * (n / 2) + block_col / 2) + part * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][k * 64 + part * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
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
            const d = asmgen.matrix.mma_sp_ordered_metadata_m16n8k64_s32_s4(
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

/// `a` is the pruned A (n/4 packed bytes per row, one byte per surviving
/// pair), `meta` one u32 per row per 64 dense k (n/64 words per row), `b`
/// dense s4 packed 2-per-byte (n/2 bytes per row), `c` s32. `n` is the dense
/// element count per row.
pub fn immaSpS4(a: [*]const u8, b: [*]const u8, meta: [*]const u32, c: [*]i32, n: u32) callconv(.kernel) void {
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
    cuda.abi.assertMatches(api.imma_sp_s4, @TypeOf(immaSpS4));
    _ = cuda.Keep(.{&immaSpS4}).__zoxide_keep_kernels;
}
