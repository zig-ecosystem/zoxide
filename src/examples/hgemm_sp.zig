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

/// Sparse HGEMM: 2:4-structured-sparse f16 GEMM on
/// mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.
///
/// Shape choice: the generated bindings (src/gen) carry no *plain*
/// `mma.sp.sync` f16 wrapper — all 32 plain mma.sp entries are the integer
/// u4/s4/u8/s8 shapes; f16 sparse exists only in the `::ordered_metadata`
/// form. That form is also sm_80, so it is verifiable on the H20 (sm_90)
/// pod, and `ordered_metadata` only changes how metadata threads are
/// selected, not the sparsity semantics.
///
/// The 2:4 contract, as read from the PTX ISA sparse-MMA section (this
/// reading is what the exact-match bench check exists to falsify on first
/// GPU run):
///
///   A is stored dense-but-pruned: of every 4 consecutive k elements only 2
///   are kept, so a k16 mma row shrinks to 8 f16 (16 B) per matrix row. The
///   A fragment is 2 .b32 regs per lane (half the dense 4): lane l = 4*g + t
///   holds a0 = the 2 kept values of row g's k-group t (dense k 4t..4t+3),
///   packed low then high in kept order, and a1 = the same for row g+8.
///   That is 4 consecutive bytes of the pruned row — exactly what
///   ldmatrix.x2 distributes (one 8x8-b16 matrix per 8 rows; lanes 16-31
///   addresses are ignored). The dense 4-reg ldmatrix.x4 path does NOT
///   apply; the pruned row is half as wide.
///
///   B stays dense, so its fragment path is hgemm_bf16's unchanged
///   (ldmatrix.x2.trans over the [16][128] k-major tile).
///
///   Metadata is one 32-bit register per mma, supplied only by the thread
///   selected by the sparsity-selector immediate within each 4-lane group
///   (we pass 0 = lane t==0; every lane loads the same word, so any
///   selector would do). Lower 16 bits describe row g, upper 16 bits row
///   g+8. Within a row's 16 bits, k-group j (dense k 4j..4j+3) owns nibble
///   [4j+3 : 4j]: bits [4j+1:4j] = index of the first kept element, bits
///   [4j+3:4j+2] = index of the second (first < second). The host packs
///   metadata in exactly this order (see runHgemmSp in bench.zig and the
///   contract note in examples_abi.zig); a wrong bit order, a wrong row
///   pairing, or a wrong contributing thread all produce wrong-but-plausible
///   numbers on hardware, which the exact integer comparison turns into a
///   hard FAIL.
///
/// Pipeline is hgemm_bf16's: 128x128 block tile, K-slice 16 (dense k), 4
/// warps 2x2, warp tile 64x64 = 4(m) x 8(n) mma tiles. Per stage the loads
/// are A 128x16B pruned + meta 128x2B + B 16x256B dense. cp.async,
/// double-buffered, both bar.sync on the universal path.

pub const block_tile = 128;
pub const k_slice = 16; // dense k per stage; pruned A carries 8 per row
pub const threads = 128;

const a_bytes = block_tile * k_slice; // 2 KB (half of dense, 2:4 kept)
const b_bytes = k_slice * block_tile * 2; // 4 KB
const meta_bytes = block_tile * 2; // 256 B, one u16 per row

var as_mem: [2][a_bytes]u8 addrspace(.shared) = undefined;
var bs_mem: [2][b_bytes]u8 addrspace(.shared) = undefined;
var ms_mem: [2][meta_bytes]u8 addrspace(.shared) = undefined;

fn issueTileLoad(buf: usize, a: [*]const f16, b: [*]const f16, meta: [*]const u16, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    // A pruned: 128 rows x 16B = 128 chunks, one per thread. A row holds
    // n/2 kept f16; k0 dense = k0/2 kept.
    {
        const src = a + (@as(usize, block_row + tid) * (n / 2) + k0 / 2);
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][tid * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // Metadata: 128 rows x 2B = 256B = 16 chunks; threads 0..15 load 16B
    // (8 rows' worth) each. Predicated cp.async is fine; the commit below
    // stays unconditional. A row holds n/16 metadata words; k0 dense =
    // k0/16 words.
    if (tid < 16) {
        const src = meta + (@as(usize, block_row + tid * 8) * (n / 16) + k0 / 16);
        gen.async_copy.cp_async_cg_16(@ptrCast(&ms_mem[buf][tid * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    // B dense: 16 k-rows x 256B = 256 chunks, hgemm_bf16 verbatim.
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 16;
        const part = chunk % 16;
        const src = b + (@as(usize, k0 + k) * n + block_col) + part * 8;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][k * 256 + part * 16]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
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

fn loadMeta(buf: usize, r0: u32, g: u32) u32 {
    const ptr: [*]addrspace(.shared) u16 = @ptrCast(@alignCast(&ms_mem[buf]));
    const lo: u32 = ptr[r0 + g];
    const hi: u32 = ptr[r0 + g + 8];
    return lo | (hi << 16); // lower 16 = row g, upper 16 = row g+8
}

fn loadBFrag(buf: usize, c0: u32, lane: u32) [2]u32 {
    const matrix = lane / 8; // 0,1 (lanes 16-31 addresses ignored for x2)
    const row = lane % 8;
    const addr: [*]addrspace(.shared) u8 = @ptrCast(&bs_mem[buf][(matrix * 8 + row) * 256 + c0 * 2]);
    const r = gen.matrix.ldmatrix_m8n8_x2_trans_b16(addr);
    return .{ @bitCast(r.f0), @bitCast(r.f1) };
}

fn computeTile(comptime buf: usize, acc: *[4][8][4]f32, wy: u32, wx: u32, lane: u32) void {
    // Load all A fragments (4), metadata words (4) and B fragments (8) for
    // this warp once.
    var af: [4][2]u32 = undefined;
    var mf: [4]u32 = undefined;
    var bf: [8][2]u32 = undefined;
    inline for (0..4) |mi| {
        af[mi] = loadAFrag(buf, wy * 64 + @as(u32, mi * 16), lane);
        mf[mi] = loadMeta(buf, wy * 64 + @as(u32, mi * 16), lane / 4);
    }
    inline for (0..8) |ni| {
        bf[ni] = loadBFrag(buf, wx * 64 + @as(u32, ni * 8), lane);
    }
    inline for (0..4) |mi| {
        inline for (0..8) |ni| {
            const d = asmgen.matrix.mma_sp_ordered_metadata_m16n8k16_f32_f16(
                acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                af[mi][0],      af[mi][1],
                bf[ni][0],      bf[ni][1],
                mf[mi],
                0, // sparsity selector: lane t==0 of each group contributes
            );
            acc[mi][ni][0] = d.f0;
            acc[mi][ni][1] = d.f1;
            acc[mi][ni][2] = d.f2;
            acc[mi][ni][3] = d.f3;
        }
    }
}

/// `a` is the 2:4-pruned A (n/2 kept f16 per row), `meta` one u16 per row
/// per 16 dense k (n/16 per row), `b` dense f16, `c` f32. `n` is the dense
/// element count per row, so the launch shape matches the dense hgemm
/// family and the same n gates apply.
pub fn hgemmSp(a: [*]const f16, b: [*]const f16, meta: [*]const u16, c: [*]f32, n: u32) callconv(.kernel) void {
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
    cuda.abi.assertMatches(api.hgemm_sp, @TypeOf(hgemmSp));
    _ = cuda.Keep(.{&hgemmSp}).__zoxide_keep_kernels;
}
