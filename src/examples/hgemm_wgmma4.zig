const cuda = @import("cuda");
const gen = cuda.gen;
const wg = cuda.wgmma;
const api = @import("examples_abi");

fn cpAsyncWaitGroup(comptime num: u32) void {
    gen.async_copy.cp_async_wait_group(num);
}

/// HGEMM v6: one `wgmma.mma_async.sync.aligned.m64n128k16` per K-stage — the
/// shape CUTLASS uses — on stock Zig. Requires sm_90a.
///
/// ## How this gets past the 15-output limit
///
/// Every earlier wgmma kernel here is stuck on `m64n16k16` because
/// `m64nNk16` needs N/2 accumulator registers and each one has to be an
/// inline-asm output operand, while Zig's AstGen rejects more than 15. I had
/// recorded that as needing a compiler patch. It does not.
///
/// PTX register declarations are function-scoped, and LLVM splices inline asm
/// into the function body verbatim. So a `.reg` declaration in one asm statement
/// is visible to every later one:
///
///   asm volatile (".reg .f32 %zacc<64>;");                  // once
///   asm volatile ("wgmma... {%zacc0,...,%zacc63}, ...");     // zero outputs
///   asm volatile ("mov.f32 %[o], %zacc7;" : [o] "=f" (v));   // one output
///
/// The accumulator lives in registers LLVM does not know about, so it never
/// appears in an operand list and the limit does not apply. The epilogue reads
/// values back one at a time, one output operand each. `m64n128k16` — 64
/// accumulators — becomes expressible, as does anything wider.
///
/// ## What that costs, and the risks
///
/// LLVM cannot see these registers, so it cannot account for them. It allocates
/// its own registers as though the 64 f32 were not there, and ptxas has to fit
/// both. If the total exceeds what the SM can give, ptxas spills — and spilling
/// is measured on this hardware at 0.55x throughput for 16 registers and 0.93x
/// for a single byte, so that failure mode is severe rather than marginal. The
/// kernel's reported register count and any spill warning are the things to read
/// first.
///
/// Two correctness conditions, both checked in the generated PTX rather than
/// assumed:
///
///   The declaration must appear exactly once. If LLVM duplicates the statement
///   that carries it — loop unrolling, branch cloning — ptxas rejects the
///   redefinition. It is therefore issued unconditionally, before the loop, and
///   CI asserts a count of one.
///
///   It must precede every use. `asm volatile` statements keep their order
///   relative to each other, and the declaration is the first one in the
///   function, so nothing can be hoisted above it.
///
/// ## Shape
///
/// Block tile 64x128, one warpgroup, K-slice 16, cp.async double buffering, and
/// a single wgmma per stage with both operands read from shared memory through
/// descriptors. Per stage that reads A 2048 B + B 4096 B = 6144 B, the same as
/// `hgemm_wgmma3` achieves by putting A in registers — but with one instruction
/// instead of eight. That difference is the thing this kernel exists to measure:
/// the per-instruction cost of narrow N, which every previous round could only
/// point at.
///
/// Accumulator register i maps to (CUTLASS `CLayout_64xN`):
///   m = warp*16 + lane/4 + 8*((i/2) % 2)
///   n = (lane%4)*2 + (i % 2) + (i/4)*8

pub const tile_m = 64;
pub const tile_n = 128;
pub const k_slice = 16;
pub const threads = 128;
pub const acc_regs = tile_n / 2; // 64

const a_bytes = tile_m * k_slice * 2; // 2 KB
const b_bytes = k_slice * tile_n * 2; // 4 KB

var as_mem: [2][a_bytes]u8 align(128) addrspace(.shared) = undefined;
var bs_mem: [2][b_bytes]u8 align(128) addrspace(.shared) = undefined;

/// `%zacc0,%zacc1,...` for the instruction's accumulator list.
fn accList(comptime n: u32) []const u8 {
    comptime {
        var s: []const u8 = "";
        for (0..n) |i| {
            if (i != 0) s = s ++ ",";
            s = s ++ "%zacc" ++ decimal(i);
        }
        return s;
    }
}

fn decimal(comptime n: u32) []const u8 {
    comptime {
        if (n == 0) return "0";
        var buf: [10]u8 = undefined;
        var i: usize = buf.len;
        var v = n;
        while (v != 0) {
            i -= 1;
            buf[i] = '0' + @as(u8, @intCast(v % 10));
            v /= 10;
        }
        const out = buf[i..].*;
        return &out;
    }
}

/// A is core-matrix packed again, because it feeds a descriptor here rather than
/// `ldmatrix` as in v5:
///   A (64m x 16k, K-major):  offset(m, kb) = (m/8)*256 + kb*128 + (m%8)*16
///   B (16k x 128n, N-major): offset(k, nb) = nb*256 + k*16
fn issueTileLoad(buf: u32, a: [*]const f16, b: [*]const f16, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) void {
    {
        const m = tid / 2;
        const kb = tid % 2;
        const src = a + (@as(usize, block_row + m) * n + k0 + kb * 8);
        const dst = (m / 8) * 256 + kb * 128 + (m % 8) * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&as_mem[buf][dst]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    inline for (0..2) |i| {
        const chunk = tid * 2 + i;
        const k = chunk / 16;
        const nb = chunk % 16;
        const src = b + (@as(usize, k0 + k) * n + block_col + nb * 8);
        const dst = nb * 256 + k * 16;
        gen.async_copy.cp_async_cg_16(@ptrCast(&bs_mem[buf][dst]), @addrSpaceCast(@as([*]const u8, @ptrCast(src))));
    }
    gen.async_copy.cp_async_commit_group();
}

/// One `m64n128k16`, accumulating into `%zacc`. No output operands at all.
inline fn issueStage(buf: u32, scale_d: bool) void {
    const desc_a = wg.descriptor(wg.smemAddr(&as_mem[buf][0]), 128, 256, .none);
    const desc_b = wg.descriptor(wg.smemAddr(&bs_mem[buf][0]), 128, 256, .none);
    asm volatile ("wgmma.fence.sync.aligned;" ::: .{ .memory = true });
    asm volatile (
        \\{
        \\.reg .pred p;
        \\setp.ne.b32 p, %[sd], 0;
        \\wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {
    ++ accList(acc_regs) ++ "}, %[da], %[db], p, 1, 1, 0, 1;\n}"
        :
        : [da] "l" (desc_a),
          [db] "l" (desc_b),
          [sd] "r" (@as(u32, @intFromBool(scale_d))),
        : .{ .memory = true });
    asm volatile ("wgmma.commit_group.sync.aligned;" ::: .{ .memory = true });
}

/// Read one accumulator back. One output operand, so 64 of these are fine.
inline fn readAcc(comptime i: u32) f32 {
    var v: f32 = undefined;
    asm volatile ("mov.f32 %[o], %zacc" ++ decimal(i) ++ ";"
        : [o] "=f" (v)
        :
        : .{});
    return v;
}

pub fn hgemmWgmma4(a: [*]const f16, b: [*]const f16, c: [*]f32, n: u32) callconv(.kernel) void {
    // Declared once, unconditionally, before anything that uses it. CI asserts
    // this appears exactly once in the PTX.
    asm volatile (".reg .f32 %zacc<" ++ decimal(acc_regs) ++ ">;" ::: .{});

    const tid = cuda.threadIdx().x;
    const warp = tid / 32;
    const lane = cuda.laneId();
    const block_row = cuda.blockIdx().y * tile_m;
    const block_col = cuda.blockIdx().x * tile_n;

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
        // scale_d = false on the first stage overwrites, so the accumulator needs
        // no zeroing pass — which matters more here, since zeroing 64 registers
        // LLVM cannot see would take 64 more asm statements.
        issueStage(kt % 2, kt != 0);
        asm volatile ("wgmma.wait_group.sync.aligned 0;" ::: .{ .memory = true });
        cuda.syncThreads();
    }

    inline for (0..acc_regs) |i| {
        const m = warp * 16 + lane / 4 + 8 * ((i / 2) % 2);
        const col = (lane % 4) * 2 + (i % 2) + (i / 4) * 8;
        c[@as(usize, block_row + m) * n + block_col + col] = readAcc(i);
    }
}

comptime {
    cuda.abi.assertMatches(api.hgemm, @TypeOf(hgemmWgmma4));
    _ = cuda.Keep(.{&hgemmWgmma4}).__zoxide_keep_kernels;
}
