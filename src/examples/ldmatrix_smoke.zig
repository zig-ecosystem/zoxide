//! Compile smoke for the generated ldmatrix / stmatrix / movmatrix bindings.
//!
//! Zig compiles lazily: a generated wrapper nobody calls is never type-checked,
//! let alone lowered. This file references the three matrix families in src/gen
//! (ldmatrix — 18 decls, stmatrix — 4 decls, both from gen.matrix in
//! intrinsics.zig, and movmatrix — 1 decl from asmgen.matrix in
//! intrinsics_asm.zig) so `zig build kernels` proves they compile, and the CI
//! grep assertions prove they reach PTX.
//!
//! Two limitations of Zig 0.16's NVPTX backend shape this file:
//!
//!   - The b8 and b8x16 ldmatrix variants (12 of the 18) are only selectable
//!     for sm_100a+; on sm_90a LLVM aborts with "Cannot select: intrinsic
//!     %llvm.nvvm.ldmatrix...". The whole example therefore builds for sm_100a.
//!   - The four stmatrix NVVM intrinsics are never lowered by the backend at
//!     any target — they reach PTX as an unresolved `call` to
//!     `llvm.nvvm.stmatrix...`, not as the instruction. The wrappers are
//!     unusable as-is, so stmatrix is covered here by hand-written inline asm
//!     (same templates the generator would emit) and the gen decls are on the
//!     unavailable list.
//!
//! The calls are formally correct, not a working pipeline: nothing here is a
//! sequence you could run. hgemm_mma2 / hgemm_wgmma3 cover the semantics; this
//! covers the bindings.
const cuda = @import("cuda");
const gen = cuda.gen;
const asmgen = cuda.asm_gen;
const abi = @import("examples_abi").ldmatrix_smoke;

var smem: [256]u8 addrspace(.shared) = undefined;

inline fn smemPtr() [*]addrspace(.shared) u8 {
    return @ptrCast(&smem);
}

inline fn acc2(acc: *u32, v: gen.Agg3) void {
    acc.* +%= @as(u32, @bitCast(v.f0)) +% @as(u32, @bitCast(v.f1));
}

inline fn acc4(acc: *u32, v: gen.Agg5) void {
    acc.* +%= @as(u32, @bitCast(v.f0)) +% @as(u32, @bitCast(v.f1)) +%
        @as(u32, @bitCast(v.f2)) +% @as(u32, @bitCast(v.f3));
}

/// ldmatrix (all 18 decls): x1 variants return one register, x2 a 2-register
/// aggregate (Agg3), x4 a 4-register aggregate (Agg5).
pub fn ldmatrixSmoke(src: [*]const u8, diag: [*]u32) callconv(.kernel) void {
    _ = src;
    const p = smemPtr();
    var acc: u32 = 0;

    // m8n8 b16: x1/x2/x4, plain and trans.
    acc +%= @bitCast(gen.matrix.ldmatrix_m8n8_x1_b16(p));
    acc +%= @bitCast(gen.matrix.ldmatrix_m8n8_x1_trans_b16(p));
    acc2(&acc, gen.matrix.ldmatrix_m8n8_x2_b16(p));
    acc2(&acc, gen.matrix.ldmatrix_m8n8_x2_trans_b16(p));
    acc4(&acc, gen.matrix.ldmatrix_m8n8_x4_b16(p));
    acc4(&acc, gen.matrix.ldmatrix_m8n8_x4_trans_b16(p));

    // m16n16 b8: x1/x2 trans.
    acc2(&acc, gen.matrix.ldmatrix_m16n16_x1_trans_b8(p));
    acc4(&acc, gen.matrix.ldmatrix_m16n16_x2_trans_b8(p));

    // b8x16 packed variants (6-bit pairs in 32-bit regs).
    acc2(&acc, gen.matrix.ldmatrix_m16n16_x1_trans_b8x16_b6x16_p32(p));
    acc4(&acc, gen.matrix.ldmatrix_m16n16_x2_trans_b8x16_b6x16_p32(p));
    acc +%= @bitCast(gen.matrix.ldmatrix_m8n16_x1_b8x16_b6x16_p32(p));
    acc2(&acc, gen.matrix.ldmatrix_m8n16_x2_b8x16_b6x16_p32(p));
    acc4(&acc, gen.matrix.ldmatrix_m8n16_x4_b8x16_b6x16_p32(p));

    // b8x16 packed variants (4-bit pairs in 64-bit regs).
    acc2(&acc, gen.matrix.ldmatrix_m16n16_x1_trans_b8x16_b4x16_p64(p));
    acc4(&acc, gen.matrix.ldmatrix_m16n16_x2_trans_b8x16_b4x16_p64(p));
    acc +%= @bitCast(gen.matrix.ldmatrix_m8n16_x1_b8x16_b4x16_p64(p));
    acc2(&acc, gen.matrix.ldmatrix_m8n16_x2_b8x16_b4x16_p64(p));
    acc4(&acc, gen.matrix.ldmatrix_m8n16_x4_b8x16_b4x16_p64(p));

    diag[0] = acc;
}

/// Shared addresses are 32-bit; the inline asm takes them as a "r" operand.
/// Same zero-extension as src/wgmma.zig's smemAddr.
inline fn smemAddr(ptr: anytype) u32 {
    return @truncate(@intFromPtr(ptr));
}

fn stmatrixX2(comptime trans: bool, addr: u32, r0: u32, r1: u32) void {
    const tmpl = if (trans)
        "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%[addr]], {%[r0], %[r1]};"
    else
        "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%[addr]], {%[r0], %[r1]};";
    asm volatile (tmpl
        :
        : [addr] "r" (addr), [r0] "r" (r0), [r1] "r" (r1),
    );
}

fn stmatrixX4(comptime trans: bool, addr: u32, r0: u32, r1: u32, r2: u32, r3: u32) void {
    const tmpl = if (trans)
        "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%[addr]], {%[r0], %[r1], %[r2], %[r3]};"
    else
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%[addr]], {%[r0], %[r1], %[r2], %[r3]};";
    asm volatile (tmpl
        :
        : [addr] "r" (addr), [r0] "r" (r0), [r1] "r" (r1), [r2] "r" (r2), [r3] "r" (r3),
    );
}

/// stmatrix (4 forms, inline asm — see the file header) + movmatrix (1 decl,
/// the generated asm wrapper).
pub fn stmatrixMovSmoke(src: [*]const u8, diag: [*]u32) callconv(.kernel) void {
    _ = src;
    const addr = smemAddr(&smem);

    stmatrixX2(false, addr, 0, 1);
    stmatrixX2(true, addr, 0, 1);
    stmatrixX4(false, addr, 0, 1, 2, 3);
    stmatrixX4(true, addr, 0, 1, 2, 3);

    // movmatrix: warp-level 8x8 b16 transpose, register to register.
    diag[0] = asmgen.matrix.movmatrix_trans_b16(smem[0]);
}

comptime {
    cuda.abi.assertMatches(abi.signature, @TypeOf(ldmatrixSmoke));
    cuda.abi.assertMatches(abi.signature, @TypeOf(stmatrixMovSmoke));
    _ = cuda.Keep(.{ &ldmatrixSmoke, &stmatrixMovSmoke }).__zoxide_keep_kernels;
}
