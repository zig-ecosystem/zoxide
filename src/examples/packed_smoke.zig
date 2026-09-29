//! Compile smoke for the generated packed-data bindings.
//!
//! Zig compiles lazily: a generated wrapper nobody calls is never type-checked,
//! let alone lowered. This file references every wrapper of the eight packed
//! families in src/gen (packed_alu 30, packed_conversion 18, packed_atomic 2,
//! dotprod 4, prmt 7, clc 6, extended_minmax 52, integer_minmax 8 — 127 decls
//! total; the two gen.dotprod.dp2a_* wrappers do not lower (LLVM has no NVPTX
//! selection pattern for them) and are covered by hand-written asm instead) so
//! `zig build kernels` proves they all compile, and the CI grep assertions
//! prove they reach PTX. Built for sm_100a: the fp8 cvt intrinsics are gated
//! on sm_89+ and clusterlaunchcontrol on sm_90a+ in LLVM's NVPTX backend, and
//! sm_100a covers both.
//!
//! The calls are formally correct, not a working algorithm: nothing here is a
//! sequence you could run. f16_native covers real f16x2 arithmetic; this
//! covers the bindings.
const cuda = @import("cuda");
const gen = cuda.gen;
const asmgen = cuda.asm_gen;
const abi = @import("examples_abi").packed_smoke;

var clc_resp: [16]u8 align(16) addrspace(.shared) = undefined;
var clc_mbar: [8]u8 align(8) addrspace(.shared) = undefined;

/// dp2a.<lo|hi>.<ty>.<ty>: hand-written stand-in for gen.dotprod.dp2a_* —
/// see the call site for why the NVVM form is unusable.
fn dp2aAsm(comptime ty: []const u8, a: i32, b: i32, hi: bool, c: i32) i32 {
    if (hi) {
        return asm ("dp2a.hi." ++ ty ++ "." ++ ty ++ " %[out0], %[in0], %[in1], %[in2];"
            : [out0] "=r" (-> i32),
            : [in0] "r" (a), [in1] "r" (b), [in2] "r" (c),
        );
    }
    return asm ("dp2a.lo." ++ ty ++ "." ++ ty ++ " %[out0], %[in0], %[in1], %[in2];"
        : [out0] "=r" (-> i32),
        : [in0] "r" (a), [in1] "r" (b), [in2] "r" (c),
    );
}

/// packed_alu (30): f16x2 (13), bf16x2 (9), f32x2 (8).
pub fn packedAluSmoke(src: [*]const u32, diag: [*]u32) callconv(.kernel) void {
    const a = src[0];
    const b = src[1];
    const c = src[2];
    const a64 = @as(u64, a) << 32 | b;
    const b64 = @as(u64, b) << 32 | c;
    const c64 = @as(u64, c) << 32 | a;
    var acc: u32 = 0;

    // f16x2 (13).
    acc +%= asmgen.f16x2.max_f16x2(a, b);
    acc +%= asmgen.f16x2.fma_relu_f16x2(a, b, c);
    acc +%= asmgen.f16x2.fma_ftz_relu_f16x2(a, b, c);
    acc +%= asmgen.f16x2.min_f16x2(a, b);
    acc +%= asmgen.f16x2.abs_f16x2(a);
    acc +%= asmgen.f16x2.fma_sat_f16x2(a, b, c);
    acc +%= asmgen.f16x2.fma_ftz_f16x2(a, b, c);
    acc +%= asmgen.f16x2.mul_f16x2(a, b);
    acc +%= asmgen.f16x2.neg_f16x2(a);
    acc +%= asmgen.f16x2.fma_ftz_sat_f16x2(a, b, c);
    acc +%= asmgen.f16x2.sub_f16x2(a, b);
    acc +%= asmgen.f16x2.fma_f16x2(a, b, c);
    acc +%= asmgen.f16x2.add_f16x2(a, b);

    // bf16x2 (9).
    acc +%= asmgen.bf16x2.sub_bf16x2(a, b);
    acc +%= asmgen.bf16x2.fma_bf16x2(a, b, c);
    acc +%= asmgen.bf16x2.mul_bf16x2(a, b);
    acc +%= asmgen.bf16x2.max_bf16x2(a, b);
    acc +%= asmgen.bf16x2.abs_bf16x2(a);
    acc +%= asmgen.bf16x2.min_bf16x2(a, b);
    acc +%= asmgen.bf16x2.add_bf16x2(a, b);
    acc +%= asmgen.bf16x2.neg_bf16x2(a);
    acc +%= asmgen.bf16x2.fma_relu_bf16x2(a, b, c);

    // f32x2 (8): u64 lanes, folded into the low word.
    acc +%= @truncate(asmgen.f32x2.sub_f32x2(a64, b64));
    acc +%= @truncate(asmgen.f32x2.fma_f32x2(a64, b64, c64));
    acc +%= @truncate(asmgen.f32x2.add_f32x2(a64, b64));
    acc +%= @truncate(asmgen.f32x2.add_ftz_f32x2(a64, b64));
    acc +%= @truncate(asmgen.f32x2.mul_ftz_f32x2(a64, b64));
    acc +%= @truncate(asmgen.f32x2.sub_ftz_f32x2(a64, b64));
    acc +%= @truncate(asmgen.f32x2.mul_f32x2(a64, b64));
    acc +%= @truncate(asmgen.f32x2.fma_ftz_f32x2(a64, b64, c64));

    diag[0] = acc;
}

/// extended_minmax (52) + integer_minmax (8).
pub fn minmaxSmoke(src: [*]const u32, diag: [*]u32) callconv(.kernel) void {
    const a = src[0];
    const b = src[1];
    const ha: u16 = @truncate(a);
    const hb: u16 = @truncate(b);
    const fa: f32 = @bitCast(a);
    const fb: f32 = @bitCast(b);
    var acc: u32 = 0;

    // extended_minmax, f16 (16).
    acc +%= asmgen.f16.min_ftz_nan_xorsign_abs_f16(ha, hb);
    acc +%= asmgen.f16.max_ftz_nan_f16(ha, hb);
    acc +%= asmgen.f16.min_nan_f16(ha, hb);
    acc +%= asmgen.f16.min_ftz_xorsign_abs_f16(ha, hb);
    acc +%= asmgen.f16.max_nan_xorsign_abs_f16(ha, hb);
    acc +%= asmgen.f16.min_xorsign_abs_f16(ha, hb);
    acc +%= asmgen.f16.max_ftz_f16(ha, hb);
    acc +%= asmgen.f16.min_f16(ha, hb);
    acc +%= asmgen.f16.max_f16(ha, hb);
    acc +%= asmgen.f16.min_ftz_nan_f16(ha, hb);
    acc +%= asmgen.f16.max_xorsign_abs_f16(ha, hb);
    acc +%= asmgen.f16.max_ftz_xorsign_abs_f16(ha, hb);
    acc +%= asmgen.f16.min_nan_xorsign_abs_f16(ha, hb);
    acc +%= asmgen.f16.max_nan_f16(ha, hb);
    acc +%= asmgen.f16.max_ftz_nan_xorsign_abs_f16(ha, hb);
    acc +%= asmgen.f16.min_ftz_f16(ha, hb);

    // extended_minmax, f16x2 (14).
    acc +%= asmgen.f16x2.max_xorsign_abs_f16x2(a, b);
    acc +%= asmgen.f16x2.max_ftz_nan_xorsign_abs_f16x2(a, b);
    acc +%= asmgen.f16x2.max_ftz_xorsign_abs_f16x2(a, b);
    acc +%= asmgen.f16x2.max_nan_xorsign_abs_f16x2(a, b);
    acc +%= asmgen.f16x2.min_nan_xorsign_abs_f16x2(a, b);
    acc +%= asmgen.f16x2.min_ftz_xorsign_abs_f16x2(a, b);
    acc +%= asmgen.f16x2.max_ftz_f16x2(a, b);
    acc +%= asmgen.f16x2.max_ftz_nan_f16x2(a, b);
    acc +%= asmgen.f16x2.min_nan_f16x2(a, b);
    acc +%= asmgen.f16x2.min_xorsign_abs_f16x2(a, b);
    acc +%= asmgen.f16x2.max_nan_f16x2(a, b);
    acc +%= asmgen.f16x2.min_ftz_f16x2(a, b);
    acc +%= asmgen.f16x2.min_ftz_nan_xorsign_abs_f16x2(a, b);
    acc +%= asmgen.f16x2.min_ftz_nan_f16x2(a, b);

    // extended_minmax, bf16 (8).
    acc +%= asmgen.bf16.max_bf16(ha, hb);
    acc +%= asmgen.bf16.min_nan_bf16(ha, hb);
    acc +%= asmgen.bf16.max_xorsign_abs_bf16(ha, hb);
    acc +%= asmgen.bf16.max_nan_xorsign_abs_bf16(ha, hb);
    acc +%= asmgen.bf16.max_nan_bf16(ha, hb);
    acc +%= asmgen.bf16.min_nan_xorsign_abs_bf16(ha, hb);
    acc +%= asmgen.bf16.min_xorsign_abs_bf16(ha, hb);
    acc +%= asmgen.bf16.min_bf16(ha, hb);

    // extended_minmax, bf16x2 (6).
    acc +%= asmgen.bf16x2.max_nan_xorsign_abs_bf16x2(a, b);
    acc +%= asmgen.bf16x2.max_xorsign_abs_bf16x2(a, b);
    acc +%= asmgen.bf16x2.min_nan_bf16x2(a, b);
    acc +%= asmgen.bf16x2.max_nan_bf16x2(a, b);
    acc +%= asmgen.bf16x2.min_xorsign_abs_bf16x2(a, b);
    acc +%= asmgen.bf16x2.min_nan_xorsign_abs_bf16x2(a, b);

    // extended_minmax, f32 (8).
    acc +%= @as(u32, @bitCast(asmgen.float.min_ftz_nan_xorsign_abs_f32(fa, fb)));
    acc +%= @as(u32, @bitCast(asmgen.float.min_xorsign_abs_f32(fa, fb)));
    acc +%= @as(u32, @bitCast(asmgen.float.max_nan_xorsign_abs_f32(fa, fb)));
    acc +%= @as(u32, @bitCast(asmgen.float.min_ftz_xorsign_abs_f32(fa, fb)));
    acc +%= @as(u32, @bitCast(asmgen.float.max_xorsign_abs_f32(fa, fb)));
    acc +%= @as(u32, @bitCast(asmgen.float.min_nan_xorsign_abs_f32(fa, fb)));
    acc +%= @as(u32, @bitCast(asmgen.float.max_ftz_xorsign_abs_f32(fa, fb)));
    acc +%= @as(u32, @bitCast(asmgen.float.max_ftz_nan_xorsign_abs_f32(fa, fb)));

    // integer_minmax (8): s32 relu forms, s16x2/u16x2 packed forms.
    acc +%= asmgen.int.min_relu_s32(a, b);
    acc +%= asmgen.int.max_relu_s32(a, b);
    acc +%= asmgen.i16x2.max_s16x2(a, b);
    acc +%= asmgen.i16x2.min_u16x2(a, b);
    acc +%= asmgen.i16x2.max_u16x2(a, b);
    acc +%= asmgen.i16x2.max_relu_s16x2(a, b);
    acc +%= asmgen.i16x2.min_s16x2(a, b);
    acc +%= asmgen.i16x2.min_relu_s16x2(a, b);

    diag[0] = acc;
}

/// packed_conversion (18) + packed_atomic (2).
pub fn cvtAtomicSmoke(src: [*]const u32, diag: [*]u32) callconv(.kernel) void {
    const a = src[0];
    const b = src[1];
    const ha: u16 = @truncate(a);
    const fa: f32 = @bitCast(a);
    const fb: f32 = @bitCast(b);
    var acc: u32 = 0;

    // packed_conversion, NVVM fp8 forms (4): (f32, f32) -> i16.
    acc +%= @as(u16, @bitCast(gen.convert.cvt_rn_satfinite_relu_e4m3x2_f32(fa, fb)));
    acc +%= @as(u16, @bitCast(gen.convert.cvt_rn_satfinite_e5m2x2_f32(fa, fb)));
    acc +%= @as(u16, @bitCast(gen.convert.cvt_rn_satfinite_e4m3x2_f32(fa, fb)));
    acc +%= @as(u16, @bitCast(gen.convert.cvt_rn_satfinite_relu_e5m2x2_f32(fa, fb)));

    // packed_conversion, asm forms (14): fp8<->f16x2 and f32x2->f16x2/bf16x2.
    acc +%= asmgen.convert.cvt_rn_f16x2_e4m3x2(ha);
    acc +%= asmgen.convert.cvt_rn_satfinite_e5m2x2_f16x2(a);
    acc +%= asmgen.convert.cvt_rn_satfinite_relu_e4m3x2_f16x2(a);
    acc +%= asmgen.convert.cvt_f16x2_f32(fa, fb);
    acc +%= asmgen.convert.cvt_rz_bf16x2_f32(fa, fb);
    acc +%= asmgen.convert.cvt_f32x2_bf16x2(fa, fb);
    acc +%= asmgen.convert.cvt_rn_relu_f16x2_f32(fa, fb);
    acc +%= asmgen.convert.cvt_rn_relu_f16x2_e4m3x2(ha);
    acc +%= asmgen.convert.cvt_rz_f16x2_f32(fa, fb);
    acc +%= asmgen.convert.cvt_rn_satfinite_e4m3x2_f16x2(a);
    acc +%= asmgen.convert.cvt_rn_f16x2_e5m2x2(ha);
    acc +%= asmgen.convert.cvt_rn_satfinite_relu_e5m2x2_f16x2(a);
    acc +%= asmgen.convert.cvt_rn_relu_bf16x2_f32(fa, fb);
    acc +%= asmgen.convert.cvt_rn_relu_f16x2_e5m2x2(ha);

    // packed_atomic (2): the wrappers take the address as u64.
    acc +%= asmgen.atomic.atom_add_f16x2(@intFromPtr(diag), a);
    acc +%= asmgen.atomic.atom_add_bf16x2(@intFromPtr(diag), b);

    diag[0] = acc;
}

/// dotprod (4) + prmt (7) + clc (6).
pub fn dotPrmtClcSmoke(src: [*]const u32, diag: [*]u32) callconv(.kernel) void {
    const a: i32 = @bitCast(src[0]);
    const b: i32 = @bitCast(src[1]);
    const c: i32 = @bitCast(src[2]);
    const hi = (src[3] & 1) != 0;
    var acc: u32 = 0;

    // dotprod (4). dp4a lowers from NVVM; dp2a does not ("Cannot select:
    // intrinsic %llvm.nvvm.idp2a.u.u" — LLVM declares the intrinsic but has no
    // NVPTX selection pattern for it), so the two dp2a wrappers are covered by
    // hand-written asm here, following the stmatrix precedent in
    // ldmatrix_smoke. The bool operand selects the lo/hi 16-bit halves.
    acc +%= @bitCast(gen.dotprod.dp4a_u32(a, b, c));
    acc +%= @bitCast(gen.dotprod.dp4a_s32(a, b, c));
    acc +%= @bitCast(dp2aAsm("u32", a, b, hi, c));
    acc +%= @bitCast(dp2aAsm("s32", a, b, hi, c));

    // prmt (7): three-operand forms, then the two-operand mode forms.
    acc +%= @bitCast(gen.prmt.prmt(a, b, c));
    acc +%= @bitCast(gen.prmt.prmt_f4e(a, b, c));
    acc +%= @bitCast(gen.prmt.prmt_b4e(a, b, c));
    acc +%= @bitCast(gen.prmt.prmt_ecl(a, b));
    acc +%= @bitCast(gen.prmt.prmt_rc8(a, b));
    acc +%= @bitCast(gen.prmt.prmt_ecr(a, b));
    acc +%= @bitCast(gen.prmt.prmt_rc16(a, b));

    // clc (6): try_cancel writes a 128-bit response to shared memory via the
    // mbarrier; the query forms consume one. The i128 here is a formal value —
    // a real launch would read it back out of clc_resp.
    gen.clc.clc_try_cancel(@ptrCast(&clc_resp), @ptrCast(&clc_mbar));
    gen.clc.clc_try_cancel_multicast(@ptrCast(&clc_resp), @ptrCast(&clc_mbar));
    const resp: i128 = @as(i128, src[4]) << 64 | src[5];
    acc +%= @bitCast(gen.clc.clc_query_get_first_ctaid_x(resp));
    acc +%= @bitCast(gen.clc.clc_query_get_first_ctaid_y(resp));
    acc +%= @bitCast(gen.clc.clc_query_get_first_ctaid_z(resp));
    acc +%= @intFromBool(gen.clc.clc_query_is_canceled(resp));

    diag[0] = acc;
}

comptime {
    cuda.abi.assertMatches(abi.signature, @TypeOf(packedAluSmoke));
    cuda.abi.assertMatches(abi.signature, @TypeOf(minmaxSmoke));
    cuda.abi.assertMatches(abi.signature, @TypeOf(cvtAtomicSmoke));
    cuda.abi.assertMatches(abi.signature, @TypeOf(dotPrmtClcSmoke));
    _ = cuda.Keep(.{ &packedAluSmoke, &minmaxSmoke, &cvtAtomicSmoke, &dotPrmtClcSmoke }).__zoxide_keep_kernels;
}
