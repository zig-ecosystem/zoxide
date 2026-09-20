const cuda = @import("cuda");
const gen = cuda.gen;
const asmgen = cuda.asm_gen;

/// Smoke test for asm-template-generated bindings (src/gen/instrinsics_asm.zig):
/// exercises single-output asm (abs.bf16x2), packed u64 asm (mul.rn.f32x2),
/// multi-output asm (mma.sync.aligned.m16n8k16), plus one intrinsic (prmt)
/// for contrast.
pub fn asmSmoke(a: u32, b: u32, out: [*]f32, out32: [*]u32) callconv(.kernel) void {
    // abs.bf16x2: single in/out, "=r,r"
    out32[0] = asmgen.bf16x2.abs_bf16x2(a);

    // mul.rn.f32x2: packed f32 pairs in u64, "=l,l,l"
    out32[1] = @truncate(asmgen.f32x2.mul_f32x2(@as(u64, a) << 32 | b, @as(u64, b) << 32 | a));

    // prmt (NVVM intrinsic path, for contrast)
    out32[2] = @bitCast(gen.prmt.prmt_b4e(@bitCast(a), @bitCast(b), 0x5410));

    // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32:
    // D = A*B + C, fragments zeroed — correctness is verified on the pod;
    // here we prove the multi-output asm plumbing.
    const d = asmgen.matrix.mma_m16n8k16_f32_f16(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0);
    out[0] = d.f0 + d.f1 + d.f2 + d.f3;
}

comptime {
    _ = cuda.Keep(.{&asmSmoke}).__zoxide_keep_kernels;
}
