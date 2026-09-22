const cuda = @import("cuda");

/// Zig's own `f16` and `@Vector(2, f16)` on NVPTX, with no wrapper layer.
///
/// This example exists to record a measurement and to guard it. Comparing zoxide
/// against cuda-oxide, its `cuda-device` crate carries hand-written `f16.rs`,
/// `f16x2.rs` and `bf16x2.rs` modules, and I had listed "no f16/bf16 ergonomics
/// layer" as a real gap. Checking the generated PTX says otherwise: the language
/// already lowers to the packed instructions directly.
///
///   b[i] = a[i] * a[i]                  -> mul.rn.f16
///   @mulAdd(f16, ...)                   -> fma.rn.f16
///   v * v         on @Vector(2, f16)    -> mul.rn.f16x2
///   @mulAdd(@Vector(2, f16), ...)       -> fma.rn.f16x2
///   @max          on @Vector(2, f16)    -> max.f16x2
///   loads of @Vector(2, f16)            -> ld.global.b32  (properly merged)
///
/// So a wrapper layer for ordinary f16 arithmetic would duplicate the language.
/// What the generated bindings in `src/gen/` still buy is the variants Zig has no
/// syntax for — `ftz`, `nan`, `xorsign_abs`, `relu`, `sat` — which is 66 wrappers
/// across f16/f16x2/bf16/bf16x2.
///
/// CI asserts the instructions above still appear, because this is a property of
/// Zig's NVPTX lowering rather than of anything in this repository, and a
/// regression upstream would otherwise be invisible until someone measured.

const V2 = @Vector(2, f16);

pub fn f16Native(a: [*]const f16, b: [*]f16, va: [*]const V2, vb: [*]V2, n: u32) callconv(.kernel) void {
    const i = cuda.globalThreadId();
    if (i >= n) return;

    // Scalar: mul.rn.f16, then fma.rn.f16.
    b[i] = a[i] * a[i];
    b[i + n] = @mulAdd(f16, a[i], a[i], a[i]);

    // Packed 2-wide: mul.rn.f16x2, fma.rn.f16x2, max.f16x2.
    vb[i] = va[i] * va[i];
    vb[i + n] = @mulAdd(V2, va[i], va[i], va[i]);
    vb[i + 2 * n] = @max(va[i], vb[i]);
}

comptime {
    _ = cuda.Keep(.{&f16Native}).__zoxide_keep_kernels;
}
