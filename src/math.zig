//! Device-side scalar math: the libdevice counterpart for zoxide kernels.
//!
//! Two tiers, mirroring libdevice's `__nv_fast_*` / `__nv_*` split:
//!
//!   - `*Fast` functions wrap a single PTX approx instruction via the
//!     generated `asm_gen` bindings (`sin.approx.f32`, `lg2.approx.f32`, …).
//!     Accuracy matches CUDA's `-use_fast_math` (~2^-21 relative, reduced
//!     input domain), because these are the same instructions `__nv_fast_*`
//!     lowers to.
//!   - The rest are Zig implementations of the classic algorithms
//!     (Cody–Waite argument reduction + minimax polynomials, coefficients
//!     from musl/cephes). They aim for a few ulps, not bit-equality with
//!     libdevice — see docs/libdevice-path.md for why we do not link the
//!     real libdevice bitcode.
//!
//! f32 only. Zig's own `@sin`/`@cos`/`@exp`/`@log` do not work here: LLVM
//! has no libcall for them on nvptx ("no libcall available for fexp",
//! "Cannot select: fsin"), which is precisely the gap this file fills.
//! `@sqrt`/`@max`/`@min`/`@abs`/`@floor` do lower natively and are
//! re-exported under CUDA names.

const asm_gen = @import("gen/instrinsics_asm.zig");

const pi_2: f32 = 0x1.921fb6p0;
const two_over_pi_f64: f64 = 0x1.45f306dc9c883p-1;
const pi_over_2_f64: f64 = 0x1.921fb54442d18p0;
const log2e: f32 = 0x1.715476p0;
const ln2_hi: f32 = 0x1.62e400p-1;
const ln2_lo: f32 = 0x1.7f7d1cp-20;
const ln2: f32 = 0x1.62e430p-1;

// --- fast tier: one PTX instruction each ---

/// `__nv_fast_sinf` — sin.approx.f32.
pub fn sinFast(x: f32) f32 {
    return asm_gen.@"float".@"sin_approx_f32"(x);
}

/// `__nv_fast_cosf` — cos.approx.f32.
pub fn cosFast(x: f32) f32 {
    return asm_gen.@"float".@"cos_approx_f32"(x);
}

/// `__nv_fast_log2f` — lg2.approx.f32.
pub fn log2Fast(x: f32) f32 {
    return asm_gen.@"float".@"lg2_approx_f32"(x);
}

/// `__nv_fast_expf` — ex2.approx.f32(x * log2(e)).
pub fn expFast(x: f32) f32 {
    return asm_gen.@"float".@"ex2_approx_f32"(x * log2e);
}

/// `__nv_fast_logf` — lg2.approx.f32(x) * ln(2).
pub fn logFast(x: f32) f32 {
    return asm_gen.@"float".@"lg2_approx_f32"(x) * ln2;
}

/// `__nv_fast_tanhf` — tanh.approx.f32 (sm_75+).
pub fn tanhFast(x: f32) f32 {
    return asm_gen.@"float".@"tanh_approx_f32"(x);
}

/// `__frsqrt_rn`'s approx sibling — rsqrt.approx.f32.
pub fn rsqrtFast(x: f32) f32 {
    return asm_gen.@"float".@"rsqrt_approx_f32"(x);
}

// --- native tier: Zig builtins that already lower to one PTX instruction ---

/// `__nv_fsqrt_rn` — sqrt.rn.f32 (IEEE correctly rounded).
pub fn sqrt(x: f32) f32 {
    return @sqrt(x);
}

/// `__nv_fmaxf` — max.f32.
pub fn fmax(a: f32, b: f32) f32 {
    return @max(a, b);
}

/// `__nv_fminf` — min.f32.
pub fn fmin(a: f32, b: f32) f32 {
    return @min(a, b);
}

/// `__nv_fabsf` — abs.f32.
pub fn fabs(x: f32) f32 {
    return @abs(x);
}

/// `__nv_floorf` — cvt.rmi.f32.f32.
pub fn floor(x: f32) f32 {
    return @floor(x);
}

// --- software tier: no PTX instruction exists ---

/// Quadrant-reduce x by pi/2. Returns r in [-pi/4, pi/4] and the quadrant
/// count. Done in f64: for f32 inputs the double-precision reduction is
/// exact enough that the f32 result is unaffected until |x| approaches
/// 2^24, beyond which we accept the drift (libdevice switches to a slow
/// path there; we don't).
fn reducePi2(x: f32) struct { r: f32, q: i32 } {
    const xd: f64 = x;
    const q: i32 = @intFromFloat(@round(xd * two_over_pi_f64));
    const r: f32 = @floatCast(xd - @as(f64, @floatFromInt(q)) * pi_over_2_f64);
    return .{ .r = r, .q = q };
}

fn sinPoly(r: f32) f32 {
    const z = r * r;
    // musl __sindf coefficients.
    return r + r * z * (-0x1.555548p-3 + z * (0x1.110df4p-7 + z * -0x1.9f42eap-13));
}

fn cosPoly(r: f32) f32 {
    const z = r * r;
    // musl __cosdf coefficients.
    return 1 + z * (-0x1.fffff6p-2 + z * (0x1.5554b6p-5 + z * (-0x1.6c0c1ep-10 + z * 0x1.99342ep-16)));
}

/// `__nv_sinf` — a few ulps for moderate |x|; see reducePi2 for the domain.
pub fn sin(x: f32) f32 {
    const rq = reducePi2(x);
    return switch (@as(u2, @truncate(@as(u32, @bitCast(rq.q))))) {
        0 => sinPoly(rq.r),
        1 => cosPoly(rq.r),
        2 => -sinPoly(rq.r),
        3 => -cosPoly(rq.r),
    };
}

/// `__nv_cosf`.
pub fn cos(x: f32) f32 {
    const rq = reducePi2(x);
    return switch (@as(u2, @truncate(@as(u32, @bitCast(rq.q))))) {
        0 => cosPoly(rq.r),
        1 => -sinPoly(rq.r),
        2 => -cosPoly(rq.r),
        3 => sinPoly(rq.r),
    };
}

/// `__nv_expf`. e^x = 2^n * P(r) with r = x - n*ln2 (hi/lo split), n built
/// by integer exponent injection. |n| is clamped so the injection stays in
/// the normal range; past that the true result overflows/underflows anyway.
pub fn exp(x: f32) f32 {
    const n0: i32 = @intFromFloat(@round(x * log2e));
    const n = @min(@max(n0, -126), 127);
    const nf: f32 = @floatFromInt(n);
    const r = (x - nf * ln2_hi) - nf * ln2_lo;
    // cephes expf polynomial for e^r on [-ln2/2, ln2/2].
    const p = 1 + r * (1 + r * (0x1.000000p-1 + r * (0x1.55557ap-3 + r * (0x1.555736p-5 + r * 0x1.122e9cp-7))));
    const scale: f32 = @bitCast(@as(u32, @intCast(n + 127)) << 23);
    return p * scale;
}

/// `__nv_logf` for x > 0. Splits x = m * 2^e with m in [sqrt(2)/2, sqrt(2)),
/// then log(1+f) by the musl logf polynomial. x <= 0 yields nan / -inf by
/// the same arithmetic libdevice would produce (no explicit domain check).
pub fn log(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    var e: i32 = @as(i32, @intCast((bits >> 23) & 0xff)) - 126;
    var m: f32 = @bitCast((bits & 0x007fffff) | 0x3f000000); // [0.5, 1)
    if (m < 0x1.6a09e6p-1) { // sqrt(2)/2: recentre to [sqrt2/2, sqrt2)
        m *= 2;
        e -= 1;
    }
    const f = m - 1;
    // musl logf coefficients; the -f²/2 term is folded into the polynomial.
    const p = f * (1 + f * (-0x1.000000p-1 + f * (0x1.5555b2p-2 + f * (-0x1.000ca6p-2 + f * (0x1.996e12p-3 + f * (-0x1.24b0c2p-3 + f * 0x1.f06f46p-4))))));
    return @as(f32, @floatFromInt(e)) * ln2 + p;
}

/// `__nv_atanf`. |x| > 1 folds to pi/2 - atan(1/|x|); the cephes atanf
/// polynomial covers [0, 1].
pub fn atan(x: f32) f32 {
    const ax = @abs(x);
    const big = ax > 1;
    const t = if (big) 1 / ax else ax;
    const z = t * t;
    const p = t + t * z * (-0x1.5554c4p-2 + z * (0x1.9978c6p-3 + z * (-0x1.22c6ccp-3 + z * (0x1.b5a7b4p-4 + z * (-0x1.34ab32p-4 + z * (0x1.5de2c8p-5 + z * -0x1.07a1b4p-7))))));
    const v = if (big) pi_2 - p else p;
    return if (x < 0) -v else v;
}

/// `__nv_cbrtf`. Halley-style Newton from the standard 1/3-exponent bit
/// seed; three iterations land within a few ulps for normal inputs.
pub fn cbrt(x: f32) f32 {
    const ax = @abs(x);
    var y: f32 = @bitCast(@as(u32, 0x2a5137a0) + (@as(u32, @bitCast(ax)) / 3));
    y = (2 * y + ax / (y * y)) / 3;
    y = (2 * y + ax / (y * y)) / 3;
    y = y + (ax - y * y * y) / (3 * y * y) * 0.5;
    return if (x < 0) -y else y;
}
