//! Launch signatures for the bundled example kernels, imported by both the
//! kernels themselves and by `zoxide bench`.
//!
//! The point is the same as for a downstream package: host and device are
//! compiled separately for different targets, so a kernel's parameters can
//! change without the launcher noticing, and `cuLaunchKernel` takes `void**` so
//! nothing checks it at runtime either. With the signature declared once, a
//! change on one side is a compile error on the other.
//!
//! This had a real motivation. `bench` used to hand-pack `?*anyopaque` arrays,
//! and when the hgemm kernels moved from byte pointers to `f16` pointers nothing
//! in the build would have complained had the packing been wrong.
//!
//! Declared without `callconv(.kernel)`: that convention resolves per target and
//! is `unreachable` on host architectures, so the type cannot be named there.

/// The f32 GEMM family: `sgemm_naive`, `sgemm_tiled`, `sgemm_reg`, `sgemm_opt`,
/// `sgemm_opt2`, `sgemm_swz`.
pub const sgemm = fn (a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) void;

/// The f16 tensor-core family: `hgemm_mma`, `hgemm_mma2`, `hgemm_wgmma`,
/// `hgemm_wgmma2`, `hgemm_wgmma3`. f16 inputs, f32 accumulation.
pub const hgemm = fn (a: [*]const f16, b: [*]const f16, c: [*]f32, n: u32) void;

/// Shape of the `const_vs_ldg` comparison, shared because the harness computes
/// the expected result from it.
///
/// These were duplicated at first, and the duplicate drifted the moment `trips`
/// was raised on the device side only. The expected value scales linearly with
/// `trips`, so the check reported `max rel err 15.000009` — exactly `64/4 - 1`,
/// which is the kind of number that names its own cause. The measurement would
/// have run and produced plausible timings either way; only the correctness
/// check caught it.
pub const const_vs_ldg = struct {
    /// Entries in the table. Both the `.const` bank and the global array.
    pub const bank_len = 64;
    /// Times each kernel walks the whole table. Sized so a launch takes long
    /// enough that launch overhead is not part of the measurement.
    pub const trips = 64;

    pub const signature = fn (out: [*]f32, in: [*]const f32, n: u32) void;
};
