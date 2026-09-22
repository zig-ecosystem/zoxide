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
