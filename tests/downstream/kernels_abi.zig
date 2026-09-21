//! Kernel signatures, imported by both the device kernel and the host program.
//!
//! Declared without `callconv(.kernel)`: that convention resolves per target and
//! is `unreachable` on host architectures, so the type cannot be named there.
//! Only the parameter list matters for launching, and the device side asserts its
//! definition matches with `abi.assertMatches`.

pub const scale = fn (x: [*]const f32, y: [*]f32, k: f32, n: u32) void;
