//! CUDA constant memory: the PTX `.const` state space, written by the host.
//!
//! Distinct from `dev_global.zig`, which uses ordinary global memory read through
//! the read-only data cache. This one is the real constant bank: a separate 64 KB
//! window with its own addressing.
//!
//! Not a faster one, though — `const_vs_ldg` measured 1.02x against
//! `ld.global.nc` for a warp-uniform read on H20, and 0.30x when the index
//! diverges. Use a bank for `__constant__` semantics and to keep tables out of
//! the parameter list, not for throughput.
//!
//! Zig cannot declare it. `var x addrspace(.constant)` is rejected outright, and
//! the immutable form silently lands in `.global` with `ld.global.nc`, which is
//! a different thing that also cannot be written from the host. The declaration
//! is therefore emitted as module-scope inline assembly, which Zig passes
//! through verbatim.
//!
//! Note what that means for the symbol name: module-scope asm is not mangled, so
//! the host looks up `const_scales`, not `const_bank_$_const_scales`. Unlike a
//! kernel or a device global.
//!
//! `pad_before` is not decoration. `ld.const` with a bare byte offset reads from
//! the start of the constant window, which works by accident when the bank is
//! the only object in it. Padding ahead of the table means a kernel that gets
//! the addressing wrong produces wrong values instead of passing.
const cuda = @import("cuda");

const bank_len = 64;
const Scales = cuda.ConstBank("const_scales", f32, bank_len);

comptime {
    // Deliberately displaces the table from offset 0; see above.
    asm (".const .align 4 .b8 pad_before[64];");
    asm (Scales.declaration);
}

pub fn scaleByConst(out: [*]f32, in: [*]const f32, n: u32) callconv(.kernel) void {
    const i = cuda.globalThreadId();
    if (i < n) out[i] = in[i] * Scales.get(i % bank_len);
}

comptime {
    _ = cuda.Keep(.{&scaleByConst}).__zoxide_keep_kernels;
}
