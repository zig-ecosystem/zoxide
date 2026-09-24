//! Device global written by the host, read by name on the device.
//!
//! This is the pattern behind CUDA's `__constant__` / `cudaMemcpyToSymbol`: the
//! host uploads a small table once, then every launch reads it without it
//! occupying a kernel parameter.
//!
//! There is exactly one thing to get right, and it is not the one that looks
//! hard. Measured on H20 (sm_90, driver 550.90.07):
//!
//! Visibility is a non-issue. A module-scope `var` reaches the PTX as plain
//! `.global` with no `.visible`, and `cuModuleGetGlobal` resolves it anyway —
//! verified by stripping `.visible` back off a working module and watching it
//! keep working. (Which is lucky, because asking Zig for external linkage fails
//! every way it can be asked on nvptx: `export var x addrspace(.global)` and
//! `@export(&x, .strong)` give "Alias and aliasee types don't match", and a
//! plain `export var x` aborts the compiler outright with "NVPTX aliasee must be
//! a non-kernel function definition". Zig lowers `export` on a variable to an
//! LLVM alias; NVPTX only accepts aliases whose aliasee is a non-kernel
//! function.)
//!
//! The real hazard is constant folding. LLVM assumes nothing outside the module
//! writes a module-scope global, so an ordinary read gets folded against the
//! initialiser and the symbol disappears from the PTX — and whether that happens
//! depends on the initialiser, which makes it a trap rather than an error:
//!
//!     = .{ 10, 20, 30, 40 }   symbol kept, real load emitted
//!     = .{ 0, 0, 0, 0 }       folded to constant zero, symbol gone,
//!                             host uploads silently ignored
//!
//! All-zero is the natural placeholder. Reading through `cuda.ldg()` (inline asm
//! `ld.global.nc`) is opaque to the optimiser and keeps the symbol either way.
//! `.nc` is the read-only data cache, CUDA's `__ldg`; safe here because host
//! writes land between launches and that cache does not survive a kernel
//! boundary.
//!
//! `zoxide run kernels/dev_global.ptx` launches twice with different tables, so
//! a symbol that resolves but ignores host writes still fails.
const cuda = @import("cuda");

/// Read-only on the device; the host is the only writer. Deliberately not
/// `const`: a `const` would be a compile-time value with no storage to upload
/// into.
var dev_bias: [4]f32 addrspace(.global) = .{ 0, 0, 0, 0 };

pub fn addBias(out: [*]f32, n: u32) callconv(.kernel) void {
    const i = cuda.globalThreadId();
    // Must go through `ldg`: a plain `dev_bias[i % 4]` is folded against the
    // initialiser and the symbol is dropped. See the note on point 1 above.
    if (i < n) out[i] = @as(f32, @floatFromInt(i)) + cuda.ldg(f32, &dev_bias[i % 4]);
}

comptime {
    _ = cuda.Keep(.{&addBias}).__zoxide_keep_kernels;
}
