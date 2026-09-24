//! Device global written by the host, read by name on the device.
//!
//! This is the pattern behind CUDA's `__constant__` / `cudaMemcpyToSymbol`: the
//! host uploads a small table once, then every launch reads it without it
//! occupying a kernel parameter.
//!
//! Zig cannot express it on its own. A module-scope `var` reaches the PTX as
//! plain `.global`, with no `.visible`, so ptxas keeps the symbol module-local
//! and `cuModuleGetGlobal` cannot resolve it. Asking for external linkage fails
//! in the NVPTX backend every way it can be asked (zig 0.16.0 / LLVM 21.1.8):
//! `export var x addrspace(.global)` and `@export(&x, .strong)` both give
//! "Alias and aliasee types don't match", and a plain `export var x` aborts the
//! compiler with "NVPTX aliasee must be a non-kernel function definition" —
//! because Zig lowers `export` on a variable to an LLVM alias and NVPTX only
//! accepts aliases whose aliasee is a non-kernel function.
//!
//! So the PTX is post-processed to add `.visible`:
//!
//!     zoxide ptx-export kernels/dev_global.ptx --globals dev_bias
//!
//! Two things have to hold for this to actually work, and only a GPU can confir
//! the second:
//!
//!   1. LLVM must not fold the initialiser into the code. It does not: nothing
//!      in the module writes `dev_bias`, yet the kernel still emits a real
//!      `ld.global.nc.b32` from the symbol. `.nc` is the read-only data cache
//!      (what `__ldg` gives in CUDA), which is safe here because host writes
//!      land between launches and the cache does not survive a kernel boundary.
//!   2. ptxas must accept the promoted declaration and expose the symbol, so
//!      `cuModuleGetGlobal` finds it.
//!
//! `zoxide run kernels/dev_global.ptx` checks both, and launches twice with
//! different tables so that a symbol which resolves but ignores host writes
//! still fails.
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
