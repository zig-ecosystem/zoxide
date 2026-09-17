// Freestanding CUDA kernel: c = a + b, one f32 element per thread.
//
// Thread-index special registers are read via LLVM NVVM intrinsics rather
// than inline asm: zig asm templates are emitted verbatim (no `$0`/`%0`
// operand substitution), so `mov.u32 $0, %tid.x;` would leak a literal `$0`
// into the PTX and ptxas would reject it ("Unknown symbol '$0'").

extern fn @"llvm.nvvm.read.ptx.sreg.tid.x"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.x"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() i32;

fn threadIdxX() u32 {
    return @bitCast(@"llvm.nvvm.read.ptx.sreg.tid.x"());
}

fn blockIdxX() u32 {
    return @bitCast(@"llvm.nvvm.read.ptx.sreg.ctaid.x"());
}

fn blockDimX() u32 {
    return @bitCast(@"llvm.nvvm.read.ptx.sreg.ntid.x"());
}

pub fn vectorAdd(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const gid = blockIdxX() * blockDimX() + threadIdxX();
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

// Keep the kernel alive across DCE. `export fn` / `@export` on an
// nvptx_kernel function hits "NVPTX aliasee must be a non-kernel function
// definition" (LLVM rejects aliases to kernels), so a dummy non-kernel export
// materializes a pointer to the kernel instead. The kernel symbol is mangled
// to `kernel_$_vectorAdd` in the PTX.
export fn __zoxide_keep_kernels() *const anyopaque {
    return @ptrCast(&vectorAdd);
}
