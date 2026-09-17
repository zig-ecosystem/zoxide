// Freestanding CUDA kernel: c = a + b, one f32 element per thread.

fn threadIdxX() u32 {
    return asm volatile ("mov.u32 $0, %tid.x;"
        : [ret] "=r" (-> u32),
    );
}

fn blockIdxX() u32 {
    return asm volatile ("mov.u32 $0, %ctaid.x;"
        : [ret] "=r" (-> u32),
    );
}

fn blockDimX() u32 {
    return asm volatile ("mov.u32 $0, %ntid.x;"
        : [ret] "=r" (-> u32),
    );
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
