const cuda = @import("cuda");

pub fn vectorAdd(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const gid = cuda.globalThreadId();
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

comptime {
    _ = cuda.Keep(.{&vectorAdd}).__zoxide_keep_kernels;
}
