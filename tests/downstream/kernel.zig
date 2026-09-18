const cuda = @import("cuda");

pub fn scale(x: [*]const f32, y: [*]f32, k: f32, n: u32) callconv(.kernel) void {
    const gid = cuda.globalThreadId();
    if (gid < n) y[gid] = x[gid] * k;
}

comptime {
    _ = cuda.Keep(.{&scale}).__zoxide_keep_kernels;
}
