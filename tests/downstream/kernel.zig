const cuda = @import("cuda");
const api = @import("kernels_abi");

pub fn scale(x: [*]const f32, y: [*]f32, k: f32, n: u32) callconv(.kernel) void {
    const gid = cuda.globalThreadId();
    if (gid < n) y[gid] = x[gid] * k;
}

comptime {
    // Drift between this definition and the shared declaration the host launches
    // through is a compile error here, not wrong numbers at runtime.
    cuda.abi.assertMatches(api.scale, @TypeOf(scale));
    _ = cuda.Keep(.{&scale}).__zoxide_keep_kernels;
}
