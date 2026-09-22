const cuda = @import("cuda");
const api = @import("examples_abi");

/// Baseline SGEMM: one thread per output element, straight global-memory
/// reads. C = A * B, all N x N row-major f32.
pub fn sgemmNaive(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const gid = cuda.globalThreadId();
    if (gid >= n *% n) return;
    const row = gid / n;
    const col = gid % n;
    var acc: f32 = 0;
    var k: u32 = 0;
    while (k < n) : (k += 1) {
        acc += a[row * n + k] * b[k * n + col];
    }
    c[gid] = acc;
}

comptime {
    // Drift from the signature `zoxide bench` launches through is a compile
    // error here rather than a silently mis-packed argument list.
    cuda.abi.assertMatches(api.sgemm, @TypeOf(sgemmNaive));
    _ = cuda.Keep(.{&sgemmNaive}).__zoxide_keep_kernels;
}
