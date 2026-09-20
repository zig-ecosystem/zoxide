const cuda = @import("cuda");

/// printf smoke kernel: each thread of block 0 prints its tid and a float.
/// Run with 1 block of 8 threads; expected host stdout after sync:
///   debug_print: tid=0 x=1.500000
///   ... (tid 0..7)
pub fn debugPrint(marker: u32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    if (cuda.blockIdx().x == 0 and tid < 8) {
        cuda.printf("debug_print: tid=%d x=%f marker=%d\n", .{ tid, @as(f32, 1.5), marker });
    }
}

comptime {
    _ = cuda.Keep(.{&debugPrint}).__zoxide_keep_kernels;
}
