//! Downstream host program: load the cubin built alongside it, launch the
//! kernel through the compile-time-checked API, verify on the host.
const std = @import("std");
const gpu = @import("zoxide_host");
const api = @import("kernels_abi");

const ptx = @embedFile("kernel_ptx");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const n = 1 << 20;
    const block = 256;

    var drv = gpu.Driver.load() catch |e| {
        std.debug.print("no libcuda ({s}) — needs a GPU host\n", .{@errorName(e)});
        return 0;
    };
    defer drv.unload();
    var ctx = try gpu.Context.init(&drv);

    var name_buf: [128]u8 = undefined;
    std.debug.print("device: {s}\n", .{ctx.name(&name_buf)});

    const mod = try ctx.moduleFromPtx(ptx);
    const scale = try mod.kernel(api.scale, gpu.symbol("downstream_kernel", "scale"));

    const host_x = try gpa.alloc(f32, n);
    defer gpa.free(host_x);
    const host_y = try gpa.alloc(f32, n);
    defer gpa.free(host_y);
    for (host_x, 0..) |*v, i| v.* = @floatFromInt(i % 1000);

    const dx = try ctx.allocSlice(f32, n);
    defer ctx.freeSlice(dx);
    const dy = try ctx.allocSlice(f32, n);
    defer ctx.freeSlice(dy);
    try ctx.upload(dx, host_x);

    try scale.launch(
        .{ .x = gpu.gridFor(n, block) },
        .{ .x = block },
        .{ dx, dy, @as(f32, 2.5), @as(u32, n) },
    );
    try ctx.synchronize();
    try ctx.download(host_y, dy);

    var bad: usize = 0;
    for (host_x, host_y) |x, y| {
        if (y != x * 2.5) bad += 1;
    }
    if (bad != 0) {
        std.debug.print("FAIL: {d}/{d} elements wrong\n", .{ bad, n });
        return 1;
    }
    std.debug.print("PASS: downstream typed launch, {d} elements exact\n", .{n});
    return 0;
}
