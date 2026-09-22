//! Downstream host program: load the PTX built alongside it, launch the kernel
//! through the compile-time-checked API, verify on the host.
//!
//! Also exercises the parts of the host API that a real pipeline needs and a
//! demo does not: a non-blocking stream, page-locked staging buffers so the
//! transfers actually overlap, a device-side fill, and the driver's own
//! occupancy figures.
const std = @import("std");
const gpu = @import("zoxide_host");
const api = @import("kernels_abi");

const ptx = @embedFile("kernel_ptx");

const n = 1 << 20;
const block = 256;

pub fn main(init: std.process.Init) !u8 {
    _ = init;

    var drv = gpu.Driver.load() catch |e| {
        std.debug.print("no libcuda ({s}) — needs a GPU host\n", .{@errorName(e)});
        return 0;
    };
    defer drv.unload();
    var ctx = try gpu.Context.init(&drv);

    var name_buf: [128]u8 = undefined;
    const info = try ctx.info();
    std.debug.print("device: {s} ({d} SMs, {d} MB L2)\n", .{
        ctx.name(&name_buf),
        info.sms,
        info.l2_bytes >> 20,
    });

    const mod = try ctx.moduleFromPtx(ptx);
    // "kernel" is kernel.zig's stem — the PTX symbol prefix comes from the root
    // source file's name, not the build artifact's.
    const scale = mod.kernel(api.scale, gpu.symbol("kernel", "scale")) catch |e| {
        std.debug.print("FAIL: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
        return 1;
    };

    // Resource use and occupancy as the driver sees them, not as inferred from
    // the PTX — the register limit matters and is invisible in source.
    const res = try scale.resources();
    const blocks_per_sm = try scale.occupancy(block, 0);
    std.debug.print("kernel: {d} regs/thread, {d} B shared, {d} blocks/SM at {d} threads\n", .{
        res.regs_per_thread, res.shared_bytes, blocks_per_sm, block,
    });
    if (res.local_bytes != 0) {
        std.debug.print("warning: {d} B/thread spilled\n", .{res.local_bytes});
        return 1;
    }

    // Page-locked staging. Ordinary heap memory would make the async copies
    // below synchronous without saying so.
    const hx = try ctx.allocPinned(f32, n);
    defer hx.free();
    const hy = try ctx.allocPinned(f32, n);
    defer hy.free();
    for (hx.items, 0..) |*v, i| v.* = @floatFromInt(i % 1000);

    const dx = try ctx.allocSlice(f32, n);
    defer ctx.freeSlice(dx);
    const dy = try ctx.allocSlice(f32, n);
    defer ctx.freeSlice(dy);

    const stream = try ctx.createStream(true);
    defer stream.destroy();

    // Poison the output first, on the device, so a kernel that fails to write
    // cannot pass by leaving zeros behind.
    try ctx.fillBytesAsync(dy, 0xff, stream);
    try ctx.uploadAsync(dx, hx.items, stream);
    try scale.launchOn(stream, .{ .x = gpu.gridFor(n, block) }, .{ .x = block }, 0, .{
        dx, dy, @as(f32, 2.5), @as(u32, n),
    });
    try ctx.downloadAsync(hy.items, dy, stream);
    try stream.sync();

    var bad: usize = 0;
    for (hx.items, hy.items) |x, y| {
        if (y != x * 2.5) bad += 1;
    }
    if (bad != 0) {
        std.debug.print("FAIL: {d}/{d} elements wrong\n", .{ bad, n });
        return 1;
    }
    std.debug.print("PASS: downstream typed launch on a stream, {d} elements exact\n", .{n});
    return 0;
}
