//! `zoxide bench` — SGEMM performance harness with GPU event timing.

const std = @import("std");
const cu = @import("cuda_driver.zig");

const h20_fp32_peak_gflops: f64 = 44000;

pub const BenchArgs = struct {
    input: []const u8,
    n: usize = 4096,
    iters: u32 = 10,
    kernel_name: ?[]const u8 = null,
    arch: []const u8 = "sm_90",
};

fn xorshift(state: *u32) u32 {
    var x = state.*;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state.* = x;
    return x;
}

fn fillRandom(buf: []f32, seed: u32) void {
    var s = seed;
    for (buf) |*v| {
        // uniform in [-1, 1)
        v.* = @as(f32, @floatFromInt(xorshift(&s) >> 8)) / @as(f32, 1 << 23) - 1.0;
    }
}

pub fn benchMain(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *std.process.Environ.Map,
    args: BenchArgs,
    assemblePtx: *const fn (gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, in_ptx: []const u8, out_cubin: []const u8, arch: []const u8) anyerror!void,
    out: *std.Io.Writer,
) !u8 {
    const stem = std.fs.path.stem(std.fs.path.basename(args.input));
    const tiled = std.mem.eql(u8, stem, "sgemm_tiled");
    const naive = std.mem.eql(u8, stem, "sgemm_naive");
    if (!tiled and !naive) {
        try out.print("error: bench supports sgemm_naive/sgemm_tiled inputs (got '{s}')\n", .{args.input});
        return 1;
    }
    const n = args.n;

    // Resolve input to cubin bytes (assemble via ptxas when given PTX).
    var tmp_cubin: ?[]u8 = null;
    defer if (tmp_cubin) |p| {
        std.Io.Dir.deleteFileAbsolute(io, p) catch {};
        gpa.free(p);
    };
    var cubin_path: []const u8 = args.input;
    if (std.mem.endsWith(u8, args.input, ".ptx")) {
        const p = try std.fmt.allocPrint(gpa, "/tmp/zoxide-bench-{d}.cubin", .{std.c.getpid()});
        tmp_cubin = p;
        try out.print("assembling {s} -> {s} (ptxas, {s})\n", .{ args.input, p, args.arch });
        assemblePtx(gpa, io, env, args.input, p, args.arch) catch {
            try out.print("error: failed to assemble PTX (is ptxas available? see 'zoxide doctor')\n", .{});
            return 1;
        };
        cubin_path = p;
    } else if (!std.mem.endsWith(u8, args.input, ".cubin")) {
        try out.print("error: input must be a .ptx or .cubin file\n", .{});
        return 1;
    }

    const cubin = std.Io.Dir.cwd().readFileAlloc(io, cubin_path, gpa, .unlimited) catch |e| {
        try out.print("error: cannot read '{s}': {s}\n", .{ cubin_path, @errorName(e) });
        return 1;
    };
    defer gpa.free(cubin);

    const kernel_name = args.kernel_name orelse
        try std.fmt.allocPrint(gpa, "{s}_$_sgemm{s}", .{ stem, if (tiled) "Tiled" else "Naive" });
    defer if (args.kernel_name == null) gpa.free(kernel_name);

    var drv = cu.Driver.load() catch |e| {
        switch (e) {
            error.LibraryNotFound => try out.print(
                \\error: libcuda not found (tried libcuda.so.1, libcuda.so, libcuda.dylib).
                \\  bench needs an NVIDIA GPU. Use the gnu dynamic build on the pod.
                \\
            , .{}),
            else => try out.print("error: failed to load libcuda: {s}\n", .{@errorName(e)}),
        }
        return 1;
    };
    defer drv.unload();

    var ctx = cu.Context.init(&drv) catch {
        try out.print("error: CUDA init failed: {s}\n", .{drv.lastError()});
        return 1;
    };
    var name_buf: [128]u8 = undefined;
    try out.print("device: {s}\n", .{ctx.name(&name_buf)});

    const mod = ctx.module(cubin) catch {
        try out.print("error: cuModuleLoadData failed: {s}\n", .{drv.lastError()});
        return 1;
    };
    const namez = try gpa.dupeZ(u8, kernel_name);
    defer gpa.free(namez);
    const func = mod.function(namez) catch {
        try out.print("error: kernel '{s}' not found: {s}\n", .{ kernel_name, drv.lastError() });
        return 1;
    };

    // Host buffers.
    const elems = n * n;
    const bytes = elems * @sizeOf(f32);
    const a = try gpa.alloc(f32, elems);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, elems);
    defer gpa.free(b);
    const c = try gpa.alloc(f32, elems);
    defer gpa.free(c);
    fillRandom(a, 0x12345678);
    fillRandom(b, 0x9abcdef0);
    @memset(c, 0);

    const da = try ctx.alloc(bytes);
    defer ctx.free(da);
    const db = try ctx.alloc(bytes);
    defer ctx.free(db);
    const dc = try ctx.alloc(bytes);
    defer ctx.free(dc);
    try ctx.copyHtoD(da, std.mem.sliceAsBytes(a));
    try ctx.copyHtoD(db, std.mem.sliceAsBytes(b));
    try ctx.copyHtoD(dc, std.mem.sliceAsBytes(c));

    var arg_a = da;
    var arg_b = db;
    var arg_c = dc;
    var arg_n: u32 = @intCast(n);
    var params = [_]?*anyopaque{ &arg_a, &arg_b, &arg_c, &arg_n };

    const grid_x: u32 = @intCast((n + 31) / 32);
    const grid_y: u32 = grid_x;
    const start = try ctx.eventCreate();
    defer start.destroy();
    const stop = try ctx.eventCreate();
    defer stop.destroy();

    var best_ms: f32 = std.math.floatMax(f32);
    var it: u32 = 0;
    while (it < args.iters) : (it += 1) {
        try start.record();
        if (tiled) {
            try func.launch(grid_x, grid_y, 1, 32, 32, 1, &params);
        } else {
            try func.launch(@intCast((elems + 255) / 256), 1, 1, 256, 1, 1, &params);
        }
        try stop.record();
        try stop.sync();
        const ms = try start.elapsedMs(stop);
        if (ms < best_ms) best_ms = ms;
    }

    const flops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const gflops = flops / (@as(f64, best_ms) * 1e6);
    try out.print("bench: {s} n={d} iters={d}\n", .{ stem, n, args.iters });
    try out.print("best: {d:.3} ms over {d} iters\n", .{ best_ms, args.iters });
    try out.print("GFLOPS: {d:.1} ({d:.1}% of H20 FP32 peak ~{d:.0} GFLOPS)\n", .{ gflops, gflops / h20_fp32_peak_gflops * 100, h20_fp32_peak_gflops });

    // Verify: copy back and check 256 deterministic samples against an f64
    // CPU dot product (relative tolerance for f32 accumulation).
    try ctx.copyDtoH(std.mem.sliceAsBytes(c), dc);
    var bad: usize = 0;
    var max_rel: f64 = 0;
    var si: usize = 0;
    var rng: u32 = 0xdeadbeef;
    while (si < 256) : (si += 1) {
        const idx = xorshift(&rng) % elems;
        const row: usize = idx / n;
        const col: usize = idx % n;
        var want: f64 = 0;
        var k: usize = 0;
        while (k < n) : (k += 1) {
            want += @as(f64, a[row * n + k]) * @as(f64, b[k * n + col]);
        }
        const got: f64 = c[idx];
        const denom = @max(@abs(want), 1.0);
        const rel = @abs(got - want) / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-2) bad += 1;
    }
    if (bad > 0) {
        try out.print("FAIL: {d}/256 samples off, max rel err {d:.4}\n", .{ bad, max_rel });
        return 1;
    }
    try out.print("PASS: 256/256 samples within rel err 1e-2 (max {d:.6})\n", .{max_rel});
    return 0;
}
