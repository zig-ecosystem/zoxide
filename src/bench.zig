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
    /// ptxas --maxrregcount. Caps registers per thread, trading spills for
    /// occupancy; register pressure is usually what binds a tensor-core
    /// kernel's residency, not shared memory.
    max_regs: ?u32 = null,
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
    assemblePtx: *const fn (gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, in_ptx: []const u8, out_cubin: []const u8, arch: []const u8, max_regs: ?u32) anyerror!void,
    out: *std.Io.Writer,
) !u8 {
    const stem = std.fs.path.stem(std.fs.path.basename(args.input));
    const tiled = std.mem.eql(u8, stem, "sgemm_tiled");
    const naive = std.mem.eql(u8, stem, "sgemm_naive");
    const reg = std.mem.eql(u8, stem, "sgemm_reg");
    const opt = std.mem.eql(u8, stem, "sgemm_opt");
    const opt2 = std.mem.eql(u8, stem, "sgemm_opt2");
    const swz = std.mem.eql(u8, stem, "sgemm_swz");
    const hgemm1 = std.mem.eql(u8, stem, "hgemm_mma");
    const hgemm2 = std.mem.eql(u8, stem, "hgemm_mma2");
    const hgemm3 = std.mem.eql(u8, stem, "hgemm_wgmma");
    const hgemm4 = std.mem.eql(u8, stem, "hgemm_wgmma2");
    const hgemm5 = std.mem.eql(u8, stem, "hgemm_wgmma3");
    const wgmma = hgemm3 or hgemm4 or hgemm5;
    const hgemm = hgemm1 or hgemm2 or wgmma;
    // Block tile (m, n). hgemm_wgmma uses one warpgroup over a 64x128 tile;
    // the mma.sync kernels use square tiles.
    const hgemm_tile_m: usize = if (wgmma) 64 else if (hgemm2) 128 else 64;
    const hgemm_tile_n: usize = if (wgmma) 128 else hgemm_tile_m;
    if (!tiled and !naive and !reg and !opt and !opt2 and !swz and !hgemm) {
        try out.print("error: bench supports sgemm_*, hgemm_mma* or hgemm_wgmma inputs (got '{s}')\n", .{args.input});
        return 1;
    }
    const regblocked = reg or opt or opt2 or swz;
    if (hgemm and args.n % 16 != 0) {
        try out.print("error: hgemm_mma requires n % 16 == 0 (got {d})\n", .{args.n});
        return 1;
    }
    // hgemm_wgmma tiles N by 128 and has no bounds guard in the epilogue.
    if (wgmma and args.n % 128 != 0) {
        try out.print("error: hgemm_wgmma* requires n % 128 == 0 (got {d})\n", .{args.n});
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
        assemblePtx(gpa, io, env, args.input, p, args.arch, args.max_regs) catch {
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

    const kernel_name = args.kernel_name orelse if (hgemm5)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmWgmma3", .{stem})
    else if (hgemm4)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmWgmma2", .{stem})
    else if (hgemm3)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmWgmma", .{stem})
    else if (hgemm2)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmMma2", .{stem})
    else if (hgemm)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmMma", .{stem})
    else
        try std.fmt.allocPrint(gpa, "{s}_$_sgemm{s}", .{ stem, if (tiled) "Tiled" else if (reg) "Reg" else if (opt) "Opt" else if (opt2) "Opt2" else if (swz) "Swz" else "Naive" });
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
    const dev_info: ?cu.Context.Info = ctx.info() catch null;
    if (dev_info) |di| {
        try out.print("  {d} SMs, {d:.0} MB L2, {d:.0} KB shared/SM\n", .{
            di.sms,
            @as(f64, @floatFromInt(di.l2_bytes)) / (1 << 20),
            @as(f64, @floatFromInt(di.shared_per_sm)) / (1 << 10),
        });
    }

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

    // Resource use and occupancy straight from the driver. Deriving these by
    // reading shared-memory totals out of the PTX ignores the register limit and
    // goes stale as soon as the kernel changes.
    reportOccupancy(func, if (hgemm or regblocked) 128 else 1024, dev_info, out) catch {};

    if (hgemm) {
        return runHgemm(gpa, &ctx, func, n, args.iters, out, hgemm_tile_m, hgemm_tile_n, dev_info);
    }

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
    const grid_reg: u32 = @intCast((n + 127) / 128);
    const start = try ctx.eventCreate();
    defer start.destroy();
    const stop = try ctx.eventCreate();
    defer stop.destroy();

    var best_ms: f32 = std.math.floatMax(f32);
    var it: u32 = 0;
    while (it < args.iters) : (it += 1) {
        try start.record();
        if (regblocked) {
            try func.launch(grid_reg, grid_reg, 1, 16, 16, 1, &params);
        } else if (tiled) {
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
    return finishVerify(gpa, &ctx, dc, a, b, c, n, out);

}

fn finishVerify(gpa: std.mem.Allocator, ctx: *cu.Context, dc: u64, a: []f32, b: []f32, c: []f32, n: usize, out: *std.Io.Writer) !u8 {
    _ = gpa;
    const elems = n * n;
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

const h20_fp16_peak_gflops: f64 = 148000;

fn reportOccupancy(func: cu.Function, block: u32, dev_info: ?cu.Context.Info, out: *std.Io.Writer) !void {
    const regs = try func.attr(.num_regs);
    const shared = try func.attr(.shared_size_bytes);
    const spill = try func.attr(.local_size_bytes);
    const blocks = try func.occupancy(block, 0);
    try out.print("kernel: {d} regs/thread, {d} B static shared, {d} blocks/SM at {d} threads", .{ regs, shared, blocks, block });
    if (dev_info) |di| {
        const threads = blocks * block;
        try out.print(" ({d}/2048 threads = {d:.0}% occupancy, {d} SMs)", .{
            threads,
            @as(f64, @floatFromInt(threads)) / 2048.0 * 100,
            di.sms,
        });
    }
    try out.print("\n", .{});
    if (spill != 0) {
        try out.print("  warning: {d} B/thread spilled to local memory\n", .{spill});
    }
}

/// HGEMM harness: f16 inputs (small ints, exact in f16), f32 accumulate.
fn runHgemm(gpa: std.mem.Allocator, ctx: *cu.Context, func: cu.Function, n: usize, iters: u32, out: *std.Io.Writer, tile_m: usize, tile_n: usize, dev_info: ?cu.Context.Info) !u8 {
    const elems = n * n;
    const ah = try gpa.alloc(f16, elems);
    defer gpa.free(ah);
    const bh = try gpa.alloc(f16, elems);
    defer gpa.free(bh);
    const a = try gpa.alloc(f32, elems); // f32 mirrors for the CPU reference
    defer gpa.free(a);
    const b = try gpa.alloc(f32, elems);
    defer gpa.free(b);
    var rng: u32 = 0x2468ace0;
    for (ah, 0..) |*v, i| {
        const x: i32 = @intCast(xorshift(&rng) % 5);
        const val: f32 = @floatFromInt(x - 2); // -2..2, exact in f16
        v.* = @floatCast(val);
        a[i] = val;
    }
    for (bh, 0..) |*v, i| {
        const x: i32 = @intCast(xorshift(&rng) % 5);
        const val: f32 = @floatFromInt(x - 2);
        v.* = @floatCast(val);
        b[i] = val;
    }

    const hbytes = elems * @sizeOf(f16);
    const cbytes = elems * @sizeOf(f32);
    const da = try ctx.alloc(hbytes);
    defer ctx.free(da);
    const db = try ctx.alloc(hbytes);
    defer ctx.free(db);
    const dc = try ctx.alloc(cbytes);
    defer ctx.free(dc);
    try ctx.copyHtoD(da, std.mem.sliceAsBytes(ah));
    try ctx.copyHtoD(db, std.mem.sliceAsBytes(bh));

    var arg_a = da;
    var arg_b = db;
    var arg_c = dc;
    var arg_n: u32 = @intCast(n);
    var params = [_]?*anyopaque{ &arg_a, &arg_b, &arg_c, &arg_n };
    // grid.x walks N, grid.y walks M.
    const grid_x: u32 = @intCast((n + tile_n - 1) / tile_n);
    const grid_y: u32 = @intCast((n + tile_m - 1) / tile_m);

    const start = try ctx.eventCreate();
    defer start.destroy();
    const stop = try ctx.eventCreate();
    defer stop.destroy();
    var best_ms: f32 = std.math.floatMax(f32);
    var it: u32 = 0;
    while (it < iters) : (it += 1) {
        try start.record();
        try func.launch(grid_x, grid_y, 1, 128, 1, 1, &params);
        try stop.record();
        try stop.sync();
        const ms = try start.elapsedMs(stop);
        if (ms < best_ms) best_ms = ms;
    }

    const flops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const gflops = flops / (@as(f64, best_ms) * 1e6);
    try out.print("bench: hgemm(tile={d}x{d}) n={d} iters={d}\n", .{ tile_m, tile_n, n, iters });
    try out.print("best: {d:.3} ms over {d} iters\n", .{ best_ms, iters });
    try out.print("GFLOPS: {d:.1} ({d:.1}% of H20 FP16 tensor peak ~{d:.0} GFLOPS)\n", .{ gflops, gflops / h20_fp16_peak_gflops * 100, h20_fp16_peak_gflops });

    // Global-traffic accounting, and whether the operands still fit in L2.
    // Each block streams its whole tile-row of A and tile-column of B, so
    // demand traffic is (n/tile_m)*(n/tile_n) blocks * (tile_m + tile_n) * n
    // f16 elements. Comparing that against L2 is the cheap way to tell a
    // bandwidth-bound result from a compute-bound one: once A and B fit in L2,
    // repeat reads stop reaching DRAM, so if throughput jumps at smaller n the
    // kernel was traffic-limited.
    const blocks = ((n + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n);
    const demand_bytes: f64 = @floatFromInt(blocks * (tile_m + tile_n) * n * 2);
    const gbs = demand_bytes / (@as(f64, best_ms) * 1e-3) / 1e9;
    try out.print("global reads: {d:.2} GB demand -> {d:.2} TB/s (L2 absorbs repeats; DRAM is lower)\n", .{ demand_bytes / 1e9, gbs / 1000 });
    if (dev_info) |di| {
        const operands_bytes: f64 = @floatFromInt(2 * n * n * 2); // A + B in f16
        const l2: f64 = @floatFromInt(di.l2_bytes);
        try out.print("A+B working set: {d:.0} MB vs {d:.0} MB L2 — {s}\n", .{
            operands_bytes / (1 << 20),
            l2 / (1 << 20),
            if (operands_bytes <= l2) "fits, so repeat reads stay on chip" else "exceeds L2, repeat reads reach DRAM",
        });
    }

    const c = try gpa.alloc(f32, elems);
    defer gpa.free(c);
    @memset(c, 0);
    return finishVerify(gpa, ctx, dc, a, b, c, n, out);
}
