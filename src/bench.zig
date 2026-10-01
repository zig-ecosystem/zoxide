//! `zoxide bench` — SGEMM performance harness with GPU event timing.

const std = @import("std");
const cu = @import("cuda_driver.zig");
// bench launches through the same public host API it ships, so the typed-launch
// path is exercised by the project's own tooling and not only by a test. It also
// means a kernel whose parameters change without examples_abi following is a
// compile error here.
const gpu = @import("host.zig");
const api = @import("examples_abi.zig");

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
    const hgemm6 = std.mem.eql(u8, stem, "hgemm_wgmma4");
    const hgemm_tma = std.mem.eql(u8, stem, "hgemm_tma");
    const hgemm_bf16 = std.mem.eql(u8, stem, "hgemm_bf16");
    const imma_s8 = std.mem.eql(u8, stem, "imma_s8");
    const imma_s4 = std.mem.eql(u8, stem, "imma_s4");
    const hgemm_sp = std.mem.eql(u8, stem, "hgemm_sp");
    const hgemm_wgmma_bf16 = std.mem.eql(u8, stem, "hgemm_wgmma_bf16");
    const imma_sp_s8 = std.mem.eql(u8, stem, "imma_sp_s8");
    const imma_sp_s4 = std.mem.eql(u8, stem, "imma_sp_s4");
    const wgmma = hgemm3 or hgemm4 or hgemm5 or hgemm6 or hgemm_tma or hgemm_wgmma_bf16;
    const hgemm = hgemm1 or hgemm2 or wgmma;
    // Block tile (m, n). hgemm_wgmma uses one warpgroup over a 64x128 tile;
    // the mma.sync kernels use square tiles.
    const hgemm_tile_m: usize = if (wgmma) 64 else if (hgemm2 or hgemm_bf16 or imma_s8 or imma_s4 or hgemm_sp or imma_sp_s8 or imma_sp_s4) 128 else 64;
    const hgemm_tile_n: usize = if (wgmma) 128 else hgemm_tile_m;
    if (!tiled and !naive and !reg and !opt and !opt2 and !swz and !hgemm and !hgemm_bf16 and !imma_s8 and !imma_s4 and !hgemm_sp and !imma_sp_s8 and !imma_sp_s4 and !hgemm_wgmma_bf16) {
        try out.print("error: bench supports sgemm_*, hgemm_mma*, hgemm_wgmma*, hgemm_tma, hgemm_bf16, imma_s*, *_sp_* or hgemm_wgmma_bf16 inputs (got '{s}')\n", .{args.input});
        return 1;
    }
    const regblocked = reg or opt or opt2 or swz;
    if ((hgemm or hgemm_bf16) and args.n % 16 != 0) {
        try out.print("error: hgemm requires n % 16 == 0 (got {d})\n", .{args.n});
        return 1;
    }
    // hgemm_wgmma tiles N by 128 and has no bounds guard in the epilogue.
    if (wgmma and args.n % 128 != 0) {
        try out.print("error: hgemm_wgmma* requires n % 128 == 0 (got {d})\n", .{args.n});
        return 1;
    }
    // imma_s8 tiles 128x128 with a 32-deep K slice and, like the wgmma kernels,
    // has no bounds guard in its epilogue — a partial tile writes out of range
    // rather than producing a wrong number, so this is a hard gate.
    if (imma_s8 and args.n % 128 != 0) {
        try out.print("error: imma_s8 requires n % 128 == 0 (got {d})\n", .{args.n});
        return 1;
    }
    // imma_s4: same 128x128 tiling and unguarded epilogue as imma_s8, and the
    // 64-deep K slice plus two-s4-per-byte packing additionally need n even —
    // n % 128 covers all of it.
    if (imma_s4 and args.n % 128 != 0) {
        try out.print("error: imma_s4 requires n % 128 == 0 (got {d})\n", .{args.n});
        return 1;
    }
    // hgemm_sp: 128x128 tiles, 16-deep dense K slice and 4-wide sparsity
    // groups, no epilogue bounds guard — n % 128 covers all three.
    if (hgemm_sp and args.n % 128 != 0) {
        try out.print("error: hgemm_sp requires n % 128 == 0 (got {d})\n", .{args.n});
        return 1;
    }
    // imma_sp_s8: 128x128 tiles, 32-deep dense K slice, 4-wide sparsity
    // groups, unguarded epilogue — n % 128 covers all of it.
    if (imma_sp_s8 and args.n % 128 != 0) {
        try out.print("error: imma_sp_s8 requires n % 128 == 0 (got {d})\n", .{args.n});
        return 1;
    }
    // imma_sp_s4: 128x128 tiles, 64-deep dense K slice, 8-wide pair-clustered
    // sparsity chunks, unguarded epilogue — n % 128 covers all of it.
    if (imma_sp_s4 and args.n % 128 != 0) {
        try out.print("error: imma_sp_s4 requires n % 128 == 0 (got {d})\n", .{args.n});
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
        assemblePtx(gpa, io, env, args.input, p, args.arch, args.max_regs) catch |e| {
            switch (e) {
                // ptxas prints its own diagnostics before this point.
                error.PtxasRejected => try out.print("error: ptxas rejected {s} for {s} (diagnostics above)\n", .{ args.input, args.arch }),
                error.PtxasNotFound => try out.print("error: ptxas not found; see 'zoxide doctor'\n", .{}),
                else => try out.print("error: could not assemble PTX: {s}\n", .{@errorName(e)}),
            }
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

    const kernel_name = args.kernel_name orelse if (hgemm_tma)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmTma", .{stem})
    else if (hgemm_bf16)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmBf16", .{stem})
    else if (imma_s8)
        try std.fmt.allocPrint(gpa, "{s}_$_immaS8", .{stem})
    else if (imma_s4)
        try std.fmt.allocPrint(gpa, "{s}_$_immaS4", .{stem})
    else if (hgemm_sp)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmSp", .{stem})
    else if (hgemm_wgmma_bf16)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmWgmmaBf16", .{stem})
    else if (imma_sp_s8)
        try std.fmt.allocPrint(gpa, "{s}_$_immaSpS8", .{stem})
    else if (imma_sp_s4)
        try std.fmt.allocPrint(gpa, "{s}_$_immaSpS4", .{stem})
    else if (hgemm6)
        try std.fmt.allocPrint(gpa, "{s}_$_hgemmWgmma4", .{stem})
    else if (hgemm5)
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

    var ctx = gpu.Context.init(&drv) catch {
        try out.print("error: CUDA init failed: {s}\n", .{drv.lastError()});
        return 1;
    };
    var name_buf: [128]u8 = undefined;
    try out.print("device: {s}\n", .{ctx.name(&name_buf)});
    // Reported rather than swallowed: the device line is how a reader tells which
    // GPU produced a number, and silently omitting it hides that the query broke.
    const dev_info: ?cu.Context.Info = ctx.info() catch |e| blk: {
        try out.print("  (device attributes unavailable: {s}: {s})\n", .{ @errorName(e), drv.lastError() });
        break :blk null;
    };
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

    // Resource use and occupancy straight from the driver. Deriving these by
    // reading shared-memory totals out of the PTX ignores the register limit and
    // goes stale as soon as the kernel changes.

    if (hgemm_tma) {
        const kern = mod.kernel(api.hgemm_tma.signature, namez) catch |e| {
            try out.print("error: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
            return 1;
        };
        reportOccupancy(kern.inner, api.hgemm_tma.threads, dev_info, out) catch |e|
            try out.print("kernel: resource/occupancy query failed ({s}): {s}\n", .{ @errorName(e), drv.lastError() });
        return runHgemmTma(gpa, &ctx, kern, n, args.iters, out, dev_info);
    }
    if (hgemm_bf16 or hgemm_wgmma_bf16) {
        const kern = mod.kernel(api.hgemm_bf16, namez) catch |e| {
            try out.print("error: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
            return 1;
        };
        reportOccupancy(kern.inner, 128, dev_info, out) catch |e|
            try out.print("kernel: resource/occupancy query failed ({s}): {s}\n", .{ @errorName(e), drv.lastError() });
        return runHgemmBf16(gpa, &ctx, kern, n, args.iters, out, stem, hgemm_tile_m, hgemm_tile_n, dev_info);
    }
    if (imma_s8) {
        const kern = mod.kernel(api.imma_s8, namez) catch |e| {
            try out.print("error: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
            return 1;
        };
        reportOccupancy(kern.inner, 128, dev_info, out) catch |e|
            try out.print("kernel: resource/occupancy query failed ({s}): {s}\n", .{ @errorName(e), drv.lastError() });
        return runImmaS8(gpa, &ctx, kern, n, args.iters, out, hgemm_tile_m, hgemm_tile_n, dev_info);
    }
    if (imma_s4) {
        const kern = mod.kernel(api.imma_s4, namez) catch |e| {
            try out.print("error: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
            return 1;
        };
        reportOccupancy(kern.inner, 128, dev_info, out) catch |e|
            try out.print("kernel: resource/occupancy query failed ({s}): {s}\n", .{ @errorName(e), drv.lastError() });
        return runImmaS4(gpa, &ctx, kern, n, args.iters, out, hgemm_tile_m, hgemm_tile_n, dev_info);
    }
    if (hgemm_sp) {
        const kern = mod.kernel(api.hgemm_sp, namez) catch |e| {
            try out.print("error: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
            return 1;
        };
        reportOccupancy(kern.inner, 128, dev_info, out) catch |e|
            try out.print("kernel: resource/occupancy query failed ({s}): {s}\n", .{ @errorName(e), drv.lastError() });
        return runHgemmSp(gpa, &ctx, kern, n, args.iters, out, hgemm_tile_m, hgemm_tile_n, dev_info);
    }
    if (imma_sp_s8) {
        const kern = mod.kernel(api.imma_sp_s8, namez) catch |e| {
            try out.print("error: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
            return 1;
        };
        reportOccupancy(kern.inner, 128, dev_info, out) catch |e|
            try out.print("kernel: resource/occupancy query failed ({s}): {s}\n", .{ @errorName(e), drv.lastError() });
        return runImmaSpS8(gpa, &ctx, kern, n, args.iters, out, hgemm_tile_m, hgemm_tile_n, dev_info);
    }
    if (imma_sp_s4) {
        const kern = mod.kernel(api.imma_sp_s4, namez) catch |e| {
            try out.print("error: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
            return 1;
        };
        reportOccupancy(kern.inner, 128, dev_info, out) catch |e|
            try out.print("kernel: resource/occupancy query failed ({s}): {s}\n", .{ @errorName(e), drv.lastError() });
        return runImmaSpS4(gpa, &ctx, kern, n, args.iters, out, hgemm_tile_m, hgemm_tile_n, dev_info);
    }
    if (hgemm) {
        const kern = mod.kernel(api.hgemm, namez) catch |e| {
            try out.print("error: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
            return 1;
        };
        reportOccupancy(kern.inner, 128, dev_info, out) catch |e|
            try out.print("kernel: resource/occupancy query failed ({s}): {s}\n", .{ @errorName(e), drv.lastError() });
        return runHgemm(gpa, &ctx, kern, n, args.iters, out, hgemm_tile_m, hgemm_tile_n, dev_info);
    }
    const kern = mod.kernel(api.sgemm, namez) catch |e| {
        try out.print("error: {s}: {s}\n", .{ @errorName(e), drv.lastError() });
        return 1;
    };
    reportOccupancy(kern.inner, if (regblocked) 128 else 1024, dev_info, out) catch |e|
        try out.print("kernel: resource/occupancy query failed ({s}): {s}\n", .{ @errorName(e), drv.lastError() });

    // Host buffers.
    const elems = n * n;
    const a = try gpa.alloc(f32, elems);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, elems);
    defer gpa.free(b);
    const c = try gpa.alloc(f32, elems);
    defer gpa.free(c);
    fillRandom(a, 0x12345678);
    fillRandom(b, 0x9abcdef0);
    @memset(c, 0);

    const da = try ctx.allocSlice(f32, elems);
    defer ctx.freeSlice(da);
    const db = try ctx.allocSlice(f32, elems);
    defer ctx.freeSlice(db);
    const dc = try ctx.allocSlice(f32, elems);
    defer ctx.freeSlice(dc);
    try ctx.upload(da, a);
    try ctx.upload(db, b);
    try ctx.zero(dc);

    // Checked against api.sgemm at compile time.
    const kargs = .{ da, db, dc, @as(u32, @intCast(n)) };

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
            try kern.launch(.{ .x = grid_reg, .y = grid_reg }, .{ .x = 16, .y = 16 }, kargs);
        } else if (tiled) {
            try kern.launch(.{ .x = grid_x, .y = grid_y }, .{ .x = 32, .y = 32 }, kargs);
        } else {
            try kern.launch(.{ .x = @intCast((elems + 255) / 256) }, .{ .x = 256 }, kargs);
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

fn finishVerify(gpa: std.mem.Allocator, ctx: *gpu.Context, dc: gpu.Slice(f32), a: []f32, b: []f32, c: []f32, n: usize, out: *std.Io.Writer) !u8 {
    _ = gpa;
    const elems = n * n;
    try ctx.download(c, dc);
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
fn runHgemm(gpa: std.mem.Allocator, ctx: *gpu.Context, kern: gpu.Kernel(api.hgemm), n: usize, iters: u32, out: *std.Io.Writer, tile_m: usize, tile_n: usize, dev_info: ?cu.Context.Info) !u8 {
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
    const da = try ctx.allocSlice(f16, elems);
    defer ctx.freeSlice(da);
    const db = try ctx.allocSlice(f16, elems);
    defer ctx.freeSlice(db);
    const dc = try ctx.allocSlice(f32, elems);
    defer ctx.freeSlice(dc);
    try ctx.upload(da, ah);
    try ctx.upload(db, bh);
    // Poisoned rather than zeroed: a kernel that writes nothing then cannot
    // pass verification by leaving plausible zeros behind.
    try ctx.fillBytes(dc, 0xff);
    _ = .{ hbytes, cbytes };

    // Slice(f16) for A and B, Slice(f32) for C, checked against api.hgemm —
    // which is also what the kernels assert their own definitions against.
    const kargs = .{ da, db, dc, @as(u32, @intCast(n)) };
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
        try kern.launch(.{ .x = grid_x, .y = grid_y }, .{ .x = 128 }, kargs);
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

/// H20 BF16 tensor peak equals the FP16 peak (148 TFLOPS dense).
const h20_bf16_peak_gflops: f64 = 148000;

/// bf16 bit pattern of an f32 value. Zig has no native bf16; the pattern is
/// the high 16 bits of the f32 encoding (round-to-nearest is irrelevant for
/// the values used here — see below).
fn f32ToBf16Bits(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}

/// HGEMM bf16 harness: same shape as runHgemm, but inputs are u16 bf16 bit
/// patterns. Values are small integers in -2..2, which are exact in bf16
/// (8-bit mantissa), so the f64 CPU reference and the rel-err-1e-2 check are
/// not hiding any precision slack: products are exact, f32 accumulation of
/// n=4096 integer products stays far inside f32's exact-integer range, and
/// the comparison is effectively exact — the same trick hgemm_mma2 uses,
/// applied to a storage type Zig cannot name.
fn runHgemmBf16(gpa: std.mem.Allocator, ctx: *gpu.Context, kern: gpu.Kernel(api.hgemm_bf16), n: usize, iters: u32, out: *std.Io.Writer, stem: []const u8, tile_m: usize, tile_n: usize, dev_info: ?cu.Context.Info) !u8 {
    const elems = n * n;
    const ah = try gpa.alloc(u16, elems);
    defer gpa.free(ah);
    const bh = try gpa.alloc(u16, elems);
    defer gpa.free(bh);
    const a = try gpa.alloc(f32, elems); // f32 mirrors for the CPU reference
    defer gpa.free(a);
    const b = try gpa.alloc(f32, elems);
    defer gpa.free(b);
    var rng: u32 = 0x2468ace0;
    for (ah, 0..) |*v, i| {
        const x: i32 = @intCast(xorshift(&rng) % 5);
        const val: f32 = @floatFromInt(x - 2); // -2..2, exact in bf16
        v.* = f32ToBf16Bits(val);
        a[i] = val;
    }
    for (bh, 0..) |*v, i| {
        const x: i32 = @intCast(xorshift(&rng) % 5);
        const val: f32 = @floatFromInt(x - 2);
        v.* = f32ToBf16Bits(val);
        b[i] = val;
    }

    const da = try ctx.allocSlice(u16, elems);
    defer ctx.freeSlice(da);
    const db = try ctx.allocSlice(u16, elems);
    defer ctx.freeSlice(db);
    const dc = try ctx.allocSlice(f32, elems);
    defer ctx.freeSlice(dc);
    try ctx.upload(da, ah);
    try ctx.upload(db, bh);
    // Poisoned rather than zeroed: a kernel that writes nothing then cannot
    // pass verification by leaving plausible zeros behind.
    try ctx.fillBytes(dc, 0xff);

    // Slice(u16) for A and B, Slice(f32) for C, checked against
    // api.hgemm_bf16 — which is also what the kernel asserts itself against.
    const kargs = .{ da, db, dc, @as(u32, @intCast(n)) };
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
        try kern.launch(.{ .x = grid_x, .y = grid_y }, .{ .x = 128 }, kargs);
        try stop.record();
        try stop.sync();
        const ms = try start.elapsedMs(stop);
        if (ms < best_ms) best_ms = ms;
    }

    const flops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const gflops = flops / (@as(f64, best_ms) * 1e6);
    try out.print("bench: {s}(tile={d}x{d}) n={d} iters={d}\n", .{ stem, tile_m, tile_n, n, iters });
    try out.print("best: {d:.3} ms over {d} iters\n", .{ best_ms, iters });
    try out.print("GFLOPS: {d:.1} ({d:.1}% of H20 BF16 tensor peak ~{d:.0} GFLOPS)\n", .{ gflops, gflops / h20_bf16_peak_gflops * 100, h20_bf16_peak_gflops });

    const blocks = ((n + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n);
    const demand_bytes: f64 = @floatFromInt(blocks * (tile_m + tile_n) * n * 2);
    const gbs = demand_bytes / (@as(f64, best_ms) * 1e-3) / 1e9;
    try out.print("global reads: {d:.2} GB demand -> {d:.2} TB/s (L2 absorbs repeats; DRAM is lower)\n", .{ demand_bytes / 1e9, gbs / 1000 });
    if (dev_info) |di| {
        const operands_bytes: f64 = @floatFromInt(2 * n * n * 2); // A + B in bf16
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

/// H20 INT8 tensor peak, 2x the FP16 number from the same spec sheet.
const h20_int8_peak_gops: f64 = 296000;

/// IMMA s8 variant. Structurally runHgemmBf16, with one difference that is the
/// whole point of the shape: s8 x s8 -> s32 is exact, so verification is an
/// equality over the full i8 input range rather than a relative tolerance.
///
/// The full range matters. Restricting inputs to a few small values (as the
/// f16/bf16 benches do, because those need values representable in 16 bits)
/// would leave sign extension and the little-endian byte packing in the B
/// fragment gather untested — `-128` and `127` are exactly the operands that
/// catch a wrong shift or a missing `@bitCast`. Overflow is not a concern: the
/// worst-case |sum| is n * 127 * 128, which stays inside i32 for any n below
/// ~132000.
fn runImmaS8(gpa: std.mem.Allocator, ctx: *gpu.Context, kern: gpu.Kernel(api.imma_s8), n: usize, iters: u32, out: *std.Io.Writer, tile_m: usize, tile_n: usize, dev_info: ?cu.Context.Info) !u8 {
    const elems = n * n;
    const ah = try gpa.alloc(i8, elems);
    defer gpa.free(ah);
    const bh = try gpa.alloc(i8, elems);
    defer gpa.free(bh);
    var rng: u32 = 0x13579bdf;
    for (ah) |*v| v.* = @bitCast(@as(u8, @truncate(xorshift(&rng))));
    for (bh) |*v| v.* = @bitCast(@as(u8, @truncate(xorshift(&rng))));

    const da = try ctx.allocSlice(i8, elems);
    defer ctx.freeSlice(da);
    const db = try ctx.allocSlice(i8, elems);
    defer ctx.freeSlice(db);
    const dc = try ctx.allocSlice(i32, elems);
    defer ctx.freeSlice(dc);
    try ctx.upload(da, ah);
    try ctx.upload(db, bh);
    // Poisoned rather than zeroed, same reasoning as the bf16 path: 0xff as
    // i32 is -1, a value the kernel never legitimately leaves behind for this
    // input, so "wrote nothing" cannot pass.
    try ctx.fillBytes(dc, 0xff);

    const kargs = .{ da, db, dc, @as(u32, @intCast(n)) };
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
        try kern.launch(.{ .x = grid_x, .y = grid_y }, .{ .x = 128 }, kargs);
        try stop.record();
        try stop.sync();
        const ms = try start.elapsedMs(stop);
        if (ms < best_ms) best_ms = ms;
    }

    // Two integer ops per MAC, same convention as the float variants, reported
    // as GOPS because these are not floating-point operations.
    const ops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const gops = ops / (@as(f64, best_ms) * 1e6);
    try out.print("bench: imma_s8(tile={d}x{d}) n={d} iters={d}\n", .{ tile_m, tile_n, n, iters });
    try out.print("best: {d:.3} ms over {d} iters\n", .{ best_ms, iters });
    try out.print("GOPS: {d:.1} ({d:.1}% of H20 INT8 tensor peak ~{d:.0} GOPS)\n", .{ gops, gops / h20_int8_peak_gops * 100, h20_int8_peak_gops });

    const blocks = ((n + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n);
    // 1 byte per element, half the bf16 traffic for the same tiling.
    const demand_bytes: f64 = @floatFromInt(blocks * (tile_m + tile_n) * n);
    const gbs = demand_bytes / (@as(f64, best_ms) * 1e-3) / 1e9;
    try out.print("global reads: {d:.2} GB demand -> {d:.2} TB/s (L2 absorbs repeats; DRAM is lower)\n", .{ demand_bytes / 1e9, gbs / 1000 });
    if (dev_info) |di| {
        const operands_bytes: f64 = @floatFromInt(2 * n * n);
        const l2: f64 = @floatFromInt(di.l2_bytes);
        try out.print("A+B working set: {d:.0} MB vs {d:.0} MB L2 — {s}\n", .{
            operands_bytes / (1 << 20),
            l2 / (1 << 20),
            if (operands_bytes <= l2) "fits, so repeat reads stay on chip" else "exceeds L2, repeat reads reach DRAM",
        });
    }

    const c = try gpa.alloc(i32, elems);
    defer gpa.free(c);
    try ctx.download(c, dc);
    var bad: usize = 0;
    var first_bad: struct { row: usize, col: usize, want: i64, got: i32 } = undefined;
    var vrng: u32 = 0xdeadbeef;
    var si: usize = 0;
    while (si < 256) : (si += 1) {
        const idx = xorshift(&vrng) % elems;
        const row: usize = idx / n;
        const col: usize = idx % n;
        var want: i64 = 0;
        var k: usize = 0;
        while (k < n) : (k += 1) {
            want += @as(i64, ah[row * n + k]) * @as(i64, bh[k * n + col]);
        }
        if (want != c[idx]) {
            if (bad == 0) first_bad = .{ .row = row, .col = col, .want = want, .got = c[idx] };
            bad += 1;
        }
    }
    if (bad > 0) {
        try out.print("FAIL: {d}/256 samples wrong; first at ({d},{d}) want {d} got {d}\n", .{
            bad, first_bad.row, first_bad.col, first_bad.want, first_bad.got,
        });
        return 1;
    }
    try out.print("PASS: 256/256 samples exact (integer MMA, no tolerance)\n", .{});
    return 0;
}

/// H20 INT4 tensor peak. The repo's spec-table sources (the same sheet the
/// FP32/FP16/INT8 numbers come from) list no INT4 figure for the H20 — INT4
/// is not a marketed Hopper datapoint — so this is the conventional
/// assumption of 2x the INT8 rate, stated here rather than presented as a
/// measurement. Treat the percentage as "vs a plausible ceiling", not "vs
/// spec".
const h20_int4_peak_gops: f64 = 592000;

/// Unpack one logical s4 from the packed host buffer (see
/// examples_abi.packS4 for the nibble convention).
fn s4At(bytes: []const u8, idx: usize) i64 {
    const byte = bytes[idx / 2];
    const nib: u8 = if (idx % 2 == 0) byte & 0xF else byte >> 4;
    // Sign-extend the 4-bit two's-complement value via the top nibble.
    const shifted: i8 = @bitCast(nib << 4);
    return shifted >> 4;
}

/// IMMA s4 variant. Structurally runImmaS8: exact integer verification over
/// the full s4 input range [-8, 7] — the nibble packing and sign extension
/// in the B fragment gather are exactly what `-8`/`7` catch, and overflow is
/// impossible (worst case |sum| = n * 8 * 8 = 262144 at n=4096, deep inside
/// i32). Inputs are packed two per byte through the shared `api.packS4` so
/// host and device cannot drift on the nibble order.
fn runImmaS4(gpa: std.mem.Allocator, ctx: *gpu.Context, kern: gpu.Kernel(api.imma_s4), n: usize, iters: u32, out: *std.Io.Writer, tile_m: usize, tile_n: usize, dev_info: ?cu.Context.Info) !u8 {
    const elems = n * n;
    const packed_elems = elems / 2; // n % 128 == 0 is enforced above
    const ah = try gpa.alloc(u8, packed_elems);
    defer gpa.free(ah);
    const bh = try gpa.alloc(u8, packed_elems);
    defer gpa.free(bh);
    var rng: u32 = 0x0fdb9753;
    // Each draw's low nibble, sign-extended from bit 3 — covers the full s4
    // range [-8, 7] uniformly.
    for (ah) |*v| {
        const lo: i8 = @as(i8, @bitCast(@as(u8, @truncate(xorshift(&rng) & 0xF)) << 4)) >> 4;
        const hi: i8 = @as(i8, @bitCast(@as(u8, @truncate(xorshift(&rng) & 0xF)) << 4)) >> 4;
        v.* = api.packS4(lo, hi);
    }
    for (bh) |*v| {
        const lo: i8 = @as(i8, @bitCast(@as(u8, @truncate(xorshift(&rng) & 0xF)) << 4)) >> 4;
        const hi: i8 = @as(i8, @bitCast(@as(u8, @truncate(xorshift(&rng) & 0xF)) << 4)) >> 4;
        v.* = api.packS4(lo, hi);
    }

    const da = try ctx.allocSlice(u8, packed_elems);
    defer ctx.freeSlice(da);
    const db = try ctx.allocSlice(u8, packed_elems);
    defer ctx.freeSlice(db);
    const dc = try ctx.allocSlice(i32, elems);
    defer ctx.freeSlice(dc);
    try ctx.upload(da, ah);
    try ctx.upload(db, bh);
    // Poisoned rather than zeroed, same reasoning as the bf16 path.
    try ctx.fillBytes(dc, 0xff);

    const kargs = .{ da, db, dc, @as(u32, @intCast(n)) };
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
        try kern.launch(.{ .x = grid_x, .y = grid_y }, .{ .x = 128 }, kargs);
        try stop.record();
        try stop.sync();
        const ms = try start.elapsedMs(stop);
        if (ms < best_ms) best_ms = ms;
    }

    // Two integer ops per MAC, same convention as imma_s8.
    const ops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const gops = ops / (@as(f64, best_ms) * 1e6);
    try out.print("bench: imma_s4(tile={d}x{d}) n={d} iters={d}\n", .{ tile_m, tile_n, n, iters });
    try out.print("best: {d:.3} ms over {d} iters\n", .{ best_ms, iters });
    try out.print("GOPS: {d:.1} ({d:.1}% of assumed H20 INT4 tensor peak ~{d:.0} GOPS; see source comment)\n", .{ gops, gops / h20_int4_peak_gops * 100, h20_int4_peak_gops });

    const blocks = ((n + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n);
    // Half a byte per element, half the imma_s8 traffic for the same tiling.
    const demand_bytes: f64 = @floatFromInt(blocks * (tile_m + tile_n) * n / 2);
    const gbs = demand_bytes / (@as(f64, best_ms) * 1e-3) / 1e9;
    try out.print("global reads: {d:.2} GB demand -> {d:.2} TB/s (L2 absorbs repeats; DRAM is lower)\n", .{ demand_bytes / 1e9, gbs / 1000 });
    if (dev_info) |di| {
        const operands_bytes: f64 = @floatFromInt(2 * n * n / 2);
        const l2: f64 = @floatFromInt(di.l2_bytes);
        try out.print("A+B working set: {d:.0} MB vs {d:.0} MB L2 — {s}\n", .{
            operands_bytes / (1 << 20),
            l2 / (1 << 20),
            if (operands_bytes <= l2) "fits, so repeat reads stay on chip" else "exceeds L2, repeat reads reach DRAM",
        });
    }

    const c = try gpa.alloc(i32, elems);
    defer gpa.free(c);
    try ctx.download(c, dc);
    var bad: usize = 0;
    var first_bad: struct { row: usize, col: usize, want: i64, got: i32 } = undefined;
    var vrng: u32 = 0xdeadbeef;
    var si: usize = 0;
    while (si < 256) : (si += 1) {
        const idx = xorshift(&vrng) % elems;
        const row: usize = idx / n;
        const col: usize = idx % n;
        var want: i64 = 0;
        var k: usize = 0;
        while (k < n) : (k += 1) {
            want += s4At(ah, row * n + k) * s4At(bh, k * n + col);
        }
        if (want != c[idx]) {
            if (bad == 0) first_bad = .{ .row = row, .col = col, .want = want, .got = c[idx] };
            bad += 1;
        }
    }
    if (bad > 0) {
        try out.print("FAIL: {d}/256 samples wrong; first at ({d},{d}) want {d} got {d}\n", .{
            bad, first_bad.row, first_bad.col, first_bad.want, first_bad.got,
        });
        return 1;
    }
    try out.print("PASS: 256/256 samples exact (integer MMA, no tolerance)\n", .{});
    return 0;
}

/// H20 sparse FP16 tensor peak. Sparse is marketed as 2x the dense rate;
/// like imma_s4's INT4 figure this is an assumption (the repo's spec sources
/// carry no separate sparse number), and the bench output says so.
const h20_fp16_sparse_peak_gflops: f64 = 296000;

/// Sparse HGEMM harness. The host generates a dense f16 A (small integers,
/// exact in f16), prunes it 2:4 itself — dropping 2 of every 4 k elements at
/// pseudo-random positions — and packs the kept values plus metadata in
/// exactly the bit order examples_abi.hgemm_sp documents. The CPU reference
/// is dense matmul over the *pruned* A (dropped positions zeroed): the
/// sparse matrix the hardware multiplies is by definition the pruned one,
/// so a wrong metadata/fragment reading on the device is a hard mismatch,
/// not a tolerance question. Values are integers and f32 accumulation of
/// them is exact, so finishVerify's 1e-2 gate is again effectively an exact
/// comparison.
fn runHgemmSp(gpa: std.mem.Allocator, ctx: *gpu.Context, kern: gpu.Kernel(api.hgemm_sp), n: usize, iters: u32, out: *std.Io.Writer, tile_m: usize, tile_n: usize, dev_info: ?cu.Context.Info) !u8 {
    const elems = n * n;
    const a_ref = try gpa.alloc(f32, elems); // pruned dense A, for the CPU reference
    defer gpa.free(a_ref);
    const b = try gpa.alloc(f32, elems);
    defer gpa.free(b);
    const ap = try gpa.alloc(f16, elems / 2); // pruned packed A, n/2 per row
    defer gpa.free(ap);
    const mh = try gpa.alloc(u16, elems / 16); // one word per row per 16 k
    defer gpa.free(mh);
    const bh = try gpa.alloc(f16, elems);
    defer gpa.free(bh);

    // The six ways to keep 2 of 4, first index < second.
    const combos = [6][2]u2{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 }, .{ 2, 3 } };
    var rng: u32 = 0x5eed1234;
    for (0..n) |row| {
        var kb: usize = 0; // 16-wide k blocks, one metadata word each
        while (kb < n / 16) : (kb += 1) {
            var word: u16 = 0;
            inline for (0..4) |j| {
                const group_base = row * n + kb * 16 + 4 * j;
                const combo = combos[xorshift(&rng) % 6];
                inline for (0..2) |e| {
                    const kept = combo[e];
                    const x: i32 = @intCast(xorshift(&rng) % 5);
                    const val: f32 = @floatFromInt(x - 2); // -2..2, exact in f16
                    a_ref[group_base + kept] = val;
                    ap[(row * (n / 2)) + kb * 8 + 2 * j + e] = @floatCast(val);
                }
                // Dropped positions are zeros in the sparse matrix.
                var dropped: [2]u2 = undefined;
                var di: usize = 0;
                inline for (0..4) |cand| {
                    if (cand != combo[0] and cand != combo[1]) {
                        dropped[di] = cand;
                        di += 1;
                    }
                }
                a_ref[group_base + dropped[0]] = 0;
                a_ref[group_base + dropped[1]] = 0;
                // Nibble j: low 2 bits = first kept index, high 2 = second.
                word |= (@as(u16, combo[0]) | (@as(u16, combo[1]) << 2)) << (4 * j);
            }
            mh[row * (n / 16) + kb] = word;
        }
    }
    for (bh, 0..) |*v, i| {
        const x: i32 = @intCast(xorshift(&rng) % 5);
        const val: f32 = @floatFromInt(x - 2);
        v.* = @floatCast(val);
        b[i] = val;
    }

    const da = try ctx.allocSlice(f16, elems / 2);
    defer ctx.freeSlice(da);
    const db = try ctx.allocSlice(f16, elems);
    defer ctx.freeSlice(db);
    const dm = try ctx.allocSlice(u16, elems / 16);
    defer ctx.freeSlice(dm);
    const dc = try ctx.allocSlice(f32, elems);
    defer ctx.freeSlice(dc);
    try ctx.upload(da, ap);
    try ctx.upload(db, bh);
    try ctx.upload(dm, mh);
    // Poisoned rather than zeroed, same reasoning as the bf16 path.
    try ctx.fillBytes(dc, 0xff);

    const kargs = .{ da, db, dm, dc, @as(u32, @intCast(n)) };
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
        try kern.launch(.{ .x = grid_x, .y = grid_y }, .{ .x = 128 }, kargs);
        try stop.record();
        try stop.sync();
        const ms = try start.elapsedMs(stop);
        if (ms < best_ms) best_ms = ms;
    }

    const flops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const gflops = flops / (@as(f64, best_ms) * 1e6);
    try out.print("bench: hgemm_sp(tile={d}x{d}) n={d} iters={d}\n", .{ tile_m, tile_n, n, iters });
    try out.print("best: {d:.3} ms over {d} iters\n", .{ best_ms, iters });
    try out.print("GFLOPS: {d:.1} ({d:.1}% of assumed H20 sparse FP16 peak ~{d:.0} GFLOPS; 2x dense, see source comment)\n", .{ gflops, gflops / h20_fp16_sparse_peak_gflops * 100, h20_fp16_sparse_peak_gflops });

    const blocks = ((n + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n);
    // A moves half its dense bytes (2:4 kept), B is dense, metadata is negligible.
    const demand_bytes: f64 = @floatFromInt(blocks * (tile_m + 2 * tile_n) * n);
    const gbs = demand_bytes / (@as(f64, best_ms) * 1e-3) / 1e9;
    try out.print("global reads: {d:.2} GB demand -> {d:.2} TB/s (L2 absorbs repeats; DRAM is lower)\n", .{ demand_bytes / 1e9, gbs / 1000 });
    if (dev_info) |di| {
        const operands_bytes: f64 = @floatFromInt(3 * n * n); // A pruned + B dense, in bytes
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
    return finishVerify(gpa, ctx, dc, a_ref, b, c, n, out);
}

/// H20 sparse INT8 tensor peak. Sparse is marketed as 2x dense; same
/// assumption caveat as imma_s4's INT4 figure — no sourced H20 sparse-int8
/// number exists in this repo, so the percentage is against a plausible
/// ceiling, not a spec.
const h20_int8_sparse_peak_gops: f64 = 592000;

/// Sparse IMMA s8 harness. The host generates a dense s8 A over the full
/// [-128, 127] range (imma_s8's reasoning: sign extension and byte packing
/// are what full range exercises), prunes it 2:4 per 4-wide k chunk at
/// pseudo-random positions, and packs kept values plus u32 metadata words in
/// the bit order examples_abi.imma_sp_s8 contracts. Verification is exact
/// integer equality against a dense CPU reference over the *pruned* matrix —
/// a wrong metadata nibble order or row-to-thread assignment fails hard.
fn runImmaSpS8(gpa: std.mem.Allocator, ctx: *gpu.Context, kern: gpu.Kernel(api.imma_sp_s8), n: usize, iters: u32, out: *std.Io.Writer, tile_m: usize, tile_n: usize, dev_info: ?cu.Context.Info) !u8 {
    const elems = n * n;
    const a_ref = try gpa.alloc(i32, elems); // pruned dense A, for the CPU reference
    defer gpa.free(a_ref);
    const b = try gpa.alloc(i32, elems); // dense B mirrors
    defer gpa.free(b);
    const ap = try gpa.alloc(i8, elems / 2); // pruned packed A, n/2 per row
    defer gpa.free(ap);
    const mh = try gpa.alloc(u32, elems / 32); // one word per row per 32 k
    defer gpa.free(mh);
    const bh = try gpa.alloc(i8, elems);
    defer gpa.free(bh);

    // The six ways to keep 2 of 4, first index < second.
    const combos = [6][2]u2{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 }, .{ 2, 3 } };
    var rng: u32 = 0xc0ffee11;
    for (0..n) |row| {
        var kb: usize = 0; // 32-wide k blocks, one metadata word each
        while (kb < n / 32) : (kb += 1) {
            var word: u32 = 0;
            inline for (0..8) |j| {
                const group_base = row * n + kb * 32 + 4 * j;
                const combo = combos[xorshift(&rng) % 6];
                inline for (0..2) |e| {
                    const kept = combo[e];
                    const val: i8 = @bitCast(@as(u8, @truncate(xorshift(&rng))));
                    a_ref[group_base + kept] = val;
                    ap[(row * (n / 2)) + kb * 16 + 2 * j + e] = val;
                }
                var dropped: [2]u2 = undefined;
                var di: usize = 0;
                inline for (0..4) |cand| {
                    if (cand != combo[0] and cand != combo[1]) {
                        dropped[di] = cand;
                        di += 1;
                    }
                }
                a_ref[group_base + dropped[0]] = 0;
                a_ref[group_base + dropped[1]] = 0;
                // Nibble j: low 2 bits = first kept index, high 2 = second.
                word |= (@as(u32, combo[0]) | (@as(u32, combo[1]) << 2)) << (4 * j);
            }
            mh[row * (n / 32) + kb] = word;
        }
    }
    for (bh, 0..) |*v, i| {
        const val: i8 = @bitCast(@as(u8, @truncate(xorshift(&rng))));
        v.* = val;
        b[i] = val;
    }

    const da = try ctx.allocSlice(i8, elems / 2);
    defer ctx.freeSlice(da);
    const db = try ctx.allocSlice(i8, elems);
    defer ctx.freeSlice(db);
    const dm = try ctx.allocSlice(u32, elems / 32);
    defer ctx.freeSlice(dm);
    const dc = try ctx.allocSlice(i32, elems);
    defer ctx.freeSlice(dc);
    try ctx.upload(da, ap);
    try ctx.upload(db, bh);
    try ctx.upload(dm, mh);
    // Poisoned rather than zeroed, same reasoning as the other paths.
    try ctx.fillBytes(dc, 0xff);

    const kargs = .{ da, db, dm, dc, @as(u32, @intCast(n)) };
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
        try kern.launch(.{ .x = grid_x, .y = grid_y }, .{ .x = 128 }, kargs);
        try stop.record();
        try stop.sync();
        const ms = try start.elapsedMs(stop);
        if (ms < best_ms) best_ms = ms;
    }

    // Two integer ops per MAC, same convention as imma_s8.
    const ops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const gops = ops / (@as(f64, best_ms) * 1e6);
    try out.print("bench: imma_sp_s8(tile={d}x{d}) n={d} iters={d}\n", .{ tile_m, tile_n, n, iters });
    try out.print("best: {d:.3} ms over {d} iters\n", .{ best_ms, iters });
    try out.print("GOPS: {d:.1} ({d:.1}% of assumed H20 sparse INT8 peak ~{d:.0} GOPS; 2x dense, see source comment)\n", .{ gops, gops / h20_int8_sparse_peak_gops * 100, h20_int8_sparse_peak_gops });

    const blocks = ((n + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n);
    // A moves half its dense bytes (2:4 kept), B dense, metadata negligible.
    const demand_bytes: f64 = @floatFromInt(blocks * (tile_m / 2 + tile_n) * n);
    const gbs = demand_bytes / (@as(f64, best_ms) * 1e-3) / 1e9;
    try out.print("global reads: {d:.2} GB demand -> {d:.2} TB/s (L2 absorbs repeats; DRAM is lower)\n", .{ demand_bytes / 1e9, gbs / 1000 });
    if (dev_info) |di| {
        const operands_bytes: f64 = @floatFromInt(n * n + n * n / 2); // B dense + A pruned
        const l2: f64 = @floatFromInt(di.l2_bytes);
        try out.print("A+B working set: {d:.0} MB vs {d:.0} MB L2 — {s}\n", .{
            operands_bytes / (1 << 20),
            l2 / (1 << 20),
            if (operands_bytes <= l2) "fits, so repeat reads stay on chip" else "exceeds L2, repeat reads reach DRAM",
        });
    }

    const c = try gpa.alloc(i32, elems);
    defer gpa.free(c);
    try ctx.download(c, dc);
    var bad: usize = 0;
    var first_bad: struct { row: usize, col: usize, want: i64, got: i32 } = undefined;
    var vrng: u32 = 0xdeadbeef;
    var si: usize = 0;
    while (si < 256) : (si += 1) {
        const idx = xorshift(&vrng) % elems;
        const row: usize = idx / n;
        const col: usize = idx % n;
        var want: i64 = 0;
        var k: usize = 0;
        while (k < n) : (k += 1) {
            want += @as(i64, a_ref[row * n + k]) * @as(i64, b[k * n + col]);
        }
        if (want != c[idx]) {
            if (bad == 0) first_bad = .{ .row = row, .col = col, .want = want, .got = c[idx] };
            bad += 1;
        }
    }
    if (bad > 0) {
        try out.print("FAIL: {d}/256 samples wrong; first at ({d},{d}) want {d} got {d}\n", .{
            bad, first_bad.row, first_bad.col, first_bad.want, first_bad.got,
        });
        return 1;
    }
    try out.print("PASS: 256/256 samples exact (sparse integer MMA, no tolerance)\n", .{});
    return 0;
}

/// Sparse IMMA s4 harness. The host builds a dense s4 A over the full [-8,7]
/// range, prunes it 4:8 pair-clustered (two of four 2-wide sub-chunks survive
/// per 8-wide chunk — NOT plain 2:4; the sub-chunk granularity is the whole
/// difference from imma_sp_s8), and packs: one byte per surviving pair via
/// api.packS4, one u32 metadata word per row per 64 k. Verification is exact
/// integer equality against a dense CPU reference over the pruned matrix.
/// Overflow bound: |product| <= 64, |sum| <= 64 * 4096 = 262144 — deep in i32.
///
/// No peak percentage is printed: the only INT4 ceiling available is already
/// an assumption (2x INT8, no sourced H20 figure), and sparse = 2x that would
/// be an assumption squared. Raw GOPS only.
fn runImmaSpS4(gpa: std.mem.Allocator, ctx: *gpu.Context, kern: gpu.Kernel(api.imma_sp_s4), n: usize, iters: u32, out: *std.Io.Writer, tile_m: usize, tile_n: usize, dev_info: ?cu.Context.Info) !u8 {
    const elems = n * n;
    const a_ref = try gpa.alloc(i32, elems); // pruned dense A, for the CPU reference
    defer gpa.free(a_ref);
    const b = try gpa.alloc(i32, elems); // dense B mirrors (logical s4 values)
    defer gpa.free(b);
    const ap = try gpa.alloc(u8, elems / 4); // pruned packed A, n/4 bytes per row
    defer gpa.free(ap);
    const mh = try gpa.alloc(u32, elems / 64); // one word per row per 64 k
    defer gpa.free(mh);
    const bh = try gpa.alloc(u8, elems / 2); // dense packed B, n/2 per row
    defer gpa.free(bh);

    // The six ways to keep 2 of 4 sub-chunks, first index < second.
    const combos = [6][2]u2{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 }, .{ 2, 3 } };
    var rng: u32 = 0x5ca1ab1e;
    const drawS4 = struct {
        fn f(r: *u32) i8 {
            return @as(i8, @bitCast(@as(u8, @truncate(xorshift(r) & 0xF)) << 4)) >> 4;
        }
    }.f;
    for (0..n) |row| {
        var kb: usize = 0; // 64-wide k blocks, one metadata word each
        while (kb < n / 64) : (kb += 1) {
            var word: u32 = 0;
            inline for (0..8) |j| {
                const chunk_base = row * n + kb * 64 + 8 * j;
                const combo = combos[xorshift(&rng) % 6];
                // Draw all four sub-chunks; only the surviving two keep their
                // values in the reference and the packed output.
                var pairs: [4][2]i8 = undefined;
                inline for (0..4) |p| {
                    pairs[p] = .{ drawS4(&rng), drawS4(&rng) };
                    a_ref[chunk_base + 2 * p] = pairs[p][0];
                    a_ref[chunk_base + 2 * p + 1] = pairs[p][1];
                }
                inline for (0..2) |e| {
                    const p = combo[e];
                    ap[(row * (n / 4)) + kb * 16 + 2 * j + e] = api.packS4(pairs[p][0], pairs[p][1]);
                }
                inline for (0..4) |p| {
                    if (p != combo[0] and p != combo[1]) {
                        a_ref[chunk_base + 2 * p] = 0;
                        a_ref[chunk_base + 2 * p + 1] = 0;
                    }
                }
                // Nibble j: low 2 bits = first surviving sub-chunk, high 2 = second.
                word |= (@as(u32, combo[0]) | (@as(u32, combo[1]) << 2)) << (4 * j);
            }
            mh[row * (n / 64) + kb] = word;
        }
    }
    // Dense B, packed two per byte.
    for (0..elems / 2) |i| {
        const lo = drawS4(&rng);
        const hi = drawS4(&rng);
        bh[i] = api.packS4(lo, hi);
        b[2 * i] = lo;
        b[2 * i + 1] = hi;
    }

    const da = try ctx.allocSlice(u8, elems / 4);
    defer ctx.freeSlice(da);
    const db = try ctx.allocSlice(u8, elems / 2);
    defer ctx.freeSlice(db);
    const dm = try ctx.allocSlice(u32, elems / 64);
    defer ctx.freeSlice(dm);
    const dc = try ctx.allocSlice(i32, elems);
    defer ctx.freeSlice(dc);
    try ctx.upload(da, ap);
    try ctx.upload(db, bh);
    try ctx.upload(dm, mh);
    // Poisoned rather than zeroed, same reasoning as the other paths.
    try ctx.fillBytes(dc, 0xff);

    const kargs = .{ da, db, dm, dc, @as(u32, @intCast(n)) };
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
        try kern.launch(.{ .x = grid_x, .y = grid_y }, .{ .x = 128 }, kargs);
        try stop.record();
        try stop.sync();
        const ms = try start.elapsedMs(stop);
        if (ms < best_ms) best_ms = ms;
    }

    // Two integer ops per MAC, same convention as imma_s8.
    const ops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const gops = ops / (@as(f64, best_ms) * 1e6);
    try out.print("bench: imma_sp_s4(tile={d}x{d}) n={d} iters={d}\n", .{ tile_m, tile_n, n, iters });
    try out.print("best: {d:.3} ms over {d} iters\n", .{ best_ms, iters });
    try out.print("GOPS: {d:.1} (no peak %%: would be 2x the already-assumed INT4 ceiling — assumption squared, omitted)\n", .{gops});

    const blocks = ((n + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n);
    // A moves a quarter of its dense element count in bytes (2:4 kept, 2 per
    // byte), B dense at half a byte per element, metadata negligible.
    const demand_bytes: f64 = @floatFromInt(blocks * (tile_m / 4 + tile_n / 2) * n);
    const gbs = demand_bytes / (@as(f64, best_ms) * 1e-3) / 1e9;
    try out.print("global reads: {d:.2} GB demand -> {d:.2} TB/s (L2 absorbs repeats; DRAM is lower)\n", .{ demand_bytes / 1e9, gbs / 1000 });
    if (dev_info) |di| {
        const operands_bytes: f64 = @floatFromInt(n * n / 2 + n * n / 4); // B dense + A pruned
        const l2: f64 = @floatFromInt(di.l2_bytes);
        try out.print("A+B working set: {d:.0} MB vs {d:.0} MB L2 — {s}\n", .{
            operands_bytes / (1 << 20),
            l2 / (1 << 20),
            if (operands_bytes <= l2) "fits, so repeat reads stay on chip" else "exceeds L2, repeat reads reach DRAM",
        });
    }

    const c = try gpa.alloc(i32, elems);
    defer gpa.free(c);
    try ctx.download(c, dc);
    var bad: usize = 0;
    var first_bad: struct { row: usize, col: usize, want: i64, got: i32 } = undefined;
    var vrng: u32 = 0xdeadbeef;
    var si: usize = 0;
    while (si < 256) : (si += 1) {
        const idx = xorshift(&vrng) % elems;
        const row: usize = idx / n;
        const col: usize = idx % n;
        var want: i64 = 0;
        var k: usize = 0;
        while (k < n) : (k += 1) {
            want += @as(i64, a_ref[row * n + k]) * @as(i64, b[k * n + col]);
        }
        if (want != c[idx]) {
            if (bad == 0) first_bad = .{ .row = row, .col = col, .want = want, .got = c[idx] };
            bad += 1;
        }
    }
    if (bad > 0) {
        try out.print("FAIL: {d}/256 samples wrong; first at ({d},{d}) want {d} got {d}\n", .{
            bad, first_bad.row, first_bad.col, first_bad.want, first_bad.got,
        });
        return 1;
    }
    try out.print("PASS: 256/256 samples exact (sparse integer MMA, no tolerance)\n", .{});
    return 0;
}

/// TMA variant of runHgemm (S2): same data, timing loop and verification as
/// `hgemm_wgmma3`, plus the two descriptors the kernel loads through.
///
/// The expect_tx byte counts are cross-checked against the descriptors that
/// were actually encoded — `TensorMap.tileBytes()` — rather than recomputed
/// here, because that count written wrong does not fault: too low and the
/// barrier releases on partial data, too high and it never releases.
fn runHgemmTma(gpa: std.mem.Allocator, ctx: *gpu.Context, kern: gpu.Kernel(api.hgemm_tma.signature), n: usize, iters: u32, out: *std.Io.Writer, dev_info: ?cu.Context.Info) !u8 {
    const p = api.hgemm_tma;
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

    const da = try ctx.allocSlice(f16, elems);
    defer ctx.freeSlice(da);
    const db = try ctx.allocSlice(f16, elems);
    defer ctx.freeSlice(db);
    const dc = try ctx.allocSlice(f32, elems);
    defer ctx.freeSlice(dc);
    try ctx.upload(da, ah);
    try ctx.upload(db, bh);
    // Poisoned rather than zeroed: a kernel that writes nothing then cannot
    // pass verification by leaving plausible zeros behind.
    try ctx.fillBytes(dc, 0xff);

    // Both tensors are n x n row-major f16, described innermost-first. A is
    // one box per stage (16 k x 64 m, plain row-major — the ldmatrix layout);
    // B is one 8-col x 16-k box per wgmma n-block, reproducing the core-matrix
    // packing the wgmma descriptor reads (see hgemm_tma.zig). No swizzle.
    const dim = [_]u64{ n, n };
    const strides = [_]u64{n * @sizeOf(f16)};
    const box_a = [_]u32{ p.k_slice, p.tile_m };
    const box_b = [_]u32{ 8, p.k_slice };
    const map_a = ctx.inner.encodeTensorMap(f16, da.ptr, dim[0..], strides[0..], box_a[0..], .{}) catch {
        try out.print("FAIL: encodeTensorMap(A): {s}\n", .{ctx.inner.drv.lastError()});
        return 1;
    };
    const map_b = ctx.inner.encodeTensorMap(f16, db.ptr, dim[0..], strides[0..], box_b[0..], .{}) catch {
        try out.print("FAIL: encodeTensorMap(B): {s}\n", .{ctx.inner.drv.lastError()});
        return 1;
    };
    // Cross-check the byte counts the kernel will wait on against the
    // descriptors just encoded, before they can deadlock a launch.
    if (map_a.tileBytes() != p.a_tile_bytes) {
        try out.print("FAIL: A descriptor tile is {d} bytes, kernel reserves {d}\n", .{ map_a.tileBytes(), p.a_tile_bytes });
        return 1;
    }
    if (map_b.tileBytes() * p.b_subtiles != p.b_tile_bytes) {
        try out.print("FAIL: B descriptor tile is {d} x {d} bytes, kernel reserves {d}\n", .{ map_b.tileBytes(), p.b_subtiles, p.b_tile_bytes });
        return 1;
    }

    const ddesc_a = try ctx.allocSlice(u8, @sizeOf(cu.CUtensorMap));
    defer ctx.freeSlice(ddesc_a);
    const ddesc_b = try ctx.allocSlice(u8, @sizeOf(cu.CUtensorMap));
    defer ctx.freeSlice(ddesc_b);
    try ctx.upload(ddesc_a, std.mem.asBytes(&map_a.map));
    try ctx.upload(ddesc_b, std.mem.asBytes(&map_b.map));

    const kargs = .{ da, db, dc, @as(u32, @intCast(n)), ddesc_a.ptr, ddesc_b.ptr };
    // grid.x walks N, grid.y walks M.
    const grid_x: u32 = @intCast((n + p.tile_n - 1) / p.tile_n);
    const grid_y: u32 = @intCast((n + p.tile_m - 1) / p.tile_m);

    const start = try ctx.eventCreate();
    defer start.destroy();
    const stop = try ctx.eventCreate();
    defer stop.destroy();
    var best_ms: f32 = std.math.floatMax(f32);
    var it: u32 = 0;
    while (it < iters) : (it += 1) {
        try start.record();
        try kern.launch(.{ .x = grid_x, .y = grid_y }, .{ .x = p.threads }, kargs);
        try stop.record();
        try stop.sync();
        const ms = try start.elapsedMs(stop);
        if (ms < best_ms) best_ms = ms;
    }

    const flops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const gflops = flops / (@as(f64, best_ms) * 1e6);
    try out.print("bench: hgemm_tma(tile={d}x{d}) n={d} iters={d}\n", .{ p.tile_m, p.tile_n, n, iters });
    try out.print("best: {d:.3} ms over {d} iters\n", .{ best_ms, iters });
    try out.print("GFLOPS: {d:.1} ({d:.1}% of H20 FP16 tensor peak ~{d:.0} GFLOPS)\n", .{ gflops, gflops / h20_fp16_peak_gflops * 100, h20_fp16_peak_gflops });

    const blocks = ((n + p.tile_m - 1) / p.tile_m) * ((n + p.tile_n - 1) / p.tile_n);
    const demand_bytes: f64 = @floatFromInt(blocks * (p.tile_m + p.tile_n) * n * 2);
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
