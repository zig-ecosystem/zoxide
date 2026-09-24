//! `zoxide run` — end-to-end harness: load a cubin (or PTX, assembled via
//! ptxas), launch one of the known example kernels, verify results on the host.

const std = @import("std");
const cu = @import("cuda_driver.zig");

pub const RunArgs = struct {
    input: []const u8,
    kernel_name: ?[]const u8 = null,
    n: usize = 1 << 20, // elements for vector_add; derived for others
    grid: ?u32 = null,
    block: ?u32 = null,
    arch: []const u8 = "sm_90",
    /// ptxas --maxrregcount passthrough.
    max_regs: ?u32 = null,
};

const Example = struct {
    stem: []const u8,
    entry: []const u8,
    default_block: u32,
};

const examples = [_]Example{
    .{ .stem = "vector_add", .entry = "vectorAdd", .default_block = 256 },
    .{ .stem = "shared_reverse", .entry = "sharedReverse", .default_block = 256 },
    .{ .stem = "warp_reduce", .entry = "warpReduce", .default_block = 256 },
    .{ .stem = "atomic_counter", .entry = "atomicCounter", .default_block = 256 },
    .{ .stem = "debug_print", .entry = "debugPrint", .default_block = 32 },
    // One warpgroup, fixed: wgmma is a warpgroup-wide instruction.
    .{ .stem = "wgmma_smoke", .entry = "wgmmaSmoke", .default_block = 128 },
    .{ .stem = "dev_global", .entry = "addBias", .default_block = 256 },
    .{ .stem = "const_bank", .entry = "scaleByConst", .default_block = 256 },
};

fn findExample(path: []const u8) ?Example {
    const base = std.fs.path.basename(path);
    const stem = std.fs.path.stem(base);
    for (examples) |e| {
        if (std.mem.eql(u8, stem, e.stem)) return e;
    }
    return null;
}

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *std.process.Environ.Map,
    args: RunArgs,
    // Injected to reuse the CLI's ptxas logic without a circular import.
    assemblePtx: *const fn (gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, in_ptx: []const u8, out_cubin: []const u8, arch: []const u8, max_regs: ?u32) anyerror!void,
    out: *std.Io.Writer,
) !u8 {
    const ex = findExample(args.input) orelse {
        try out.print("error: '{s}' does not look like a known example (vector_add/shared_reverse/warp_reduce/atomic_counter/debug_print/wgmma_smoke)\n", .{args.input});
        return 1;
    };

    // Resolve input to cubin bytes.
    var tmp_cubin: ?[]u8 = null;
    defer if (tmp_cubin) |p| {
        std.Io.Dir.deleteFileAbsolute(io, p) catch {};
        gpa.free(p);
    };
    var cubin_path: []const u8 = args.input;
    if (std.mem.endsWith(u8, args.input, ".ptx")) {
        const p = try std.fmt.allocPrint(gpa, "/tmp/zoxide-run-{d}.cubin", .{std.c.getpid()});
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

    const kernel_name = args.kernel_name orelse
        try std.fmt.allocPrint(gpa, "{s}_$_{s}", .{ ex.stem, ex.entry });
    defer if (args.kernel_name == null) gpa.free(kernel_name);
    try out.print("kernel: {s}\n", .{kernel_name});

    var drv = cu.Driver.load() catch |e| {
        switch (e) {
            error.LibraryNotFound => try out.print(
                \\error: libcuda not found (tried libcuda.so.1, libcuda.so, libcuda.dylib).
                \\  This command needs an NVIDIA driver. Note: a statically linked musl
                \\  build cannot dlopen libcuda reliably — use the gnu dynamic build on the pod.
                \\
            , .{}),
            error.SymbolMissing => try out.print("error: libcuda found but a required symbol is missing\n", .{}),
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
        try out.print("error: kernel '{s}' not found in module: {s}\n", .{ kernel_name, drv.lastError() });
        return 1;
    };

    const r = if (std.mem.eql(u8, ex.stem, "vector_add"))
        try runVectorAdd(gpa, &ctx, func, args, out)
    else if (std.mem.eql(u8, ex.stem, "shared_reverse"))
        try runSharedReverse(gpa, &ctx, func, args, out)
    else if (std.mem.eql(u8, ex.stem, "warp_reduce"))
        try runWarpReduce(gpa, &ctx, func, args, out)
    else if (std.mem.eql(u8, ex.stem, "debug_print"))
        try runDebugPrint(&ctx, func, out)
    else if (std.mem.eql(u8, ex.stem, "wgmma_smoke"))
        try runWgmmaSmoke(gpa, &ctx, func, out)
    else if (std.mem.eql(u8, ex.stem, "dev_global"))
        try runDevGlobal(gpa, &ctx, mod, func, out)
    else if (std.mem.eql(u8, ex.stem, "const_bank"))
        try runConstBank(gpa, &ctx, mod, func, out)
    else
        try runAtomicCounter(gpa, &ctx, func, args, out);
    return r;
}

fn fail(out: *std.Io.Writer, comptime fmt: []const u8, a: anytype) !u8 {
    try out.print("FAIL: " ++ fmt ++ "\n", a);
    return 1;
}

/// Verify real CUDA constant memory — PTX `.const`, not read-only global.
///
/// The symbol name is the give-away that this is a different mechanism: module-
/// scope assembly is emitted verbatim, so it is `const_scales`, with none of the
/// `<stem>_$_` mangling a kernel or a Zig-declared global gets.
///
/// Two rounds with different tables, for the same reason as `dev_global`: a bank
/// that resolves but is not re-read would pass a single round. The kernel also
/// reads at an offset behind a padding object, so wrong addressing — feeding
/// `ld.const` a bare byte offset instead of a symbol-relative one — produces
/// wrong values rather than accidentally correct ones.
fn runConstBank(
    gpa: std.mem.Allocator,
    ctx: *cu.Context,
    mod: cu.Module,
    func: cu.Function,
    out: *std.Io.Writer,
) !u8 {
    const symbol = "const_scales";
    const bank_len = 64;
    const g = mod.global(symbol) catch {
        try out.print(
            "FAIL: cannot resolve constant bank '{s}': {s}\n" ++
                "  note: module-scope asm is not mangled, so the symbol is '{s}', " ++
                "not '<stem>_$_{s}'\n",
            .{ symbol, ctx.drv.lastError(), symbol, symbol },
        );
        return 1;
    };
    const want_bytes = bank_len * @sizeOf(f32);
    if (g.bytes != want_bytes) {
        return fail(out, "constant bank '{s}' is {d} bytes, expected {d}", .{ symbol, g.bytes, want_bytes });
    }
    try out.print("resolved constant bank '{s}': {d} bytes (.const state space)\n", .{ symbol, g.bytes });

    const n: usize = 4096;
    const block: u32 = 256;
    const grid: u32 = @intCast((n + block - 1) / block);
    const din = try ctx.alloc(n * @sizeOf(f32));
    defer ctx.free(din);
    const dout = try ctx.alloc(n * @sizeOf(f32));
    defer ctx.free(dout);

    const input = try gpa.alloc(f32, n);
    defer gpa.free(input);
    for (input, 0..) |*v, i| v.* = @floatFromInt(i % 97);
    try ctx.copyHtoD(din, std.mem.sliceAsBytes(input));

    const host = try gpa.alloc(f32, n);
    defer gpa.free(host);

    var scales: [bank_len]f32 = undefined;
    for (0..2) |round| {
        for (&scales, 0..) |*s, i| {
            s.* = if (round == 0)
                @floatFromInt(i + 1)
            else
                -0.5 * @as(f32, @floatFromInt(i)) + 0.25;
        }
        try ctx.copyHtoD(g.ptr, std.mem.sliceAsBytes(scales[0..]));

        var arg_out = dout;
        var arg_in = din;
        var arg_n: u32 = @intCast(n);
        var params = [_]?*anyopaque{ &arg_out, &arg_in, &arg_n };
        try func.launch(grid, 1, 1, block, 1, 1, &params);
        try ctx.synchronize();
        try ctx.copyDtoH(std.mem.sliceAsBytes(host), dout);

        var bad: usize = 0;
        var first_bad: usize = 0;
        var max_err: f64 = 0;
        for (host, 0..) |v, i| {
            const want = @as(f64, input[i]) * @as(f64, scales[i % bank_len]);
            const err = @abs(@as(f64, v) - want);
            if (err > max_err) max_err = err;
            if (err != 0) {
                if (bad == 0) first_bad = i;
                bad += 1;
            }
        }
        if (bad > 0) {
            try out.print(
                "FAIL: round {d}: {d}/{d} mismatches, max err {d}; out[{d}]={d} wanted {d}\n",
                .{
                    round, bad, n, max_err, first_bad, host[first_bad],
                    @as(f64, input[first_bad]) * @as(f64, scales[first_bad % bank_len]),
                },
            );
            if (round == 1) try out.print(
                "  round 0 passed and round 1 did not: the bank is not being re-read\n",
                .{},
            );
            return 1;
        }
        try out.print("  round {d}: {d}/{d} exact\n", .{ round, n, n });
    }

    try out.print("PASS: const_bank .const state space, host-written, 2 rounds exact\n", .{});
    return 0;
}

/// Verify the "host writes once, device reads by name" path end to end.
///
/// Three things can go wrong and they are worth separating, because only a GPU
/// can tell them apart:
///
///   1. `cuModuleGetGlobal` cannot find the symbol — wrong mangled name, or the
///      global was folded away and is not in the PTX at all.
///   2. The symbol resolves but its size is wrong, so the name matched something
///      other than the intended declaration.
///   3. The symbol resolves and writes appear to succeed, but the kernel ignores
///      them because the read was constant-folded against the initialiser.
///      Catching this is why the kernel runs twice with different tables.
fn runDevGlobal(
    gpa: std.mem.Allocator,
    ctx: *cu.Context,
    mod: cu.Module,
    func: cu.Function,
    out: *std.Io.Writer,
) !u8 {
    const symbol = "dev_global_$_dev_bias";
    const g = mod.global(symbol) catch {
        try out.print(
            "FAIL: cannot resolve device global '{s}': {s}\n",
            .{ symbol, ctx.drv.lastError() },
        );
        return 1;
    };
    const want_bytes = 4 * @sizeOf(f32);
    if (g.bytes != want_bytes) {
        return fail(out, "device global '{s}' is {d} bytes, expected {d}", .{ symbol, g.bytes, want_bytes });
    }
    try out.print("resolved device global '{s}': {d} bytes\n", .{ symbol, g.bytes });

    const n: usize = 1024;
    const block: u32 = 256;
    const grid: u32 = @intCast((n + block - 1) / block);
    const dout = try ctx.alloc(n * @sizeOf(f32));
    defer ctx.free(dout);
    const host = try gpa.alloc(f32, n);
    defer gpa.free(host);

    // Two different tables. The second is what proves host writes reach the
    // kernel rather than the initialiser having been baked in.
    const tables = [2][4]f32{
        .{ 1, 2, 3, 4 },
        .{ -100.5, 0.25, 7, 65536 },
    };
    for (tables, 0..) |table, round| {
        try ctx.copyHtoD(g.ptr, std.mem.sliceAsBytes(table[0..]));
        var arg_out = dout;
        var arg_n: u32 = @intCast(n);
        var params = [_]?*anyopaque{ &arg_out, &arg_n };
        try func.launch(grid, 1, 1, block, 1, 1, &params);
        try ctx.synchronize();
        try ctx.copyDtoH(std.mem.sliceAsBytes(host), dout);

        var bad: usize = 0;
        var max_err: f64 = 0;
        var first_bad: usize = 0;
        for (host, 0..) |v, i| {
            const want = @as(f64, @floatFromInt(i)) + @as(f64, table[i % 4]);
            const err = @abs(@as(f64, v) - want);
            if (err > max_err) max_err = err;
            if (err != 0) {
                if (bad == 0) first_bad = i;
                bad += 1;
            }
        }
        if (bad > 0) {
            try out.print(
                "FAIL: round {d} table {{{d}, {d}, {d}, {d}}}: {d}/{d} mismatches, " ++
                    "max err {d}; out[{d}]={d} wanted {d}\n",
                .{
                    round,          table[0], table[1],   table[2],
                    table[3],       bad,      n,          max_err,
                    first_bad,      host[first_bad],
                    @as(f64, @floatFromInt(first_bad)) + @as(f64, table[first_bad % 4]),
                },
            );
            // Round 0 passing and round 1 failing is the folded-read signature:
            // the kernel is using whatever was baked in, not what we uploaded.
            if (round == 1) try out.print(
                "  round 0 passed and round 1 did not: the kernel is not re-reading " ++
                    "the global. Check that it reads through cuda.ldg().\n",
                .{},
            );
            return 1;
        }
        try out.print("  round {d}: 1024/1024 exact with bias {{{d}, {d}, {d}, {d}}}\n", .{
            round, table[0], table[1], table[2], table[3],
        });
    }

    try out.print(
        "PASS: dev_global host-written device global read by name, 2 rounds exact\n",
        .{},
    );
    return 0;
}

fn runVectorAdd(gpa: std.mem.Allocator, ctx: *cu.Context, func: cu.Function, args: RunArgs, out: *std.Io.Writer) !u8 {
    const n = args.n;
    const block: u32 = args.block orelse 256;
    const grid: u32 = args.grid orelse @intCast((n + block - 1) / block);
    try out.print("vector_add: n={d} grid={d} block={d}\n", .{ n, grid, block });

    const bytes = n * @sizeOf(f32);
    const a = try gpa.alloc(f32, n);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, n);
    defer gpa.free(b);
    const c = try gpa.alloc(f32, n);
    defer gpa.free(c);
    for (a, 0..) |*v, i| v.* = @floatFromInt(i);
    for (b, 0..) |*v, i| v.* = @floatFromInt(2 * i);
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

    var arg_da = da;
    var arg_db = db;
    var arg_dc = dc;
    var arg_n: u32 = @intCast(n);
    var params = [_]?*anyopaque{ &arg_da, &arg_db, &arg_dc, &arg_n };
    try func.launch(grid, 1, 1, block, 1, 1, &params);
    try ctx.synchronize();
    try ctx.copyDtoH(std.mem.sliceAsBytes(c), dc);

    var bad: usize = 0;
    var max_err: f64 = 0;
    for (c, 0..) |v, i| {
        const want: f64 = @floatFromInt(3 * i);
        const err = @abs(@as(f64, v) - want);
        if (err > max_err) max_err = err;
        if (err != 0) bad += 1;
    }
    if (bad > 0) return fail(out, "vector_add: {d}/{d} mismatches, max err {d}", .{ bad, n, max_err });
    try out.print("PASS: vector_add n={d}, max err {d}\n", .{ n, max_err });
    return 0;
}

fn runSharedReverse(gpa: std.mem.Allocator, ctx: *cu.Context, func: cu.Function, args: RunArgs, out: *std.Io.Writer) !u8 {
    const block: u32 = 256; // must match block_size in shared_reverse.zig
    const grid: u32 = args.grid orelse @intCast(@max(1, args.n / block));
    const n: usize = @as(usize, grid) * block;
    try out.print("shared_reverse: n={d} grid={d} block={d}\n", .{ n, grid, block });

    const bytes = n * @sizeOf(f32);
    const input = try gpa.alloc(f32, n);
    defer gpa.free(input);
    const result = try gpa.alloc(f32, n);
    defer gpa.free(result);
    for (input, 0..) |*v, i| v.* = @floatFromInt(i);
    @memset(result, 0);

    const din = try ctx.alloc(bytes);
    defer ctx.free(din);
    const dout = try ctx.alloc(bytes);
    defer ctx.free(dout);
    try ctx.copyHtoD(din, std.mem.sliceAsBytes(input));
    try ctx.copyHtoD(dout, std.mem.sliceAsBytes(result));

    var arg_in = din;
    var arg_out = dout;
    var params = [_]?*anyopaque{ &arg_in, &arg_out };
    try func.launch(grid, 1, 1, block, 1, 1, &params);
    try ctx.synchronize();
    try ctx.copyDtoH(std.mem.sliceAsBytes(result), dout);

    var bad: usize = 0;
    for (0..grid) |g| {
        for (0..block) |t| {
            if (result[g * block + t] != input[g * block + (block - 1 - t)]) bad += 1;
        }
    }
    if (bad > 0) return fail(out, "shared_reverse: {d}/{d} mismatches", .{ bad, n });
    try out.print("PASS: shared_reverse n={d}\n", .{n});
    return 0;
}

fn runWarpReduce(gpa: std.mem.Allocator, ctx: *cu.Context, func: cu.Function, args: RunArgs, out: *std.Io.Writer) !u8 {
    const block: u32 = 256; // must match block_size in warp_reduce.zig
    const grid: u32 = args.grid orelse @intCast(@max(1, args.n / block));
    const n: usize = @as(usize, grid) * block;
    try out.print("warp_reduce: n={d} grid={d} block={d}\n", .{ n, grid, block });

    const input = try gpa.alloc(f32, n);
    defer gpa.free(input);
    for (input, 0..) |*v, i| v.* = @floatFromInt(i % 8);

    const din = try ctx.alloc(n * @sizeOf(f32));
    defer ctx.free(din);
    const dout = try ctx.alloc(grid * @sizeOf(f32));
    defer ctx.free(dout);
    try ctx.copyHtoD(din, std.mem.sliceAsBytes(input));

    var arg_in = din;
    var arg_out = dout;
    var params = [_]?*anyopaque{ &arg_in, &arg_out };
    try func.launch(grid, 1, 1, block, 1, 1, &params);
    try ctx.synchronize();

    const sums = try gpa.alloc(f32, grid);
    defer gpa.free(sums);
    try ctx.copyDtoH(std.mem.sliceAsBytes(sums), dout);

    var bad: usize = 0;
    var max_err: f64 = 0;
    for (0..grid) |g| {
        var want: f64 = 0;
        for (0..block) |t| want += input[g * block + t];
        const err = @abs(@as(f64, sums[g]) - want);
        if (err > max_err) max_err = err;
        if (err > 1e-3) bad += 1;
    }
    if (bad > 0) return fail(out, "warp_reduce: {d}/{d} block sums wrong, max err {d}", .{ bad, grid, max_err });
    try out.print("PASS: warp_reduce grid={d}, max err {d}\n", .{ grid, max_err });
    return 0;
}

fn runAtomicCounter(gpa: std.mem.Allocator, ctx: *cu.Context, func: cu.Function, args: RunArgs, out: *std.Io.Writer) !u8 {
    const block: u32 = 256;
    const grid: u32 = args.grid orelse 64;
    try out.print("atomic_counter: grid={d} block={d}\n", .{ grid, block });

    var zero32: u32 = 0;
    var zero_f: f32 = 0;
    const hist_zeros = try gpa.alloc(u32, 16);
    defer gpa.free(hist_zeros);
    @memset(hist_zeros, 0);

    const dcount = try ctx.alloc(@sizeOf(u32));
    defer ctx.free(dcount);
    const dfsum = try ctx.alloc(@sizeOf(f32));
    defer ctx.free(dfsum);
    const dhist = try ctx.alloc(16 * @sizeOf(u32));
    defer ctx.free(dhist);
    try ctx.copyHtoD(dcount, std.mem.asBytes(&zero32));
    try ctx.copyHtoD(dfsum, std.mem.asBytes(&zero_f));
    try ctx.copyHtoD(dhist, std.mem.sliceAsBytes(hist_zeros));

    var arg_count = dcount;
    var arg_fsum = dfsum;
    var arg_hist = dhist;
    var params = [_]?*anyopaque{ &arg_count, &arg_fsum, &arg_hist };
    try func.launch(grid, 1, 1, block, 1, 1, &params);
    try ctx.synchronize();

    try ctx.copyDtoH(std.mem.asBytes(&zero32), dcount);
    try ctx.copyDtoH(std.mem.asBytes(&zero_f), dfsum);
    try ctx.copyDtoH(std.mem.sliceAsBytes(hist_zeros), dhist);

    const total: u64 = @as(u64, grid) * block;
    var bad: usize = 0;
    if (zero32 != total) bad += 1;
    if (zero_f != @as(f32, @floatFromInt(total))) bad += 1;
    // bins 1..15: grid * (block/16) each; bin 0 gets + block per block
    for (hist_zeros, 0..) |v, k| {
        const want: u32 = grid * (block / 16) + if (k == 0) grid * block else 0;
        if (v != want) bad += 1;
    }
    if (bad > 0) return fail(out, "atomic_counter: count={d} (want {d}), fsum={d}, hist={any}", .{ zero32, total, zero_f, hist_zeros });
    try out.print("PASS: atomic_counter count={d} fsum={d}\n", .{ zero32, zero_f });
    return 0;
}

/// wgmma_smoke checks itself on the device and reports a per-thread mismatch
/// count, so the host side is just "launch one warpgroup, expect all zeros".
fn runWgmmaSmoke(gpa: std.mem.Allocator, ctx: *cu.Context, func: cu.Function, out: *std.Io.Writer) !u8 {
    const threads = 128;
    try out.print("wgmma_smoke: 1 block x {d} threads (sm_90a, m64n16k16)\n", .{threads});

    const counts = try gpa.alloc(u32, threads);
    defer gpa.free(counts);
    @memset(counts, 0xffffffff);
    const dump = try gpa.alloc(f32, threads * 16);
    defer gpa.free(dump);
    @memset(dump, 0);

    const dcounts = try ctx.alloc(threads * @sizeOf(u32));
    defer ctx.free(dcounts);
    const ddump = try ctx.alloc(threads * 16 * @sizeOf(f32));
    defer ctx.free(ddump);
    try ctx.copyHtoD(dcounts, std.mem.sliceAsBytes(counts));
    try ctx.copyHtoD(ddump, std.mem.sliceAsBytes(dump));

    var arg_counts = dcounts;
    var arg_dump = ddump;
    var params = [_]?*anyopaque{ &arg_counts, &arg_dump };
    try func.launch(1, 1, 1, threads, 1, 1, &params);
    try ctx.synchronize();

    try ctx.copyDtoH(std.mem.sliceAsBytes(counts), dcounts);
    try ctx.copyDtoH(std.mem.sliceAsBytes(dump), ddump);

    var bad: u32 = 0;
    var first_bad: usize = 0;
    for (counts, 0..) |v, i| {
        if (v != 0) {
            if (bad == 0) first_bad = i;
            bad += v;
        }
    }
    if (bad != 0) {
        const t = first_bad;
        return fail(out, "wgmma_smoke: {d} mismatched accumulator elements; first bad thread {d}, got/expect pairs {any}", .{ bad, t, dump[t * 16 ..][0..16] });
    }
    try out.print("PASS: wgmma_smoke 1024 accumulator elements exact (descriptor + core-matrix packing + CLayout mapping)\n", .{});
    return 0;
}

fn runDebugPrint(ctx: *cu.Context, func: cu.Function, out: *std.Io.Writer) !u8 {
    var marker: u32 = 42;
    var params = [_]?*anyopaque{&marker};
    try func.launch(1, 1, 1, 32, 1, 1, &params);
    try ctx.synchronize();
    // vprintf output is flushed to host stdout during synchronization;
    // it is not capturable programmatically, so this is fail-open.
    try out.print("PASS(launch): debug_print ran; expect 8 lines 'debug_print: tid=N x=1.500000 marker=42' on stdout above (verified visually — device printf is not capturable)\n", .{});
    return 0;
}
