const std = @import("std");
const run_cmd = @import("run.zig");
const scaffold = @import("scaffold.zig");
const bench_cmd = @import("bench.zig");
const gen_cmd = @import("gen.zig");
const ptx = @import("ptx.zig");
const toolwrap = @import("toolwrap.zig");
const emb = @import("embedded.zig");
const embedded_kernels = @import("embedded_kernels");

/// Baseline GPU: NVIDIA H20 (Hopper, compute capability 9.0).
const default_sm = "sm_90";
// `sm_90a` is the architecture-specific Hopper target: `wgmma.*` and the
// TMA family are only available there (LLVM predicate `hasSM90a`, ptxas
// `-arch sm_90a`). Such code is not forward-compatible with later
// architectures, so it stays opt-in.
const valid_arches = [_][]const u8{ "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_120" };

fn validateArch(arch: []const u8) bool {
    if (!std.mem.startsWith(u8, arch, "sm_") or arch.len <= 3) return false;
    // Digits, optionally followed by a single architecture-specific suffix
    // ('a' = arch-specific, 'f' = family-specific).
    const body = if (arch[arch.len - 1] == 'a' or arch[arch.len - 1] == 'f') arch[3 .. arch.len - 1] else arch[3..];
    if (body.len == 0) return false;
    for (body) |c| if (!std.ascii.isDigit(c)) return false;
    for (valid_arches) |v| if (std.mem.eql(u8, arch, v)) return true;
    return false;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();

    const args = try std.process.Args.toSlice(init.minimal.args, arena);

    if (args.len < 2) {
        usage();
        return 1;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "ptx")) {
        return cmdPtx(gpa, io, args[2..]);
    } else if (std.mem.eql(u8, cmd, "cubin")) {
        return cmdCubin(gpa, io, init.environ_map, args[2..]);
    } else if (std.mem.eql(u8, cmd, "doctor")) {
        return cmdDoctor(gpa, io, init.environ_map, args[2..]);
    } else if (std.mem.eql(u8, cmd, "run")) {
        return cmdRun(gpa, io, init.environ_map, args[2..]);
    } else if (std.mem.eql(u8, cmd, "gen")) {
        var buf: [8192]u8 = undefined;
        var w = std.Io.File.stdout().writerStreaming(io, &buf);
        defer w.interface.flush() catch {};
        return gen_cmd.genMain(gpa, io, args[2..], &w.interface);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        return cmdBench(gpa, io, init.environ_map, args[2..]);
    } else if (std.mem.eql(u8, cmd, "lint")) {
        var buf: [8192]u8 = undefined;
        var w = std.Io.File.stdout().writerStreaming(io, &buf);
        defer w.interface.flush() catch {};
        return cmdLint(gpa, io, args[2..], &w.interface);
    } else if (std.mem.eql(u8, cmd, "sanitize")) {
        return toolwrap.run(gpa, io, init.environ_map, args[2..], toolwrap.sanitize_tool);
    } else if (std.mem.eql(u8, cmd, "debug")) {
        return toolwrap.run(gpa, io, init.environ_map, args[2..], toolwrap.debug_tool);
    } else if (std.mem.eql(u8, cmd, "new")) {
        return cmdNew(gpa, io, init.environ_map, args[2..]);
    } else if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        usage();
        return 0;
    }
    std.debug.print("error: unknown subcommand '{s}'\n", .{cmd});
    usage();
    return 1;
}

fn usage() void {
    std.debug.print(
        \\zoxide — Zig-native CUDA kernel toolchain skeleton
        \\
        \\usage:
        \\  zoxide ptx <kernel.zig> -o out.ptx [--arch sm_XX]     compile Zig source to PTX via zig (default {s})
        \\  zoxide cubin <in.ptx> -o out.cubin [--arch sm_XX]     assemble PTX via ptxas (default {s})
        \\  zoxide bench <kernel.ptx|name> [--n N] [--iters K] [--arch sm_XX] [--maxrregcount N]
        \\                                                       GEMM bench: event timing + CPU-reference check
        \\  zoxide gen <cuda-oxide/intrinsics> [-o src/gen/intrinsics.zig]
        \\                                                       regenerate the intrinsics bindings
        \\  zoxide new <name> [--dir path] [--zoxide-path dir]    scaffold a host+device package
        \\  zoxide doctor [--arch sm_XX]                          probe zig / nvptx / ptxas / libNVVM / GPU
        \\  zoxide run <example.ptx|.cubin> [--kernel name] [--n N] [--grid G --block B] [--arch sm_XX]
        \\                                                       run an example kernel on the GPU and verify results
        \\  zoxide lint <file.ptx> [--census]                   lint PTX (structure + known-bad patterns);
        \\                                                       --census prints the instruction census instead
        \\  zoxide sanitize [opts] [--] <command...>            run under compute-sanitizer
        \\  zoxide debug [opts] [--] <command...>               run under cuda-gdb
        \\  supported arch values: {s}
        \\
    , .{ default_sm, default_sm, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_100a sm_120" });
}

const Parsed = struct { positional: []const u8, output: []const u8, arch: []const u8 };

fn parseIoArgs(args: []const [:0]const u8, allow_arch: bool) !Parsed {
    var positional: ?[]const u8 = null;
    var output: ?[]const u8 = null;
    var arch: []const u8 = default_sm;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "-o")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            output = args[i];
        } else if (allow_arch and std.mem.eql(u8, a, "--arch")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            arch = args[i];
        } else if (positional == null) {
            positional = a;
        } else {
            return error.UnexpectedArg;
        }
    }
    if (positional == null or output == null) return error.MissingArgs;
    return .{ .positional = positional.?, .output = output.?, .arch = arch };
}

fn cmdPtx(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !u8 {
    const parsed = parseIoArgs(args, true) catch {
        std.debug.print("error: expected 'zoxide ptx <kernel.zig> -o out.ptx [--arch sm_XX]'\n", .{});
        return 1;
    };
    if (!validateArch(parsed.arch)) {
        std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ parsed.arch, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_100a sm_120" });
        return 1;
    }
    const emit_arg = try std.fmt.allocPrint(gpa, "-femit-asm={s}", .{parsed.output});
    defer gpa.free(emit_arg);

    if (!fileExists(io, parsed.positional)) {
        std.debug.print("error: input file not found: '{s}'\n", .{parsed.positional});
        return 1;
    }

    const result = std.process.run(gpa, io, .{
        .argv = &.{
            "zig",           "build-lib", parsed.positional,
            "-target",       "nvptx64-cuda", "-mcpu",
            parsed.arch,     "-O",        "ReleaseFast",
            "-fstrip",       "-fno-ubsan-rt", "-fno-emit-bin", emit_arg,
        },
    }) catch |err| {
        std.debug.print("error: failed to spawn 'zig' ({s}); is zig on PATH?\n", .{@errorName(err)});
        return 1;
    };
    defer gpa.free(result.stdout);
    defer gpa.free(result.stderr);
    if (!termOk(result.term)) {
        std.debug.print("error: zig failed compiling '{s}':\n{s}\n", .{ parsed.positional, result.stderr });
        return 1;
    }

    std.debug.print("wrote PTX: {s}\n", .{parsed.output});
    return 0;
}

/// `zoxide lint <file.ptx> [--census]` — structural lint over the lossless
/// text view in src/ptx.zig, or an instruction census with --census.
///
/// Exit 1 on any finding. This is the internal consumer the module was
/// built for: the checks are the ones this repository has been burned by
/// (positional-asm residue, debug targets, instructions outside a body),
/// and the census replaces the hand-grep counting behind the
/// upstream-asm-output-limit and tma-plan analyses.
fn cmdLint(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8, out: *std.Io.Writer) !u8 {
    var positional: ?[]const u8 = null;
    var census_mode = false;
    for (args) |a| {
        if (std.mem.eql(u8, a, "--census")) {
            census_mode = true;
        } else if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            try out.print(
                \\usage: zoxide lint <file.ptx> [--census]
                \\  lint: structure (braces, body placement, .target) and known-bad
                \\        patterns (unsubstituted $N / %[name] asm operands, debug
                \\        target). Exit 1 on any finding.
                \\  --census: print the instruction census, count by
                \\        mnemonic+modifiers, descending.
                \\
            , .{});
            return 0;
        } else if (positional == null) {
            positional = a;
        } else {
            std.debug.print("error: unexpected argument '{s}'\n", .{a});
            return 1;
        }
    }
    var path = positional orelse {
        std.debug.print("error: expected 'zoxide lint <file.ptx|name> [--census]'\n", .{});
        return 1;
    };
    const resolved: ?[]u8 = resolveKernelInput(gpa, io, path) catch |e| switch (e) {
        error.UnknownKernel => return 1,
        else => return e,
    };
    defer if (resolved) |p| {
        cleanupResolved(gpa, io, p);
    };
    if (resolved) |p| path = p;
    const src = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch |e| {
        std.debug.print("error: cannot read '{s}': {s}\n", .{ path, @errorName(e) });
        return 1;
    };
    defer gpa.free(src);

    const doc = ptx.parse(gpa, src) catch |e| {
        std.debug.print("error: parse failed on '{s}': {s}\n", .{ path, @errorName(e) });
        return 1;
    };
    defer {
        for (doc.stmts) |s| gpa.free(s.modifiers);
        gpa.free(doc.stmts);
    }
    if (!doc.roundTrips()) {
        // The lossless contract is asserted, not assumed: a parse that drops
        // bytes would make every lint position after it misleading.
        std.debug.print("error: internal: parse of '{s}' is not lossless\n", .{path});
        return 1;
    }

    if (census_mode) {
        var map = doc.census(gpa) catch |e| {
            std.debug.print("error: census failed: {s}\n", .{@errorName(e)});
            return 1;
        };
        defer {
            var it = map.iterator();
            while (it.next()) |e| gpa.free(e.key_ptr.*);
            map.deinit();
        }
        const Entry = struct { key: []const u8, count: u32 };
        var entries = std.array_list.Managed(Entry).init(gpa);
        defer entries.deinit();
        var it = map.iterator();
        while (it.next()) |e| try entries.append(.{ .key = e.key_ptr.*, .count = e.value_ptr.* });
        std.mem.sort(Entry, entries.items, {}, struct {
            fn lt(_: void, a: Entry, b: Entry) bool {
                if (a.count != b.count) return a.count > b.count;
                return std.mem.order(u8, a.key, b.key) == .lt;
            }
        }.lt);
        try out.print("census: {s} ({d} instruction statements, {d} classes)\n", .{ path, blk: {
            var t: u32 = 0;
            for (entries.items) |e| t += e.count;
            break :blk t;
        }, entries.items.len });
        for (entries.items) |e| try out.print("  {d: >6}  {s}\n", .{ e.count, e.key });
        return 0;
    }

    const findings = ptx.lint(gpa, doc) catch |e| {
        std.debug.print("error: lint failed: {s}\n", .{@errorName(e)});
        return 1;
    };
    defer gpa.free(findings);
    if (findings.len != 0) {
        for (findings) |f| {
            if (f.line != 0)
                try out.print("{s}:{d}: {s}\n", .{ path, f.line, f.msg })
            else
                try out.print("{s}: {s}\n", .{ path, f.msg });
        }
        try out.print("FAIL: {s}: {d} finding(s)\n", .{ path, findings.len });
        return 1;
    }
    try out.print("PASS: {s}: {d} statements, structure clean\n", .{ path, doc.stmts.len });
    return 0;
}

fn cmdCubin(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, args: []const [:0]const u8) !u8 {
    const parsed = parseIoArgs(args, true) catch {
        std.debug.print("error: expected 'zoxide cubin <in.ptx> -o out.cubin [--arch sm_XX]'\n", .{});
        return 1;
    };
    if (!validateArch(parsed.arch)) {
        std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ parsed.arch, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_100a sm_120" });
        return 1;
    }
    const ptxas = findPtxas(gpa, io, env) orelse {
        std.debug.print(
            \\error: ptxas not found.
            \\  Looked on PATH, in $CUDA_HOME/bin, and at /usr/local/cuda/bin/ptxas.
            \\  Install the CUDA toolkit or add ptxas to PATH to assemble PTX into cubin.
            \\
        , .{});
        return 1;
    };
    defer gpa.free(ptxas);

    const arch_arg = try std.fmt.allocPrint(gpa, "-arch={s}", .{parsed.arch});
    defer gpa.free(arch_arg);
    const result = std.process.run(gpa, io, .{
        .argv = &.{ ptxas, arch_arg, "-o", parsed.output, parsed.positional },
    }) catch |err| {
        std.debug.print("error: failed to run ptxas ({s})\n", .{@errorName(err)});
        return 1;
    };
    defer gpa.free(result.stdout);
    defer gpa.free(result.stderr);
    if (!termOk(result.term)) {
        std.debug.print("error: ptxas failed:\n{s}\n", .{result.stderr});
        return 1;
    }
    std.debug.print("wrote cubin: {s}\n", .{parsed.output});
    return 0;
}

/// Assemble PTX to cubin via ptxas (shared by `cubin` and `run`).
fn assemblePtx(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, in_ptx: []const u8, out_cubin: []const u8, arch: []const u8, max_regs: ?u32) !void {
    const ptxas = findPtxas(gpa, io, env) orelse return error.PtxasNotFound;
    defer gpa.free(ptxas);
    const arch_arg = try std.fmt.allocPrint(gpa, "-arch={s}", .{arch});
    defer gpa.free(arch_arg);
    // Capping registers per thread trades spills for occupancy. Worth having as
    // a knob because register pressure, not shared memory, is what usually binds
    // a tensor-core kernel's residency, and the tradeoff is not predictable from
    // source.
    var argv_buf: [6][]const u8 = undefined;
    var argc: usize = 0;
    argv_buf[argc] = ptxas;
    argc += 1;
    argv_buf[argc] = arch_arg;
    argc += 1;
    var reg_arg_buf: [32]u8 = undefined;
    if (max_regs) |m| {
        argv_buf[argc] = try std.fmt.bufPrint(&reg_arg_buf, "--maxrregcount={d}", .{m});
        argc += 1;
    }
    argv_buf[argc] = "-o";
    argc += 1;
    argv_buf[argc] = out_cubin;
    argc += 1;
    argv_buf[argc] = in_ptx;
    argc += 1;
    const result = try std.process.run(gpa, io, .{
        .argv = argv_buf[0..argc],
    });
    defer gpa.free(result.stdout);
    defer gpa.free(result.stderr);
    if (!termOk(result.term)) {
        // ptxas has already said exactly what is wrong — an unsupported
        // instruction for the target, a bad operand, an arch mismatch. Discarding
        // that and reporting "failed to assemble" made the caller print "is ptxas
        // available?", which is actively misleading when ptxas is present and
        // simply rejected the input. Assembling a wgmma kernel for sm_90 instead
        // of sm_90a produced precisely that.
        const diag = std.mem.trim(u8, result.stderr, " \t\r\n");
        if (diag.len != 0) {
            std.debug.print("ptxas: {s}\n", .{diag});
        }
        const extra = std.mem.trim(u8, result.stdout, " \t\r\n");
        if (extra.len != 0) {
            std.debug.print("ptxas: {s}\n", .{extra});
        }
        return error.PtxasRejected;
    }
}

/// Version tag the scaffold points at for `zig fetch --save`. Bump with releases.
const scaffold_version = "v0.0.14-alpha";

fn cmdNew(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, args: []const [:0]const u8) !u8 {
    var na: scaffold.NewArgs = .{ .name = "", .version = scaffold_version };
    var have_name = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--dir")) {
            i += 1;
            if (i >= args.len) return usageErr("new: --dir requires a value");
            na.dir = args[i];
        } else if (std.mem.eql(u8, a, "--zoxide-path")) {
            i += 1;
            if (i >= args.len) return usageErr("new: --zoxide-path requires a value");
            na.zoxide_path = args[i];
        } else if (!have_name) {
            na.name = a;
            have_name = true;
        } else {
            return usageErr("new: unexpected argument");
        }
    }
    if (!have_name) return usageErr("expected 'zoxide new <name> [--dir path] [--zoxide-path dir]'");
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writerStreaming(io, &buf);
    defer w.interface.flush() catch {};
    return scaffold.run(gpa, io, env, na, &w.interface);
}

fn cmdRun(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, args: []const [:0]const u8) !u8 {
    var ra: run_cmd.RunArgs = .{ .input = "" };
    var have_input = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const needValue = struct {
            fn get(rest: []const [:0]const u8, idx: *usize) ?[]const u8 {
                idx.* += 1;
                if (idx.* >= rest.len) return null;
                return rest[idx.*];
            }
        }.get;
        if (std.mem.eql(u8, a, "--kernel")) {
            ra.kernel_name = needValue(args, &i) orelse return usageErr("run: --kernel requires a value");
        } else if (std.mem.eql(u8, a, "--n")) {
            const v = needValue(args, &i) orelse return usageErr("run: --n requires a value");
            ra.n = std.fmt.parseInt(usize, v, 10) catch return usageErr("run: --n must be a positive integer");
        } else if (std.mem.eql(u8, a, "--grid")) {
            const v = needValue(args, &i) orelse return usageErr("run: --grid requires a value");
            ra.grid = std.fmt.parseInt(u32, v, 10) catch return usageErr("run: --grid must be a positive integer");
        } else if (std.mem.eql(u8, a, "--block")) {
            const v = needValue(args, &i) orelse return usageErr("run: --block requires a value");
            ra.block = std.fmt.parseInt(u32, v, 10) catch return usageErr("run: --block must be a positive integer");
        } else if (std.mem.eql(u8, a, "--arch")) {
            const v = needValue(args, &i) orelse return usageErr("run: --arch requires a value");
            if (!validateArch(v)) {
                std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ v, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_100a sm_120" });
                return 1;
            }
            ra.arch = v;
        } else if (std.mem.eql(u8, a, "--maxrregcount")) {
            const v = needValue(args, &i) orelse return usageErr("run: --maxrregcount requires a value");
            ra.max_regs = std.fmt.parseInt(u32, v, 10) catch return usageErr("run: --maxrregcount must be a positive integer");
        } else if (!have_input) {
            ra.input = a;
            have_input = true;
        } else {
            return usageErr("run: unexpected argument");
        }
    }
    if (!have_input) return usageErr("expected 'zoxide run <example.ptx|.cubin|name> [--kernel name] [--n N] [--grid G --block B] [--arch sm_XX]'");
    const resolved: ?[]u8 = resolveKernelInput(gpa, io, ra.input) catch |e| switch (e) {
        error.UnknownKernel => return 1,
        else => return e,
    };
    defer if (resolved) |p| {
        cleanupResolved(gpa, io, p);
    };
    if (resolved) |p| ra.input = p;
    if (!fileExists(io, ra.input)) {
        std.debug.print("error: input file not found: '{s}'\n", .{ra.input});
        return 1;
    }

    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writerStreaming(io, &buf);
    const out = &w.interface;
    defer out.flush() catch {};
    return run_cmd.run(gpa, io, env, ra, assemblePtx, out);
}

fn cmdBench(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, args: []const [:0]const u8) !u8 {
    var ba: bench_cmd.BenchArgs = .{ .input = "" };
    var have_input = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--n")) {
            i += 1;
            if (i >= args.len) return usageErr("bench: --n requires a value");
            ba.n = std.fmt.parseInt(usize, args[i], 10) catch return usageErr("bench: --n must be a positive integer");
        } else if (std.mem.eql(u8, a, "--iters")) {
            i += 1;
            if (i >= args.len) return usageErr("bench: --iters requires a value");
            ba.iters = std.fmt.parseInt(u32, args[i], 10) catch return usageErr("bench: --iters must be a positive integer");
        } else if (std.mem.eql(u8, a, "--kernel")) {
            i += 1;
            if (i >= args.len) return usageErr("bench: --kernel requires a value");
            ba.kernel_name = args[i];
        } else if (std.mem.eql(u8, a, "--arch")) {
            i += 1;
            if (i >= args.len) return usageErr("bench: --arch requires a value");
            if (!validateArch(args[i])) {
                std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ args[i], "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_100a sm_120" });
                return 1;
            }
            ba.arch = args[i];
        } else if (std.mem.eql(u8, a, "--maxrregcount")) {
            i += 1;
            if (i >= args.len) return usageErr("bench: --maxrregcount requires a value");
            ba.max_regs = std.fmt.parseInt(u32, args[i], 10) catch return usageErr("bench: --maxrregcount must be a positive integer");
        } else if (!have_input) {
            ba.input = a;
            have_input = true;
        } else {
            return usageErr("bench: unexpected argument");
        }
    }
    if (!have_input) return usageErr("expected 'zoxide bench <kernel.ptx|name> [--n N] [--iters K]'");
    const resolved: ?[]u8 = resolveKernelInput(gpa, io, ba.input) catch |e| switch (e) {
        error.UnknownKernel => return 1,
        else => return e,
    };
    defer if (resolved) |p| {
        cleanupResolved(gpa, io, p);
    };
    if (resolved) |p| ba.input = p;
    if (!fileExists(io, ba.input)) {
        std.debug.print("error: input file not found: '{s}'\n", .{ba.input});
        return 1;
    }
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writerStreaming(io, &buf);
    defer w.interface.flush() catch {};
    return bench_cmd.benchMain(gpa, io, env, ba, assemblePtx, &w.interface);
}

fn usageErr(msg: []const u8) u8 {
    std.debug.print("error: {s}\n", .{msg});
    return 1;
}

/// Resolve a run/bench/lint input: an explicit path that exists wins;
/// otherwise a bare kernel name resolves against the PTX embedded in this
/// binary (build-time `-Dembed-kernels`, default on). Returns null to use
/// the input as-is; otherwise a freshly written temp .ptx path the caller
/// must free and delete. A bare name that matches nothing is an error
/// listing the embedded stems, because "unknown kernel" and "forgot to build
/// kernels/" look identical from the CLI without it.
fn resolveKernelInput(gpa: std.mem.Allocator, io: std.Io, input: []const u8) !?[]u8 {
    if (fileExists(io, input)) return null;
    const stem = emb.stemOf(input) orelse return null; // path-shaped: let the caller report it
    const bytes = emb.lookup(embedded_kernels.kernels, stem) orelse {
        const avail = try emb.stems(gpa, embedded_kernels.kernels);
        defer gpa.free(avail);
        std.debug.print("error: '{s}' is not a file and not an embedded kernel; available:", .{input});
        for (avail) |s| std.debug.print(" {s}", .{s});
        std.debug.print("\n", .{});
        return error.UnknownKernel;
    };
    // The stem must survive as the file's basename: run/bench key kernel
    // selection on it.
    const dir = try std.fmt.allocPrint(gpa, "/tmp/zoxide-embedded-{d}", .{std.c.getpid()});
    defer gpa.free(dir);
    std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    const p = try std.fmt.allocPrint(gpa, "{s}/{s}.ptx", .{ dir, stem });
    errdefer gpa.free(p);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = bytes });
    return p;
}

/// Delete a temp file from resolveKernelInput, and its (now empty) pid dir.
fn cleanupResolved(gpa: std.mem.Allocator, io: std.Io, p: []u8) void {
    std.Io.Dir.deleteFileAbsolute(io, p) catch {};
    if (std.fs.path.dirname(p)) |d| std.Io.Dir.deleteDirAbsolute(io, d) catch {};
    gpa.free(p);
}

fn cmdDoctor(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, args: []const [:0]const u8) !u8 {
    var want_arch: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--arch")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("error: --arch requires a value\n", .{});
                return 1;
            }
            want_arch = args[i];
        } else {
            std.debug.print("error: unexpected argument '{s}' for doctor\n", .{args[i]});
            return 1;
        }
    }
    if (want_arch) |a| {
        if (!validateArch(a)) {
            std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ a, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_100a sm_120" });
            return 1;
        }
    }

    var any_fail = false;

    // --- compile workflow: zig -> PTX ---
    var zig_ok = false;
    if (std.process.run(gpa, io, .{ .argv = &.{ "zig", "version" } }) catch null) |res| {
        defer gpa.free(res.stdout);
        defer gpa.free(res.stderr);
        if (termOk(res.term)) {
            zig_ok = true;
            print(io, "[ok  ] zig: {s}", .{res.stdout});
        } else {
            print(io, "[warn] zig: present but 'zig version' failed\n", .{});
        }
    } else {
        print(io, "[warn] zig: not found on PATH (only needed to compile .zig -> PTX)\n", .{});
    }

    var nvptx_ok = false;
    if (zig_ok) {
        if (std.process.run(gpa, io, .{ .argv = &.{ "zig", "targets" } }) catch null) |res| {
            defer gpa.free(res.stdout);
            defer gpa.free(res.stderr);
            if (termOk(res.term) and std.mem.indexOf(u8, res.stdout, "\"nvptx64\"") != null) {
                nvptx_ok = true;
                print(io, "[ok  ] nvptx64 target: available\n", .{});
            } else {
                print(io, "[warn] nvptx64 target: NOT available in this zig build\n", .{});
            }
        } else {
            print(io, "[warn] nvptx64 target: unknown (no zig)\n", .{});
        }
    } else {
        print(io, "[warn] nvptx64 target: unknown (no zig)\n", .{});
    }

    // --- assemble workflow: PTX -> cubin ---
    var ptxas_ok = false;
    if (findPtxas(gpa, io, env)) |p| {
        defer gpa.free(p);
        ptxas_ok = true;
        print(io, "[ok  ] ptxas: {s}\n", .{p});
    } else {
        print(io, "[warn] ptxas: not found (PATH, $CUDA_HOME/bin, /usr/local/cuda/bin); only needed for 'zoxide cubin'\n", .{});
    }

    // --- future LTOIR route ---
    if (findLibNvvm()) |name| {
        print(io, "[ok  ] libNVVM: found ({s})\n", .{name});
    } else {
        print(io, "[info] libNVVM: not found (only needed for a future LTOIR route)\n", .{});
    }

    // --- run workflow: GPU ---
    const probe = probeGpu(gpa, io);
    defer probe.deinit(gpa);
    var gpu: ?GpuInfo = null;
    switch (probe) {
        .ok => |g| {
            gpu = g;
            print(io, "[ok  ] gpu: {s} (compute capability {s}, driver {s})\n", .{ g.name, g.cc, g.driver });
            if (want_arch) |a| {
                const gpu_sm = ccToSm(g.cc);
                if (!std.mem.eql(u8, a, &gpu_sm)) {
                    print(io, "[fail] arch check: requested {s} does not match this GPU ({s}); cubins are not forward-compatible across major CC\n", .{ a, gpu_sm });
                    any_fail = true;
                } else {
                    print(io, "[ok  ] arch check: {s} matches this GPU\n", .{a});
                }
            }
        },
        .not_installed => print(io, "[warn] gpu: nvidia-smi not found; fine on a dev machine\n", .{}),
        // Distinguished from "not installed" on purpose. nvidia-smi being present
        // and failing means something is wrong that its own message identifies —
        // typically a container that did not receive the device, or a
        // driver/library version mismatch. Reporting that as "no GPU here" sends
        // the reader looking in the wrong place.
        .failed => |msg| print(io,
            \\[warn] gpu: nvidia-smi is installed but failed. Its message:
            \\         {s}
            \\       In a container this usually means the device was not passed
            \\       through, or the driver and the userspace libraries disagree.
            \\
        , .{msg}),
        .unparsable => |msg| print(io,
            \\[warn] gpu: nvidia-smi succeeded but its output could not be parsed:
            \\         {s}
            \\
        , .{msg}),
    }

    print(io, "summary:\n", .{});
    if (zig_ok and nvptx_ok) {
        print(io, "  compile (zig -> PTX):      ready\n", .{});
    } else if (zig_ok) {
        print(io, "  compile (zig -> PTX):      unavailable (no nvptx64 backend)\n", .{});
    } else {
        print(io, "  compile (zig -> PTX):      unavailable (zig not found)\n", .{});
    }
    print(io, "  assemble (ptxas -> cubin): {s}\n", .{if (ptxas_ok) "ready" else "unavailable (ptxas not found)"});
    if (gpu) |g| {
        const gpu_sm = ccToSm(g.cc);
        print(io, "  run (GPU):                 ready — {s}, {s}\n", .{ g.name, gpu_sm });
    } else {
        print(io, "  run (GPU):                 unavailable (no GPU visible)\n", .{});
    }

    return if (any_fail) 1 else 0;
}

const GpuInfo = struct { name: []u8, cc: []u8, driver: []u8 };

/// Run nvidia-smi and parse "name, compute_cap, driver_version" for the first GPU.
/// Returns null (caller-owned fields otherwise) when nvidia-smi is absent/fails.
/// Outcome of probing for a GPU.
///
/// Previously all of these collapsed to `null` and `doctor` reported "no GPU
/// visible (nvidia-smi missing or failed); fine on a dev machine". That is
/// actively misleading in the case that matters most: inside a container,
/// `nvidia-smi` is usually installed and *fails*, with a message that names the
/// cause — a device mapping the container did not get, or a driver/library
/// version mismatch. Both look identical to "no GPU here" unless the message
/// survives, and doctor's whole job is to surface it.
const GpuProbe = union(enum) {
    ok: GpuInfo,
    /// Could not be spawned: not installed, or not on PATH. Expected on a dev
    /// machine and in CI.
    not_installed,
    /// Ran and failed. Owns nvidia-smi's own diagnosis.
    failed: []u8,
    /// Ran successfully but produced nothing usable. Owns what it printed.
    unparsable: []u8,

    fn deinit(self: GpuProbe, gpa: std.mem.Allocator) void {
        switch (self) {
            .ok => |g| {
                gpa.free(g.name);
                gpa.free(g.cc);
                gpa.free(g.driver);
            },
            .failed, .unparsable => |m| gpa.free(m),
            .not_installed => {},
        }
    }
};

fn probeGpu(gpa: std.mem.Allocator, io: std.Io) GpuProbe {
    const res = std.process.run(gpa, io, .{
        .argv = &.{ "nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader" },
    }) catch return .not_installed;
    if (!termOk(res.term)) {
        gpa.free(res.stdout);
        // nvidia-smi's own words; this is the whole point.
        const msg = std.mem.trim(u8, res.stderr, " \r\n\t");
        if (msg.len == 0) {
            gpa.free(res.stderr);
            return .{ .failed = gpa.dupe(u8, "nvidia-smi exited non-zero with no message") catch return .not_installed };
        }
        const owned = gpa.dupe(u8, msg) catch {
            gpa.free(res.stderr);
            return .not_installed;
        };
        gpa.free(res.stderr);
        return .{ .failed = owned };
    }
    defer gpa.free(res.stderr);
    defer gpa.free(res.stdout);
    var lines = std.mem.splitScalar(u8, res.stdout, '\n');
    const first = std.mem.trim(u8, lines.first(), " \r\t");
    const unparsable = struct {
        fn make(a: std.mem.Allocator, raw: []const u8) GpuProbe {
            const shown = if (raw.len == 0) "empty output" else raw;
            return .{ .unparsable = a.dupe(u8, shown) catch return .not_installed };
        }
    }.make;
    if (first.len == 0) return unparsable(gpa, std.mem.trim(u8, res.stdout, " \r\n\t"));
    var fields = std.mem.splitScalar(u8, first, ',');
    const name = std.mem.trim(u8, fields.next() orelse return unparsable(gpa, first), " ");
    const cc = std.mem.trim(u8, fields.next() orelse return unparsable(gpa, first), " ");
    const driver = std.mem.trim(u8, fields.next() orelse return unparsable(gpa, first), " ");
    return .{ .ok = .{
        .name = gpa.dupe(u8, name) catch return .not_installed,
        .cc = gpa.dupe(u8, cc) catch return .not_installed,
        .driver = gpa.dupe(u8, driver) catch return .not_installed,
    } };
}

/// Map a compute capability like "9.0" to an sm arch string like "sm_90".
/// CC values are major.minor with single digits, so the result is always 5 chars.
fn ccToSm(cc: []const u8) [5]u8 {
    var buf: [5]u8 = .{ 's', 'm', '_', '?', '?' };
    var n: usize = 3;
    for (cc) |c| {
        if (c == '.') continue;
        if (!std.ascii.isDigit(c)) break;
        if (n >= buf.len) break;
        buf[n] = c;
        n += 1;
    }
    return buf;
}

fn print(io: std.Io, comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, fmt, args) catch return;
    std.Io.File.stdout().writeStreamingAll(io, msg) catch {};
}

fn termOk(term: std.process.Child.Term) bool {
    return switch (term) {
        .exited => |code| code == 0,
        else => false,
    };
}

fn fileExists(io: std.Io, path: []const u8) bool {
    if (std.fs.path.isAbsolute(path)) {
        std.Io.Dir.accessAbsolute(io, path, .{}) catch return false;
    } else {
        std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    }
    return true;
}

/// Locate ptxas via the shared tool probe (PATH, $CUDA_HOME/bin,
/// /usr/local/cuda/bin). Returned slice is caller-owned.
fn findPtxas(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map) ?[]u8 {
    return toolwrap.probe(gpa, io, env, .{
        .exe = "ptxas",
        .sub = "cubin",
        .purpose = "assemble PTX into cubin",
    });
}

fn findLibNvvm() ?[]const u8 {
    const candidates = [_][]const u8{
        "libnvvm.dylib",
        "libNVVM.dylib",
        "libnvvm.so",
        "libnvvm.so.4",
    };
    for (candidates) |name| {
        if (std.DynLib.open(name)) |lib| {
            var l = lib;
            l.close();
            return name;
        } else |_| {}
    }
    return null;
}
