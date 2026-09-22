const std = @import("std");
const run_cmd = @import("run.zig");
const scaffold = @import("scaffold.zig");
const bench_cmd = @import("bench.zig");
const gen_cmd = @import("gen.zig");

/// Baseline GPU: NVIDIA H20 (Hopper, compute capability 9.0).
const default_sm = "sm_90";
// `sm_90a` is the architecture-specific Hopper target: `wgmma.*` and the
// TMA family are only available there (LLVM predicate `hasSM90a`, ptxas
// `-arch sm_90a`). Such code is not forward-compatible with later
// architectures, so it stays opt-in.
const valid_arches = [_][]const u8{ "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_120" };

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
        \\  zoxide new <name> [--dir path] [--zoxide-path dir]    scaffold a host+device package
        \\  zoxide doctor [--arch sm_XX]                          probe zig / nvptx / ptxas / libNVVM / GPU
        \\  zoxide run <example.ptx|.cubin> [--kernel name] [--n N] [--grid G --block B] [--arch sm_XX]
        \\                                                       run an example kernel on the GPU and verify results
        \\  supported arch values: {s}
        \\
    , .{ default_sm, default_sm, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_120" });
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
        std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ parsed.arch, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_120" });
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

fn cmdCubin(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, args: []const [:0]const u8) !u8 {
    const parsed = parseIoArgs(args, true) catch {
        std.debug.print("error: expected 'zoxide cubin <in.ptx> -o out.cubin [--arch sm_XX]'\n", .{});
        return 1;
    };
    if (!validateArch(parsed.arch)) {
        std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ parsed.arch, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_120" });
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
    if (!termOk(result.term)) return error.PtxasFailed;
}

/// Version tag the scaffold points at for `zig fetch --save`. Bump with releases.
const scaffold_version = "v0.0.10-alpha";

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
                std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ v, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_120" });
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
    if (!have_input) return usageErr("expected 'zoxide run <example.ptx|.cubin> [--kernel name] [--n N] [--grid G --block B] [--arch sm_XX]'");
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
                std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ args[i], "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_120" });
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
    if (!have_input) return usageErr("expected 'zoxide bench <sgemm_naive|sgemm_tiled>.ptx [--n N] [--iters K]'");
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
            std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ a, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_120" });
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
    var gpu: ?GpuInfo = null;
    defer if (gpu) |g| {
        gpa.free(g.name);
        gpa.free(g.cc);
        gpa.free(g.driver);
    };
    if (probeGpu(gpa, io)) |g| {
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
    } else {
        print(io, "[warn] gpu: no GPU visible (nvidia-smi missing or failed); fine on a dev machine\n", .{});
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
fn probeGpu(gpa: std.mem.Allocator, io: std.Io) ?GpuInfo {
    const res = std.process.run(gpa, io, .{
        .argv = &.{ "nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader" },
    }) catch return null;
    defer gpa.free(res.stderr);
    if (!termOk(res.term)) {
        gpa.free(res.stdout);
        return null;
    }
    defer gpa.free(res.stdout);
    var lines = std.mem.splitScalar(u8, res.stdout, '\n');
    const first = std.mem.trim(u8, lines.first(), " \r\t");
    if (first.len == 0) return null;
    var fields = std.mem.splitScalar(u8, first, ',');
    const name = std.mem.trim(u8, fields.next() orelse return null, " ");
    const cc = std.mem.trim(u8, fields.next() orelse return null, " ");
    const driver = std.mem.trim(u8, fields.next() orelse return null, " ");
    return .{
        .name = gpa.dupe(u8, name) catch return null,
        .cc = gpa.dupe(u8, cc) catch return null,
        .driver = gpa.dupe(u8, driver) catch return null,
    };
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

/// Locate ptxas: PATH probe first, then $CUDA_HOME/bin, then /usr/local/cuda/bin.
/// Returned slice is caller-owned.
fn findPtxas(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map) ?[]u8 {
    if (std.process.run(gpa, io, .{ .argv = &.{ "ptxas", "--version" } }) catch null) |res| {
        gpa.free(res.stdout);
        gpa.free(res.stderr);
        if (termOk(res.term)) return gpa.dupe(u8, "ptxas") catch null;
    }
    if (env.get("CUDA_HOME")) |home| {
        const cand = std.fs.path.join(gpa, &.{ home, "bin", "ptxas" }) catch null;
        if (cand) |c| {
            if (fileExists(io, c)) return c;
            gpa.free(c);
        }
    }
    const fallback = "/usr/local/cuda/bin/ptxas";
    if (fileExists(io, fallback)) return gpa.dupe(u8, fallback) catch null;
    return null;
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
