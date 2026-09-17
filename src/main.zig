const std = @import("std");

/// Baseline GPU: NVIDIA H20 (Hopper, compute capability 9.0).
const default_sm = "sm_90";
const valid_arches = [_][]const u8{ "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120" };

fn validateArch(arch: []const u8) bool {
    if (!std.mem.startsWith(u8, arch, "sm_") or arch.len <= 3) return false;
    for (arch[3..]) |c| if (!std.ascii.isDigit(c)) return false;
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
        \\  zoxide doctor [--arch sm_XX]                          probe zig / nvptx / ptxas / libNVVM / GPU
        \\  supported arch values: {s}
        \\
    , .{ default_sm, default_sm, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120" });
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
        std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ parsed.arch, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120" });
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
        std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ parsed.arch, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120" });
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
            std.debug.print("error: invalid arch '{s}'; expected one of: {s}\n", .{ a, "sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120" });
            return 1;
        }
    }

    var ok = true;

    print(io, "zig: ", .{});
    if (std.process.run(gpa, io, .{ .argv = &.{ "zig", "version" } }) catch null) |res| {
        defer gpa.free(res.stdout);
        defer gpa.free(res.stderr);
        if (termOk(res.term)) {
            print(io, "{s}", .{res.stdout});
        } else {
            print(io, "present but 'zig version' failed\n", .{});
            ok = false;
        }
    } else {
        print(io, "not found on PATH\n", .{});
        ok = false;
    }

    print(io, "nvptx64 target: ", .{});
    if (std.process.run(gpa, io, .{ .argv = &.{ "zig", "targets" } }) catch null) |res| {
        defer gpa.free(res.stdout);
        defer gpa.free(res.stderr);
        if (termOk(res.term) and std.mem.indexOf(u8, res.stdout, "\"nvptx64\"") != null) {
            print(io, "available\n", .{});
        } else {
            print(io, "NOT available\n", .{});
            ok = false;
        }
    } else {
        print(io, "unknown (no zig)\n", .{});
        ok = false;
    }

    print(io, "ptxas: ", .{});
    if (findPtxas(gpa, io, env)) |p| {
        defer gpa.free(p);
        print(io, "{s}\n", .{p});
    } else {
        print(io, "not found (PATH, $CUDA_HOME/bin, /usr/local/cuda/bin)\n", .{});
    }

    print(io, "libNVVM: ", .{});
    if (findLibNvvm()) |name| {
        print(io, "found ({s})\n", .{name});
    } else {
        print(io, "not found (dlopen probe failed for libnvvm variants)\n", .{});
    }

    print(io, "gpu: ", .{});
    if (probeGpu(gpa, io)) |gpu| {
        defer gpa.free(gpu.name);
        defer gpa.free(gpu.cc);
        defer gpa.free(gpu.driver);
        print(io, "{s} (compute capability {s}, driver {s})\n", .{ gpu.name, gpu.cc, gpu.driver });
        if (want_arch) |a| {
            const gpu_sm = ccToSm(gpu.cc);
            if (!std.mem.eql(u8, a, &gpu_sm)) {
                print(io, "warning: requested arch {s} does not match this GPU ({s}); cubins are not forward-compatible across major CC\n", .{ a, gpu_sm });
            } else {
                print(io, "arch check: {s} matches this GPU\n", .{a});
            }
        }
    } else {
        print(io, "no GPU visible (nvidia-smi missing or failed)\n", .{});
    }

    print(io, "doctor: {s}\n", .{if (ok) "core toolchain OK" else "problems found"});
    return if (ok) 0 else 1;
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
