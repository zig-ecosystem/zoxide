//! `zoxide sanitize` / `zoxide debug` — thin wrappers that locate
//! compute-sanitizer / cuda-gdb and replace the current process with them.
//!
//! Thin is deliberate. Both tools have their own large option languages; the
//! wrapper's whole job is (a) finding the binary the same way the rest of
//! the toolchain finds ptxas, and (b) passing the user's arguments through
//! untouched. Argument syntax is the tool's own: everything after the
//! subcommand is forwarded verbatim, so
//!
//!     zoxide sanitize --tool memcheck ./zoxide run kernels/vector_add.ptx
//!     zoxide debug --args ./zoxide run kernels/vector_add.ptx
//!
//! both work. A leading `--` is accepted and stripped, purely for
//! readability — the tools do not need it.
//!
//! Execution is `std.process.replace` (exec), not spawn-and-capture: the
//! wrapper's exit status *is* the tool's, and stdio stays attached, which is
//! what keeps cuda-gdb interactive. The PATH probe runs the candidate with
//! `--version` first, matching findPtxas: a binary that exists but fails to
//! run is a broken toolchain, worth a note, not a silent fallthrough.

const std = @import("std");

pub const Tool = struct {
    /// Executable name as probed on PATH.
    exe: []const u8,
    /// Subcommand name, for messages.
    sub: []const u8,
    /// One-line purpose, for --help and the not-found message.
    purpose: []const u8,
};

pub const sanitize_tool = Tool{
    .exe = "compute-sanitizer",
    .sub = "sanitize",
    .purpose = "run a command under compute-sanitizer (memcheck/racecheck/...)",
};

pub const debug_tool = Tool{
    .exe = "cuda-gdb",
    .sub = "debug",
    .purpose = "run a command under cuda-gdb",
};

/// The probe order, as a pure function of the environment: bare name (PATH),
/// $CUDA_HOME/bin, /usr/local/cuda/bin. Returned slices are caller-owned;
/// the list itself too.
pub fn probeCandidates(gpa: std.mem.Allocator, env: ?*const std.process.Environ.Map, exe: []const u8) ![][]u8 {
    var list = std.array_list.Managed([]u8).init(gpa);
    errdefer {
        for (list.items) |p| gpa.free(p);
        list.deinit();
    }
    try list.append(try gpa.dupe(u8, exe));
    if (env) |e| {
        if (e.get("CUDA_HOME")) |home| {
            try list.append(try std.fs.path.join(gpa, &.{ home, "bin", exe }));
        }
    }
    try list.append(try std.fs.path.join(gpa, &.{ "/usr/local/cuda/bin", exe }));
    return list.toOwnedSlice();
}

/// Build the argv for process replacement: the resolved tool path, then the
/// user's arguments, minus one leading `--` if present. Caller owns the
/// slice (not the elements, which borrow).
pub fn assembleArgv(gpa: std.mem.Allocator, resolved: []const u8, user_args: []const [:0]const u8) ![][]const u8 {
    const args = if (user_args.len > 0 and std.mem.eql(u8, user_args[0], "--")) user_args[1..] else user_args;
    var argv = try gpa.alloc([]const u8, args.len + 1);
    argv[0] = resolved;
    for (args, 1..) |a, i| argv[i] = a;
    return argv;
}

fn fileExists(io: std.Io, path: []const u8) bool {
    if (std.fs.path.isAbsolute(path)) {
        std.Io.Dir.accessAbsolute(io, path, .{}) catch return false;
    } else {
        std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    }
    return true;
}

fn termOk(term: std.process.Child.Term) bool {
    return switch (term) {
        .exited => |code| code == 0,
        else => false,
    };
}

/// Locate the tool, following the probe order from `probeCandidates`.
/// The PATH entry is validated by running `<exe> --version`, matching the
/// ptxas probe. Returned slice is caller-owned.
pub fn probe(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, tool: Tool) ?[]u8 {
    // A binary on PATH that fails `--version` is a broken toolchain, not an
    // absent one; say so rather than falling through in silence.
    if (std.process.run(gpa, io, .{ .argv = &.{ tool.exe, "--version" } }) catch null) |res| {
        gpa.free(res.stdout);
        gpa.free(res.stderr);
        if (termOk(res.term)) return gpa.dupe(u8, tool.exe) catch null;
        std.debug.print("note: a {s} on PATH failed `--version`; ignoring it and looking in CUDA_HOME\n", .{tool.exe});
    }
    if (env.get("CUDA_HOME")) |home| {
        const cand = std.fs.path.join(gpa, &.{ home, "bin", tool.exe }) catch null;
        if (cand) |c| {
            if (fileExists(io, c)) return c;
            gpa.free(c);
        }
    }
    const fallback = std.fs.path.join(gpa, &.{ "/usr/local/cuda/bin", tool.exe }) catch null;
    if (fallback) |f| {
        if (fileExists(io, f)) return f;
        gpa.free(f);
    }
    return null;
}

/// Entry point for `zoxide <tool.sub> [args...]`. Returns the exit status;
/// on success it does not return at all (process replaced).
pub fn run(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, args: []const [:0]const u8, tool: Tool) !u8 {
    for (args) |a| {
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            std.debug.print(
                \\usage: zoxide {s} [tool options] [--] <command...>
                \\  {s}.
                \\  Arguments are forwarded to {s} verbatim; a leading `--` is
                \\  accepted for readability but not required.
                \\  example: zoxide {s} --tool memcheck ./zoxide run kernels/vector_add.ptx
                \\
            , .{ tool.sub, tool.purpose, tool.exe, tool.sub });
            return 0;
        }
    }
    if (args.len == 0) {
        std.debug.print("error: zoxide {s}: nothing to run; try 'zoxide {s} --help'\n", .{ tool.sub, tool.sub });
        return 1;
    }
    const resolved = probe(gpa, io, env, tool) orelse {
        // Mirror `zoxide cubin` on a missing ptxas: for a subcommand whose
        // whole job is this tool, absence is an error, not a warn.
        std.debug.print(
            \\error: {s} not found.
            \\  Looked on PATH, in $CUDA_HOME/bin, and at /usr/local/cuda/bin/{s}.
            \\  Install the CUDA toolkit to {s}.
            \\
        , .{ tool.exe, tool.exe, tool.purpose });
        return 1;
    };
    defer gpa.free(resolved);

    const argv = try assembleArgv(gpa, resolved, args);
    defer gpa.free(argv);
    // Reaching the call's result means the exec failed: replace returns the
    // error as a value and never returns on success.
    const err = std.process.replace(io, .{ .argv = argv });
    std.debug.print("error: failed to exec {s}: {s}\n", .{ resolved, @errorName(err) });
    return 1;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "probeCandidates: PATH first, CUDA_HOME second, /usr/local/cuda last" {
    const alloc = std.testing.allocator;
    var env = std.process.Environ.Map.init(alloc);
    defer env.deinit();
    try env.put("CUDA_HOME", "/opt/cuda");
    const cands = try probeCandidates(alloc, &env, "compute-sanitizer");
    defer {
        for (cands) |c| alloc.free(c);
        alloc.free(cands);
    }
    try std.testing.expectEqual(@as(usize, 3), cands.len);
    try std.testing.expectEqualStrings("compute-sanitizer", cands[0]);
    try std.testing.expectEqualStrings("/opt/cuda/bin/compute-sanitizer", cands[1]);
    try std.testing.expectEqualStrings("/usr/local/cuda/bin/compute-sanitizer", cands[2]);
}

test "probeCandidates: no CUDA_HOME skips the middle entry" {
    const alloc = std.testing.allocator;
    var env = std.process.Environ.Map.init(alloc);
    defer env.deinit();
    const cands = try probeCandidates(alloc, &env, "cuda-gdb");
    defer {
        for (cands) |c| alloc.free(c);
        alloc.free(cands);
    }
    try std.testing.expectEqual(@as(usize, 2), cands.len);
    try std.testing.expectEqualStrings("cuda-gdb", cands[0]);
    try std.testing.expectEqualStrings("/usr/local/cuda/bin/cuda-gdb", cands[1]);
}

test "assembleArgv: verbatim forwarding, tool first" {
    const alloc = std.testing.allocator;
    const args = [_][:0]const u8{ "--tool", "memcheck", "./zoxide", "run", "x.ptx" };
    const argv = try assembleArgv(alloc, "/opt/cuda/bin/compute-sanitizer", &args);
    defer alloc.free(argv);
    try std.testing.expectEqual(@as(usize, 6), argv.len);
    try std.testing.expectEqualStrings("/opt/cuda/bin/compute-sanitizer", argv[0]);
    try std.testing.expectEqualStrings("memcheck", argv[2]);
    try std.testing.expectEqualStrings("x.ptx", argv[5]);
}

test "assembleArgv: a single leading -- is stripped, anything else kept" {
    const alloc = std.testing.allocator;
    const args = [_][:0]const u8{ "--", "./zoxide", "run", "x.ptx" };
    const argv = try assembleArgv(alloc, "cuda-gdb", &args);
    defer alloc.free(argv);
    try std.testing.expectEqual(@as(usize, 4), argv.len);
    try std.testing.expectEqualStrings("./zoxide", argv[1]);

    // A `--` in any other position is the tool's own argument and stays.
    const args2 = [_][:0]const u8{ "--tool", "--", "memcheck" };
    const argv2 = try assembleArgv(alloc, "cuda-gdb", &args2);
    defer alloc.free(argv2);
    try std.testing.expectEqual(@as(usize, 4), argv2.len);
    try std.testing.expectEqualStrings("--", argv2[2]);
}
