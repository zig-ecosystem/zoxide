//! `zoxide new` — generate a working host+device package.
//!
//! Getting a first GPU program running in Zig currently means copying
//! `tests/downstream/` by hand: a build.zig that compiles one module for
//! nvptx64 and another for the host, a shared signature module so the two
//! cannot drift, and the `@embedFile` wiring between them. None of that is
//! guessable, and all of it is boilerplate.
//!
//! The generated package is deliberately the same shape as
//! `tests/downstream/`, so the integration test and the scaffold cannot
//! disagree about what the recommended layout is.

const std = @import("std");

pub const NewArgs = struct {
    name: []const u8,
    /// Where to create the package directory. Defaults to the name itself.
    dir: ?[]const u8 = null,
    /// Local path to a zoxide checkout. When set, the generated package depends
    /// on it by path, which needs no network. Otherwise the caller is told which
    /// `zig fetch --save` to run.
    zoxide_path: ?[]const u8 = null,
    /// Version tag used in the printed `zig fetch --save` command.
    version: []const u8,
};

/// Replace `@@KERNEL@@` and `@@NAME@@` in a template.
fn substitute(gpa: std.mem.Allocator, template: []const u8, kernel: []const u8, name: []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    var rest = template;
    while (rest.len != 0) {
        const at = std.mem.indexOf(u8, rest, "@@") orelse {
            try buf.appendSlice(gpa, rest);
            break;
        };
        try buf.appendSlice(gpa, rest[0..at]);
        if (std.mem.startsWith(u8, rest[at..], "@@KERNEL@@")) {
            try buf.appendSlice(gpa, kernel);
            rest = rest[at + "@@KERNEL@@".len ..];
        } else if (std.mem.startsWith(u8, rest[at..], "@@NAME@@")) {
            try buf.appendSlice(gpa, name);
            rest = rest[at + "@@NAME@@".len ..];
        } else {
            try buf.appendSlice(gpa, "@@");
            rest = rest[at + 2 ..];
        }
    }
    return buf.toOwnedSlice(gpa);
}

/// Identifier-safe form of a package name, for `.name = .<ident>` in the zon.
fn identOf(gpa: std.mem.Allocator, name: []const u8) ![]u8 {
    const out = try gpa.alloc(u8, name.len);
    for (name, 0..) |c, i| {
        out[i] = switch (c) {
            'a'...'z', 'A'...'Z', '0'...'9', '_' => c,
            else => '_',
        };
    }
    return out;
}

fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64) return false;
    // Must start with a letter or underscore so the zon identifier is valid.
    switch (name[0]) {
        'a'...'z', 'A'...'Z', '_' => {},
        else => return false,
    }
    for (name) |c| switch (c) {
        'a'...'z', 'A'...'Z', '0'...'9', '_', '-' => {},
        else => return false,
    };
    return true;
}

const abi_zig =
    \\//! Kernel signatures, imported by both the device kernel and the host
    \\//! program so the two cannot drift apart.
    \\//!
    \\//! Declared without `callconv(.kernel)` on purpose: that calling convention
    \\//! resolves per target, and on a host architecture
    \\//! `std.builtin.CallingConvention.kernel` is `unreachable`, so the type
    \\//! cannot even be named there. Only the parameter list matters for
    \\//! launching, and the device side asserts its definition matches.
    \\
    \\pub const scale = fn (x: [*]const f32, y: [*]f32, k: f32, n: u32) void;
    \\
;

const kernel_zig =
    \\const cuda = @import("cuda");
    \\const api = @import("kernels_abi");
    \\
    \\/// y[i] = x[i] * k
    \\pub fn scale(x: [*]const f32, y: [*]f32, k: f32, n: u32) callconv(.kernel) void {
    \\    const gid = cuda.globalThreadId();
    \\    if (gid < n) y[gid] = x[gid] * k;
    \\}
    \\
    \\comptime {
    \\    // Changing these parameters without updating kernels_abi.zig is a
    \\    // compile error here, rather than wrong numbers at runtime.
    \\    cuda.abi.assertMatches(api.scale, @TypeOf(scale));
    \\    // Keeps the kernel alive through dead-code elimination. Without this the
    \\    // NVPTX backend has nothing referencing it and emits no entry point.
    \\    _ = cuda.Keep(.{&scale}).__zoxide_keep_kernels;
    \\}
    \\
;

const main_zig_template =
    \\const std = @import("std");
    \\const gpu = @import("zoxide_host");
    \\const api = @import("kernels_abi");
    \\
    \\const ptx = @embedFile("kernel_ptx");
    \\
    \\const n = 1 << 20;
    \\const block = 256;
    \\
    \\pub fn main(init: std.process.Init) !u8 {
    \\    _ = init;
    \\
    \\    var drv = gpu.Driver.load() catch |e| {
    \\        std.debug.print("no libcuda ({s}) — this needs an NVIDIA GPU\n", .{@errorName(e)});
    \\        return 0;
    \\    };
    \\    defer drv.unload();
    \\    var ctx = try gpu.Context.init(&drv);
    \\
    \\    var name_buf: [128]u8 = undefined;
    \\    std.debug.print("device: {s}\n", .{ctx.name(&name_buf)});
    \\
    \\    // PTX is JIT-compiled by the driver, so no ptxas is needed to build.
    \\    const mod = try ctx.moduleFromPtx(ptx);
    \\    // "kernel" here is kernel.zig's stem — the PTX symbol prefix comes from
    \\    // the root source file's name, not from the build artifact's name.
    \\    const scale = try mod.kernel(api.scale, gpu.symbol("@@KERNEL@@", "scale"));
    \\
    \\    // Register and shared-memory use as the driver sees it. A non-zero spill
    \\    // is a performance bug worth noticing early.
    \\    const res = try scale.resources();
    \\    std.debug.print("kernel: {d} regs/thread, {d} B shared, {d} blocks/SM\n", .{
    \\        res.regs_per_thread,
    \\        res.shared_bytes,
    \\        try scale.occupancy(block, 0),
    \\    });
    \\
    \\    // Page-locked staging, so the async copies below actually overlap.
    \\    const hx = try ctx.allocPinned(f32, n);
    \\    defer hx.free();
    \\    const hy = try ctx.allocPinned(f32, n);
    \\    defer hy.free();
    \\    for (hx.items, 0..) |*v, i| v.* = @floatFromInt(i % 1000);
    \\
    \\    const dx = try ctx.allocSlice(f32, n);
    \\    defer ctx.freeSlice(dx);
    \\    const dy = try ctx.allocSlice(f32, n);
    \\    defer ctx.freeSlice(dy);
    \\
    \\    const stream = try ctx.createStream(true);
    \\    defer stream.destroy();
    \\
    \\    try ctx.uploadAsync(dx, hx.items, stream);
    \\    // The argument tuple is checked against api.scale at compile time: wrong
    \\    // count, order, width or a host pointer here is a compile error.
    \\    try scale.launchOn(stream, .{ .x = gpu.gridFor(n, block) }, .{ .x = block }, 0, .{
    \\        dx, dy, @as(f32, 2.5), @as(u32, n),
    \\    });
    \\    try ctx.downloadAsync(hy.items, dy, stream);
    \\    try stream.sync();
    \\
    \\    var bad: usize = 0;
    \\    for (hx.items, hy.items) |x, y| {
    \\        if (y != x * 2.5) bad += 1;
    \\    }
    \\    if (bad != 0) {
    \\        std.debug.print("FAIL: {d}/{d} wrong\n", .{ bad, n });
    \\        return 1;
    \\    }
    \\    std.debug.print("PASS: {d} elements exact\n", .{n});
    \\    return 0;
    \\}
    \\
;

const build_zig_template =
    \\const std = @import("std");
    \\
    \\pub fn build(b: *std.Build) void {
    \\    const target = b.standardTargetOptions(.{});
    \\    const optimize = b.standardOptimizeOption(.{});
    \\
    \\    const zoxide = b.dependency("zoxide", .{});
    \\    const zx = @import("zoxide");
    \\
    \\    // Shared by both sides, so a signature change breaks the build rather
    \\    // than producing wrong results.
    \\    const abi = b.createModule(.{ .root_source_file = b.path("kernels_abi.zig") });
    \\
    \\    // Device side: kernel.zig -> PTX for nvptx64. The Compile step is
    \\    // returned rather than just its output path so the shared ABI module can
    \\    // be added to it.
    \\    const kernel_obj = zx.addNvptxKernelObject(b, "@@KERNEL@@", b.path("kernel.zig"), zoxide.path("src/cuda.zig"), .{});
    \\    kernel_obj.root_module.addImport("kernels_abi", abi);
    \\
    \\    // Host side: embeds the PTX and launches it.
    \\    const exe = b.addExecutable(.{
    \\        .name = "@@NAME@@",
    \\        .root_module = b.createModule(.{
    \\            .root_source_file = b.path("main.zig"),
    \\            .target = target,
    \\            .optimize = optimize,
    \\            .link_libc = true, // the host runner dlopens libcuda
    \\        }),
    \\    });
    \\    exe.root_module.addImport("zoxide_host", zoxide.module("zoxide_host"));
    \\    exe.root_module.addImport("kernels_abi", abi);
    \\    exe.root_module.addAnonymousImport("kernel_ptx", .{ .root_source_file = kernel_obj.getEmittedAsm() });
    \\    b.installArtifact(exe);
    \\
    \\    const run = b.addRunArtifact(exe);
    \\    if (b.args) |args| run.addArgs(args);
    \\    const run_step = b.step("run", "Build and run (needs an NVIDIA GPU)");
    \\    run_step.dependOn(&run.step);
    \\
    \\    // `zig build ptx` leaves the generated PTX where you can read it, which
    \\    // is worth doing: things like an accidental register spill are visible
    \\    // there and nowhere else.
    \\    const ptx_step = b.step("ptx", "Emit the kernel PTX to zig-out/kernels/");
    \\    ptx_step.dependOn(&b.addInstallFileWithDir(
    \\        kernel_obj.getEmittedAsm(),
    \\        .{ .custom = "kernels" },
    \\        "@@KERNEL@@.ptx",
    \\    ).step);
    \\}
    \\
;

const gitignore = ".zig-cache/\nzig-out/\n";

fn readmeFor(gpa: std.mem.Allocator, name: []const u8) ![]u8 {
    return std.fmt.allocPrint(gpa,
        \\# {[n]s}
        \\
        \\A CUDA program in Zig: the kernel compiles to PTX for nvptx64, the host
        \\program embeds it and launches it.
        \\
        \\```sh
        \\zig build run     # needs an NVIDIA GPU
        \\zig build ptx     # emit the PTX to zig-out/kernels/ and read it
        \\```
        \\
        \\## Layout
        \\
        \\| file | role |
        \\| --- | --- |
        \\| `kernels_abi.zig` | kernel signatures, imported by both sides |
        \\| `kernel.zig` | device code, compiled for nvptx64 |
        \\| `main.zig` | host code, embeds the PTX and launches |
        \\
        \\`kernels_abi.zig` is the single source of truth. The device side asserts
        \\its definition matches, and the host side uses it to check launch
        \\arguments, so changing a kernel's parameters without updating the host is
        \\a compile error instead of wrong numbers at runtime.
        \\
        \\Two things that are not obvious:
        \\
        \\The shared declaration omits `callconv(.kernel)`. That convention
        \\resolves per target and is `unreachable` on host architectures, so the
        \\type cannot be named there at all. Only the parameter list matters for
        \\launching.
        \\
        \\`kernel.zig` ends with `cuda.Keep(...)`. Without something referencing
        \\the kernel, dead-code elimination removes it and the PTX comes out with
        \\no entry point.
        \\
    , .{ .n = name });
}

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: *std.process.Environ.Map,
    args: NewArgs,
    out: *std.Io.Writer,
) !u8 {
    if (!validName(args.name)) {
        try out.print("error: '{s}' is not a usable package name (letters, digits, '_' and '-'; must not start with a digit)\n", .{args.name});
        return 1;
    }
    const dir_path = args.dir orelse args.name;
    const ident = try identOf(gpa, args.name);
    defer gpa.free(ident);

    const cwd = std.Io.Dir.cwd();
    if (cwd.access(io, dir_path, .{})) |_| {
        try out.print("error: '{s}' already exists; refusing to overwrite\n", .{dir_path});
        return 1;
    } else |_| {}
    cwd.createDirPath(io, dir_path) catch |e| {
        try out.print("error: cannot create '{s}': {s}\n", .{ dir_path, @errorName(e) });
        return 1;
    };

    // Fingerprint starts at zero and is filled in below from the compiler's own
    // answer, rather than by reimplementing its hash — that would silently rot
    // the moment Zig changes it.
    // Zig requires a path dependency to be expressed relative to the build root,
    // so an absolute --zoxide-path has to be rewritten relative to the package
    // being created.
    var rel_dep: ?[]u8 = null;
    defer if (rel_dep) |r| gpa.free(r);
    const dep_block = if (args.zoxide_path) |p| blk: {
        // Both ends are resolved through the filesystem rather than combined
        // lexically. A purely textual relative path breaks across symlinks: on
        // macOS /tmp is a link to /private/tmp, so "/tmp/x/y" is three levels
        // below the root textually but four in reality, and the dependency path
        // lands somewhere that does not exist.
        //
        // The sentinel type is kept in both branches: `realPathFileAlloc`
        // allocates len+1 for the NUL, so coercing a fallback to `[]u8` makes
        // `free` pass the wrong length.
        const from: [:0]u8 = cwd.realPathFileAlloc(io, dir_path, gpa) catch
            try gpa.dupeZ(u8, dir_path);
        defer gpa.free(from);
        const to: [:0]u8 = cwd.realPathFileAlloc(io, p, gpa) catch
            try gpa.dupeZ(u8, p);
        defer gpa.free(to);
        const r = std.fs.path.relative(gpa, from, env, from, to) catch null;
        if (r) |rr| {
            rel_dep = rr;
            break :blk try std.fmt.allocPrint(gpa, "        .zoxide = .{{ .path = \"{s}\" }},\n", .{rr});
        }
        break :blk try std.fmt.allocPrint(gpa, "        .zoxide = .{{ .path = \"{s}\" }},\n", .{p});
    } else try gpa.dupe(u8, "");
    defer gpa.free(dep_block);

    const zon = try std.fmt.allocPrint(gpa,
        \\.{{
        \\    .name = .{[ident]s},
        \\    .version = "0.0.0",
        \\    .minimum_zig_version = "0.16.0",
        \\    .fingerprint = 0x0,
        \\    .dependencies = .{{
        \\{[dep]s}    }},
        \\    .paths = .{{
        \\        "build.zig",
        \\        "build.zig.zon",
        \\        "kernels_abi.zig",
        \\        "kernel.zig",
        \\        "main.zig",
        \\    }},
        \\}}
        \\
    , .{ .ident = ident, .dep = dep_block });
    defer gpa.free(zon);

    // Deliberately "kernel", matching kernel.zig: the PTX symbol prefix is the
    // root source file's stem, not the object name, so making them differ would
    // produce a package whose host side cannot find its own kernel. That is
    // exactly the bug this template had before the generated PTX was inspected.
    const kernel_obj_name = try gpa.dupe(u8, "kernel");
    defer gpa.free(kernel_obj_name);
    // Plain substitution rather than a format string: the templates contain the
    // generated program's own `{s}` / `{d}` placeholders, which a formatter would
    // try to consume.
    const build_zig = try substitute(gpa, build_zig_template, kernel_obj_name, args.name);
    defer gpa.free(build_zig);
    const main_zig = try substitute(gpa, main_zig_template, kernel_obj_name, args.name);
    defer gpa.free(main_zig);
    const readme = try readmeFor(gpa, args.name);
    defer gpa.free(readme);

    var dir = cwd.openDir(io, dir_path, .{}) catch |e| {
        try out.print("error: cannot open '{s}': {s}\n", .{ dir_path, @errorName(e) });
        return 1;
    };
    defer dir.close(io);

    const files = [_]struct { name: []const u8, data: []const u8 }{
        .{ .name = "build.zig.zon", .data = zon },
        .{ .name = "build.zig", .data = build_zig },
        .{ .name = "kernels_abi.zig", .data = abi_zig },
        .{ .name = "kernel.zig", .data = kernel_zig },
        .{ .name = "main.zig", .data = main_zig },
        .{ .name = ".gitignore", .data = gitignore },
        .{ .name = "README.md", .data = readme },
    };
    for (files) |f| {
        dir.writeFile(io, .{ .sub_path = f.name, .data = f.data }) catch |e| {
            try out.print("error: cannot write {s}/{s}: {s}\n", .{ dir_path, f.name, @errorName(e) });
            return 1;
        };
        try out.print("  {s}/{s}\n", .{ dir_path, f.name });
    }

    const fp_ok = fixFingerprint(gpa, io, env, dir_path) catch false;

    try out.print("\ncreated {s}\n", .{dir_path});
    if (!fp_ok) {
        try out.print(
            \\
            \\note: could not fill in build.zig.zon's fingerprint automatically.
            \\      Run `zig build` in {s} — the error message states the value to use.
            \\
        , .{dir_path});
    }
    if (args.zoxide_path == null) {
        try out.print(
            \\
            \\next: add the zoxide dependency, then run it
            \\  cd {s}
            \\  zig fetch --save git+https://github.com/zig-ecosystem/zoxide#{s}
            \\  zig build run
            \\
        , .{ dir_path, args.version });
    } else {
        try out.print("\nnext:\n  cd {s}\n  zig build run\n", .{dir_path});
    }
    return 0;
}

/// Ask the compiler what the fingerprint should be and write it in.
///
/// The value is a 32-bit id derived from the package name in the high half plus
/// an arbitrary low half, and `zig` reports the correct one in its error
/// message. Harvesting that is version-proof in a way that reimplementing the
/// hash would not be.
fn fixFingerprint(gpa: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map, dir_path: []const u8) !bool {
    _ = env;
    const result = std.process.run(gpa, io, .{
        .argv = &.{ "zig", "build", "--help" },
        .cwd = .{ .path = dir_path },
    }) catch return false;
    defer gpa.free(result.stdout);
    defer gpa.free(result.stderr);

    const marker = "use this value: ";
    const at = std.mem.indexOf(u8, result.stderr, marker) orelse return false;
    const rest = result.stderr[at + marker.len ..];
    var end: usize = 0;
    while (end < rest.len) : (end += 1) {
        switch (rest[end]) {
            '0'...'9', 'a'...'f', 'A'...'F', 'x', 'X' => {},
            else => break,
        }
    }
    if (end == 0) return false;
    const value = rest[0..end];

    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{}) catch return false;
    defer dir.close(io);
    const zon = dir.readFileAlloc(io, "build.zig.zon", gpa, .limited(64 * 1024)) catch return false;
    defer gpa.free(zon);
    const needle = ".fingerprint = 0x0,";
    const pos = std.mem.indexOf(u8, zon, needle) orelse return false;

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, zon[0..pos]);
    try buf.appendSlice(gpa, ".fingerprint = ");
    try buf.appendSlice(gpa, value);
    try buf.appendSlice(gpa, ",");
    try buf.appendSlice(gpa, zon[pos + needle.len ..]);
    dir.writeFile(io, .{ .sub_path = "build.zig.zon", .data = buf.items }) catch return false;
    return true;
}

test "validName rejects what the zon cannot express" {
    try std.testing.expect(validName("hello"));
    try std.testing.expect(validName("my-gpu-thing"));
    try std.testing.expect(validName("_x1"));
    try std.testing.expect(!validName(""));
    try std.testing.expect(!validName("1abc"));
    try std.testing.expect(!validName("has space"));
    try std.testing.expect(!validName("dots.bad"));
}

test "identOf maps dashes to underscores" {
    const gpa = std.testing.allocator;
    const got = try identOf(gpa, "my-gpu-thing");
    defer gpa.free(got);
    try std.testing.expectEqualStrings("my_gpu_thing", got);
}
