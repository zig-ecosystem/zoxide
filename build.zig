const std = @import("std");

// Target GPU baseline: NVIDIA H20 (Hopper, CC 9.0).
const sm_model = &std.Target.nvptx.cpu.sm_90;

fn addKernel(b: *std.Build, name: []const u8, source: std.Build.LazyPath) *std.Build.Step.InstallFile {
    const kernel = b.addObject(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = source,
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .nvptx64,
                .os_tag = .cuda,
                .cpu_model = .{ .explicit = sm_model },
            }),
            .optimize = .ReleaseFast,
            .strip = true,
        }),
    });
    kernel.root_module.addImport("cuda", b.createModule(.{
        .root_source_file = b.path("src/cuda.zig"),
        .target = kernel.root_module.resolved_target.?,
        .optimize = .ReleaseFast,
        .strip = true,
    }));
    // Zig's UBSan runtime hooks generate LLVM aliases, which the NVPTX
    // backend rejects when they target kernel functions.
    kernel.bundle_ubsan_rt = false;
    const out_name = b.fmt("{s}.ptx", .{name});
    return b.addInstallFileWithDir(kernel.getEmittedAsm(), .{ .custom = "kernels" }, out_name);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zoxide",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const example_names = [_][]const u8{
        "vector_add",
        "shared_reverse",
        "warp_reduce",
        "atomic_counter",
    };

    // `zig build kernels`: compile every kernel in src/examples/ to
    // zig-out/kernels/<name>.ptx (sm_90).
    const kernels_step = b.step("kernels", "Compile all kernels in src/examples/ to PTX (zig-out/kernels/)");
    for (example_names) |name| {
        const source = b.path(b.fmt("src/examples/{s}.zig", .{name}));
        kernels_step.dependOn(&addKernel(b, name, source).step);
    }

    // `zig build kernel`: single default kernel (kept for compatibility).
    const kernel_step = b.step("kernel", "Compile src/kernel.zig to PTX (zig-out/kernels/kernel.ptx)");
    kernel_step.dependOn(&addKernel(b, "kernel", b.path("src/kernel.zig")).step);
}
