const std = @import("std");

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

    // Device side: compile src/kernel.zig to PTX for nvptx64.
    const kernel_step = b.step("kernel", "Compile src/kernel.zig to PTX (zig-out/kernel.ptx)");
    const kernel = b.addObject(.{
        .name = "kernel",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/kernel.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .nvptx64,
                .os_tag = .cuda,
                // Target GPU baseline is NVIDIA H20 (Hopper, CC 9.0).
                .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_90 },
            }),
            .optimize = .ReleaseFast,
            .strip = true,
        }),
    });
    // Zig's UBSan runtime hooks generate LLVM aliases, which the NVPTX
    // backend rejects when they target kernel functions.
    kernel.bundle_ubsan_rt = false;
    const asm_path = kernel.getEmittedAsm();
    const install_asm = b.addInstallFileWithDir(asm_path, .prefix, "kernel.ptx");
    kernel_step.dependOn(&install_asm.step);
}
