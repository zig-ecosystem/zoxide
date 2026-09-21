//! Downstream integration test: one package containing both the device kernel
//! and the host program that launches it, wired only through zoxide's public
//! build helpers and modules. If this builds, a new user can follow the same
//! shape.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zoxide = b.dependency("zoxide", .{});
    const zx = @import("zoxide");

    // Kernel signatures, shared by both sides so they cannot drift.
    const abi = b.createModule(.{ .root_source_file = b.path("kernels_abi.zig") });

    // Device side: kernel.zig -> PTX. The object is returned rather than just
    // its output path so the shared ABI module can be added to it.
    const kernel_obj = zx.addNvptxKernelObject(b, "downstream_kernel", b.path("kernel.zig"), zoxide.path("src/cuda.zig"), .{});
    kernel_obj.root_module.addImport("kernels_abi", abi);
    const ptx = kernel_obj.getEmittedAsm();

    // Host side: embeds the PTX and launches it through the typed API.
    const exe = b.addExecutable(.{
        .name = "downstream",
        .root_module = b.createModule(.{
            .root_source_file = b.path("main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true, // the host runner dlopens libcuda
        }),
    });
    exe.root_module.addImport("zoxide_host", zoxide.module("zoxide_host"));
    exe.root_module.addImport("kernels_abi", abi);
    exe.root_module.addAnonymousImport("kernel_ptx", .{ .root_source_file = ptx });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    const run_step = b.step("run", "Launch the downstream kernel (needs a GPU)");
    run_step.dependOn(&run.step);
}
