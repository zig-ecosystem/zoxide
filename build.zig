const std = @import("std");

// Target GPU baseline: NVIDIA H20 (Hopper, CC 9.0).
pub const default_sm_model = &std.Target.nvptx.cpu.sm_90;

pub const CudaOptions = struct {
    sm: *const std.Target.Cpu.Model = default_sm_model,
    optimize: std.builtin.OptimizeMode = .ReleaseFast,
};

/// Create the device-side `cuda` module (src/cuda.zig) targeting nvptx64.
///
/// `root` is a LazyPath to this package's `src/cuda.zig`: `b.path(...)` when
/// used inside zoxide itself, or `zoxide_dep.path("src/cuda.zig")` from a
/// downstream package. Downstream usage:
///
/// ```zig
/// const zoxide = b.dependency("zoxide", .{});
/// const cuda = @import("zoxide").addCudaModule(b, zoxide.path("src/cuda.zig"), .{});
/// kernel.root_module.addImport("cuda", cuda);
/// ```
pub fn addCudaModule(b: *std.Build, root: std.Build.LazyPath, opts: CudaOptions) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = root,
        .target = b.resolveTargetQuery(.{
            .cpu_arch = .nvptx64,
            .os_tag = .cuda,
            .cpu_model = .{ .explicit = opts.sm },
        }),
        .optimize = opts.optimize,
        .strip = true,
    });
}

/// Compile one kernel source file to PTX and install it as
/// <prefix>/kernels/<name>.ptx. The `cuda` import is wired automatically.
/// Works for downstream packages too: pass `dep.path("src/cuda.zig")` as
/// cuda_root.
pub fn addNvptxKernel(
    b: *std.Build,
    name: []const u8,
    source: std.Build.LazyPath,
    cuda_root: std.Build.LazyPath,
    opts: CudaOptions,
) *std.Build.Step.InstallFile {
    const kernel = b.addObject(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = source,
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .nvptx64,
                .os_tag = .cuda,
                .cpu_model = .{ .explicit = opts.sm },
            }),
            .optimize = opts.optimize,
            .strip = true,
        }),
    });
    kernel.root_module.addImport("cuda", addCudaModule(b, cuda_root, opts));
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
            // libc everywhere: the host runner dlopens libcuda. For gnu
            // targets this yields a dynamic binary (DlDynLib); musl static
            // still links statically but falls back to ElfDynLib.
            .link_libc = true,
        }),
    });
    b.installArtifact(exe);

    // Expose the device library for `dep.module("cuda")` consumers.
    _ = b.addModule("cuda", .{
        .root_source_file = b.path("src/cuda.zig"),
        .target = b.resolveTargetQuery(.{
            .cpu_arch = .nvptx64,
            .os_tag = .cuda,
            .cpu_model = .{ .explicit = default_sm_model },
        }),
        .optimize = .ReleaseFast,
        .strip = true,
    });

    const example_names = [_][]const u8{
        "vector_add",
        "shared_reverse",
        "warp_reduce",
        "atomic_counter",
        "intrinsics_smoke",
        "sgemm_naive",
        "sgemm_tiled",
        "sgemm_reg",
        "sgemm_opt",
        "sgemm_opt2",
        "sgemm_swz",
        "debug_print",
        "asm_smoke",
        "hgemm_mma",
    };

    // `zig build kernels`: compile every kernel in src/examples/ to
    // zig-out/kernels/<name>.ptx (sm_90).
    const kernels_step = b.step("kernels", "Compile all kernels in src/examples/ to PTX (zig-out/kernels/)");
    for (example_names) |name| {
        const source = b.path(b.fmt("src/examples/{s}.zig", .{name}));
        kernels_step.dependOn(&addNvptxKernel(b, name, source, b.path("src/cuda.zig"), .{}).step);
    }

    // `zig build kernel`: single default kernel (kept for compatibility).
    const kernel_step = b.step("kernel", "Compile src/kernel.zig to PTX (zig-out/kernels/kernel.ptx)");
    kernel_step.dependOn(&addNvptxKernel(b, "kernel", b.path("src/kernel.zig"), b.path("src/cuda.zig"), .{}).step);
}
