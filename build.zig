const std = @import("std");

// Target GPU baseline: NVIDIA H20 (Hopper, CC 9.0).
pub const default_sm_model = &std.Target.nvptx.cpu.sm_90;
// Architecture-specific Hopper target. `wgmma.*` and the TMA/`tcgen` family are
// gated on `hasSM90a` in LLVM's NVPTX backend, so kernels using them must be
// built for sm_90a rather than plain sm_90. Code compiled for sm_90a is not
// forward-compatible with later architectures (no PTX JIT to sm_100), which is
// why it is opt-in per kernel instead of the project default.
pub const sm_90a_model = &std.Target.nvptx.cpu.sm_90a;

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
/// Compile step for one kernel source, targeting nvptx64 with the `cuda` module
/// wired up. Returned rather than consumed so callers can add further imports to
/// the kernel's root module — sharing a signature declaration with the host, for
/// instance — before taking `getEmittedAsm()`.
///
/// ```zig
/// const obj = zx.addNvptxKernelObject(b, "my_kernels", b.path("kernel.zig"), dep.path("src/cuda.zig"), .{});
/// obj.root_module.addImport("kernels_abi", abi);
/// exe.root_module.addAnonymousImport("kernel_ptx", .{ .root_source_file = obj.getEmittedAsm() });
/// ```
pub fn addNvptxKernelObject(
    b: *std.Build,
    name: []const u8,
    source: std.Build.LazyPath,
    cuda_root: std.Build.LazyPath,
    opts: CudaOptions,
) *std.Build.Step.Compile {
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
    // Zig's UBSan runtime hooks generate LLVM aliases, which the NVPTX backend
    // rejects when they target kernel functions.
    kernel.bundle_ubsan_rt = false;
    return kernel;
}

/// Emitted PTX for one kernel source, for `@embedFile` into a host program.
/// Use `addNvptxKernelObject` when the kernel needs imports beyond `cuda`.
pub fn addNvptxKernelPath(
    b: *std.Build,
    name: []const u8,
    source: std.Build.LazyPath,
    cuda_root: std.Build.LazyPath,
    opts: CudaOptions,
) std.Build.LazyPath {
    return addNvptxKernelObject(b, name, source, cuda_root, opts).getEmittedAsm();
}

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

    // Host-side API for `dep.module("zoxide_host")` consumers: module loading,
    // typed device allocations and compile-time-checked kernel launches. Target
    // follows the consumer, so it is left unset here.
    _ = b.addModule("zoxide_host", .{
        .root_source_file = b.path("src/host.zig"),
    });

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

    const Example = struct { name: []const u8, sm: *const std.Target.Cpu.Model = default_sm_model };
    const examples = [_]Example{
        .{ .name = "vector_add" },
        .{ .name = "shared_reverse" },
        .{ .name = "warp_reduce" },
        .{ .name = "atomic_counter" },
        .{ .name = "intrinsics_smoke" },
        .{ .name = "sgemm_naive" },
        .{ .name = "sgemm_tiled" },
        .{ .name = "sgemm_reg" },
        .{ .name = "sgemm_opt" },
        .{ .name = "sgemm_opt2" },
        .{ .name = "sgemm_swz" },
        .{ .name = "debug_print" },
        .{ .name = "asm_smoke" },
        .{ .name = "hgemm_mma" },
        .{ .name = "hgemm_mma2" },
        .{ .name = "wgmma_smoke", .sm = sm_90a_model },
        .{ .name = "hgemm_wgmma", .sm = sm_90a_model },
        .{ .name = "hgemm_wgmma2", .sm = sm_90a_model },
        .{ .name = "hgemm_wgmma3", .sm = sm_90a_model },
        .{ .name = "f16_native" },
    };

    // `zig build kernels`: compile every kernel in src/examples/ to
    // zig-out/kernels/<name>.ptx (sm_90 unless the entry overrides it).
    const kernels_step = b.step("kernels", "Compile all kernels in src/examples/ to PTX (zig-out/kernels/)");
    for (examples) |ex| {
        const source = b.path(b.fmt("src/examples/{s}.zig", .{ex.name}));
        kernels_step.dependOn(&addNvptxKernel(b, ex.name, source, b.path("src/cuda.zig"), .{ .sm = ex.sm }).step);
    }

    // `zig build kernel`: single default kernel (kept for compatibility).
    // `zig build test`: host-side unit tests (symbol mangling, Slice, Kernel
    // signature reflection). Device code cannot be unit-tested here — it needs a
    // GPU, which is what scripts/pod-verify.sh is for.
    const host_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/host.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const test_step = b.step("test", "Run host-side unit tests");
    test_step.dependOn(&b.addRunArtifact(host_tests).step);

    const kernel_step = b.step("kernel", "Compile src/kernel.zig to PTX (zig-out/kernels/kernel.ptx)");
    kernel_step.dependOn(&addNvptxKernel(b, "kernel", b.path("src/kernel.zig"), b.path("src/cuda.zig"), .{}).step);
}
