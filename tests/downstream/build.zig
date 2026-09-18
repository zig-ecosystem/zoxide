const std = @import("std");

pub fn build(b: *std.Build) void {
    const zoxide = b.dependency("zoxide", .{});
    const cuda = @import("zoxide").addCudaModule(b, zoxide.path("src/cuda.zig"), .{});

    const kernel = b.addObject(.{
        .name = "downstream_kernel",
        .root_module = b.createModule(.{
            .root_source_file = b.path("kernel.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .nvptx64,
                .os_tag = .cuda,
                .cpu_model = .{ .explicit = @import("zoxide").default_sm_model },
            }),
            .optimize = .ReleaseFast,
            .strip = true,
        }),
    });
    kernel.root_module.addImport("cuda", cuda);
    kernel.bundle_ubsan_rt = false;

    const install = b.addInstallFileWithDir(kernel.getEmittedAsm(), .{ .custom = "kernels" }, "downstream_kernel.ptx");
    b.getInstallStep().dependOn(&install.step);
}
