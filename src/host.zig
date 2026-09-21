//! Host-side public API: load a cubin or PTX-derived module, allocate device
//! memory, and launch kernels with the argument list checked at compile time.
//!
//! Until now this lived inside the `zoxide` CLI and downstream users got only a
//! `.ptx` file, leaving them to write their own libcuda FFI. This is that layer,
//! exposed as a module.
//!
//! The reason it is worth more than a convenience wrapper: `cuLaunchKernel`
//! takes `void**`, one untyped pointer per argument. Passing the wrong number of
//! arguments, the wrong order, or a 32-bit value where the kernel expects 64
//! produces no diagnostic at all — just a kernel reading garbage, usually
//! showing up as wrong results rather than a crash. `Kernel(Signature).launch`
//! checks the argument tuple against the kernel's actual Zig signature, so those
//! three mistakes become compile errors.
//!
//! ```zig
//! const gpu = @import("zoxide_host");
//!
//! var drv = try gpu.Driver.load();
//! defer drv.unload();
//! var ctx = try gpu.Context.init(&drv);
//! const mod = try ctx.module(cubin_bytes);
//!
//! // The signature is the kernel's own, spelled once on the host side.
//! // Note: no `callconv(.kernel)` — that convention cannot be named on a host
//! // target. Better still, declare it in a file the device imports too and let
//! // `abi.assertMatches` keep the two in step. See src/kernel_abi.zig.
//! const Scale = fn ([*]const f32, [*]f32, f32, u32) void;
//! const scale = try mod.kernel(Scale, gpu.symbol("my_kernels", "scale"));
//!
//! const dx = try ctx.allocSlice(f32, n);
//! defer ctx.freeSlice(dx);
//! try ctx.upload(dx, host_x);
//!
//! try scale.launch(.{ .x = gpu.gridFor(n, 256) }, .{ .x = 256 }, .{ dx, dy, 2.0, @as(u32, n) });
//! ```

const std = @import("std");
const cu = @import("cuda_driver.zig");
pub const abi = @import("kernel_abi.zig");

/// Driver errors, plus the checks this layer adds on top.
pub const Error = cu.Error || error{
    /// A host slice's length did not match the device buffer's.
    LengthMismatch,
    OutOfMemory,
};
pub const Driver = cu.Driver;
pub const Event = cu.Event;
pub const DevicePtr = cu.CUdeviceptr;

/// Launch geometry. `y`/`z` default to 1, which is what almost every launch
/// wants and what is easy to get wrong when passing six bare integers.
pub const Dims = struct {
    x: u32,
    y: u32 = 1,
    z: u32 = 1,
};

/// Blocks needed to cover `n` items with `block` threads each.
pub fn gridFor(n: usize, block: u32) u32 {
    return @intCast((n + block - 1) / block);
}

/// Zig's NVPTX backend names an exported kernel `<object>_$_<decl path>`, with
/// `$_` joining namespace components. Spelling that by hand is easy to get
/// subtly wrong, and the failure mode is a runtime "kernel not found".
pub fn symbol(comptime object: []const u8, comptime decl_path: []const u8) [:0]const u8 {
    // Materialised as a static constant rather than returned straight out of a
    // comptime block, so the function can be called from runtime code.
    const S = struct {
        const joined = blk: {
            var out: []const u8 = object;
            var it = std.mem.splitScalar(u8, decl_path, '.');
            while (it.next()) |part| out = out ++ "_$_" ++ part;
            break :blk out;
        };
        const z: [joined.len:0]u8 = (joined ++ "\x00")[0..joined.len :0].*;
    };
    return &S.z;
}

/// A typed device allocation. Carrying the element type is what lets `launch`
/// reject a buffer of the wrong type, and carrying the length is what lets
/// `upload`/`download` check sizes instead of trusting a byte count.
pub fn Slice(comptime T: type) type {
    return struct {
        ptr: DevicePtr,
        len: usize,

        pub const Elem = T;
        const Self = @This();

        pub fn bytes(self: Self) usize {
            return self.len * @sizeOf(T);
        }

        /// Sub-range, for feeding part of a buffer to a kernel.
        pub fn slice(self: Self, start: usize, end: usize) Self {
            std.debug.assert(start <= end and end <= self.len);
            return .{ .ptr = self.ptr + start * @sizeOf(T), .len = end - start };
        }
    };
}

fn isSlice(comptime S: type) bool {
    return @typeInfo(S) == .@"struct" and @hasDecl(S, "Elem") and
        S == Slice(S.Elem);
}

pub const Context = struct {
    inner: cu.Context,

    pub fn init(drv: *Driver) Error!Context {
        return .{ .inner = try cu.Context.init(drv) };
    }

    pub fn name(self: *Context, buf: []u8) []const u8 {
        return self.inner.name(buf);
    }

    pub fn info(self: *Context) Error!cu.Context.Info {
        return self.inner.info();
    }

    pub fn synchronize(self: *Context) Error!void {
        return self.inner.synchronize();
    }

    pub fn eventCreate(self: *Context) Error!Event {
        return self.inner.eventCreate();
    }

    /// Load a module from cubin bytes.
    pub fn module(self: *Context, image: []const u8) Error!Module {
        return .{ .inner = try self.inner.module(image) };
    }

    /// Load a module from PTX text, letting the driver JIT it.
    ///
    /// This is the low-friction path: PTX is what `zig build` produces directly,
    /// so a downstream package needs no ptxas at build time and can just
    /// `@embedFile` the result. The tradeoff is that a bad kernel surfaces here
    /// as a load failure instead of at build time, and the first load pays JIT
    /// cost. Pre-assembling with ptxas and using `module` avoids both, at the
    /// price of pinning the build to one architecture.
    ///
    /// Must be NUL-terminated, which is what `@embedFile` yields.
    pub fn moduleFromPtx(self: *Context, ptx: [:0]const u8) Error!Module {
        return .{ .inner = try self.inner.module(ptx.ptr[0 .. ptx.len + 1]) };
    }

    pub fn allocSlice(self: *Context, comptime T: type, len: usize) Error!Slice(T) {
        return .{ .ptr = try self.inner.alloc(len * @sizeOf(T)), .len = len };
    }

    pub fn freeSlice(self: *Context, s: anytype) void {
        self.inner.free(s.ptr);
    }

    /// Host to device. Lengths must match; a short copy into a longer device
    /// buffer is almost always a bug, and silently allowing it hides it.
    pub fn upload(self: *Context, dst: anytype, src: anytype) Error!void {
        const Elem = @TypeOf(dst).Elem;
        const host: []const Elem = src;
        if (host.len != dst.len) return error.LengthMismatch;
        return self.inner.copyHtoD(dst.ptr, std.mem.sliceAsBytes(host));
    }

    /// Device to host. `dst` is a host slice of the device buffer's element type.
    pub fn download(self: *Context, dst: anytype, src: anytype) Error!void {
        const Elem = @TypeOf(src).Elem;
        const host: []Elem = dst;
        if (host.len != src.len) return error.LengthMismatch;
        return self.inner.copyDtoH(std.mem.sliceAsBytes(host), src.ptr);
    }

    /// Zero a device buffer by uploading zeros — the driver's async memset is not
    /// bound yet, so this costs a host allocation and a transfer.
    pub fn memsetZero(self: *Context, s: anytype, gpa: std.mem.Allocator) Error!void {
        const T = @TypeOf(s).Elem;
        const zeros = gpa.alloc(T, s.len) catch return error.OutOfMemory;
        defer gpa.free(zeros);
        @memset(zeros, std.mem.zeroes(T));
        return self.inner.copyHtoD(s.ptr, std.mem.sliceAsBytes(zeros));
    }
};

pub const Module = struct {
    inner: cu.Module,

    /// Look up a kernel and bind it to its signature. `Signature` is a function
    /// type matching the kernel's Zig declaration, e.g.
    /// `fn ([*]const f32, [*]f32, u32) callconv(.kernel) void`.
    pub fn kernel(self: Module, comptime Signature: type, name: [:0]const u8) Error!Kernel(Signature) {
        return .{ .inner = try self.inner.function(name) };
    }

    /// Escape hatch for callers that cannot name the signature at comptime.
    pub fn rawFunction(self: Module, name: [:0]const u8) Error!cu.Function {
        return self.inner.function(name);
    }
};

/// A kernel bound to its signature, so `launch` can check arguments.
pub fn Kernel(comptime Signature: type) type {
    // Spelled without `callconv(.kernel)`: that convention resolves per target
    // and is `unreachable` on host architectures, so the type is unspellable
    // here. Only the parameter list matters for launching.
    const param_types = abi.paramTypes(Signature);

    // The value actually handed to cuLaunchKernel for each parameter: a device
    // address for pointer parameters, the value itself otherwise.
    const AbiTuple = comptime blk: {
        var types: [param_types.len]type = undefined;
        for (param_types, 0..) |P, i| {
            types[i] = switch (@typeInfo(P)) {
                .pointer => DevicePtr,
                else => P,
            };
        }
        break :blk std.meta.Tuple(&types);
    };

    return struct {
        inner: cu.Function,
        const Self = @This();

        pub const params = param_types;

        /// Launch with the argument tuple checked against the kernel signature.
        ///
        /// Pointer parameters take a `Slice(T)` whose element type must match
        /// the pointee; scalars must match exactly, with no implicit widening,
        /// because a u32 passed where the kernel reads u64 reads adjacent
        /// garbage and produces wrong answers rather than an error.
        pub fn launch(self: Self, grid: Dims, block: Dims, args: anytype) Error!void {
            const Args = @TypeOf(args);
            const args_info = switch (@typeInfo(Args)) {
                .@"struct" => |s| s,
                else => @compileError("launch expects an argument tuple, got " ++ @typeName(Args)),
            };
            if (args_info.fields.len != param_types.len) {
                @compileError(std.fmt.comptimePrint(
                    "kernel takes {d} argument(s), got {d}",
                    .{ param_types.len, args_info.fields.len },
                ));
            }

            var packed_args: AbiTuple = undefined;
            var ptrs: [param_types.len]?*anyopaque = undefined;

            inline for (param_types, 0..) |P, i| {
                const arg = args[i];
                const A = @TypeOf(arg);
                switch (@typeInfo(P)) {
                    .pointer => |ptr| {
                        if (!comptime isSlice(A)) {
                            @compileError(std.fmt.comptimePrint(
                                "argument {d}: kernel wants {s}, so pass a Slice({s}), got {s}",
                                .{ i, @typeName(P), @typeName(ptr.child), @typeName(A) },
                            ));
                        }
                        if (A.Elem != ptr.child) {
                            @compileError(std.fmt.comptimePrint(
                                "argument {d}: kernel wants {s} but got Slice({s})",
                                .{ i, @typeName(P), @typeName(A.Elem) },
                            ));
                        }
                        packed_args[i] = arg.ptr;
                    },
                    else => {
                        if (A != P) {
                            @compileError(std.fmt.comptimePrint(
                                "argument {d}: kernel wants {s}, got {s} (no implicit conversion —" ++
                                    " a mismatched width reads adjacent memory on the device)",
                                .{ i, @typeName(P), @typeName(A) },
                            ));
                        }
                        packed_args[i] = arg;
                    },
                }
                ptrs[i] = @ptrCast(&packed_args[i]);
            }

            return self.inner.launch(grid.x, grid.y, grid.z, block.x, block.y, block.z, &ptrs);
        }
    };
}

test "symbol mangling matches the NVPTX backend's naming" {
    try std.testing.expectEqualStrings("k_$_scale", symbol("k", "scale"));
    try std.testing.expectEqualStrings("m_$_Ns_$_inner", symbol("m", "Ns.inner"));
}

test "gridFor rounds up" {
    try std.testing.expectEqual(@as(u32, 4), gridFor(1000, 256));
    try std.testing.expectEqual(@as(u32, 4), gridFor(1024, 256));
    try std.testing.expectEqual(@as(u32, 5), gridFor(1025, 256));
    try std.testing.expectEqual(@as(u32, 0), gridFor(0, 256));
}

test "Slice tracks element type and length" {
    const S = Slice(f32);
    const s: S = .{ .ptr = 0x1000, .len = 100 };
    try std.testing.expectEqual(@as(usize, 400), s.bytes());
    const sub = s.slice(10, 20);
    try std.testing.expectEqual(@as(DevicePtr, 0x1028), sub.ptr);
    try std.testing.expectEqual(@as(usize, 10), sub.len);
    try std.testing.expect(isSlice(S));
    try std.testing.expect(!isSlice(f32));
}

test "Kernel derives its parameter list from the signature" {
    const K = Kernel(fn ([*]const f32, [*]f32, f32, u32) void);
    try std.testing.expectEqual(@as(usize, 4), K.params.len);
    // Type values are comptime-only, so compare inside a comptime block.
    comptime {
        std.debug.assert(K.params[0] == [*]const f32);
        std.debug.assert(K.params[3] == u32);
    }
}

test "kernel_abi is re-exported for the shared-declaration pattern" {
    const Declared = fn ([*]const f32, u32) void;
    abi.assertMatches(Declared, Declared);
}
