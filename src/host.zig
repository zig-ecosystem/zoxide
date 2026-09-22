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
pub const Stream = cu.Stream;
pub const DevicePtr = cu.CUdeviceptr;

/// Page-locked host memory, typed.
///
/// This exists because the async copy API would otherwise be misleading:
/// `cuMemcpy*Async` issued from ordinary pageable memory is asynchronous in name
/// only. The driver has to stage such memory through an internal pinned buffer
/// and blocks while it does, so the transfer does not overlap with compute and
/// the stream API silently buys nothing. Transfers that need to overlap must
/// come from here.
pub fn Pinned(comptime T: type) type {
    return struct {
        inner: cu.PinnedHost,
        items: []T,

        const Self = @This();
        pub const Elem = T;

        pub fn free(self: Self) void {
            self.inner.free();
        }
    };
}

/// Kernel resource use, from the loaded module rather than guessed from source.
pub const Resources = struct {
    regs_per_thread: u32,
    shared_bytes: u64,
    /// Local (spill) memory per thread. Non-zero means the kernel spilled.
    local_bytes: u64,
    max_threads_per_block: u32,
};

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

/// Zig's NVPTX backend names an exported kernel `<root source file stem>_$_<decl
/// path>`, with `$_` joining namespace components. For `kernel.zig` containing
/// `pub fn scale`, that is `kernel_$_scale`.
///
/// Note it is the **root source file's** name, not the object or artifact name.
/// Those often coincide — zoxide's own examples are `src/examples/<name>.zig`
/// built as object `<name>` — which makes the distinction easy to miss, and the
/// failure mode is a runtime "kernel not found" rather than a build error. An
/// earlier version of this comment said `<object>`, and the scaffold generated
/// from it produced packages that could not find their own kernels.
pub fn symbol(comptime root_source_stem: []const u8, comptime decl_path: []const u8) [:0]const u8 {
    // Materialised as a static constant rather than returned straight out of a
    // comptime block, so the function can be called from runtime code.
    const S = struct {
        const joined = blk: {
            var out: []const u8 = root_source_stem;
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

/// Device launch limits, read once at context creation.
///
/// Cached rather than queried per launch: validating geometry is then pure
/// arithmetic, so a benchmark loop issuing thousands of launches pays nothing.
pub const Limits = struct {
    max_threads_per_block: u32,
    max_block: [3]u32,
    max_grid: [3]u32,
    max_shared_per_block: u64,

    /// Permissive fallback when the attributes cannot be read, so a query
    /// failure degrades to the previous behaviour (the driver rejects it) rather
    /// than to spurious rejections here.
    pub const unknown: Limits = .{
        .max_threads_per_block = std.math.maxInt(u32),
        .max_block = @splat(std.math.maxInt(u32)),
        .max_grid = @splat(std.math.maxInt(u32)),
        .max_shared_per_block = std.math.maxInt(u64),
    };
};

pub const Context = struct {
    inner: cu.Context,
    limits: Limits,

    pub fn init(drv: *Driver) Error!Context {
        var inner = try cu.Context.init(drv);
        return .{ .inner = inner, .limits = readLimits(&inner) };
    }

    fn readLimits(inner: *cu.Context) Limits {
        const q = struct {
            fn get(c: *cu.Context, a: cu.Context.Attr, fallback: u64) u64 {
                return c.attr(a) catch fallback;
            }
        }.get;
        const max = std.math.maxInt(u32);
        return .{
            .max_threads_per_block = @intCast(q(inner, .max_threads_per_block, max)),
            .max_block = .{
                @intCast(q(inner, .max_block_dim_x, max)),
                @intCast(q(inner, .max_block_dim_y, max)),
                @intCast(q(inner, .max_block_dim_z, max)),
            },
            .max_grid = .{
                @intCast(q(inner, .max_grid_dim_x, max)),
                @intCast(q(inner, .max_grid_dim_y, max)),
                @intCast(q(inner, .max_grid_dim_z, max)),
            },
            .max_shared_per_block = q(inner, .max_shared_memory_per_block, std.math.maxInt(u64)),
        };
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

    /// `non_blocking` streams do not serialise against the legacy default
    /// stream, which is what you want when streams should genuinely overlap.
    pub fn createStream(self: *Context, non_blocking: bool) Error!Stream {
        return self.inner.streamCreate(non_blocking);
    }

    /// Page-locked host buffer of `len` items. See `Pinned` for why ordinary
    /// memory will not do for overlapping transfers.
    pub fn allocPinned(self: *Context, comptime T: type, len: usize) Error!Pinned(T) {
        const raw = try self.inner.allocPinned(len * @sizeOf(T));
        return .{ .inner = raw, .items = @as([*]T, @ptrCast(@alignCast(raw.bytes.ptr)))[0..len] };
    }

    /// Host to device on `stream`. `src` should be a `Pinned(T).items` slice for
    /// the copy to actually overlap.
    pub fn uploadAsync(self: *Context, dst: anytype, src: anytype, stream: Stream) Error!void {
        const Elem = @TypeOf(dst).Elem;
        const host: []const Elem = src;
        if (host.len != dst.len) return error.LengthMismatch;
        return self.inner.copyHtoDAsync(dst.ptr, std.mem.sliceAsBytes(host), stream);
    }

    /// Device to host on `stream`.
    pub fn downloadAsync(self: *Context, dst: anytype, src: anytype, stream: Stream) Error!void {
        const Elem = @TypeOf(src).Elem;
        const host: []Elem = dst;
        if (host.len != src.len) return error.LengthMismatch;
        return self.inner.copyDtoHAsync(std.mem.sliceAsBytes(host), src.ptr, stream);
    }

    /// Device to device on `stream`. Element types and lengths must match.
    pub fn copyAsync(self: *Context, dst: anytype, src: anytype, stream: Stream) Error!void {
        if (@TypeOf(dst).Elem != @TypeOf(src).Elem) {
            @compileError("copyAsync between Slice(" ++ @typeName(@TypeOf(src).Elem) ++
                ") and Slice(" ++ @typeName(@TypeOf(dst).Elem) ++ ")");
        }
        if (dst.len != src.len) return error.LengthMismatch;
        return self.inner.copyDtoDAsync(dst.ptr, src.ptr, dst.bytes(), stream);
    }

    /// Fill every byte of a device buffer, on the device.
    pub fn fillBytes(self: *Context, s: anytype, value: u8) Error!void {
        return self.inner.memsetD8(s.ptr, value, s.bytes());
    }

    pub fn fillBytesAsync(self: *Context, s: anytype, value: u8, stream: Stream) Error!void {
        return self.inner.memsetD8Async(s.ptr, value, s.bytes(), stream);
    }

    /// Zero a device buffer. Now a device-side memset; it used to allocate a
    /// host buffer of zeros and transfer it.
    pub fn zero(self: *Context, s: anytype) Error!void {
        return self.inner.memsetD8(s.ptr, 0, s.bytes());
    }

    pub fn zeroAsync(self: *Context, s: anytype, stream: Stream) Error!void {
        return self.inner.memsetD8Async(s.ptr, 0, s.bytes(), stream);
    }

    /// Load a module from cubin bytes.
    pub fn module(self: *Context, image: []const u8) Error!Module {
        return .{ .inner = try self.inner.module(image), .limits = self.limits };
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
        return .{ .inner = try self.inner.module(ptx.ptr[0 .. ptx.len + 1]), .limits = self.limits };
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

};

pub const Module = struct {
    inner: cu.Module,
    limits: Limits,

    /// Look up a kernel and bind it to its signature. `Signature` is a function
    /// type matching the kernel's Zig declaration, e.g.
    /// `fn ([*]const f32, [*]f32, u32) callconv(.kernel) void`.
    pub fn kernel(self: Module, comptime Signature: type, name: [:0]const u8) Error!Kernel(Signature) {
        const f = try self.inner.function(name);
        // A kernel's own thread ceiling is derived from its register use and can
        // be well below the device's, so it has to come from the function rather
        // than the device.
        const own_max = f.attr(.max_threads_per_block) catch @as(u64, std.math.maxInt(u32));
        return .{ .inner = f, .limits = self.limits, .max_threads = @intCast(own_max) };
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
        limits: Limits,
        /// This kernel's own threads-per-block ceiling, which its register use can
        /// push below the device's.
        max_threads: u32,
        const Self = @This();

        pub const params = param_types;

        /// Resident blocks per SM at this block size, as the driver computes it
        /// from the kernel's actual register and shared-memory use.
        ///
        /// Worth having rather than deriving: working it out by hand means
        /// reading shared-memory totals out of the PTX and dividing by the SM
        /// budget, which ignores the register limit entirely and is exactly the
        /// sort of arithmetic that silently goes stale when a kernel changes.
        pub fn occupancy(self: Self, block_size: u32, dynamic_shared: usize) Error!u32 {
            return self.inner.occupancy(block_size, dynamic_shared);
        }

        /// Registers per thread, static shared bytes, spill bytes. A non-zero
        /// `local_bytes` means the kernel spilled, which is usually a
        /// performance bug worth failing a build over.
        pub fn resources(self: Self) Error!Resources {
            return .{
                .regs_per_thread = @intCast(try self.inner.attr(.num_regs)),
                .shared_bytes = try self.inner.attr(.shared_size_bytes),
                .local_bytes = try self.inner.attr(.local_size_bytes),
                .max_threads_per_block = @intCast(try self.inner.attr(.max_threads_per_block)),
            };
        }

        /// Launch with the argument tuple checked against the kernel signature.
        ///
        /// Pointer parameters take a `Slice(T)` whose element type must match
        /// the pointee; scalars must match exactly, with no implicit widening,
        /// because a u32 passed where the kernel reads u64 reads adjacent
        /// garbage and produces wrong answers rather than an error.
        /// Launch on the default stream.
        pub fn launch(self: Self, grid: Dims, block: Dims, args: anytype) Error!void {
            return self.launchOn(null, grid, block, 0, args);
        }

        /// Check launch geometry against the device's limits and this kernel's own.
        ///
        /// Without this the driver returns `CUDA_ERROR_INVALID_VALUE` or
        /// `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, neither of which says which
        /// dimension was wrong or what the limit is. The most confusing case is a
        /// block size the device allows but this kernel does not, because its
        /// register use lowers the ceiling — nothing in the source hints at it.
        ///
        /// `msg` receives a description when the geometry is rejected.
        pub fn checkGeometry(self: Self, grid: Dims, block: Dims, dynamic_shared: u32, msg: *[]const u8, buf: []u8) bool {
            const threads = @as(u64, block.x) * block.y * block.z;
            if (threads == 0 or grid.x == 0 or grid.y == 0 or grid.z == 0) {
                msg.* = std.fmt.bufPrint(buf, "empty launch: grid {d}x{d}x{d}, block {d}x{d}x{d}", .{
                    grid.x, grid.y, grid.z, block.x, block.y, block.z,
                }) catch "empty launch";
                return false;
            }
            if (threads > self.max_threads) {
                msg.* = std.fmt.bufPrint(buf, "block of {d} threads ({d}x{d}x{d}) exceeds this kernel's limit of {d}" ++
                    "; a kernel's register use can cap it below the device's {d}", .{
                    threads, block.x,      block.y,                  block.z,
                    self.max_threads,      self.limits.max_threads_per_block,
                }) catch "block too large for this kernel";
                return false;
            }
            const bd = [3]u32{ block.x, block.y, block.z };
            const gd = [3]u32{ grid.x, grid.y, grid.z };
            const axis = [3]u8{ 'x', 'y', 'z' };
            for (bd, self.limits.max_block, axis) |v, lim, ax| {
                if (v > lim) {
                    msg.* = std.fmt.bufPrint(buf, "block dim {c} is {d}, device maximum is {d}", .{ ax, v, lim }) catch "block dim too large";
                    return false;
                }
            }
            for (gd, self.limits.max_grid, axis) |v, lim, ax| {
                if (v > lim) {
                    msg.* = std.fmt.bufPrint(buf, "grid dim {c} is {d}, device maximum is {d}", .{ ax, v, lim }) catch "grid dim too large";
                    return false;
                }
            }
            if (dynamic_shared > self.limits.max_shared_per_block) {
                msg.* = std.fmt.bufPrint(buf, "dynamic shared memory {d} B exceeds the device's {d} B per block", .{
                    dynamic_shared, self.limits.max_shared_per_block,
                }) catch "dynamic shared too large";
                return false;
            }
            return true;
        }

        /// Launch on `stream`, optionally with dynamic shared memory.
        /// Pass a null stream for the legacy default stream.
        pub fn launchOn(self: Self, stream: ?Stream, grid: Dims, block: Dims, dynamic_shared: u32, args: anytype) Error!void {
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

            // Arithmetic against cached limits, so this costs nothing per launch.
            var why: []const u8 = "";
            var why_buf: [256]u8 = undefined;
            if (!self.checkGeometry(grid, block, dynamic_shared, &why, &why_buf)) {
                self.inner.drv.setError(why);
                return error.InvalidLaunchGeometry;
            }
            return self.inner.launchOn(
                if (stream) |st| st.s else null,
                grid.x,
                grid.y,
                grid.z,
                block.x,
                block.y,
                block.z,
                dynamic_shared,
                &ptrs,
            );
        }
    };
}

test "checkGeometry names the dimension and the limit" {
    // Geometry validation is pure arithmetic over cached limits, so it is
    // testable without a GPU — which is the point: these are launch failures that
    // otherwise only appear at runtime as an opaque driver code.
    const K = Kernel(fn ([*]f32, u32) void);
    const k: K = .{
        .inner = undefined,
        .limits = .{
            .max_threads_per_block = 1024,
            .max_block = .{ 1024, 1024, 64 },
            .max_grid = .{ 2147483647, 65535, 65535 },
            .max_shared_per_block = 49152,
        },
        // Below the device's 1024: this kernel's registers cap it at 512.
        .max_threads = 512,
    };
    var buf: [256]u8 = undefined;
    var msg: []const u8 = "";

    try std.testing.expect(k.checkGeometry(.{ .x = 100 }, .{ .x = 512 }, 0, &msg, &buf));

    // The confusing case: allowed by the device, not by this kernel.
    try std.testing.expect(!k.checkGeometry(.{ .x = 1 }, .{ .x = 1024 }, 0, &msg, &buf));
    try std.testing.expect(std.mem.indexOf(u8, msg, "exceeds this kernel's limit of 512") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg, "device's 1024") != null);

    // Per-axis block limit, named.
    try std.testing.expect(!k.checkGeometry(.{ .x = 1 }, .{ .x = 2, .y = 2, .z = 100 }, 0, &msg, &buf));
    try std.testing.expect(std.mem.indexOf(u8, msg, "block dim z is 100") != null);

    // Per-axis grid limit, named.
    try std.testing.expect(!k.checkGeometry(.{ .x = 1, .y = 70000 }, .{ .x = 64 }, 0, &msg, &buf));
    try std.testing.expect(std.mem.indexOf(u8, msg, "grid dim y is 70000") != null);

    // Dynamic shared memory.
    try std.testing.expect(!k.checkGeometry(.{ .x = 1 }, .{ .x = 64 }, 65536, &msg, &buf));
    try std.testing.expect(std.mem.indexOf(u8, msg, "dynamic shared memory 65536 B") != null);

    // A zero in any dimension launches nothing; the driver accepts it silently.
    try std.testing.expect(!k.checkGeometry(.{ .x = 0 }, .{ .x = 64 }, 0, &msg, &buf));
    try std.testing.expect(std.mem.indexOf(u8, msg, "empty launch") != null);
    try std.testing.expect(!k.checkGeometry(.{ .x = 1 }, .{ .x = 64, .y = 0 }, 0, &msg, &buf));
    try std.testing.expect(std.mem.indexOf(u8, msg, "empty launch") != null);
}

test "Limits.unknown accepts anything, so a failed query degrades to driver behaviour" {
    const K = Kernel(fn ([*]f32, u32) void);
    const k: K = .{ .inner = undefined, .limits = Limits.unknown, .max_threads = std.math.maxInt(u32) };
    var buf: [256]u8 = undefined;
    var msg: []const u8 = "";
    try std.testing.expect(k.checkGeometry(.{ .x = 1 << 30 }, .{ .x = 4096 }, 1 << 20, &msg, &buf));
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
