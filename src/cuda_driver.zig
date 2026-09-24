//! Hand-written CUDA driver API bindings via dlopen — no cuda.h, no @cImport.
//!
//! The library is loaded at runtime: `libcuda.so.1` first (pod runtime images
//! often lack the dev symlink), then `libcuda.so`, then `libcuda.dylib` for
//! completeness. Types follow the CUDA driver ABI exactly (64-bit Linux/macOS).

const std = @import("std");

pub const CUdevice = c_int;
pub const CUdeviceptr = u64; // 64-bit only
pub const CUcontext = ?*anyopaque;
pub const CUmodule = ?*anyopaque;
pub const CUfunction = ?*anyopaque;
pub const CUstream = ?*anyopaque; // null = default stream
pub const CUevent = ?*anyopaque;

/// `CUtensorMap` — the TMA descriptor. Opaque 128 bytes that the driver fills in
/// and the kernel receives by pointer; the layout is deliberately not public, so
/// it is only ever passed around, never inspected.
///
/// 64-byte alignment is a hardware requirement, not a suggestion: `cp.async.bulk.
/// tensor` reads the descriptor through a dedicated path and a misaligned one
/// faults at launch rather than at encode time.
pub const CUtensorMap = extern struct {
    opaque_bytes: [128]u8 align(64) = @splat(0),
};

/// CUresult is a c_int. Common codes (authoritative names come from
/// cuGetErrorName at runtime):
///   0 SUCCESS, 1 INVALID_VALUE, 2 OUT_OF_MEMORY, 3 NOT_INITIALIZED,
///   34 STUB_LIBRARY, 100 NO_DEVICE, 101 INVALID_DEVICE, 200 INVALID_IMAGE,
///   201 INVALID_CONTEXT, 218 INVALID_PTX, 219 INVALID_GRAPHICS_CONTEXT,
///   300 INVALID_SOURCE, 301 FILE_NOT_FOUND, 304 INVALID_HANDLE,
///   500 NOT_FOUND, 700 NOT_READY, 701 ILLEGAL_ADDRESS,
///   719 LAUNCH_FAILURE, 999 UNKNOWN.

/// `CUDA_ERROR_NOT_READY`. Returned by `cuStreamQuery` for work still in
/// flight, which is a status rather than a failure, so it must not go through
/// `check`.
pub const cuda_error_not_ready: c_int = 600;

/// `CUDA_ERROR_NOT_FOUND`. Returned by `cuModuleGetFunction` for a name the
/// module does not export, which is common enough to deserve its own error: the
/// symbol is `<root source file stem>_$_<decl>`, and guessing the artifact name
/// instead produces exactly this.
pub const cuda_error_not_found: c_int = 500;

/// The `CUresult` codes worth distinguishing. Everything else collapses to
/// `CudaCall` with the driver's own text in `lastError()`.
///
/// The motivation is that `try` discards the message. A caller who writes
/// `try ctx.module(bytes)` and gets `error.CudaCall` has learned nothing, and the
/// most common causes each have a specific fix — a cubin built for the wrong
/// architecture, PTX the driver will not accept, a launch geometry the kernel's
/// register use cannot support. Those deserve names.
const CuResult = struct {
    const invalid_value = 1;
    const out_of_memory = 2;
    const no_binary_for_gpu = 209;
    const invalid_ptx = 218;
    const not_found = 500;
    const not_ready = 600;
    const illegal_address = 700;
    const launch_out_of_resources = 701;
    const launch_timeout = 702;
    const launch_failed = 719;
};

pub const Error = error{
    CudaInit,
    /// Any driver failure without a more specific mapping. The driver's message
    /// is in `lastError()`.
    CudaCall,
    LibraryNotFound,
    SymbolMissing,
    /// A module does not export the requested kernel name.
    KernelNotFound,
    /// A module does not export the requested device global. Distinct from
    /// `KernelNotFound` because the usual cause is different: the global was
    /// constant-folded away by LLVM and is not in the PTX at all. See
    /// `Module.global`.
    GlobalNotFound,
    /// The device is out of memory.
    CudaOutOfMemory,
    /// The cubin contains no code for this GPU — usually built for another `sm_`.
    ArchMismatch,
    /// The driver rejected the PTX. Often an instruction the target does not
    /// support, such as `wgmma` without `sm_90a`.
    InvalidPtx,
    /// A kernel dereferenced memory it does not own. Asynchronous, so it usually
    /// surfaces at the next synchronisation rather than at the offending launch.
    IllegalAddress,
    /// The launch needs more registers, shared memory or threads per block than
    /// the device can provide for this kernel.
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchFailed,
    InvalidValue,
    /// Launch geometry rejected before reaching the driver, with a description in
    /// `lastError()`.
    InvalidLaunchGeometry,
};

pub const Driver = struct {
    lib: std.DynLib,
    err_buf: [256]u8 = undefined,
    err_len: usize = 0,

    // Function pointers (populated by load()).
    cuInit: *const fn (flags: c_uint) callconv(.c) c_int,
    cuDeviceGetCount: *const fn (count: *c_int) callconv(.c) c_int,
    cuDeviceGet: *const fn (device: *CUdevice, ordinal: c_int) callconv(.c) c_int,
    cuDeviceGetName: *const fn (name: [*]u8, len: c_int, dev: CUdevice) callconv(.c) c_int,
    cuDeviceGetAttribute: *const fn (pi: *c_int, attrib: c_int, dev: CUdevice) callconv(.c) c_int,
    cuCtxCreate_v2: *const fn (pctx: *CUcontext, flags: c_uint, dev: CUdevice) callconv(.c) c_int,
    cuCtxSetCurrent: *const fn (ctx: CUcontext) callconv(.c) c_int,
    cuModuleLoadData: *const fn (module: *CUmodule, image: ?*const anyopaque) callconv(.c) c_int,
    cuModuleGetFunction: *const fn (hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) callconv(.c) c_int,
    // Resolves a device global by name, which is the only way to implement the
    // "host writes once, device reads by name" pattern (CUDA's
    // `cudaMemcpyToSymbol`). Requires the symbol to be `.visible` in the PTX.
    cuModuleGetGlobal_v2: *const fn (dptr: *CUdeviceptr, bytes: *usize, hmod: CUmodule, name: [*:0]const u8) callconv(.c) c_int,
    // Builds the TMA descriptor. Host-side only: the hardware needs the tensor
    // shape resolved before launch, which is the whole point — address generation
    // moves out of the kernel and into the copy engine.
    // Optional: CUDA 12+ only. See the note in `load` — a missing TMA symbol
    // must not stop an older driver from loading everything else.
    cuTensorMapEncodeTiled: ?*const fn (
        tensorMap: *CUtensorMap,
        dtype: c_uint,
        rank: c_uint,
        global_address: ?*anyopaque,
        global_dim: [*]const u64,
        global_strides: [*]const u64,
        box_dim: [*]const u32,
        element_strides: [*]const u32,
        interleave: c_uint,
        swizzle: c_uint,
        l2_promotion: c_uint,
        oob_fill: c_uint,
    ) callconv(.c) c_int,
    cuMemAlloc_v2: *const fn (dptr: *CUdeviceptr, bytesize: usize) callconv(.c) c_int,
    cuMemFree_v2: *const fn (dptr: CUdeviceptr) callconv(.c) c_int,
    cuMemcpyHtoD_v2: *const fn (dstDevice: CUdeviceptr, srcHost: ?*const anyopaque, byteCount: usize) callconv(.c) c_int,
    cuMemcpyDtoH_v2: *const fn (dstHost: ?*anyopaque, srcDevice: CUdeviceptr, byteCount: usize) callconv(.c) c_int,
    cuMemcpyHtoDAsync_v2: *const fn (dstDevice: CUdeviceptr, srcHost: ?*const anyopaque, byteCount: usize, hStream: CUstream) callconv(.c) c_int,
    cuMemcpyDtoHAsync_v2: *const fn (dstHost: ?*anyopaque, srcDevice: CUdeviceptr, byteCount: usize, hStream: CUstream) callconv(.c) c_int,
    cuMemcpyDtoDAsync_v2: *const fn (dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, byteCount: usize, hStream: CUstream) callconv(.c) c_int,
    cuMemsetD8_v2: *const fn (dstDevice: CUdeviceptr, uc: u8, n: usize) callconv(.c) c_int,
    cuMemsetD8Async: *const fn (dstDevice: CUdeviceptr, uc: u8, n: usize, hStream: CUstream) callconv(.c) c_int,
    cuMemsetD32_v2: *const fn (dstDevice: CUdeviceptr, ui: c_uint, n: usize) callconv(.c) c_int,
    // Page-locked host memory. Without it `cuMemcpy*Async` cannot actually
    // overlap: the driver has to stage pageable memory through an internal
    // pinned buffer and blocks while doing so, so the call is asynchronous in
    // name only.
    cuMemHostAlloc: *const fn (pp: *?*anyopaque, bytesize: usize, flags: c_uint) callconv(.c) c_int,
    cuMemFreeHost: *const fn (p: ?*anyopaque) callconv(.c) c_int,
    cuStreamCreate: *const fn (phStream: *CUstream, flags: c_uint) callconv(.c) c_int,
    cuStreamDestroy_v2: *const fn (hStream: CUstream) callconv(.c) c_int,
    cuStreamSynchronize: *const fn (hStream: CUstream) callconv(.c) c_int,
    cuStreamQuery: *const fn (hStream: CUstream) callconv(.c) c_int,
    cuOccupancyMaxActiveBlocksPerMultiprocessor: *const fn (numBlocks: *c_int, func: CUfunction, blockSize: c_int, dynamicSMemSize: usize) callconv(.c) c_int,
    cuFuncGetAttribute: *const fn (pi: *c_int, attrib: c_int, hfunc: CUfunction) callconv(.c) c_int,
    cuLaunchKernel: *const fn (
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: ?*?*anyopaque,
        extra: ?*?*anyopaque,
    ) callconv(.c) c_int,
    cuCtxSynchronize: *const fn () callconv(.c) c_int,
    cuEventCreate: *const fn (phEvent: *CUevent, flags: c_uint) callconv(.c) c_int,
    cuEventRecord: *const fn (hEvent: CUevent, hStream: CUstream) callconv(.c) c_int,
    cuEventSynchronize: *const fn (hEvent: CUevent) callconv(.c) c_int,
    cuEventElapsedTime: *const fn (pMilliseconds: *f32, hStart: CUevent, hEnd: CUevent) callconv(.c) c_int,
    cuEventDestroy: *const fn (hEvent: CUevent) callconv(.c) c_int,
    cuGetErrorString: *const fn (err: c_int, pStr: *?[*:0]const u8) callconv(.c) c_int,
    cuGetErrorName: *const fn (err: c_int, pStr: *?[*:0]const u8) callconv(.c) c_int,

    /// dlopen libcuda and resolve all symbols.
    pub fn load() Error!Driver {
        const candidates = [_][]const u8{ "libcuda.so.1", "libcuda.so", "libcuda.dylib" };
        var lib: std.DynLib = undefined;
        var opened = false;
        for (candidates) |name| {
            if (std.DynLib.open(name)) |l| {
                lib = l;
                opened = true;
                break;
            } else |_| {}
        }
        if (!opened) return error.LibraryNotFound;
        errdefer lib.close();

        var drv: Driver = undefined;
        drv.lib = lib;
        drv.err_len = 0;
        inline for (@typeInfo(Driver).@"struct".fields) |f| {
            if (comptime std.mem.startsWith(u8, f.name, "cu")) {
                // An optional field marks a symbol that may legitimately be
                // absent on an older driver. Treating those as fatal would mean
                // a CUDA 11 driver cannot load the library at all, just because
                // it lacks the TMA entry points — so absence is recorded and
                // reported at the point of use instead.
                if (comptime @typeInfo(f.type) == .optional) {
                    @field(drv, f.name) = lib.lookup(@typeInfo(f.type).optional.child, f.name);
                } else {
                    @field(drv, f.name) = lib.lookup(f.type, f.name) orelse
                        return error.SymbolMissing;
                }
            }
        }
        return drv;
    }

    pub fn unload(self: *Driver) void {
        self.lib.close();
    }

    /// Check a CUresult; on failure, record "code (NAME): description" in
    /// err_buf (readable via lastError()) and return error.CudaCall.
    pub fn check(self: *Driver, res: c_int) Error!void {
        if (res == 0) return;
        var name_ptr: ?[*:0]const u8 = null;
        var str_ptr: ?[*:0]const u8 = null;
        _ = self.cuGetErrorName(res, &name_ptr);
        _ = self.cuGetErrorString(res, &str_ptr);
        const name = if (name_ptr) |p| std.mem.span(p) else "CUDA_ERROR_?";
        const str = if (str_ptr) |p| std.mem.span(p) else "unknown error";
        const msg = std.fmt.bufPrint(&self.err_buf, "CUDA error {d} ({s}): {s}", .{ res, name, str }) catch
            "CUDA error (message truncated)";
        self.err_len = msg.len;
        return switch (res) {
            CuResult.out_of_memory => error.CudaOutOfMemory,
            CuResult.no_binary_for_gpu => error.ArchMismatch,
            CuResult.invalid_ptx => error.InvalidPtx,
            CuResult.not_found => error.KernelNotFound,
            CuResult.illegal_address => error.IllegalAddress,
            CuResult.launch_out_of_resources => error.LaunchOutOfResources,
            CuResult.launch_timeout => error.LaunchTimeout,
            CuResult.launch_failed => error.LaunchFailed,
            CuResult.invalid_value => error.InvalidValue,
            else => error.CudaCall,
        };
    }

    /// Record a message for `lastError()` from a check this layer performed
    /// itself, so caller-side diagnostics read the same way as driver ones.
    pub fn setError(self: *Driver, msg: []const u8) void {
        const n = @min(msg.len, self.err_buf.len);
        @memcpy(self.err_buf[0..n], msg[0..n]);
        self.err_len = n;
    }

    pub fn lastError(self: *const Driver) []const u8 {
        return self.err_buf[0..self.err_len];
    }
};

/// Device 0 + primary context, RAII-style.
pub const Context = struct {
    drv: *Driver,
    dev: CUdevice,
    ctx: CUcontext,

    pub fn init(drv: *Driver) Error!Context {
        try drv.check(drv.cuInit(0));
        var count: c_int = 0;
        try drv.check(drv.cuDeviceGetCount(&count));
        if (count <= 0) {
            const m = std.fmt.bufPrint(&drv.err_buf, "no CUDA devices (cuDeviceGetCount = {d})", .{count}) catch "no CUDA devices";
            drv.err_len = m.len;
            return error.CudaCall;
        }
        var dev: CUdevice = 0;
        try drv.check(drv.cuDeviceGet(&dev, 0));
        var ctx: CUcontext = null;
        try drv.check(drv.cuCtxCreate_v2(&ctx, 0, dev));
        return .{ .drv = drv, .dev = dev, .ctx = ctx };
    }

    pub fn name(self: *Context, buf: []u8) []const u8 {
        var raw: [128]u8 = undefined;
        self.drv.check(self.drv.cuDeviceGetName(&raw, raw.len, self.dev)) catch return "?";
        const len = std.mem.indexOfScalar(u8, &raw, 0) orelse raw.len;
        const n = @min(len, buf.len);
        @memcpy(buf[0..n], raw[0..n]);
        return buf[0..n];
    }

    /// `CUdevice_attribute` values we query. Only the ones that are exact and
    /// stable are listed — notably *not* the clock/bus-width pair, since
    /// deriving HBM bandwidth from them does not come out right for HBM's
    /// pseudo-channel organisation and would just be a spec guess wearing a
    /// measurement's clothes.
    pub const Attr = enum(c_int) {
        max_threads_per_block = 1,
        max_block_dim_x = 2,
        max_block_dim_y = 3,
        max_block_dim_z = 4,
        max_grid_dim_x = 5,
        max_grid_dim_y = 6,
        max_grid_dim_z = 7,
        max_shared_memory_per_block = 8,
        multiprocessor_count = 16,
        l2_cache_size = 38,
        max_shared_memory_per_multiprocessor = 81,
    };

    pub fn attr(self: *Context, a: Attr) Error!u64 {
        var v: c_int = 0;
        try self.drv.check(self.drv.cuDeviceGetAttribute(&v, @intFromEnum(a), self.dev));
        return @intCast(@max(v, 0));
    }

    pub const Info = struct {
        sms: u64,
        l2_bytes: u64,
        shared_per_sm: u64,
    };

    /// Device properties that matter for reading a bench result: how many SMs
    /// the work spreads over, and how big L2 is — the latter decides whether a
    /// given problem size still fits in cache, which is what separates a
    /// bandwidth-bound measurement from a compute-bound one.
    pub fn info(self: *Context) Error!Info {
        return .{
            .sms = try self.attr(.multiprocessor_count),
            .l2_bytes = try self.attr(.l2_cache_size),
            .shared_per_sm = try self.attr(.max_shared_memory_per_multiprocessor),
        };
    }

    pub fn module(self: *Context, image: []const u8) Error!Module {
        try self.drv.check(self.drv.cuCtxSetCurrent(self.ctx));
        var m: CUmodule = null;
        try self.drv.check(self.drv.cuModuleLoadData(&m, image.ptr));
        return .{ .drv = self.drv, .m = m };
    }

    pub fn alloc(self: *Context, bytes: usize) Error!CUdeviceptr {
        try self.drv.check(self.drv.cuCtxSetCurrent(self.ctx));
        var p: CUdeviceptr = 0;
        try self.drv.check(self.drv.cuMemAlloc_v2(&p, bytes));
        return p;
    }

    pub fn free(self: *Context, p: CUdeviceptr) void {
        self.drv.check(self.drv.cuMemFree_v2(p)) catch {};
    }

    pub fn copyHtoD(self: *Context, dst: CUdeviceptr, src: []const u8) Error!void {
        try self.drv.check(self.drv.cuMemcpyHtoD_v2(dst, src.ptr, src.len));
    }

    pub fn copyDtoH(self: *Context, dst: []u8, src: CUdeviceptr) Error!void {
        try self.drv.check(self.drv.cuMemcpyDtoH_v2(dst.ptr, src, dst.len));
    }

    pub fn synchronize(self: *Context) Error!void {
        try self.drv.check(self.drv.cuCtxSynchronize());
    }

    /// Build a TMA descriptor for a `rank`-dimensional tensor in device memory,
    /// tiled by `box`.
    ///
    /// Both arrays are innermost-first, which is the driver's order and the
    /// reverse of how a row-major matrix is usually described: for an `rows x
    /// cols` row-major matrix, `dim = .{ cols, rows }`.
    ///
    /// The element type comes from `T` rather than a separate argument, because
    /// the two disagreeing is the failure that costs the most to find — it
    /// encodes, it launches, and it produces garbage.
    ///
    /// Checked here rather than left to `CUDA_ERROR_INVALID_VALUE`:
    ///
    ///   - rank 1..5
    ///   - every box dimension in 1..256
    ///   - innermost box width against the swizzle period. This one is not an
    ///     error in the driver at all: a 128B swizzle with a 64-byte inner tile
    ///     copies successfully and delivers permuted data.
    ///   - global address 16-byte aligned
    ///   - strides multiples of 16
    ///
    /// `strides` is `rank - 1` entries in bytes, innermost stride first, matching
    /// the driver: the innermost dimension is implicitly unit-stride.
    pub fn encodeTensorMap(
        self: *Context,
        comptime T: type,
        ptr: CUdeviceptr,
        dim: []const u64,
        strides: []const u64,
        box: []const u32,
        opts: struct {
            swizzle: Swizzle = .none,
            l2_promotion: L2Promotion = .b128,
            oob_fill: OobFill = .zero,
            /// Elements skipped between loads along each dimension. All ones for
            /// a dense tile, which is nearly always what is wanted.
            element_strides: ?[]const u32 = null,
        },
    ) Error!TensorMap {
        const elem = @sizeOf(T);

        if (dim.len < 1 or dim.len > 5) {
            return self.fail("tensor rank {d} out of range; TMA supports 1..5", .{dim.len});
        }
        if (box.len != dim.len) {
            return self.fail("box has {d} dimensions but the tensor has {d}", .{ box.len, dim.len });
        }
        if (strides.len != dim.len - 1) {
            return self.fail(
                "strides needs {d} entries for a rank-{d} tensor (the innermost " ++
                    "dimension is implicitly unit-stride), got {d}",
                .{ dim.len - 1, dim.len, strides.len },
            );
        }
        for (box, 0..) |b, i| {
            if (b == 0 or b > 256) {
                return self.fail("box dimension {d} is {d}; TMA requires 1..256", .{ i, b });
            }
        }
        // The check the driver does not do.
        if (opts.swizzle != .none) {
            const inner_bytes = @as(usize, box[0]) * elem;
            const period = opts.swizzle.periodBytes();
            if (inner_bytes > period) {
                return self.fail(
                    "innermost box is {d} bytes ({d} x {d}B) but {s} swizzle has a " ++
                        "{d}-byte period; the copy would succeed and deliver permuted data",
                    .{ inner_bytes, box[0], elem, @tagName(opts.swizzle), period },
                );
            }
        }
        if (ptr % 16 != 0) {
            return self.fail("tensor address {x} is not 16-byte aligned", .{ptr});
        }
        for (strides, 0..) |s, i| {
            if (s % 16 != 0) {
                return self.fail("stride {d} is {d} bytes; TMA requires multiples of 16", .{ i, s });
            }
        }

        var ones: [5]u32 = @splat(1);
        const estrides = opts.element_strides orelse ones[0..dim.len];

        var out: TensorMap = .{
            .map = .{},
            .box = @splat(1),
            .rank = @intCast(dim.len),
            .swizzle = opts.swizzle,
            .elem_bytes = elem,
        };
        @memcpy(out.box[0..box.len], box);

        const encode = self.drv.cuTensorMapEncodeTiled orelse {
            _ = self.fail(
                "this driver has no cuTensorMapEncodeTiled; TMA needs CUDA 12 or newer",
                .{},
            );
            // Not InvalidValue: nothing is wrong with the arguments.
            return error.SymbolMissing;
        };
        try self.drv.check(self.drv.cuCtxSetCurrent(self.ctx));
        try self.drv.check(encode(
            &out.map,
            tensorDataType(T),
            @intCast(dim.len),
            @ptrFromInt(ptr),
            dim.ptr,
            strides.ptr,
            box.ptr,
            estrides.ptr,
            0, // interleave: none
            @intFromEnum(opts.swizzle),
            @intFromEnum(opts.l2_promotion),
            @intFromEnum(opts.oob_fill),
        ));
        return out;
    }

    /// Record a rejected-before-the-driver message and return `InvalidValue`, so
    /// the reason survives instead of becoming a bare error code.
    fn fail(self: *Context, comptime fmt: []const u8, args: anytype) Error {
        const m = std.fmt.bufPrint(&self.drv.err_buf, fmt, args) catch "invalid tensor map parameters";
        self.drv.err_len = m.len;
        return error.InvalidValue;
    }

    /// `non_blocking` streams do not synchronise with the legacy default
    /// stream, which is what you want when several streams should genuinely run
    /// concurrently.
    pub fn streamCreate(self: *Context, non_blocking: bool) Error!Stream {
        var st: CUstream = null;
        try self.drv.check(self.drv.cuStreamCreate(&st, if (non_blocking) 1 else 0));
        return .{ .drv = self.drv, .s = st };
    }

    /// Page-locked host memory. `cuMemcpy*Async` from ordinary pageable memory
    /// is asynchronous in name only — the driver stages it through an internal
    /// pinned buffer and blocks — so overlapping transfers with compute needs
    /// allocations from here.
    pub fn allocPinned(self: *Context, bytes: usize) Error!PinnedHost {
        var p: ?*anyopaque = null;
        try self.drv.check(self.drv.cuMemHostAlloc(&p, bytes, 0));
        return .{ .drv = self.drv, .bytes = @as([*]u8, @ptrCast(p.?))[0..bytes] };
    }

    pub fn copyHtoDAsync(self: *Context, dst: CUdeviceptr, src: []const u8, stream: Stream) Error!void {
        try self.drv.check(self.drv.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, stream.s));
    }

    pub fn copyDtoHAsync(self: *Context, dst: []u8, src: CUdeviceptr, stream: Stream) Error!void {
        try self.drv.check(self.drv.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, stream.s));
    }

    pub fn copyDtoDAsync(self: *Context, dst: CUdeviceptr, src: CUdeviceptr, bytes: usize, stream: Stream) Error!void {
        try self.drv.check(self.drv.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.s));
    }

    pub fn memsetD8(self: *Context, dst: CUdeviceptr, value: u8, bytes: usize) Error!void {
        try self.drv.check(self.drv.cuMemsetD8_v2(dst, value, bytes));
    }

    pub fn memsetD8Async(self: *Context, dst: CUdeviceptr, value: u8, bytes: usize, stream: Stream) Error!void {
        try self.drv.check(self.drv.cuMemsetD8Async(dst, value, bytes, stream.s));
    }

    /// 32-bit pattern fill; `n` counts words, not bytes.
    pub fn memsetD32(self: *Context, dst: CUdeviceptr, value: u32, n: usize) Error!void {
        try self.drv.check(self.drv.cuMemsetD32_v2(dst, value, n));
    }

    pub fn eventCreate(self: *Context) Error!Event {
        try self.drv.check(self.drv.cuCtxSetCurrent(self.ctx));
        var ev: CUevent = null;
        try self.drv.check(self.drv.cuEventCreate(&ev, 0)); // CU_EVENT_DEFAULT
        return .{ .drv = self.drv, .ev = ev };
    }
};

pub const Stream = struct {
    drv: *Driver,
    s: CUstream,

    pub fn sync(self: Stream) Error!void {
        try self.drv.check(self.drv.cuStreamSynchronize(self.s));
    }

    /// True when all previously enqueued work has completed. Unlike the other
    /// wrappers this treats `CUDA_ERROR_NOT_READY` as a value, not an error.
    pub fn done(self: Stream) Error!bool {
        const r = self.drv.cuStreamQuery(self.s);
        if (r == cuda_error_not_ready) return false;
        try self.drv.check(r);
        return true;
    }

    pub fn destroy(self: Stream) void {
        _ = self.drv.cuStreamDestroy_v2(self.s);
    }
};

/// Page-locked host allocation, required for `cuMemcpy*Async` to overlap.
pub const PinnedHost = struct {
    drv: *Driver,
    bytes: []u8,

    pub fn free(self: PinnedHost) void {
        _ = self.drv.cuMemFreeHost(self.bytes.ptr);
    }
};

pub const Event = struct {
    drv: *Driver,
    ev: CUevent,

    pub fn record(self: Event) Error!void {
        try self.drv.check(self.drv.cuEventRecord(self.ev, null));
    }

    pub fn sync(self: Event) Error!void {
        try self.drv.check(self.drv.cuEventSynchronize(self.ev));
    }

    /// Milliseconds between two recorded events.
    pub fn elapsedMs(self: Event, end: Event) Error!f32 {
        var ms: f32 = 0;
        try self.drv.check(self.drv.cuEventElapsedTime(&ms, self.ev, end.ev));
        return ms;
    }

    pub fn destroy(self: Event) void {
        self.drv.check(self.drv.cuEventDestroy(self.ev)) catch {};
    }
};

/// How the shared-memory destination of a TMA copy is swizzled.
///
/// Not cosmetic: the mode fixes how many bytes of the innermost dimension one
/// swizzle period covers, so it has to agree with the tile width. A mismatch is
/// not a fault — the copy succeeds and delivers permuted data — which is why
/// `TensorMap.encode` checks the relationship instead of passing the value
/// through.
pub const Swizzle = enum(c_uint) {
    none = 0,
    b32 = 1,
    b64 = 2,
    b128 = 3,

    /// Bytes of the innermost dimension covered by one swizzle period. The
    /// innermost box dimension must not exceed this.
    pub fn periodBytes(self: Swizzle) usize {
        return switch (self) {
            .none => 0, // no constraint
            .b32 => 32,
            .b64 => 64,
            .b128 => 128,
        };
    }
};

/// L2 prefetch hint applied to the copy.
pub const L2Promotion = enum(c_uint) { none = 0, b64 = 1, b128 = 2, b256 = 3 };

/// What a copy writes for coordinates outside the tensor. `zero` is what a GEMM
/// wants at the ragged edge; `nan` makes out-of-range reads visible instead of
/// quietly plausible.
pub const OobFill = enum(c_uint) { zero = 0, nan = 1 };

/// A TMA descriptor plus the shape it was built for.
///
/// Keeping the shape alongside the opaque bytes is what makes the kernel-side
/// contract checkable: the descriptor tells the hardware the tile geometry, and
/// nothing in `cp.async.bulk.tensor` verifies that the kernel's shared-memory
/// buffer matches it.
pub const TensorMap = struct {
    map: CUtensorMap,
    /// Innermost first, matching the driver's ordering.
    box: [5]u32,
    rank: u32,
    swizzle: Swizzle,
    elem_bytes: u32,

    /// Bytes one tile occupies in shared memory. What the kernel must reserve.
    pub fn tileBytes(self: TensorMap) usize {
        var n: usize = self.elem_bytes;
        for (self.box[0..self.rank]) |d| n *= d;
        return n;
    }
};

/// Map a Zig element type to `CUtensorMapDataType`.
///
/// Deriving this from the type rather than taking it as an argument removes the
/// mismatch that costs the most to debug: declaring an f16 tensor while handing
/// over an f32 pointer encodes cleanly, launches cleanly, and produces garbage.
fn tensorDataType(comptime T: type) c_uint {
    return switch (T) {
        u8, i8 => 0,
        u16 => 1,
        u32 => 2,
        i32 => 3,
        u64 => 4,
        i64 => 5,
        f16 => 6,
        f32 => 7,
        f64 => 9,
        else => @compileError("no CUtensorMapDataType for " ++ @typeName(T)),
    };
}

pub const Module = struct {
    drv: *Driver,
    m: CUmodule,

    pub fn function(self: Module, name: [:0]const u8) Error!Function {
        var f: CUfunction = null;
        const r = self.drv.cuModuleGetFunction(&f, self.m, name.ptr);
        if (r == cuda_error_not_found) {
            const m = std.fmt.bufPrint(
                &self.drv.err_buf,
                "kernel '{s}' not found in module; the PTX symbol is " ++
                    "<root source file stem>_$_<decl>, e.g. kernel_$_scale for kernel.zig — " ++
                    "not the build artifact's name",
                .{name},
            ) catch "kernel not found in module";
            self.drv.err_len = m.len;
            return error.KernelNotFound;
        }
        try self.drv.check(r);
        return .{ .drv = self.drv, .f = f };
    }

    /// Resolve a device global by PTX symbol name, returning its device address
    /// and size. This is the host half of "host writes once, device reads by
    /// name"; pair it with `copyHtoD`/`copyDtoH`.
    ///
    /// Measured on H20: a module-scope `var` reaches the PTX as plain `.global`
    /// with no `.visible`, and this call resolves it anyway — confirmed by
    /// stripping `.visible` off a working module and watching it keep working.
    /// So no linkage games are needed, which is fortunate: every way of asking
    /// Zig for external linkage on a variable fails on nvptx (zig 0.16.0 / LLVM
    /// 21.1.8), and a plain `export var x` aborts the compiler.
    ///
    /// What does need care is the device side. LLVM assumes nothing outside the
    /// module writes a module-scope global, so an ordinary read is folded
    /// against the initialiser and the symbol is dropped from the PTX — which
    /// surfaces here as `GlobalNotFound`. Read it with `cuda.ldg()`.
    pub fn global(self: Module, name: [:0]const u8) Error!struct { ptr: CUdeviceptr, bytes: usize } {
        var p: CUdeviceptr = 0;
        var n: usize = 0;
        const r = self.drv.cuModuleGetGlobal_v2(&p, &n, self.m, name.ptr);
        if (r == cuda_error_not_found) {
            const m = std.fmt.bufPrint(
                &self.drv.err_buf,
                "device global '{s}' not found in module. Two different causes " ++
                    "look alike here: (1) the name — the PTX symbol is " ++
                    "<root source file stem>_$_<decl>, e.g. kernel_$_dev_scale " ++
                    "for kernel.zig; (2) visibility — Zig emits module-scope " ++
                    "globals without '.visible', so the symbol is module-local " ++
                    "and absent from the cubin symbol table even when the name " ++
                    "is right. Run the PTX through 'zoxide ptx --promote-globals' " ++
                    "to fix (2)",
                .{name},
            ) catch "device global not found in module";
            self.drv.err_len = m.len;
            return error.GlobalNotFound;
        }
        try self.drv.check(r);
        return .{ .ptr = p, .bytes = n };
    }
};

pub const Function = struct {
    drv: *Driver,
    f: CUfunction,

    /// `CUfunction_attribute` values worth reporting. Register count and static
    /// shared memory are what actually determine occupancy, so having them
    /// removes the need to work it out from the PTX by hand.
    pub const Attr = enum(c_int) {
        max_threads_per_block = 0,
        shared_size_bytes = 1,
        const_size_bytes = 2,
        local_size_bytes = 3,
        num_regs = 4,
        ptx_version = 5,
        binary_version = 6,
    };

    pub fn attr(self: Function, a: Attr) Error!u64 {
        var v: c_int = 0;
        try self.drv.check(self.drv.cuFuncGetAttribute(&v, @intFromEnum(a), self.f));
        return @intCast(@max(v, 0));
    }

    /// Blocks of `block_size` threads that can be resident per SM, as the
    /// driver computes it from this kernel's register and shared-memory use.
    pub fn occupancy(self: Function, block_size: u32, dynamic_shared: usize) Error!u32 {
        var n: c_int = 0;
        try self.drv.check(self.drv.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &n,
            self.f,
            @intCast(block_size),
            dynamic_shared,
        ));
        return @intCast(@max(n, 0));
    }

    /// params: one pointer per kernel argument, each pointing at the
    /// argument value (kernelParams convention of cuLaunchKernel).
    pub fn launch(
        self: Function,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        params: []?*anyopaque,
    ) Error!void {
        return self.launchOn(null, grid_x, grid_y, grid_z, block_x, block_y, block_z, 0, params);
    }

    /// `stream` of null is the legacy default stream.
    pub fn launchOn(
        self: Function,
        stream: CUstream,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        dynamic_shared: u32,
        params: []?*anyopaque,
    ) Error!void {
        try self.drv.check(self.drv.cuLaunchKernel(
            self.f,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            dynamic_shared,
            stream,
            if (params.len > 0) @ptrCast(params.ptr) else null,
            null, // extra
        ));
    }
};
