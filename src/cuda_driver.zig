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

pub const Error = error{
    CudaInit,
    CudaCall,
    LibraryNotFound,
    SymbolMissing,
    /// A module does not export the requested kernel name.
    KernelNotFound,
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
                @field(drv, f.name) = lib.lookup(f.type, f.name) orelse
                    return error.SymbolMissing;
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
        return error.CudaCall;
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
