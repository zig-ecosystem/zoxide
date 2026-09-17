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

/// CUresult is a c_int. Common codes (authoritative names come from
/// cuGetErrorName at runtime):
///   0 SUCCESS, 1 INVALID_VALUE, 2 OUT_OF_MEMORY, 3 NOT_INITIALIZED,
///   34 STUB_LIBRARY, 100 NO_DEVICE, 101 INVALID_DEVICE, 200 INVALID_IMAGE,
///   201 INVALID_CONTEXT, 218 INVALID_PTX, 219 INVALID_GRAPHICS_CONTEXT,
///   300 INVALID_SOURCE, 301 FILE_NOT_FOUND, 304 INVALID_HANDLE,
///   500 NOT_FOUND, 700 NOT_READY, 701 ILLEGAL_ADDRESS,
///   719 LAUNCH_FAILURE, 999 UNKNOWN.

pub const Error = error{
    CudaInit,
    CudaCall,
    LibraryNotFound,
    SymbolMissing,
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
    cuCtxCreate_v2: *const fn (pctx: *CUcontext, flags: c_uint, dev: CUdevice) callconv(.c) c_int,
    cuCtxSetCurrent: *const fn (ctx: CUcontext) callconv(.c) c_int,
    cuModuleLoadData: *const fn (module: *CUmodule, image: ?*const anyopaque) callconv(.c) c_int,
    cuModuleGetFunction: *const fn (hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) callconv(.c) c_int,
    cuMemAlloc_v2: *const fn (dptr: *CUdeviceptr, bytesize: usize) callconv(.c) c_int,
    cuMemFree_v2: *const fn (dptr: CUdeviceptr) callconv(.c) c_int,
    cuMemcpyHtoD_v2: *const fn (dstDevice: CUdeviceptr, srcHost: ?*const anyopaque, byteCount: usize) callconv(.c) c_int,
    cuMemcpyDtoH_v2: *const fn (dstHost: ?*anyopaque, srcDevice: CUdeviceptr, byteCount: usize) callconv(.c) c_int,
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
};

pub const Module = struct {
    drv: *Driver,
    m: CUmodule,

    pub fn function(self: Module, name: [:0]const u8) Error!Function {
        var f: CUfunction = null;
        try self.drv.check(self.drv.cuModuleGetFunction(&f, self.m, name.ptr));
        return .{ .drv = self.drv, .f = f };
    }
};

pub const Function = struct {
    drv: *Driver,
    f: CUfunction,

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
        try self.drv.check(self.drv.cuLaunchKernel(
            self.f,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            0, // sharedMemBytes
            null, // default stream
            if (params.len > 0) @ptrCast(params.ptr) else null,
            null, // extra
        ));
    }
};
