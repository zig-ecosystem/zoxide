//! Device-side CUDA library for zoxide kernels.
//!
//! Freestanding: no host std dependencies; only LLVM NVVM intrinsics, Zig
//! builtins and inline PTX.
//!
//! Preference order for reaching an instruction: an `llvm.nvvm.*` intrinsic
//! when one exists (best typed, see `gen`), otherwise inline asm with
//! `%[name]` named operands (`asm_gen`, `wgmma`). Note that Zig supports only
//! the named operand form on nvptx — the positional spellings (`$0`, `%0`,
//! `${0}`) are emitted verbatim and rejected by ptxas. An earlier revision of
//! this comment claimed operand substitution did not work at all and that
//! inline asm was therefore unusable here; that was wrong, and the tensor-core
//! kernels are built on it.

/// 3-component index for threadIdx / blockIdx / blockDim / gridDim.
pub const Idx3 = struct { x: u32, y: u32, z: u32 };

pub const warp_size = 32;

/// Generated NVVM intrinsic bindings (see `zoxide gen`).
pub const gen = @import("gen/intrinsics.zig");
/// Generated asm-template bindings (named-operand inline asm).
pub const asm_gen = @import("gen/instrinsics_asm.zig");
/// Hopper warpgroup MMA (`wgmma.mma_async`). Requires sm_90a; hand-written
/// asm because LLVM has no mma_async intrinsic.
pub const wgmma = @import("wgmma.zig");
/// Compile-time kernel signature agreement with the host side. Use
/// `abi.assertMatches(shared_decl, @TypeOf(my_kernel))` so that changing a
/// kernel's parameters without updating the host is a compile error here.
pub const abi = @import("kernel_abi.zig");

// --- printf ---
//
// LLVM removed the llvm.nvvm.vprintf intrinsic; the NVPTX backend instead
// lowers calls to a function literally named `vprintf` into the PTX vprintf
// mechanism. ABI: format is a pointer to a NUL-terminated string in .global
// memory; valist is a byte buffer of 8-byte little-endian slots, one per
// argument (printf varargs promotion applies: f32 is passed as f64).

extern fn vprintf(fmt: ?*const anyopaque, valist: ?*const anyopaque) i32;

fn packPrintfSlot(slot: *[8]u8, val: anytype) void {
    const T = @TypeOf(val);
    const write = struct {
        fn u64le(s: *[8]u8, v: u64) void {
            s.* = @bitCast(v);
        }
    }.u64le;
    switch (@typeInfo(T)) {
        .int => |t| {
            const v: u64 = if (t.signedness == .signed)
                @bitCast(@as(i64, val))
            else
                @intCast(val);
            write(slot, v);
        },
        .float => |t| {
            // C varargs promotion: all floats become double.
            const v: u64 = if (t.bits == 64)
                @bitCast(@as(f64, val))
            else
                @bitCast(@as(f64, @floatCast(val)));
            write(slot, v);
        },
        .bool => write(slot, @intFromBool(val)),
        else => @compileError("printf: unsupported arg type " ++ @typeName(T)),
    }
}

/// Kernel-side printf: `cuda.printf("tid=%d x=%f\n", .{ tid, x });`
/// Output is flushed to host stdout on the next cuCtxSynchronize.
pub fn printf(comptime fmt: [:0]const u8, args: anytype) void {
    const S = struct {
        const fmt_g: [fmt.len:0]u8 addrspace(.global) = fmt[0..fmt.len :0].*;
    };
    const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
    if (fields.len == 0) {
        _ = vprintf(@ptrCast(@addrSpaceCast(&S.fmt_g)), null);
        return;
    }
    var buf: [fields.len * 8]u8 align(8) = undefined;
    inline for (fields, 0..) |f, i| {
        packPrintfSlot(buf[i * 8 ..][0..8], @field(args, f.name));
    }
    _ = vprintf(@ptrCast(@addrSpaceCast(&S.fmt_g)), &buf);
}

extern fn @"llvm.nvvm.read.ptx.sreg.tid.x"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.tid.y"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.tid.z"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.x"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.y"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.z"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.y"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.z"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.x"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.y"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.z"() i32;
extern fn @"llvm.nvvm.read.ptx.sreg.laneid"() i32;

extern fn @"llvm.nvvm.barrier0"() void;
extern fn @"llvm.nvvm.bar.warp.sync"(mask: i32) void;

const ShflRet = extern struct { val: i32, pred: i32 };
extern fn @"llvm.nvvm.shfl.sync.down.i32"(mask: i32, val: i32, delta: i32, pack: i32) ShflRet;
extern fn @"llvm.nvvm.shfl.sync.up.i32"(mask: i32, val: i32, delta: i32, pack: i32) ShflRet;
extern fn @"llvm.nvvm.shfl.sync.bfly.i32"(mask: i32, val: i32, delta: i32, pack: i32) ShflRet;
extern fn @"llvm.nvvm.shfl.sync.idx.i32"(mask: i32, val: i32, delta: i32, pack: i32) ShflRet;

fn sreg1(f: anytype) u32 {
    return @bitCast(f());
}

pub fn threadIdx() Idx3 {
    return .{
        .x = sreg1(@"llvm.nvvm.read.ptx.sreg.tid.x"),
        .y = sreg1(@"llvm.nvvm.read.ptx.sreg.tid.y"),
        .z = sreg1(@"llvm.nvvm.read.ptx.sreg.tid.z"),
    };
}

pub fn blockIdx() Idx3 {
    return .{
        .x = sreg1(@"llvm.nvvm.read.ptx.sreg.ctaid.x"),
        .y = sreg1(@"llvm.nvvm.read.ptx.sreg.ctaid.y"),
        .z = sreg1(@"llvm.nvvm.read.ptx.sreg.ctaid.z"),
    };
}

pub fn blockDim() Idx3 {
    return .{
        .x = sreg1(@"llvm.nvvm.read.ptx.sreg.ntid.x"),
        .y = sreg1(@"llvm.nvvm.read.ptx.sreg.ntid.y"),
        .z = sreg1(@"llvm.nvvm.read.ptx.sreg.ntid.z"),
    };
}

pub fn gridDim() Idx3 {
    return .{
        .x = sreg1(@"llvm.nvvm.read.ptx.sreg.nctaid.x"),
        .y = sreg1(@"llvm.nvvm.read.ptx.sreg.nctaid.y"),
        .z = sreg1(@"llvm.nvvm.read.ptx.sreg.nctaid.z"),
    };
}

/// Lane index within the warp (0..31).
pub fn laneId() u32 {
    return sreg1(@"llvm.nvvm.read.ptx.sreg.laneid");
}

/// Warp index within the block.
pub fn warpId() u32 {
    return threadIdx().x / warp_size;
}

/// Flattened global thread id over the x dimension: blockIdx.x * blockDim.x + threadIdx.x.
pub fn globalThreadId() u32 {
    return blockIdx().x *% blockDim().x +% threadIdx().x;
}

/// __syncthreads(): block-wide barrier.
pub fn syncThreads() void {
    @"llvm.nvvm.barrier0"();
}

/// __syncwarp(mask).
pub fn syncWarp(mask: u32) void {
    @"llvm.nvvm.bar.warp.sync"(mask);
}

/// Register class a type is loaded into. PTX asm constraints have to be written
/// as string literals — the grammar takes a token, not an expression — so each
/// class needs its own `asm` statement rather than a parameterised one.
const RegClass = enum { b16, f32, f64, b32, b64 };

fn regClass(comptime T: type) RegClass {
    return switch (T) {
        f16, u16, i16 => .b16,
        f32 => .f32,
        f64 => .f64,
        // @Vector(2, f16) is two halves in one 32-bit register: the shape wgmma
        // and the f16x2 math instructions want.
        u32, i32, @Vector(2, f16) => .b32,
        u64, i64 => .b64,
        else => @compileError("ldg: unsupported type " ++ @typeName(T)),
    };
}

/// Load through the read-only data cache — CUDA's `__ldg`, PTX `ld.global.nc`.
///
/// Beyond the caching hint, this load is opaque to the optimiser, and that is
/// what makes it the only sound way to read a device global the host writes.
/// LLVM assumes nothing outside the module writes a module-scope global, so an
/// ordinary read can be constant-folded against the initialiser and the symbol
/// dropped from the PTX entirely. Measured on zig 0.16.0 / LLVM 21.1.8: with
/// `= .{ 10, 20, 30, 40 }` the symbol survived, and with `= .{ 0, 0, 0, 0 }` —
/// the natural placeholder — the whole thing folded to a constant zero and the
/// symbol disappeared. Host uploads would then be silently ignored. Reading via
/// this function keeps the symbol and emits a real load either way.
///
/// The address must be in global memory. Kernel pointer parameters and
/// `addrspace(.global)` declarations both are.
///
/// ```zig
/// var dev_bias: [4]f32 addrspace(.global) = .{ 0, 0, 0, 0 };
/// ...
/// const b = cuda.ldg(f32, &dev_bias[i % 4]);
/// ```
///
/// Safe against host writes between launches: the read-only cache does not
/// survive a kernel boundary. It is *not* safe if something writes the same
/// location during the launch — that is what `.nc` asserts does not happen.
pub inline fn ldg(comptime T: type, p: anytype) T {
    return switch (comptime regClass(T)) {
        .b16 => @bitCast(asm volatile ("ld.global.nc.b16 %[o], [%[a]];"
            : [o] "=h" (-> u16),
            : [a] "l" (p),
            : .{})),
        .f32 => @bitCast(asm volatile ("ld.global.nc.f32 %[o], [%[a]];"
            : [o] "=f" (-> f32),
            : [a] "l" (p),
            : .{})),
        .f64 => @bitCast(asm volatile ("ld.global.nc.f64 %[o], [%[a]];"
            : [o] "=d" (-> f64),
            : [a] "l" (p),
            : .{})),
        .b32 => @bitCast(asm volatile ("ld.global.nc.b32 %[o], [%[a]];"
            : [o] "=r" (-> u32),
            : [a] "l" (p),
            : .{})),
        .b64 => @bitCast(asm volatile ("ld.global.nc.b64 %[o], [%[a]];"
            : [o] "=l" (-> u64),
            : [a] "l" (p),
            : .{})),
    };
}

const full_mask: i32 = -1; // 0xffffffff

fn shflI32(comptime kind: enum { down, up, bfly, idx }, mask: u32, val: i32, off: i32, pack: i32) i32 {
    const m: i32 = @bitCast(mask);
    return switch (kind) {
        .down => @"llvm.nvvm.shfl.sync.down.i32"(m, val, off, pack).val,
        .up => @"llvm.nvvm.shfl.sync.up.i32"(m, val, off, pack).val,
        .bfly => @"llvm.nvvm.shfl.sync.bfly.i32"(m, val, off, pack).val,
        .idx => @"llvm.nvvm.shfl.sync.idx.i32"(m, val, off, pack).val,
    };
}

/// __shfl_down_sync: read `val` from lane (laneId + delta) within the warp.
pub fn shflDownSync(comptime T: type, mask: u32, val: T, delta: u32) T {
    return shfl(.down, T, mask, val, @bitCast(delta), 31);
}

/// __shfl_up_sync: read `val` from lane (laneId - delta).
pub fn shflUpSync(comptime T: type, mask: u32, val: T, delta: u32) T {
    return shfl(.up, T, mask, val, @bitCast(delta), 0);
}

/// __shfl_xor_sync: read `val` from lane (laneId ^ lane_mask).
pub fn shflXorSync(comptime T: type, mask: u32, val: T, lane_mask: u32) T {
    return shfl(.bfly, T, mask, val, @bitCast(lane_mask), 31);
}

/// __shfl_sync: read `val` from lane `src_lane`.
pub fn shflIdxSync(comptime T: type, mask: u32, val: T, src_lane: u32) T {
    return shfl(.idx, T, mask, val, @bitCast(src_lane), 31);
}

fn shfl(comptime kind: anytype, comptime T: type, mask: u32, val: T, off: i32, pack: i32) T {
    return switch (T) {
        i32, u32 => @bitCast(shflI32(kind, mask, @bitCast(val), off, pack)),
        f32 => @bitCast(shflI32(kind, mask, @bitCast(val), off, pack)),
        else => @compileError("shfl*Sync only supports i32/u32/f32, got " ++ @typeName(T)),
    };
}

/// atomicAdd on global or shared memory. T must be u32, u64 or f32.
/// Lowers to a single `atom.*.add.*` PTX instruction via Zig's @atomicRmw
/// (no NVVM intrinsic needed).
pub fn atomicAdd(comptime T: type, ptr: anytype, val: T) T {
    return switch (T) {
        u32, u64, f32 => @atomicRmw(T, ptr, .Add, val, .monotonic),
        else => @compileError("atomicAdd only supports u32/u64/f32, got " ++ @typeName(T)),
    };
}

/// Declare a shared-memory array, e.g.:
///   const tile = cuda.shared([64]f32);
///   tile()[tid] = x;
/// Returns a pointer into the .shared address space. Note: storage is one
/// per (kernel, call site) — declare once per kernel, at file scope if it
/// must be shared by several functions.
pub fn Shared(comptime T: type, comptime len: usize) type {
    return struct {
        var data: [len]T addrspace(.shared) = undefined;
    };
}

/// Instantiate to keep kernels alive across DCE without hitting the NVPTX
/// "aliasee must be a non-kernel function" alias bug. Usage at file scope:
///
///   comptime { _ = cuda.Keep(.{ vectorAdd, otherKernel }); }
///
/// Emits `pub export fn __zoxide_keep_kernels(i: usize) *const anyopaque`
/// which materializes pointers to the kernels as regular instructions.
pub fn Keep(comptime kernels: anytype) type {
    const fields = @typeInfo(@TypeOf(kernels)).@"struct".fields;
    return struct {
        pub export fn __zoxide_keep_kernels(i: usize) *const anyopaque {
            var ptrs: [fields.len]*const anyopaque = undefined;
            inline for (fields, 0..) |f, idx| {
                ptrs[idx] = @ptrCast(@field(kernels, f.name));
            }
            return ptrs[i % fields.len];
        }
    };
}
