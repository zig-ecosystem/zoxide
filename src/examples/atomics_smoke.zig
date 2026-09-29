//! Compile smoke for the atomic extension surface beyond packed_atomic.
//!
//! The cuda-oxide catalog has no plain atom.*/red.* global-memory entries at
//! all: filtering its 1029 intrinsics on atom/atomic/red yields only the
//! packed_atomic family (2 decls, covered by the packed_atomic smoke), the
//! warp-level redux family (16, covered by warpops_smoke), the TMA bulk-tensor
//! reductions (covered by tma_smoke) and tensormap.replace.swizzle_atomicity.
//! The general atomics surface therefore comes from two other places:
//!
//!   - Zig's @atomicRmw / @cmpxchgStrong, which the NVPTX backend lowers
//!     directly to atom.global / atom.shared — surfaced through the humanized
//!     wrappers in cuda.zig (add/min/max/and/or/xor/exch/cas);
//!   - hand-written inline asm for the red.* (reduction, no return value)
//!     forms, which the LLVM NVPTX backend never selects: even a discarded
//!     @atomicRmw result still emits `atom` with a dead destination register
//!     (see atomic_counter.ptx). red.* covers add/min/max/and/or/xor plus
//!     inc/dec, which have no @atomicRmw equivalent at all.
//!
//! The calls are formally correct, not a working algorithm. atomic_counter
//! covers real atomic semantics; this covers the surface.
const cuda = @import("cuda");
const abi = @import("examples_abi").atomics_smoke;

var shared_slots: [4]u32 addrspace(.shared) = undefined;

// diag layout (word offsets; the 64-bit slots must be even so the buffer
// stays 8-byte aligned):
//   0 add.u32  1 min.u32  2 max.u32  3 min.s32  4 max.s32
//   5 and.b32  6 or.b32   7 xor.b32
//   8 add.u64  10 min.u64 12 max.s64 14 max.u64 16 add.f64
//   20 add.f32 21 exch.u32 22 cas.u32 24 exch.u64
//   31 fold-out

fn slot64(diag: [*]u32, comptime word: usize) *u64 {
    return @ptrCast(@alignCast(diag + word));
}

fn slotI64(diag: [*]u32, comptime word: usize) *i64 {
    return @ptrCast(@alignCast(diag + word));
}

fn slotF64(diag: [*]u32, comptime word: usize) *f64 {
    return @ptrCast(@alignCast(diag + word));
}

fn slotI32(diag: [*]u32, comptime word: usize) *i32 {
    return @ptrCast(diag + word);
}

fn slotF32(diag: [*]u32, comptime word: usize) *f32 {
    return @ptrCast(diag + word);
}

/// atom.global: every @atomicRmw-backed op the humanized layer exposes.
pub fn atomGlobalSmoke(src: [*]const i32, diag: [*]u32) callconv(.kernel) void {
    const sv: i32 = src[0];
    const v: u32 = @bitCast(sv);
    const f: f32 = @floatFromInt(sv);
    const d: f64 = @floatFromInt(sv);
    var acc: u32 = 0;

    // add: u32 / u64 / f32 / f64.
    acc +%= cuda.atomicAdd(u32, &diag[0], v);
    acc +%= @truncate(cuda.atomicAdd(u64, slot64(diag, 8), v));
    acc +%= @bitCast(cuda.atomicAdd(f32, slotF32(diag, 20), f));
    acc +%= @truncate(@as(u64, @bitCast(cuda.atomicAdd(f64, slotF64(diag, 16), d))));

    // min / max, signed and unsigned, 32- and 64-bit.
    acc +%= cuda.atomicMin(u32, &diag[1], v);
    acc +%= cuda.atomicMax(u32, &diag[2], v);
    acc +%= @bitCast(cuda.atomicMin(i32, slotI32(diag, 3), sv));
    acc +%= @bitCast(cuda.atomicMax(i32, slotI32(diag, 4), sv));
    acc +%= @truncate(cuda.atomicMin(u64, slot64(diag, 10), v));
    acc +%= @truncate(cuda.atomicMax(u64, slot64(diag, 14), v));
    acc +%= @truncate(@as(u64, @bitCast(cuda.atomicMax(i64, slotI64(diag, 12), sv))));

    // bitwise and / or / xor.
    acc +%= cuda.atomicAnd(u32, &diag[5], v);
    acc +%= cuda.atomicOr(u32, &diag[6], v);
    acc +%= cuda.atomicXor(u32, &diag[7], v);

    // exch and cas.
    acc +%= cuda.atomicExch(u32, &diag[21], v);
    acc +%= @truncate(cuda.atomicExch(u64, slot64(diag, 24), v));
    if (cuda.atomicCas(u32, &diag[22], 0, v)) |_| acc +%= 1;

    diag[31] = acc;
}

/// atom.shared: the same wrappers applied to shared-memory pointers.
pub fn atomSharedSmoke(src: [*]const i32, diag: [*]u32) callconv(.kernel) void {
    const v: u32 = @bitCast(src[0]);
    var acc: u32 = 0;
    acc +%= cuda.atomicAdd(u32, &shared_slots[0], v);
    acc +%= cuda.atomicMin(u32, &shared_slots[1], v);
    acc +%= @bitCast(cuda.atomicMax(i32, @as(*addrspace(.shared) i32, @ptrCast(&shared_slots[2])), src[0]));
    acc +%= cuda.atomicExch(u32, &shared_slots[3], v);
    diag[0] = acc;
}

fn red(comptime tmpl: []const u8, addr: u64, val: u32) void {
    asm volatile (tmpl
        :
        : [addr] "l" (addr),
          [val] "r" (val),
        : .{ .memory = true }
    );
}

fn redF32(comptime tmpl: []const u8, addr: u64, val: f32) void {
    asm volatile (tmpl
        :
        : [addr] "l" (addr),
          [val] "f" (val),
        : .{ .memory = true }
    );
}

/// red.global (inline asm): the no-return reduction forms the LLVM backend
/// never emits, including inc/dec which have no @atomicRmw spelling.
pub fn redGlobalSmoke(src: [*]const i32, diag: [*]u32) callconv(.kernel) void {
    const sv: i32 = src[0];
    const v: u32 = @bitCast(sv);
    const f: f32 = @floatFromInt(sv);

    red("red.global.add.u32 [%[addr]], %[val];", @intFromPtr(&diag[0]), v);
    red("red.global.min.u32 [%[addr]], %[val];", @intFromPtr(&diag[1]), v);
    red("red.global.max.s32 [%[addr]], %[val];", @intFromPtr(&diag[4]), v);
    red("red.global.and.b32 [%[addr]], %[val];", @intFromPtr(&diag[5]), v);
    red("red.global.or.b32 [%[addr]], %[val];", @intFromPtr(&diag[6]), v);
    red("red.global.xor.b32 [%[addr]], %[val];", @intFromPtr(&diag[7]), v);
    // inc/dec take a wrap-around clamp as the operand.
    red("red.global.inc.u32 [%[addr]], %[val];", @intFromPtr(&diag[2]), v);
    red("red.global.dec.u32 [%[addr]], %[val];", @intFromPtr(&diag[3]), v);
    redF32("red.global.add.f32 [%[addr]], %[val];", @intFromPtr(&diag[20]), f);
}

comptime {
    cuda.abi.assertMatches(abi.signature, @TypeOf(atomGlobalSmoke));
    cuda.abi.assertMatches(abi.signature, @TypeOf(atomSharedSmoke));
    cuda.abi.assertMatches(abi.signature, @TypeOf(redGlobalSmoke));
    _ = cuda.Keep(.{ &atomGlobalSmoke, &atomSharedSmoke, &redGlobalSmoke }).__zoxide_keep_kernels;
}
