//! Hopper warpgroup MMA (`wgmma.mma_async`) support.
//!
//! Unlike `mma.sync`, `wgmma` has **no LLVM intrinsic**: NVVM only exposes the
//! three control ops (`wgmma.fence`, `wgmma.commit_group`,
//! `wgmma.wait_group`), so `src/gen/` contains no `wgmma` MMA wrapper and the
//! instruction itself must be hand-written inline asm. Everything here is
//! gated on `sm_90a` (LLVM predicate `hasSM90a`); plain `sm_90` fails to
//! select.
//!
//! A warpgroup is 128 consecutive threads (4 warps). All 128 must execute the
//! same `wgmma`; the accumulator lives in registers spread across the
//! warpgroup, while A and B are read asynchronously by the tensor core
//! straight out of shared memory via 64-bit *matrix descriptors*.
//!
//! ## Shape ceiling: 15 asm outputs
//!
//! `m64nNk16` needs N/2 accumulator registers per thread, and every one must
//! be an inline-asm output operand. Zig's AstGen caps an `asm` expression at
//! 15 outputs (`lib/std/zig/AstGen.zig`, "too many asm outputs"), so:
//!
//!   * `m64n16k16` →  8 regs — OK, the largest shape that fits
//!   * `m64n32k16` → 16 regs — one over the limit, rejected
//!   * `m64n128k16` → 64 regs — far over
//!
//! The cap is a compiler-side ZIR encoding artifact, not a hardware limit; see
//! `docs/upstream-asm-output-limit.md`. Until it is lifted, wide-N wgmma is
//! unreachable from Zig and this module offers only `m64n16k16`, tiled over N.
//!
//! Descriptor bit layout and the canonical shared-memory layouts below follow
//! CUTLASS `cute` (BSD-3-Clause): `include/cute/arch/mma_sm90_desc.hpp` and
//! `include/cute/atom/mma_traits_sm90_gmma.hpp`.

/// Shared-memory swizzle mode (descriptor bits 62..63).
pub const Swizzle = enum(u2) {
    /// No swizzle ("interleaved"): plain core-matrix-packed layout.
    none = 0,
    b128 = 1,
    b64 = 2,
    b32 = 3,
};

/// Which logical dimension is contiguous in shared memory. Maps directly onto
/// the instruction's `imm-trans-a` / `imm-trans-b` operands.
pub const Major = enum(u1) {
    /// K contiguous (row-major A, k-major B). trans = 0.
    k = 0,
    /// M or N contiguous (column-major A, row-major B). trans = 1.
    mn = 1,
};

/// Build a 64-bit shared-memory matrix descriptor.
///
/// `smem_addr` is a shared-window byte address, 16-byte aligned.
/// `leading_byte_offset` / `stride_byte_offset` are byte strides between
/// 128-byte *core matrices*; see `coreMatrixOffset`.
pub inline fn descriptor(
    smem_addr: u32,
    leading_byte_offset: u32,
    stride_byte_offset: u32,
    comptime swizzle: Swizzle,
) u64 {
    // All three address/offset fields drop their 4 low bits (16-byte units).
    const start: u64 = (smem_addr >> 4) & 0x3fff;
    const lbo: u64 = (leading_byte_offset >> 4) & 0x3fff;
    const sbo: u64 = (stride_byte_offset >> 4) & 0x3fff;
    return start | (lbo << 16) | (sbo << 32) | (@as(u64, @intFromEnum(swizzle)) << 62);
}

/// Shared-window address of a `.shared` pointer, for `descriptor`.
pub inline fn smemAddr(ptr: anytype) u32 {
    return @truncate(@intFromPtr(ptr));
}

/// Byte offset of the 16-byte chunk holding row `row` of core matrix `block`
/// in a core-matrix-packed tile (`Swizzle.none`).
///
/// The tensor core reads shared memory as 8x8 *core matrices* of 128
/// contiguous bytes: 8 rows of 16 bytes (8 f16). A plain `[rows][cols]` array
/// is therefore not a legal wgmma operand unless its row pitch happens to be
/// 16 bytes — tiles must be packed as a sequence of core matrices.
///
/// With this packing the descriptor offsets are
/// `leading_byte_offset = 128` and
/// `stride_byte_offset = 16 * chunks_per_block`.
pub inline fn coreMatrixOffset(block: u32, row: u32, comptime chunks_per_block: u32) u32 {
    return block * (chunks_per_block * 16) + row * 16;
}

/// `wgmma.fence.sync.aligned`: orders prior register writes to the
/// accumulator against the asynchronous reads of subsequent `wgmma` ops.
/// Required once before the first `wgmma` of a group.
pub inline fn fence() void {
    asm volatile ("wgmma.fence.sync.aligned;" ::: .{ .memory = true });
}

/// `wgmma.commit_group.sync.aligned`: closes the current wgmma group so it can
/// be waited on.
pub inline fn commitGroup() void {
    asm volatile ("wgmma.commit_group.sync.aligned;" ::: .{ .memory = true });
}

/// `wgmma.wait_group.sync.aligned N`: block until at most `n` committed groups
/// are still in flight. `n` is an immediate, hence comptime.
pub inline fn waitGroup(comptime n: u32) void {
    asm volatile ("wgmma.wait_group.sync.aligned " ++ decimal(n) ++ ";" ::: .{ .memory = true });
}

/// Accumulator tile of one `m64n16k16` wgmma: 8 f32 per thread.
///
/// Register `i` maps to logical coordinate (CUTLASS `CLayout_64xN`):
///   m = warp*16 + lane/4 + 8*((i/2) % 2)
///   n = (lane%4)*2 + (i % 2) + (i/4)*8
pub const Acc64x16 = struct {
    d0: f32 = 0,
    d1: f32 = 0,
    d2: f32 = 0,
    d3: f32 = 0,
    d4: f32 = 0,
    d5: f32 = 0,
    d6: f32 = 0,
    d7: f32 = 0,

    /// Logical (m, n) within the 64x16 tile for register index `i`.
    pub inline fn coord(warp: u32, lane: u32, comptime i: u32) struct { m: u32, n: u32 } {
        return .{
            .m = warp * 16 + lane / 4 + 8 * ((i / 2) % 2),
            .n = (lane % 4) * 2 + (i % 2) + (i / 4) * 8,
        };
    }

    pub inline fn get(self: Acc64x16, comptime i: u32) f32 {
        return switch (i) {
            0 => self.d0,
            1 => self.d1,
            2 => self.d2,
            3 => self.d3,
            4 => self.d4,
            5 => self.d5,
            6 => self.d6,
            7 => self.d7,
            else => @compileError("Acc64x16 has 8 registers"),
        };
    }
};

/// `wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16`, both operands in
/// shared memory.
///
/// `scale_d = false` overwrites the accumulator instead of accumulating into
/// it — use it on the first K-tile to skip zeroing the registers.
///
/// Asm output operands must be plain identifiers in Zig, so the accumulator is
/// unpacked into locals and written back; at ReleaseFast the struct is
/// scalarized and this costs nothing.
pub inline fn mmaAsyncM64N16K16(
    acc: *Acc64x16,
    desc_a: u64,
    desc_b: u64,
    scale_d: bool,
    comptime major_a: Major,
    comptime major_b: Major,
) void {
    var d0 = acc.d0;
    var d1 = acc.d1;
    var d2 = acc.d2;
    var d3 = acc.d3;
    var d4 = acc.d4;
    var d5 = acc.d5;
    var d6 = acc.d6;
    var d7 = acc.d7;
    asm volatile (
        \\{
        \\.reg .pred p;
        \\setp.ne.b32 p, %[sd], 0;
        \\wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%[d0],%[d1],%[d2],%[d3],%[d4],%[d5],%[d6],%[d7]}, %[da], %[db], p, 1, 1,
    ++ " " ++ decimal(@intFromEnum(major_a)) ++ ", " ++ decimal(@intFromEnum(major_b)) ++ ";\n}"
        : [d0] "+f" (d0),
          [d1] "+f" (d1),
          [d2] "+f" (d2),
          [d3] "+f" (d3),
          [d4] "+f" (d4),
          [d5] "+f" (d5),
          [d6] "+f" (d6),
          [d7] "+f" (d7),
        : [da] "l" (desc_a),
          [db] "l" (desc_b),
          [sd] "r" (@as(u32, @intFromBool(scale_d))),
        : .{ .memory = true });
    acc.* = .{ .d0 = d0, .d1 = d1, .d2 = d2, .d3 = d3, .d4 = d4, .d5 = d5, .d6 = d6, .d7 = d7 };
}

/// CUTLASS's `warpgroup_fence_operand`: an empty asm taking the value as a
/// read-write operand, which forces the compiler to treat it as live here.
///
/// Needed because an in-flight `wgmma` reads its A and D registers *after* the
/// instruction issues, but the register allocator has no idea. Once the last
/// wgmma of a group has been emitted, LLVM considers that group's A fragment
/// dead and is free to recycle the physical registers for the next stage's
/// fragment — silently undoing any double buffering the source expresses. Doing
/// this to both fragment buffers at the loop's end keeps them simultaneously
/// live across the back edge, so they must land in disjoint registers.
///
/// The PTX ISA is ambiguous about whether `wgmma.fence` alone would legalise
/// overwriting a fragment that an outstanding group is reading. This does not
/// rely on that reading: the fragments are genuinely double-buffered, and this
/// only stops the compiler from collapsing them back together.
pub inline fn fenceFragment(frag: *[4]u32) void {
    inline for (0..4) |i| {
        var x = frag[i];
        asm volatile (""
            : [r] "+r" (x)
            :
            : .{});
        frag[i] = x;
    }
}

/// `wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16` with **A supplied from
/// registers** and only B read from shared memory.
///
/// This is the way out of the n16 operand-traffic penalty without a wider N.
/// In the all-shared form each of the 8 wgmma covering a 128-wide tile re-reads
/// the whole A tile, so one K-stage pulls 8*(2048+512) = 20480 B out of shared
/// memory. Loading A once into registers instead leaves 2048 + 8*512 = 6144 B —
/// exactly what a single `m64n128k16` would have read, and 3.3x less than the
/// all-shared n16 form. The accumulator is still 8 registers, so this stays
/// inside Zig's 15-output asm limit.
///
/// `a` is this thread's 4-register A fragment for the 64x16 tile, in CUTLASS
/// `ALayout_64x16` order. With `m = warp*16 + lane/4` and `k0 = (lane%4)*2`
/// that is:
///   a[0] = A[m  ][k0, k0+1]    a[1] = A[m+8][k0,   k0+1]
///   a[2] = A[m  ][k0+8, k0+9]  a[3] = A[m+8][k0+8, k0+9]
/// which is the `mma.sync m16n8k16` A-fragment layout applied to each warp's
/// own 16 rows — i.e. precisely what one `ldmatrix.x4` returns.
///
/// A is required to be K-major here; there is no `trans_a` operand because the
/// fragment layout is fixed when A comes from registers.
///
/// Hazard note for pipelined callers: the async op reads `a` and `acc` after it
/// issues, so neither may be overwritten while the group is still in flight.
/// With `waitGroup(1)` the previous stage is still running, so a caller must
/// alternate between two A fragment register sets.
pub inline fn mmaAsyncM64N16K16Rs(
    acc: *Acc64x16,
    a: [4]u32,
    desc_b: u64,
    scale_d: bool,
    comptime major_b: Major,
) void {
    var d0 = acc.d0;
    var d1 = acc.d1;
    var d2 = acc.d2;
    var d3 = acc.d3;
    var d4 = acc.d4;
    var d5 = acc.d5;
    var d6 = acc.d6;
    var d7 = acc.d7;
    asm volatile (
        \\{
        \\.reg .pred p;
        \\setp.ne.b32 p, %[sd], 0;
        \\wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%[d0],%[d1],%[d2],%[d3],%[d4],%[d5],%[d6],%[d7]}, {%[a0],%[a1],%[a2],%[a3]}, %[db], p, 1, 1,
    ++ " " ++ decimal(@intFromEnum(major_b)) ++ ";\n}"
        : [d0] "+f" (d0),
          [d1] "+f" (d1),
          [d2] "+f" (d2),
          [d3] "+f" (d3),
          [d4] "+f" (d4),
          [d5] "+f" (d5),
          [d6] "+f" (d6),
          [d7] "+f" (d7),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [db] "l" (desc_b),
          [sd] "r" (@as(u32, @intFromBool(scale_d))),
        : .{ .memory = true });
    acc.* = .{ .d0 = d0, .d1 = d1, .d2 = d2, .d3 = d3, .d4 = d4, .d5 = d5, .d6 = d6, .d7 = d7 };
}

/// Comptime decimal rendering, for splicing immediates into asm templates.
fn decimal(comptime n: u32) []const u8 {
    comptime {
        if (n == 0) return "0";
        var buf: [10]u8 = undefined;
        var i: usize = buf.len;
        var v = n;
        while (v != 0) {
            i -= 1;
            buf[i] = '0' + @as(u8, @intCast(v % 10));
            v /= 10;
        }
        const out = buf[i..].*;
        return &out;
    }
}
