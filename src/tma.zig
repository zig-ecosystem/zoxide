//! TMA (Tensor Memory Accelerator) — bulk tensor copies driven by a descriptor.
//!
//! The point is where address generation happens. `cp.async` needs the kernel to
//! compute every address in registers; `hgemm_wgmma3` spends 144 instructions on
//! address arithmetic against 14 actual loads, about 27% of the kernel. TMA hands
//! the copy engine a descriptor built once on the host plus a set of tile
//! coordinates, and the hardware does the rest.
//!
//! ## Why this is hand-written asm
//!
//! LLVM exposes `cp.async.bulk.tensor` intrinsics only for `s2g`
//! (shared to global) and the `reduce_*` variants. The direction a GEMM needs,
//! `g2s`, has no intrinsic, so the generator produced none either:
//!
//!     $ grep -c g2s src/gen/intrinsics.zig src/gen/instrinsics_asm.zig
//!     0
//!     0
//!
//! Same situation as `wgmma.mma_async`. So the copy is written out directly here.
//!
//! ## Completion is tracked by an mbarrier, not by a counter
//!
//! `cp.async` has `commit_group`/`wait_group`, which count outstanding groups.
//! Bulk tensor copies instead signal an mbarrier, and the barrier has to be told
//! in advance how many bytes to expect (`mbarrier.arrive.expect_tx`). Getting
//! that byte count wrong does not fault: the wait either returns early with
//! partial data, or never completes. `TensorMap.tileBytes()` on the host side
//! exists so the number comes from the descriptor rather than being written twice.
//!
//! ## Shared addresses are 32-bit
//!
//! Shared-memory operands in `.shared::cta` are addresses in the shared window,
//! which is 32-bit. Same convention as `wgmma.smemAddr`: truncate the pointer.
//! The generated `mbarrier_*` wrappers in `src/gen` take `u64` for these, which
//! is why this module does not use them.
const cuda = @import("cuda.zig");

/// Shared-memory address as the hardware wants it: 32-bit offset into the shared
/// window. Matching `wgmma.smemAddr`.
pub inline fn smemAddr(ptr: anytype) u32 {
    return @truncate(@intFromPtr(ptr));
}

/// An mbarrier in shared memory, used here to signal completion of bulk copies.
///
/// Phase parity is the part that trips people up. `mbarrier.try_wait.parity`
/// waits for the barrier to flip *into* the given phase, so the caller has to
/// track a toggling bit across pipeline stages. It is kept explicit rather than
/// hidden because a stale parity does not fault — it waits forever, or returns
/// immediately on data that is not there yet.
pub const Barrier = struct {
    addr: u32,

    pub inline fn at(ptr: anytype) Barrier {
        return .{ .addr = smemAddr(ptr) };
    }

    /// Initialise for `count` arrivals. One thread only; follow with a barrier so
    /// the rest of the block does not race ahead.
    pub inline fn init(self: Barrier, count: u32) void {
        asm volatile ("mbarrier.init.shared::cta.b64 [%[a]], %[c];"
            :
            : [a] "r" (self.addr),
              [c] "r" (count),
            : .{ .memory = true });
    }

    pub inline fn invalidate(self: Barrier) void {
        asm volatile ("mbarrier.inval.shared::cta.b64 [%[a]];"
            :
            : [a] "r" (self.addr),
            : .{ .memory = true });
    }

    /// Arrive and declare how many bytes of bulk copy this phase should wait for.
    /// One thread per barrier per phase.
    ///
    /// `bytes` must match what the copies actually deliver. Too few and the wait
    /// releases on incomplete data; too many and it never releases. Neither
    /// faults.
    pub inline fn arriveExpectTx(self: Barrier, bytes: u32) void {
        asm volatile ("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%[a]], %[b];"
            :
            : [a] "r" (self.addr),
              [b] "r" (bytes),
            : .{ .memory = true });
    }

    /// Arrive without an expected byte count, for the threads that are not
    /// issuing copies.
    pub inline fn arrive(self: Barrier) void {
        asm volatile ("mbarrier.arrive.release.cta.shared::cta.b64 _, [%[a]];"
            :
            : [a] "r" (self.addr),
            : .{ .memory = true });
    }

    /// Spin until the barrier reaches phase `parity`.
    ///
    /// A loop rather than `mbarrier.test_wait` because `try_wait` is the form
    /// that lets the SM sleep between polls.
    pub inline fn wait(self: Barrier, parity: u32) void {
        asm volatile (
            \\{
            \\.reg .pred %pw;
            \\$zwait:
            \\mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 %pw, [%[a]], %[p];
            \\@!%pw bra $zwait;
            \\}
            :
            : [a] "r" (self.addr),
              [p] "r" (parity),
            : .{ .memory = true });
    }
};

/// True on exactly one thread of the warp, for issuing a per-CTA operation
/// without a `threadIdx == 0` branch that also serialises the rest of the block.
pub inline fn electOne() bool {
    return asm volatile (
        \\{ .reg .pred %pe; .reg .b32 %me;
        \\  elect.sync %me|%pe, -1;
        \\  selp.b32 %[r], 1, 0, %pe;
        \\}
        : [r] "=r" (-> u32),
        :
        : .{}) != 0;
}

/// Make prior shared-memory writes visible to the async proxy (the copy engine
/// and wgmma). Needed between filling shared memory by ordinary stores and
/// letting an async operation read it.
pub inline fn fenceProxyAsync() void {
    asm volatile ("fence.proxy.async.shared::cta;" ::: .{ .memory = true });
}

/// Issue a 2D tile copy from global to shared, completing on `bar`.
///
/// `desc` is the device address of the `CUtensorMap` built by
/// `Context.encodeTensorMap`. Coordinates are in elements and innermost-first,
/// the same order as the descriptor's `box`.
///
/// Must be issued by a single thread — the copy is per-CTA, not per-thread.
/// `electOne()` below picks one.
pub inline fn load2D(dst: anytype, desc: u64, bar: Barrier, x: i32, y: i32) void {
    asm volatile (
        \\cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        \\  [%[d]], [%[t], {%[x], %[y]}], [%[b]];
        :
        : [d] "r" (smemAddr(dst)),
          [t] "l" (desc),
          [x] "r" (x),
          [y] "r" (y),
          [b] "r" (bar.addr),
        : .{ .memory = true });
}

/// 3D form, for a tensor with a batch or K-block dimension.
pub inline fn load3D(dst: anytype, desc: u64, bar: Barrier, x: i32, y: i32, z: i32) void {
    asm volatile (
        \\cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        \\  [%[d]], [%[t], {%[x], %[y], %[z]}], [%[b]];
        :
        : [d] "r" (smemAddr(dst)),
          [t] "l" (desc),
          [x] "r" (x),
          [y] "r" (y),
          [z] "r" (z),
          [b] "r" (bar.addr),
        : .{ .memory = true });
}

/// Hint the descriptor into cache. Worth issuing once early when the same
/// descriptor drives many copies, since the first access otherwise stalls on a
/// cold read of the 128-byte descriptor.
pub inline fn prefetchDescriptor(desc: u64) void {
    asm volatile ("prefetch.tensormap [%[t]];"
        :
        : [t] "l" (desc),
        : .{ .memory = true });
}

comptime {
    _ = cuda;
}
