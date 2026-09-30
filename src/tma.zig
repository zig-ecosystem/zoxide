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
//!     $ grep -c g2s src/gen/intrinsics.zig src/gen/intrinsics_asm.zig
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
//! The generated `mbarrier_*` asm wrappers follow the same rule (src/gen.zig
//! narrows bracketed shared-address operands to u32).
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
    /// A loop rather than `mbarrier.test_wait` because `try_wait` is the form that
    /// lets the SM sleep between polls.
    ///
    /// Unbounded, so a barrier that never completes hangs the launch. That is the
    /// right shape for production code and the wrong shape for a test — use
    /// `tryWaitFor` where a failure has to be reportable.
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

    /// One `try_wait` poll. True if the barrier has reached `parity`.
    ///
    /// A single self-contained instruction, which matters: the previous version
    /// put the retry loop inside the asm block and wrote its output register
    /// before reading `%[a]` and `%[p]` on later iterations. Nothing stops the
    /// compiler from assigning the output the same physical register as an input —
    /// inline asm is assumed to read all inputs before writing any output — so the
    /// barrier address or the parity could be clobbered mid-loop. It would have
    /// needed `"=&r"`; not having the loop in asm at all is better.
    pub inline fn tryWaitOnce(self: Barrier, parity: u32) bool {
        return asm volatile (
            \\{
            \\.reg .pred %pw;
            \\mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 %pw, [%[a]], %[p];
            \\selp.b32 %[r], 1, 0, %pw;
            \\}
            : [r] "=r" (-> u32),
            : [a] "r" (self.addr),
              [p] "r" (parity),
            : .{ .memory = true }) != 0;
    }

    /// Poll at most `max_polls` times. Returns false on giving up.
    ///
    /// The loop is Zig's, not PTX's, so there are no labels to collide and no
    /// register aliasing to reason about. An unbounded wait on a device turns a
    /// wrong byte count or a missing fence into a hung process with no diagnostic.
    pub inline fn tryWaitFor(self: Barrier, parity: u32, max_polls: u32) bool {
        var n = max_polls;
        while (n > 0) : (n -= 1) {
            if (self.tryWaitOnce(parity)) return true;
        }
        return false;
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

/// 1D form. Same template as 2D/3D with one coordinate; hand-written because
/// LLVM exposes no g2s intrinsic (see the module header).
pub inline fn load1D(dst: anytype, desc: u64, bar: Barrier, x: i32) void {
    asm volatile (
        \\cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        \\  [%[d]], [%[t], {%[x]}], [%[b]];
        :
        : [d] "r" (smemAddr(dst)),
          [t] "l" (desc),
          [x] "r" (x),
          [b] "r" (bar.addr),
        : .{ .memory = true });
}

/// 4D form, for a tensor with two outer (batch/K-block) dimensions.
pub inline fn load4D(dst: anytype, desc: u64, bar: Barrier, x: i32, y: i32, z: i32, w: i32) void {
    asm volatile (
        \\cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        \\  [%[d]], [%[t], {%[x], %[y], %[z], %[w]}], [%[b]];
        :
        : [d] "r" (smemAddr(dst)),
          [t] "l" (desc),
          [x] "r" (x),
          [y] "r" (y),
          [z] "r" (z),
          [w] "r" (w),
          [b] "r" (bar.addr),
        : .{ .memory = true });
}

/// 5D form — the maximum rank `CUtensorMap` supports.
pub inline fn load5D(dst: anytype, desc: u64, bar: Barrier, x: i32, y: i32, z: i32, w: i32, v: i32) void {
    asm volatile (
        \\cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        \\  [%[d]], [%[t], {%[x], %[y], %[z], %[w], %[v]}], [%[b]];
        :
        : [d] "r" (smemAddr(dst)),
          [t] "l" (desc),
          [x] "r" (x),
          [y] "r" (y),
          [z] "r" (z),
          [w] "r" (w),
          [v] "r" (v),
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

// --- L2 tile prefetch -------------------------------------------------------
//
// cp.async.bulk.prefetch.tensor is fire-and-forget: no mbarrier, no bulk
// group — it warms L2 for a later load and reports nothing. NVVM exposes one
// intrinsic per dimensionality with a (cache_hint, use_hint) trailing pair;
// the generator's per-catalog-id naming therefore mapped the plain form for
// some dims and the cache_hint form for others, leaving six catalog entries
// nominally missing (1d/5d/gather4 plain, 2d/3d/4d cache_hint). They are all
// the same intrinsics — these wrappers select the form by the flag. Coords
// are elements, innermost first, same as the descriptor's box.

fn descPtr(desc: u64) ?*anyopaque {
    return @ptrFromInt(@as(usize, @intCast(desc)));
}

pub inline fn prefetch1D(desc: u64, x: i32) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_1d_l2_cache_hint(descPtr(desc), x, 0, false);
}
pub inline fn prefetch1DCacheHint(desc: u64, x: i32, hint: u64) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_1d_l2_cache_hint(descPtr(desc), x, @bitCast(hint), true);
}
pub inline fn prefetch2D(desc: u64, x: i32, y: i32) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_2d_l2(descPtr(desc), x, y, 0, false);
}
pub inline fn prefetch2DCacheHint(desc: u64, x: i32, y: i32, hint: u64) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_2d_l2(descPtr(desc), x, y, @bitCast(hint), true);
}
pub inline fn prefetch3D(desc: u64, x: i32, y: i32, z: i32) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_3d_l2(descPtr(desc), x, y, z, 0, false);
}
pub inline fn prefetch3DCacheHint(desc: u64, x: i32, y: i32, z: i32, hint: u64) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_3d_l2(descPtr(desc), x, y, z, @bitCast(hint), true);
}
pub inline fn prefetch4D(desc: u64, x: i32, y: i32, z: i32, w: i32) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_4d_l2(descPtr(desc), x, y, z, w, 0, false);
}
pub inline fn prefetch4DCacheHint(desc: u64, x: i32, y: i32, z: i32, w: i32, hint: u64) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_4d_l2(descPtr(desc), x, y, z, w, @bitCast(hint), true);
}
pub inline fn prefetch5D(desc: u64, x: i32, y: i32, z: i32, w: i32, v: i32) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_5d_l2_cache_hint(descPtr(desc), x, y, z, w, v, 0, false);
}
pub inline fn prefetch5DCacheHint(desc: u64, x: i32, y: i32, z: i32, w: i32, v: i32, hint: u64) void {
    cuda.gen.tma.cp_async_bulk_prefetch_tensor_5d_l2_cache_hint(descPtr(desc), x, y, z, w, v, @bitCast(hint), true);
}
/// gather4: four row indices per issue. Hand-written asm, same reason as
/// g2s: the NVVM intrinsic exists but the NVPTX backend never lowers it —
/// it survives into PTX as an unresolved extern call (verified 2026-09-30;
/// that is also why the generator's per-id mapping could not have produced
/// a working wrapper either way).
pub inline fn prefetchGather4_2D(desc: u64, r0: i32, r1: i32, r2: i32, r3: i32, col: i32) void {
    asm volatile (
        \\cp.async.bulk.prefetch.tensor.gather4.2d.L2.global.tile
        \\  [%[t], {%[c], %[r0], %[r1], %[r2], %[r3]}];
        :
        : [t] "l" (desc),
          [c] "r" (col),
          [r0] "r" (r0),
          [r1] "r" (r1),
          [r2] "r" (r2),
          [r3] "r" (r3),
        : .{ .memory = true });
}
pub inline fn prefetchGather4_2DCacheHint(desc: u64, r0: i32, r1: i32, r2: i32, r3: i32, col: i32, hint: u64) void {
    asm volatile (
        \\cp.async.bulk.prefetch.tensor.gather4.2d.L2.global.tile.L2::cache_hint
        \\  [%[t], {%[c], %[r0], %[r1], %[r2], %[r3]}], %[h];
        :
        : [t] "l" (desc),
          [c] "r" (col),
          [r0] "r" (r0),
          [r1] "r" (r1),
          [r2] "r" (r2),
          [r3] "r" (r3),
          [h] "l" (hint),
        : .{ .memory = true });
}

comptime {
    _ = cuda;
}
