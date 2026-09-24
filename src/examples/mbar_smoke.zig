//! mbarrier in isolation, with TMA removed from the picture.
//!
//! `tma_smoke` hung four times. Three explanations were tried and the last one —
//! that the poll budget alone accounted for it — was wrong too, because a
//! 1024-poll kernel still failed to return. At that point the honest move is to
//! stop guessing which part of a two-part mechanism is broken and bisect it.
//!
//! Three kernels, each adding one thing:
//!
//!   barOnly        init, arrive, wait. No TMA, no expect_tx. If this hangs, the
//!                  barrier code in src/tma.zig is wrong and TMA is innocent.
//!   barExpectZero  init, arrive.expect_tx with a zero byte count, wait. Isolates
//!                  the expect_tx path: a barrier with nothing to transfer should
//!                  complete on the arrival alone.
//!   barTwoPhase    two consecutive phases, to check the parity convention. A
//!                  wrong parity is the classic mbarrier mistake and would show up
//!                  as phase 0 working and phase 1 hanging.
//!
//! Every wait is bounded and every kernel writes what it observed, so a failure
//! says which step it was and what the barrier state word held.
const cuda = @import("cuda");
const tma = cuda.tma;
const abi = @import("examples_abi").mbar_smoke;

var bar_mem: [1]u64 addrspace(.shared) = undefined;
var bar_mem2: [1]u64 addrspace(.shared) = undefined;

inline fn state(p: *addrspace(.shared) const u64) u64 {
    const v: *addrspace(.shared) volatile const u64 = @ptrCast(p);
    return v.*;
}

/// Poll budget. Small: a barrier that is going to complete does so almost
/// immediately, and a failing `mbarrier.try_wait` suspends the thread for an
/// implementation-defined interval, so a large budget does not fail fast.
const budget = 1 << 10;

inline fn waitBounded(bar: tma.Barrier, parity: u32) struct { ok: bool, polls: u32 } {
    var n: u32 = 0;
    while (n < budget) : (n += 1) {
        if (bar.tryWaitOnce(parity)) return .{ .ok = true, .polls = n };
    }
    return .{ .ok = false, .polls = n };
}

/// init -> arrive -> wait. The minimum that must work before TMA can be blamed
/// for anything.
pub fn barOnly(diag: [*]u32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const bar = tma.Barrier.at(&bar_mem);
    if (tid == 0) {
        diag[0] = 1;
        bar.init(abi.block);
        diag[1] = @truncate(state(&bar_mem[0]));
    }
    cuda.syncThreads();

    // Every thread arrives, so the count is the block size.
    bar.arrive();
    const r = waitBounded(bar, 0);
    if (tid == 0) {
        diag[2] = if (r.ok) 1 else 0;
        diag[3] = r.polls;
        diag[4] = @truncate(state(&bar_mem[0]));
    }
}

/// Adds expect_tx with nothing to transfer. Completion then depends only on the
/// arrivals, so this separates "expect_tx is mis-encoded" from "the copy never
/// delivered".
pub fn barExpectZero(diag: [*]u32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const bar = tma.Barrier.at(&bar_mem2);
    if (tid == 0) {
        diag[0] = 1;
        bar.init(1);
        tma.fenceProxyAsync();
        diag[1] = @truncate(state(&bar_mem2[0]));
        bar.arriveExpectTx(0);
        diag[5] = @truncate(state(&bar_mem2[0]));
    }
    cuda.syncThreads();

    const r = waitBounded(bar, 0);
    if (tid == 0) {
        diag[2] = if (r.ok) 1 else 0;
        diag[3] = r.polls;
        diag[4] = @truncate(state(&bar_mem2[0]));
    }
}

/// Two phases back to back. Phase 0 waits on parity 0, phase 1 on parity 1.
///
/// Worth its own kernel because a wrong parity convention is the standard
/// mbarrier error and it hides in a single-phase test: the first wait would pass
/// and only a pipelined kernel would deadlock. S2 has three pipeline stages, so
/// this has to be right before that is attempted.
pub fn barTwoPhase(diag: [*]u32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const bar = tma.Barrier.at(&bar_mem);
    if (tid == 0) {
        diag[0] = 1;
        bar.init(abi.block);
    }
    cuda.syncThreads();

    bar.arrive();
    const r0 = waitBounded(bar, 0);
    if (tid == 0) {
        diag[2] = if (r0.ok) 1 else 0;
        diag[3] = r0.polls;
    }
    if (!r0.ok) return;

    bar.arrive();
    const r1 = waitBounded(bar, 1);
    if (tid == 0) {
        diag[6] = if (r1.ok) 1 else 0;
        diag[7] = r1.polls;
        diag[4] = @truncate(state(&bar_mem[0]));
    }
}

comptime {
    cuda.abi.assertMatches(abi.signature, @TypeOf(barOnly));
    cuda.abi.assertMatches(abi.signature, @TypeOf(barExpectZero));
    cuda.abi.assertMatches(abi.signature, @TypeOf(barTwoPhase));
    _ = cuda.Keep(.{ &barOnly, &barExpectZero, &barTwoPhase }).__zoxide_keep_kernels;
}
