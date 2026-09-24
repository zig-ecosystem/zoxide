//! Minimal TMA copy: one 2D tile from global to shared, then straight back out
//! so the host can compare it byte for byte.
//!
//! No performance claim here. The question is only whether a descriptor built by
//! `Context.encodeTensorMap` drives `cp.async.bulk.tensor` correctly, since the
//! `g2s` direction is hand-written asm (LLVM exposes no intrinsic for it) and has
//! never been assembled by ptxas in this repository.
//!
//! ## Why the readout is linear
//!
//! The kernel copies shared memory out in address order rather than
//! reconstructing rows. With `swizzle = .none` that is exactly the row-major
//! tile, so the host can compare against the source directly.
//!
//! That also sets up the control. The same kernel, unchanged, run against a
//! descriptor built with `swizzle = .b128`, must produce *different* bytes —
//! because the hardware permutes 16-byte chunks within the tile. If both
//! descriptors gave the same answer, this test could not tell "TMA works" from
//! "the buffer happened to hold the right values". It is the same guard as
//! `pad_before` in `const_bank`.
//!
//! ## The byte count is the dangerous parameter
//!
//! `mbarrier.arrive.expect_tx` has to be told how many bytes to wait for. Wrong
//! low, the wait releases on partial data; wrong high, it never releases.
//! Neither faults. The count comes from `abi.tile_bytes`, shared with the host,
//! which computes it from the descriptor it actually encoded.
const cuda = @import("cuda");
const tma = cuda.tma;
const abi = @import("examples_abi").tma_smoke;

/// 128-byte aligned: bulk tensor copies require it, and swizzled ones assume a
/// 128-byte boundary for the permutation to land where the descriptor says.
var tile_mem: [abi.tile_bytes]u8 align(128) addrspace(.shared) = undefined;
/// mbarrier state is a single 64-bit word.
var bar_mem: [1]u64 addrspace(.shared) = undefined;

pub fn tmaSmoke(out: [*]u8, desc: u64, x: i32, y: i32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const bar = tma.Barrier.at(&bar_mem);

    // One thread issues: a bulk tensor copy is per-CTA, and the barrier must be
    // told the byte count exactly once.
    if (tid == 0) {
        bar.init(1);
    }
    cuda.syncThreads();

    if (tid == 0) {
        bar.arriveExpectTx(abi.tile_bytes);
        tma.load2D(&tile_mem, desc, bar, x, y);
    }
    // Every thread waits for phase 0; the copy flips the barrier when the full
    // byte count has landed.
    bar.wait(0);

    const src: [*]addrspace(.shared) const u8 = @ptrCast(&tile_mem);
    var i = tid;
    while (i < abi.tile_bytes) : (i += abi.block) {
        out[i] = src[i];
    }
}

comptime {
    cuda.abi.assertMatches(abi.signature, @TypeOf(tmaSmoke));
    _ = cuda.Keep(.{&tmaSmoke}).__zoxide_keep_kernels;
}
