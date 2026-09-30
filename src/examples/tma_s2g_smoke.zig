//! TMA s2g round trip: g2s a 2D tile into shared (tma_smoke's hand-written
//! path), then store it back out to a *different* tensor through the
//! generated `cp.async.bulk.tensor.*.global.shared::cta.tile` wrapper
//! (`gen.tma.cp_async_bulk_tensor_2d_s2g` — the direction LLVM *does*
//! expose, unlike g2s), and let the host compare the destination tensor
//! byte for byte.
//!
//! One claim: the generated s2g wrapper, driven by a host-encoded
//! descriptor, stores the tile at the coordinates it was given. The
//! destination tensor is pre-filled with a sentinel, and the store lands at
//! non-origin coordinates — a store that ignores coordinates, drops bytes,
//! or writes through the wrong descriptor leaves visible sentinel damage or
//! missing data.
//!
//! ## Completion is a bulk async-group, not an mbarrier
//!
//! g2s signals an mbarrier (`mbarrier.arrive.expect_tx` + try_wait) because
//! the copy engine deposits bytes the consumer then reads. s2g has no
//! mbarrier operand at all: completion is tracked by bulk async-group
//! semantics, `cp.async.bulk.commit_group` to close the group and
//! `cp.async.bulk.wait_group.read 0` to drain it. The `.read` form waits
//! only until the copy engine has finished *reading* shared memory (so the
//! tile buffer could be reused); it does not promise global visibility of
//! the store. For this smoke test the host's `cuCtxSynchronize` before the
//! byte comparison provides that; a pipelined kernel that overwrites the
//! shared tile is what `.read` is actually for. Both instructions come from
//! generated wrappers whose runtime i32 parameter constant-folds to the
//! immediate PTX wants, the same trick as `cp_async_wait_group` in
//! hgemm_mma2.
//!
//! The barrier wait before the store is also the ordering guarantee: the
//! mbarrier phase completes with acquire semantics, so the tile bytes are
//! visible to the subsequent async-proxy read. A `fence.proxy.async` is
//! issued anyway — it is what CUTLASS's tma_store_fence does before every
//! s2g, it costs one instruction, and it removes a visibility assumption
//! this test is not trying to make.

const cuda = @import("cuda");
const tma = cuda.tma;
const gen = cuda.gen;
const geo = @import("examples_abi").tma_smoke;
const abi = @import("examples_abi").tma_s2g_smoke;

/// 128-byte aligned: bulk tensor copies require it.
var tile_mem: [geo.tile_bytes]u8 align(128) addrspace(.shared) = undefined;
/// mbarrier state is a single 64-bit word.
var bar_mem: [1]u64 addrspace(.shared) = undefined;

/// Stage markers, same motivation as tma_smoke: a device-side failure that
/// cannot report its position costs a whole round trip.
pub const Stage = struct {
    pub const entered = 0;
    pub const initialised = 1;
    pub const g2s_issued = 2;
    pub const g2s_wait_result = 3;
    pub const s2g_issued = 4;
    pub const s2g_drained = 5;
};

pub fn tmaS2gSmoke(diag: [*]u32, desc_in: u64, desc_out: u64, x: i32, y: i32) callconv(.kernel) void {
    const tid = cuda.threadIdx().x;
    const bar = tma.Barrier.at(&bar_mem);

    if (tid == 0) diag[Stage.entered] = 1;

    // Phase 1: g2s, tma_smoke's path verbatim — this smoke assumes it and
    // does not re-claim it.
    if (tid == 0) {
        bar.init(1);
        tma.fenceProxyAsync();
        diag[Stage.initialised] = 1;
    }
    cuda.syncThreads();

    if (tid == 0) {
        bar.arriveExpectTx(geo.tile_bytes);
        tma.load2D(&tile_mem, desc_in, bar, x, y);
        diag[Stage.g2s_issued] = 1;
    }

    const budget = 1 << 10;
    var polls: u32 = 0;
    var ok = false;
    while (polls < budget) : (polls += 1) {
        if (bar.tryWaitOnce(0)) {
            ok = true;
            break;
        }
    }
    if (tid == 0) diag[Stage.g2s_wait_result] = if (ok) 1 else 0;
    if (!ok) return; // the host names the stage; nothing else is claimed

    // Phase 2: s2g. One thread issues; the copy is per-CTA.
    if (tid == 0) {
        tma.fenceProxyAsync(); // tma_store_fence: see the doc comment
        const out_desc: ?*anyopaque = @ptrFromInt(@as(usize, @intCast(desc_out)));
        gen.tma.cp_async_bulk_tensor_2d_s2g(@ptrCast(&tile_mem), out_desc, x, y, 0, false);
        diag[Stage.s2g_issued] = 1;
        gen.tma.cp_async_bulk_commit_group();
        gen.tma.cp_async_bulk_wait_group_read(0);
        diag[Stage.s2g_drained] = 1;
    }
}

comptime {
    cuda.abi.assertMatches(abi.signature, @TypeOf(tmaS2gSmoke));
    _ = cuda.Keep(.{&tmaS2gSmoke}).__zoxide_keep_kernels;
}
