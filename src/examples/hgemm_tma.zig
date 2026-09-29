//! HGEMM v6 (S2 of the TMA line): `hgemm_wgmma3` with the cp.async load path
//! replaced by TMA (`cp.async.bulk.tensor` driven by host-built descriptors).
//! Tile shape, pipeline depth and the whole wgmma compute path are identical to
//! v5 — only the way A and B tiles get into shared memory changes. The point is
//! H1: TMA moves address generation into the copy engine, so the PTX should
//! lose most of the 144 address-arithmetic instructions v5 spends against 14
//! actual loads.
//!
//! ## Shared-memory layout: unchanged from v5, and why
//!
//! v5's B tile is *core-matrix packed* — offset(k, nb) = nb*256 + k*16 bytes —
//! because that is what the wgmma MN-major descriptor (lbo 128, sbo 256,
//! Swizzle.none) reads. A single TMA box over the k×n tile would deliver plain
//! [k][n] row-major, which that descriptor cannot describe (TMA fills shared
//! memory linearly in box order; the core-matrix packing interleaves n-blocks
//! and k-rows).
//!
//! The chosen compromise: keep the packed layout exactly, and produce it with
//! 16 separate 2D loads per stage, one per 8-column n-block (box 8×16 f16 =
//! 256 B, landing at nb*256). Each sub-tile is contiguous in global memory per
//! k-row, so this needs no swizzle and no exotic descriptor geometry. The
//! wgmma descriptor path is untouched — the risk this kernel is meant to
//! retire is in the load machinery, not in re-validating a new smem layout.
//! The canonical pairing (TMA 128B swizzle + wgmma Swizzle.b128 descriptor)
//! would cut 16 issues to 1 and is the follow-up if H1 holds.
//!
//! A is simpler: v5 stores it plainly as [64][16] f16 (32 B row pitch) for
//! `ldmatrix`, which is exactly a 2D row-major tile — one box {16, 64} load.
//!
//! ## Completion: mbarrier instead of commit_group/wait_group
//!
//! One barrier per pipeline stage. Thread 0 arrives with
//! `arriveExpectTx(abi.stage_bytes)` — the byte count comes from the shared
//! ABI constants, which the host cross-checks against the descriptors it
//! actually encoded (`TensorMap.tileBytes()`); the number is written once, in
//! `examples_abi`. All threads then wait on the phase parity, bounded
//! (`tryWaitFor`) as in `tma_smoke`: a wrong byte count or a descriptor the
//! copy engine rejects must surface as a poisoned result, not a hung launch.
//! On giving up, the kernel writes NaN to its C tile — the harness poisons C
//! with 0xff and verifies, so either way the failure is loud and named.

const cuda = @import("cuda");
const tma = cuda.tma;
const wg = cuda.wgmma;
const abi = @import("examples_abi").hgemm_tma;

pub const tile_m = abi.tile_m;
pub const tile_n = abi.tile_n;
pub const k_slice = abi.k_slice;
pub const threads = abi.threads;
pub const stages = abi.stages;
pub const a_bufs = 2; // A fragment register sets, matching wgmma in-flight depth
pub const n_tiles = tile_n / 16;

var as_mem: [stages][abi.a_tile_bytes]u8 align(128) addrspace(.shared) = undefined;
var bs_mem: [stages][abi.b_tile_bytes]u8 align(128) addrspace(.shared) = undefined;
var bars: [stages]u64 addrspace(.shared) = undefined;

/// Issue one pipeline stage's loads. Single-thread only: a bulk tensor copy is
/// per-CTA and the barrier's expect_tx must be declared exactly once.
///
/// Coordinates are in elements, innermost-first, matching the descriptors the
/// host encoded: A is (k offset, m offset), each B sub-tile is (n offset, k
/// offset).
fn issueTileLoad(buf: u32, desc_a: u64, desc_b: u64, block_row: u32, block_col: u32, k0: u32) void {
    const bar = tma.Barrier.at(&bars[buf]);
    bar.arriveExpectTx(abi.stage_bytes);
    tma.load2D(&as_mem[buf], desc_a, bar, @intCast(k0), @intCast(block_row));
    inline for (0..abi.b_subtiles) |nb| {
        tma.load2D(&bs_mem[buf][nb * abi.b_subtile_bytes], desc_b, bar, @intCast(block_col + nb * 8), @intCast(k0));
    }
}

/// One `ldmatrix.x4` gives this warp its whole A fragment for the 64x16 tile.
/// Identical to v5: the A smem layout TMA produces is the same plain [64][16].
fn loadAFrag(buf: u32, warp: u32, lane: u32) [4]u32 {
    const matrix = lane / 8;
    const row = lane % 8;
    const grow = warp * 16 + row + (matrix % 2) * 8;
    const koff = (matrix / 2) * 8;
    const addr: [*]addrspace(.shared) u8 = @ptrCast(&as_mem[buf][grow * 32 + koff * 2]);
    const r = cuda.gen.matrix.ldmatrix_m8n8_x4_b16(addr);
    return .{ @bitCast(r.f0), @bitCast(r.f1), @bitCast(r.f2), @bitCast(r.f3) };
}

/// One K-stage of compute, unchanged from v5: load this warp's A fragment,
/// then 8 wgmma stepping the B descriptor 16 columns at a time.
inline fn issueStage(
    sbuf: u32,
    acc: *[n_tiles]wg.Acc64x16,
    frag: *[4]u32,
    warp: u32,
    lane: u32,
    scale_d: bool,
) void {
    frag.* = loadAFrag(sbuf, warp, lane);
    wg.fence();
    const b_base = wg.smemAddr(&bs_mem[sbuf][0]);
    inline for (0..n_tiles) |t| {
        const desc_b = wg.descriptor(b_base + @as(u32, t) * 512, 128, 256, .none);
        wg.mmaAsyncM64N16K16Rs(&acc[t], frag.*, desc_b, scale_d, .mn);
    }
    wg.commitGroup();
}

/// `a` and `b` are not read by the kernel — the descriptors carry the global
/// addresses — but stay in the signature so the launch spells out which
/// tensors the descriptors were built over.
pub fn hgemmTma(a: [*]const f16, b: [*]const f16, c: [*]f32, n: u32, desc_a: u64, desc_b: u64) callconv(.kernel) void {
    _ = a;
    _ = b;
    const tid = cuda.threadIdx().x;
    const warp = tid / 32;
    const lane = cuda.laneId();

    const block_row = cuda.blockIdx().y * tile_m;
    const block_col = cuda.blockIdx().x * tile_n;

    var acc: [n_tiles]wg.Acc64x16 = @splat(.{});
    var afrag_even: [4]u32 = @splat(0);
    var afrag_odd: [4]u32 = @splat(0);

    const ktiles = n / k_slice;

    if (tid == 0) {
        inline for (0..stages) |s| tma.Barrier.at(&bars[s]).init(1);
        // Not optional (tma_smoke hung without it): the async proxy is a
        // separate memory consumer and is not guaranteed to observe the
        // initialised barriers otherwise. init → fence → issue.
        tma.fenceProxyAsync();
        tma.prefetchDescriptor(desc_a);
        tma.prefetchDescriptor(desc_b);
    }
    cuda.syncThreads();

    if (tid == 0) {
        issueTileLoad(0, desc_a, desc_b, block_row, block_col, 0);
        if (ktiles > 1) issueTileLoad(1, desc_a, desc_b, block_row, block_col, k_slice);
    }

    // Poll budget per stage wait. A stage is 6 KB and lands in a handful of
    // polls when the machinery works; when it does not, the kernel gives up
    // and poisons C rather than hanging the launch (tma_smoke's lesson: an
    // unbounded device-side wait is an undiagnosable failure).
    const wait_budget: u32 = 1 << 16;

    var poisoned = false;
    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        const sbuf = kt % stages;
        // The barrier completes one phase per round of the stage rotation.
        const parity: u32 = (kt / stages) & 1;
        if (!tma.Barrier.at(&bars[sbuf]).tryWaitFor(parity, wait_budget)) {
            poisoned = true;
            break;
        }

        if (kt % a_bufs == 0) {
            issueStage(sbuf, &acc, &afrag_even, warp, lane, kt != 0);
        } else {
            issueStage(sbuf, &acc, &afrag_odd, warp, lane, kt != 0);
        }

        // As v5: waitGroup(1) retires stage kt-1's wgmma group, so the buffer
        // stage kt+2 is about to overwrite ((kt+2)%stages == (kt-1)%stages) is
        // no longer being read. The syncThreads keeps thread 0 from re-arming
        // the barrier while another thread is still inside this stage.
        wg.waitGroup(1);
        cuda.syncThreads();

        if (kt + 2 < ktiles and tid == 0) {
            issueTileLoad((kt + 2) % stages, desc_a, desc_b, block_row, block_col, (kt + 2) * k_slice);
        }

        wg.fenceFragment(&afrag_even);
        wg.fenceFragment(&afrag_odd);
    }

    // Also on the poisoned path: outstanding wgmma groups are unaffected by a
    // failed load wait and must be retired before the accumulator is read.
    wg.waitGroup(0);

    inline for (0..n_tiles) |t| {
        inline for (0..8) |i| {
            const p = wg.Acc64x16.coord(warp, lane, i);
            const row = block_row + p.m;
            const col = block_col + t * 16 + p.n;
            // NaN by bit pattern: std.math.nan blows the comptime branch quota.
            c[@as(usize, row) * n + col] = if (poisoned) @bitCast(@as(u32, 0x7fffffff)) else acc[t].get(i);
        }
    }
}

comptime {
    cuda.abi.assertMatches(abi.signature, @TypeOf(hgemmTma));
    _ = cuda.Keep(.{&hgemmTma}).__zoxide_keep_kernels;
}
