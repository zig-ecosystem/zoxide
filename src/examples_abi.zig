//! Launch signatures for the bundled example kernels, imported by both the
//! kernels themselves and by `zoxide bench`.
//!
//! The point is the same as for a downstream package: host and device are
//! compiled separately for different targets, so a kernel's parameters can
//! change without the launcher noticing, and `cuLaunchKernel` takes `void**` so
//! nothing checks it at runtime either. With the signature declared once, a
//! change on one side is a compile error on the other.
//!
//! This had a real motivation. `bench` used to hand-pack `?*anyopaque` arrays,
//! and when the hgemm kernels moved from byte pointers to `f16` pointers nothing
//! in the build would have complained had the packing been wrong.
//!
//! Declared without `callconv(.kernel)`: that convention resolves per target and
//! is `unreachable` on host architectures, so the type cannot be named there.

/// The f32 GEMM family: `sgemm_naive`, `sgemm_tiled`, `sgemm_reg`, `sgemm_opt`,
/// `sgemm_opt2`, `sgemm_swz`.
pub const sgemm = fn (a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) void;

/// The f16 tensor-core family: `hgemm_mma`, `hgemm_mma2`, `hgemm_wgmma`,
/// `hgemm_wgmma2`, `hgemm_wgmma3`. f16 inputs, f32 accumulation.
pub const hgemm = fn (a: [*]const f16, b: [*]const f16, c: [*]f32, n: u32) void;

/// `hgemm_bf16`: the bf16 mma shape. bf16 has no native Zig type, so inputs
/// are u16 bit patterns; f32 accumulation, same as the f16 family.
///
/// Also the launch-bounds demonstration: the kernel is written for exactly
/// 128 threads (4 warps, no bounds guards), so the contract says so —
/// `cuda.launchBounds` emits `.maxntid 128` into the PTX and the host launch
/// validates against it.
pub const hgemm_bf16 = struct {
    pub const signature = fn (a: [*]const u16, b: [*]const u16, c: [*]f32, n: u32) void;
    pub const launch_bounds = .{ .max_threads = 128 };
};

/// `imma_s8`: the int8 tensor-core shape, `mma.sync.m16n8k32.s32.s8.s8.s32`.
/// Unlike the f16/bf16 family this one is integer end to end — `i8` inputs and
/// `i32` accumulators — so the result is exact and the correctness check is an
/// equality, not a tolerance.
pub const imma_s8 = fn (a: [*]const i8, b: [*]const i8, c: [*]i32, n: u32) void;

/// `imma_s4`: the int4 tensor-core shape, `mma.sync.m16n8k64.s32.s4.s4.s32`.
/// s4 has no byte type, so A and B are packed bytes (see `packS4`) and `n` is
/// the *logical* element count per row — both sides must agree on that, which
/// is why the convention lives here and not in a comment on each side.
pub const imma_s4 = fn (a: [*]const u8, b: [*]const u8, c: [*]i32, n: u32) void;

/// s4 storage convention for `imma_s4`: two 4-bit two's-complement values per
/// byte, the even logical index in the low nibble. The kernel feeds packed
/// bytes to the mma untouched, so this nibble order is the only coupling
/// between the host packer and the hardware.
pub fn packS4(even: i8, odd: i8) u8 {
    return (@as(u8, @bitCast(even)) & 0xF) | ((@as(u8, @bitCast(odd)) & 0xF) << 4);
}

/// `hgemm_sp`: 2:4-structured-sparse f16 mma
/// (`mma.sp::ordered_metadata.m16n8k16.f32.f16.f16.f32`). `a` is the pruned A
/// (n/2 kept f16 per row, kept elements in k order), `meta` one u16 per row
/// per 16 dense k (n/16 words per row) with the bit layout below, `b` dense,
/// `n` the *dense* element count per row.
///
/// Metadata contract (the kernel reads it the same way): within a u16,
/// k-group j (dense k 4j..4j+3) owns nibble [4j+3 : 4j]; the low 2 bits of
/// the nibble are the index of the first kept element, the high 2 bits the
/// second. On the device the row-g word goes to the low 16 bits of the
/// metadata register and the row-(g+8) word to the high 16.
pub const hgemm_sp = fn (a: [*]const f16, b: [*]const f16, meta: [*]const u16, c: [*]f32, n: u32) void;

/// `imma_sp_s8`: 2:4-structured-sparse int8 mma
/// (`mma.sp::ordered_metadata.m16n8k32.s32.s8.s8.s32`). `a` is the pruned A
/// (n/2 kept s8 per row), `meta` one u32 per row per 32 dense k
/// (n/32 words per row), `b` dense s8, `c` s32, `n` the dense count per row.
///
/// Metadata contract: k-chunk j (dense k 4j..4j+3) owns nibble [4j+3 : 4j],
/// low 2 bits = first kept index, high 2 = second — same nibble order as
/// hgemm_sp. The row-to-thread assignment differs: k32's selector names a
/// thread pair (t==0 carries row g, t==1 row g+8), so the metadata register
/// holds one full row per contributing lane rather than two half-rows.
pub const imma_sp_s8 = fn (a: [*]const i8, b: [*]const i8, meta: [*]const u32, c: [*]i32, n: u32) void;

/// `imma_sp_s4`: structured-sparse int4 mma
/// (`mma.sp::ordered_metadata.m16n8k64.s32.s4.s4.s32`). The sparsity is 4:8
/// and *pair-clustered*: within each 8-wide dense k chunk, exactly two of the
/// four 2-wide sub-chunks survive whole. Kept pairs stay pairs, so under the
/// `packS4` convention each surviving pair is one byte of the pruned row —
/// the prune is byte-granular.
///
/// Buffers: `a` pruned A (n/4 packed bytes per row), `meta` one u32 per row
/// per 64 dense k (n/64 words per row; chunk j owns nibble [4j+3 : 4j], low
/// 2 bits = first surviving sub-chunk index 0..3, high 2 = second), `b` dense
/// s4 packed (n/2 bytes per row), `c` s32, `n` the dense count per row.
/// Selector contract is imma_sp_s8's: k64-s4 names a thread pair, t==0
/// carries row g, t==1 row g+8.
pub const imma_sp_s4 = fn (a: [*]const u8, b: [*]const u8, meta: [*]const u32, c: [*]i32, n: u32) void;

/// Shape of the `const_vs_ldg` comparison, shared because the harness computes
/// the expected result from it.
///
/// These were duplicated at first, and the duplicate drifted the moment `trips`
/// was raised on the device side only. The expected value scales linearly with
/// `trips`, so the check reported `max rel err 15.000009` — exactly `64/4 - 1`,
/// which is the kind of number that names its own cause. The measurement would
/// have run and produced plausible timings either way; only the correctness
/// check caught it.
pub const const_vs_ldg = struct {
    /// Entries in the table. Both the `.const` bank and the global array.
    pub const bank_len = 64;
    /// Times each kernel walks the whole table. Sized so a launch takes long
    /// enough that launch overhead is not part of the measurement.
    pub const trips = 64;

    pub const signature = fn (out: [*]f32, in: [*]const f32, n: u32) void;
};

/// `cpasync_mbar_smoke`: compile smoke for the generated cp.async + mbarrier
/// bindings (src/gen), covering all six async-copy/barrier families.
pub const cpasync_mbar_smoke = struct {
    pub const signature = fn (src: [*]const u8, diag: [*]u32) void;
};

/// `ldmatrix_smoke`: compile smoke for the generated ldmatrix / stmatrix /
/// movmatrix bindings (src/gen), covering all three matrix families.
pub const ldmatrix_smoke = struct {
    pub const signature = fn (src: [*]const u8, diag: [*]u32) void;
};

/// `warpops_smoke`: compile smoke for the generated warp-level bindings
/// (src/gen), covering redux, warp_match, vote, active_mask, warp_barrier and
/// warp_shuffle (38 decls).
pub const warpops_smoke = struct {
    pub const signature = fn (src: [*]const i32, diag: [*]u32) void;
};

/// `packed_smoke`: compile smoke for the generated packed-data bindings
/// (src/gen), covering packed_alu, packed_conversion, packed_atomic, dotprod,
/// prmt, clc, extended_minmax and integer_minmax (127 decls).
pub const packed_smoke = struct {
    pub const signature = fn (src: [*]const u32, diag: [*]u32) void;
};

/// `mbar_smoke`: mbarrier with TMA removed, to find out which half is broken.
pub const mbar_smoke = struct {
    pub const block = 128;
    pub const diag_words = 8;
    pub const signature = fn (diag: [*]u32) void;
};

/// `hgemm_tma` (S2 of the TMA line): the `hgemm_wgmma3` tile and pipeline
/// shape with TMA loads, so the byte counts the kernel waits on and the boxes
/// the host encodes are the same numbers, spelled once here. The host
/// cross-checks them against `TensorMap.tileBytes()` of the descriptors it
/// actually encoded — the expect_tx count written wrong does not fault, it
/// deadlocks or releases early.
pub const hgemm_tma = struct {
    pub const tile_m = 64;
    pub const tile_n = 128;
    pub const k_slice = 16;
    pub const threads = 128;
    pub const stages = 3;

    /// B arrives as one 8-column x 16-k-row sub-tile per wgmma n-block, each
    /// 256 B landing at nb*256 — reproducing the core-matrix-packed layout the
    /// wgmma descriptor path already reads. See hgemm_tma.zig for why a single
    /// box over the whole k×n tile does not work.
    pub const b_subtiles = tile_n / 8;
    pub const a_tile_bytes = tile_m * k_slice * 2; // 2 KB, plain [64][16] f16
    pub const b_subtile_bytes = 8 * k_slice * 2; // 256 B
    pub const b_tile_bytes = b_subtile_bytes * b_subtiles; // 4 KB
    /// What one pipeline stage delivers, and what arrive.expect_tx waits for.
    pub const stage_bytes = a_tile_bytes + b_tile_bytes;

    /// The two u64s are device addresses of the A and B `CUtensorMap`s. The
    /// matrix pointers are unused on the device (the descriptors carry the
    /// addresses) but document the binding and keep the harness symmetric.
    pub const signature = fn (a: [*]const f16, b: [*]const f16, c: [*]f32, n: u32, desc_a: u64, desc_b: u64) void;
};

/// Geometry of the `tma_smoke` tile, shared because three separate things have to
/// agree on it and only one of them is checkable at compile time:
///
///   - the shared-memory buffer the kernel reserves
///   - the byte count handed to `mbarrier.arrive.expect_tx`
///   - the `box` the host encodes into the descriptor
///
/// The byte count is the one that bites. Too low and the wait releases on partial
/// data; too high and it never releases. Neither faults, and neither is reported.
pub const tma_smoke = struct {
    /// Source tensor, row-major.
    pub const rows = 128;
    pub const cols = 128;
    /// Tile: innermost (columns) first, matching the descriptor's `box` order.
    pub const box_cols = 64;
    pub const box_rows = 8;
    /// f16 elements, so the innermost tile row is 64 x 2 = 128 bytes — exactly one
    /// period of a `.b128` swizzle, which is what makes the swizzled control a
    /// clean comparison.
    pub const elem_bytes = 2;
    pub const tile_bytes = box_cols * box_rows * elem_bytes;
    pub const block = 128;

    pub const signature = fn (out: [*]u8, diag: [*]u32, desc: u64, x: i32, y: i32) void;
    /// Stage markers, so a device-side failure reports where it stopped.
    pub const diag_words = 11;
};

/// `tma_s2g_smoke`: the s2g half of the TMA round trip — g2s a tile into
/// shared (tma_smoke's path), then store it back out through a second
/// descriptor with the generated `cp.async.bulk.tensor.*.global.shared::cta`
/// wrapper. Tile geometry is tma_smoke's, referenced in the kernel rather
/// than copied here, because descriptor, buffer and expect_tx count must
/// agree for the same reasons as above.
pub const tma_s2g_smoke = struct {
    pub const signature = fn (diag: [*]u32, desc_in: u64, desc_out: u64, x: i32, y: i32) void;
    /// Stage markers, so a device-side failure reports where it stopped.
    pub const diag_words = 6;
};

pub const atomics_smoke = struct {
    pub const signature = fn (src: [*]const i32, diag: [*]u32) void;
    /// diag layout: see src/examples/atomics_smoke.zig.
    pub const diag_words = 32;
};
