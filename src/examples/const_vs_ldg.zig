//! Does `.const` actually beat global memory for a warp-uniform read?
//!
//! The claim attached to constant memory is that its cache broadcasts, so a warp
//! reading one address is a single fetch. That is CUDA documentation, not a
//! measurement on this device, and the rest of this repository has been wrong
//! often enough when reasoning stood in for a test.
//!
//! Both kernels do the same work: walk a 64-entry table where the index is
//! uniform across every thread (`k` is a loop counter, not derived from the
//! thread id) and accumulate into a per-thread value. That is the access pattern
//! constant memory is supposed to be good at. The only difference is where the
//! table lives:
//!
//!   constUniform  .const + ld.const        (constant window, broadcast cache)
//!   ldgUniform    .global + ld.global.nc   (read-only data cache)
//!
//! The loop is long enough that the table read dominates; `in[i]` is read once
//! outside it so global bandwidth is not what is being compared.
//!
//! Interpretation, decided before running so the numbers cannot pick it:
//!   - `.const` clearly faster -> ConstBank earns its complexity
//!   - within noise -> on H20 the read-only cache already handles this pattern,
//!     and ConstBank is only worth it to free up parameter space or to match
//!     CUDA's layout; the doc comments claiming a broadcast win must be fixed
//!   - `.const` slower -> ConstBank should be marked as not recommended here
const cuda = @import("cuda");

const bank_len = 64;
const Scales = cuda.ConstBank("cv_scales", f32, bank_len);

comptime {
    asm (Scales.declaration);
}

/// Same table, ordinary global memory. Read through `ldg` so the read is opaque
/// to the optimiser — without that the loop would be folded into a constant and
/// the comparison would measure nothing.
var g_scales: [bank_len]f32 addrspace(.global) = .{0} ** bank_len;

/// Outer trips over the table. Both kernels unroll by hand over the full table
/// so the emitted shape is identical by construction — left to LLVM, the two
/// loops got different unroll factors (4x vs 8x, because the `.const` asm block
/// is larger), and comparing different amounts of ILP would measure the unroller
/// rather than the memory path.
/// Raised from 4: at 4 the whole kernel ran in 30 us, close enough to launch
/// overhead to be worth removing as a variable.
const trips = 64;

pub fn constUniform(out: [*]f32, in: [*]const f32, n: u32) callconv(.kernel) void {
    const i = cuda.globalThreadId();
    if (i >= n) return;
    const x = in[i];
    var acc: f32 = 0;
    for (0..trips) |_| {
        // Index is uniform across the warp — every thread reads the same
        // address, the case constant memory is meant to win. Comptime index, so
        // the offset folds into the instruction.
        inline for (0..bank_len) |k| acc += x * Scales.getAt(k);
    }
    out[i] = acc;
}

pub fn ldgUniform(out: [*]f32, in: [*]const f32, n: u32) callconv(.kernel) void {
    const i = cuda.globalThreadId();
    if (i >= n) return;
    const x = in[i];
    var acc: f32 = 0;
    for (0..trips) |_| {
        inline for (0..bank_len) |k| acc += x * cuda.ldg(f32, &g_scales[k]);
    }
    out[i] = acc;
}

// Divergent counterparts, to separate two explanations of the uniform result.
//
// The uniform test came out 1.11x in favour of .const, but the .const kernel also
// emits 17% fewer statements (237 vs 285), because `ld.const [sym+N]` needs no
// address arithmetic. An 11% gain next to a 17% instruction reduction does not
// isolate the broadcast cache; it is equally consistent with "fewer
// instructions".
//
// Constant memory serialises when threads in a warp hit different addresses,
// while the read-only data cache coalesces. So the mechanism is testable through
// the *change* in ratio rather than its absolute value:
//
//   broadcast is real   -> .const advantage disappears or inverts when divergent
//   just instructions   -> ratio stays roughly the same in both
//
// Both variants here index by thread id, so both need the runtime accessor. That
// puts .const back on mov/cvt/add/ld against ldg's single instruction, which is a
// confound in the opposite direction — hence reading the delta between the two
// ratios, not either one alone.
pub fn constDivergent(out: [*]f32, in: [*]const f32, n: u32) callconv(.kernel) void {
    const i = cuda.globalThreadId();
    if (i >= n) return;
    const x = in[i];
    var acc: f32 = 0;
    // Stride 7 against a 64-entry table: lanes within a warp land on distinct
    // entries, which is what constant memory is bad at.
    for (0..trips) |_| {
        inline for (0..bank_len) |k| acc += x * Scales.get((i *% 7 +% @as(u32, k)) % bank_len);
    }
    out[i] = acc;
}

pub fn ldgDivergent(out: [*]f32, in: [*]const f32, n: u32) callconv(.kernel) void {
    const i = cuda.globalThreadId();
    if (i >= n) return;
    const x = in[i];
    var acc: f32 = 0;
    for (0..trips) |_| {
        inline for (0..bank_len) |k| acc += x * cuda.ldg(f32, &g_scales[(i *% 7 +% @as(u32, k)) % bank_len]);
    }
    out[i] = acc;
}

comptime {
    _ = cuda.Keep(.{
        &constUniform,   &ldgUniform,
        &constDivergent, &ldgDivergent,
    }).__zoxide_keep_kernels;
}
