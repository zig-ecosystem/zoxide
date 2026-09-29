# Upstream: Zig's 15-output inline-asm cap, and what it actually costs

**Status:** confirmed locally against Zig 0.16.0 and `ziglang/zig` master
(`3bfb299947`), including an executed reproduction of both caps. Not yet filed
upstream; no prior art exists to file against (see below).

**Revised 2026-09-29.** The title used to say the cap "blocks wide-N Hopper
`wgmma`". It does not — a `.reg` workaround reaches every shape. The cap costs
LLVM's register accounting, not capability, and this document's conclusions were
rewritten to stop arguing otherwise.

## What the limit is

`lib/std/zig/AstGen.zig` rejects any `asm` expression with 16 or more output
operands:

```zig
if (full.outputs.len >= 16) {
    return astgen.failNode(full.outputs[16], "too many asm outputs", .{});
}
var outputs_buffer: [15]Zir.Inst.Asm.Output = undefined;
```

Inputs are capped the same way at 32 (`[31]` buffer).

Reproduced on Zig 0.16.0 with an `nvptx64-cuda` target: 15 `"+f"` outputs
compile, 16 fail with `too many asm outputs`.

## Why the number is 15

The cap is a compiler-side encoding artifact, not an LLVM or hardware limit.

ZIR's `extended` instruction header carries a single 16-bit `small` field, and
`asm` packs all of its metadata into it. Commit `06eebafadd`
("AstGen: redistribute inline asm limits", 2025-03-27) changed the split from
`5/5/5/1` bits (outputs / inputs / clobbers / volatile) to `4/5/6/1`, buying a
bit for clobbers — x86 clobber lists routinely exceed 31 entries — and paying
for it out of the output field. 4 bits is where `>= 16` comes from.

That trade is now stale. On master, clobbers are no longer length-encoded (they
became a `Ref` to a `std.builtin` value), which freed the bit budget, and
`Zir.Inst.Asm.Small` is already:

```zig
pub const Small = packed struct(u16) {
    is_volatile: bool,
    outputs_len: u7,   // 127
    inputs_len: u8,    // 255
};
```

`Sema.zirAsm` reads `small.outputs_len` / `small.inputs_len` directly, with no
`u4`/`u5` truncation. So the IR can already express 127 outputs; only the
AstGen constant and its stack buffer still say 15.

The one remaining real constraint is `Zir.Inst.Asm.output_type_bits: u32`, one
bit per output, which caps outputs at 32.

## Why zoxide cares

Hopper's `wgmma.mma_async` keeps its accumulator in registers spread across the
warpgroup, and every accumulator register has to appear as an inline-asm output
operand. `m64nNk16` needs `N/2` registers per thread:

| shape | accumulator regs | as asm output operands | via the `.reg` workaround |
| --- | --- | --- | --- |
| `m64n16k16` | 8 | yes | n/a |
| `m64n32k16` | 16 | no — one over the cap | yes |
| `m64n64k16` | 32 | no | yes |
| `m64n128k16` | 64 | no | yes (`hgemm_wgmma4`) |

> **Correction (2026-09-22).** An earlier version of this document said there was
> no way around the limit. That was wrong, and the workaround makes the cap a
> nuisance rather than a blocker. See "There is a way around it" below. The
> analysis of *why* the limit is 15 stands; the conclusion that it prevents
> wide-N wgmma does not.
>
> **Correction (2026-09-29).** The 2026-09-22 revision inserted the workaround
> section but left three later sections arguing from the superseded premise —
> that wide N is unreachable and therefore unmeasurable. Those sections are
> rewritten below. The contradiction was load-bearing for the upstream case, so
> it is worth naming what it was: this document simultaneously claimed
> `hgemm_wgmma4` *is* the wide-N measurement and that we are "barred from
> measuring it."

Splitting the asm does not help: one `wgmma` instruction needs its whole
accumulator in a single operand list. And there is no intrinsic to fall back on —
NVVM exposes only `wgmma.fence`, `wgmma.commit_group` and `wgmma.wait_group`,
never the MMA itself, so inline asm is the only path.

## There is a way around it

The limit applies to *operands*, and the accumulator does not have to be one.

PTX register declarations are function-scoped, and LLVM splices inline asm into
the function body verbatim, so a `.reg` declaration in one asm statement is
visible to every later statement in the same function:

```zig
asm volatile (".reg .f32 %zacc<64>;");                   // once, unconditionally
asm volatile ("wgmma...m64n128k16... {%zacc0,...,%zacc63}, %[da], %[db], p, 1, 1, 0, 1;"
    : : [da] "l" (desc_a), [db] "l" (desc_b), [sd] "r" (sd) : ...);  // zero outputs
asm volatile ("mov.f32 %[o], %zacc7;" : [o] "=f" (v));   // one output, 64 times
```

The accumulator lives in registers LLVM does not know about, never appears in an
operand list, and the limit does not apply. `m64n128k16` — 64 accumulators, the
shape CUTLASS uses — becomes expressible on stock Zig, as does anything wider.
`src/examples/hgemm_wgmma4.zig` does exactly this.

Verified in the generated PTX: the declaration appears once, the instruction
carries 64 registers and no output operands, and the epilogue reads them back
with 64 single-output `mov`s.

### What it costs

LLVM cannot see these registers, so it cannot account for them. It allocates its
own as though the 64 f32 were absent, and ptxas has to fit both. If the total
exceeds what the SM can provide, ptxas spills — and on this hardware spilling is
measured at 0.55x throughput for 16 registers and 0.93x for a single byte, so that
failure mode is severe rather than marginal.

Two conditions are load-bearing and neither is visible in the source, so CI
asserts both against the generated PTX:

  * the `.reg` declaration must appear exactly once, or ptxas rejects the
    redefinition — which means the statement carrying it must never be duplicated
    by unrolling or branch cloning;
  * it must precede every use.

### What this does to the case for raising the limit

It weakens it, honestly. The cap no longer blocks wide-N wgmma; it forces a
workaround that gives up LLVM's register accounting. That is still worth fixing —
a language should not require hiding state from its own compiler to express a
hardware instruction — but it is a wart, not a wall, and this document previously
overstated it.

It also removes the circularity that justified building a patched compiler.
Quantifying what narrow N costs no longer needs one — `hgemm_wgmma4` is that
experiment, built on stock Zig. **It has not been run on hardware yet:** there
is no `hgemm_wgmma4` entry under `docs/verification/`, and it is still an open
P0 item in `docs/cuda-oxide-port-plan.md`. So the measurement is unblocked, not
taken.

`src/examples/hgemm_wgmma.zig` therefore uses `m64n16k16` by choice of form
rather than by necessity: it tiles the shape 8x to cover a 128-wide N. In the
all-shared form that re-reads the A tile from shared memory 8 times per K-stage
instead of once; `hgemm_wgmma3` already recovers that by feeding A from
registers (see the operand-traffic table below), so the remaining cost of
narrow N is per-instruction, not traffic.

Raising the cap to 32 (the `output_type_bits` ceiling) would make
`m64n64k16` expressible with ordinary operands. `m64n128k16` — what CUTLASS's
reference kernels use — would need `output_type_bits` widened as well. Neither
is a capability gate any more, since the `.reg` workaround reaches both; what
the cap costs is LLVM's register accounting, which is a correctness-adjacent
property rather than a convenience (see "What it costs").

### Measured cost (H20, n=4096, 2026-09-21)

| kernel | instruction | GFLOPS | % of FP16 tensor peak |
| --- | --- | --- | --- |
| `hgemm_mma2` | `mma.sync m16n8k16` | 53839 | 36.4% |
| `hgemm_wgmma` | `wgmma.mma_async m64n16k16` | 80308 | 54.3% |
| `hgemm_wgmma2` | + 3-stage pipeline | 86279 | 58.3% |
| `hgemm_wgmma3` | + A from registers (RS) | 95175 | 64.3% |

So even the narrowest wgmma shape — the only one Zig can express — is worth
1.49x over a tuned `mma.sync` kernel, at exact results. A 3-stage pipeline then
took it to 86284 GFLOPS (58.3%), 1.60x over the baseline.

### Per-instruction n16 efficiency is the last unexplained candidate

58.3% leaves 41.7pp on the table, and three of the four candidate causes have
been eliminated by experiment on H20 (details in `docs/verification/`):

| hypothesis | status |
| --- | --- |
| pipeline draining the tensor core | eliminated — fixing it was worth 4pp |
| global traffic / DRAM bandwidth | eliminated — throughput is flat across the L2 boundary (57.9% at 36 MB working set, 58.3% at 64 MB, 57.8% at 256 MB) |
| too few independent warpgroups | eliminated, on measurement this time. My first attempt at this rested on 12 blocks per SM, derived by dividing the SM's shared-memory budget by the kernel's usage; the driver's figure is 4 blocks at 25%, because registers bind, not shared memory. Capping registers to reach 5 blocks at 31% with no spill made the kernel **slower** (0.97x), so 4 blocks already hides the latency |
| register pressure / occupancy | closed — 98 regs at 4 blocks/SM is the optimum; both more and fewer registers are worse |
| **per-instruction efficiency of n16** | **the only candidate left** |

### Correction: operand traffic had a second solution below the cap

An earlier revision of this document claimed the 3.3x operand-traffic penalty was
*invariant to tile shape*, so that only a wider N per instruction could fix it
and nothing was left to tune below the cap. That was wrong. It reasoned entirely
inside the all-shared (SS) form of `wgmma`. The RS form takes A from registers
and still has only 8 accumulators for `m64n16k16`, so it fits the cap:

| form | shared reads per K-stage | flops/byte |
| --- | --- | --- |
| all-shared n16 | 8 × (A 2048 + B 512) = 20480 B | 12.8 |
| A in registers (RS) | A 2048 + 8 × 512 = 6144 B | 42.7 |
| one n128, all-shared | A 2048 + B 4096 = 6144 B | 42.7 |

Loading A once per stage and feeding all eight wgmma from registers reads exactly
what a single `m64n128k16` would. Measured on H20 this is worth +6.0pp, taking
`hgemm_wgmma3` to 95175 GFLOPS (64.3% of peak), still exact.

The lesson is about method: elimination narrowed the cause to "the n16 shape",
and I then treated that as an atomic explanation. It was not — it had at least
two separable components, operand traffic and per-instruction cost, and the first
had a second solution that did not need this patch.

**Operand traffic now matches what a wide-N instruction would demand, and the
kernel is still 35.7pp off peak.** Whatever remains is per-instruction cost:
eight instructions doing one instruction's work. Testing that requires a wider
N — which `hgemm_wgmma4` provides on stock Zig, and which therefore is no longer
an argument for the patch. The honest statement of where this stands:

* **The remaining gap is measurable today.** Run `hgemm_wgmma4` on H20 against
  `hgemm_wgmma3`. That is a P0 item, not an upstream dependency.
* **The cap's cost is not throughput, it is accountability.** The workaround
  hides 64 live f32 from LLVM's register allocator, so ptxas has to fit
  registers nobody budgeted for. CI has to assert the absence of `st.local` to
  catch it, because the failure mode is a silent 0.55x rather than an error.
  That is the upstream case: a language should not require hiding state from its
  own compiler to name a hardware instruction.

An earlier revision argued the opposite — that the cap barred the measurement
and that this made the patch urgent. That was reasoning from a premise the same
document had already retracted two sections earlier.

## Proposed change

In `lib/std/zig/AstGen.zig`:

```diff
-    if (full.outputs.len >= 16) {
-        return astgen.failNode(full.outputs[16], "too many asm outputs", .{});
+    if (full.outputs.len > 32) {
+        return astgen.failNode(full.outputs[32], "too many asm outputs", .{});
     }
-    var outputs_buffer: [15]Zir.Inst.Asm.Output = undefined;
+    var outputs_buffer: [32]Zir.Inst.Asm.Output = undefined;
```

No ZIR or Sema change should be needed; `outputs_len` is already `u7` and
`output_type_bits` already has 32 bits. Going beyond 32 requires widening
`output_type_bits`.

## There is no prior art to argue against

Searched 2026-09-29, both trackers. **No issue, pull request, review comment or
forum thread anywhere mentions this limit.**

* Codeberg (`ziglang/zig`, the authoritative tracker since the migration): 394
  unique issues matching asm / assembly / operands / outputs / inputs scanned by
  title and body, plus every comment on the 29 asm-related issues that have any
  — zero hits for `too many asm outputs`, `15 outputs`, `31 inputs`,
  `operand limit`. `tcgen05` and `outputs_len` return nothing at all.
* GitHub (the pre-migration archive): same queries, zero hits. A control query
  (`inline asm clobbers`) returns results, so the search was working.
* `#215` (inline assembly improvements, open since 2016) and `#5241`
  (New Inline Assembly) discuss types, stack machines and clobber syntax, never
  operand counts.
* `doc/langref.html.in` does not document the limit at all — it is only
  discoverable by hitting the compile error.

`06eebafadd` itself rode in on PR `#23355`, titled
`x86_64: start rewriting overflow operations`, whose body is just
`Closes #19607` (an unrelated multiplication bug) and which has **zero
comments**. Its motive is inferable only from the adjacent commit
`7a2963efab "x86_64: add avx512 registers"`, authored 48 seconds earlier:
AVX-512 pushes the register namespace past 32, so clobber lists needed a sixth
bit. That trade was never written down in prose.

The only written trace of the limit anywhere is a regression test,
`test/cases/compile_errors/astgen_assembly_errors.zig`, added by
`f6fecfdc00 "improve assembly error test coverage"` (2025-11-13) — eight months
after the fact, asserting the error message without stating a rationale.

This is favourable for filing: there is no design consensus to overturn and no
counter-argument on record. It also means the AVX-512 motive should be stated
*for* the maintainers rather than asked about, since the change was theirs and
undocumented.

## Not yet done

- Build a patched compiler and confirm 32 outputs round-trip through Sema and
  reach the NVPTX backend intact. The reasoning above is read from the source
  (`outputs_len` is already `u7`, `output_type_bits` already 32 bits, and
  `Sema.zirAsm` no longer truncates); it has not been confirmed by building a
  patched compiler. **This is the one thing that should happen before filing** —
  the whole case rests on "the encoding already allows it", and that claim is
  currently read, not executed.
- Run `hgemm_wgmma4` on H20 against `hgemm_wgmma3` to quantify per-instruction
  n16 cost. Open P0, independent of upstream.
- File upstream. **Channel note (2026-09-29):** Zig's tracker has moved to
  Codeberg (`codeberg.org/ziglang/zig`) and issues were renumbered — migrated
  ones carry a `Migrated from: github.com/ziglang/zig/issues/NNNNN` line. The
  older note in `docs/drafts/` about GitHub issue creation being restricted to
  collaborators is obsolete; re-check current Codeberg permissions before
  assuming a fallback channel is needed.
- Report the off-by-one while filing: when `full.outputs.len == 16` exactly,
  `failNode(full.outputs[16], ...)` indexes one past the end. Same shape on the
  inputs path at 32. Small, independently valid, and it demonstrates the code
  path has had no attention.

## Suggested framing when filing

Three facts, in this order, because the third is the ask and the first two remove
the objections to it:

1. The 4-bit output field was carved out in `06eebafadd` to widen clobbers.
2. Clobbers later left the bitfield entirely (`fcafc63f3d`, "inline assembly:
   use types"), and the freed bits went back to the lengths — `outputs_len` is
   `u7`, `inputs_len` is `u8`. The reason for the 15 no longer exists.
3. AstGen's `>= 16` check and its `[15]` stack buffer were never updated to
   match. Raise them to the `output_type_bits` ceiling of 32.

Then the concrete use case: `wgmma.mma_async` accumulators (8/16/32/64 registers
by shape) and `tcgen05.ld/st` (up to 128 results, 130 operands) — NVIDIA
instructions with no LLVM intrinsic, reachable only through inline asm.
