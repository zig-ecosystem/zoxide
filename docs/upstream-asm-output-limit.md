# Upstream: Zig's 15-output inline-asm cap blocks wide-N Hopper `wgmma`

**Status:** confirmed locally against Zig 0.16.0 and `ziglang/zig` master
(`3bfb299947`). Not yet filed upstream.

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

| shape | accumulator regs | expressible in Zig |
| --- | --- | --- |
| `m64n16k16` | 8 | yes |
| `m64n32k16` | 16 | no — one over the cap |
| `m64n64k16` | 32 | no |
| `m64n128k16` | 64 | no |

There is no way around it by splitting the asm: one `wgmma` instruction needs
its whole accumulator in a single operand list. And there is no intrinsic to
fall back on — NVVM exposes only `wgmma.fence`, `wgmma.commit_group` and
`wgmma.wait_group`, never the MMA itself, so inline asm is the only path.

So `src/examples/hgemm_wgmma.zig` is stuck on `m64n16k16` and has to tile it 8x
to cover a 128-wide N. That re-reads the A tile from shared memory 8 times per
K-stage instead of once, giving away part of the shared-memory bandwidth win
that wide-N `wgmma` exists to deliver.

Raising the cap to 32 (the `output_type_bits` ceiling) would unlock
`m64n64k16`. CUTLASS's reference kernels use `m64n128k16` and would still be
out of reach at 32; lifting that too would additionally require widening
`output_type_bits`.

### Measured cost (H20, n=4096, 2026-09-21)

| kernel | instruction | GFLOPS | % of FP16 tensor peak |
| --- | --- | --- | --- |
| `hgemm_mma2` | `mma.sync m16n8k16` | 53839 | 36.4% |
| `hgemm_wgmma` | `wgmma.mma_async m64n16k16` | 80308 | 54.3% |

So even the narrowest wgmma shape — the only one Zig can express — is worth
1.49x over a tuned `mma.sync` kernel, at exact results. That is the floor, not
the ceiling: the n16 tiling re-reads the A tile from shared memory 8x per
K-stage. What `m64n64k16` would add is not yet measured, and measuring it is
exactly what this cap prevents. Quantifying it requires a patched compiler,
which makes the patch its own prerequisite.

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

## Not yet done

- Build a patched compiler and confirm 32 outputs round-trip through Sema and
  reach the NVPTX backend intact. The reasoning above is read from the source
  (`outputs_len` is already `u7`, `output_type_bits` already 32 bits, and
  `Sema.zirAsm` no longer truncates); it has not been confirmed by building a
  patched compiler.
- Measure `m64n64k16` against the `m64n16k16` kernel on H20 to quantify what the
  cap costs.
- File upstream. Note `ziglang/zig` issue creation is restricted to
  collaborators (see `docs/drafts/`), so this will need one of the fallback
  channels already documented there.
