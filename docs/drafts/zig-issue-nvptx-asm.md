# Draft: ziglang/zig issue — NVPTX inline asm operand substitution / NVVM intrinsics

> 状态：**草稿，待评审**。评审通过后发到 https://github.com/ziglang/zig/issues 。
> 目标读者：zig 编译器维护者。写作基调：最小复现 + 明确诉求 + 真实用例，不抱怨。

---

**Title:** NVPTX: inline asm operand placeholders ($0) emitted verbatim; no usable path for PTX instructions without NVVM intrinsics

## Summary

When targeting `nvptx64-cuda`, Zig `asm` templates are emitted verbatim into the PTX output — operand placeholders (`$0`, `%0`, `${0}`, named `$[x]`) are never substituted. This is [documented language behavior](https://ziglang.org/documentation/master/#asm) ("there are no template substitution sequences"), and the recommended `{reg}` constraints cannot help because NVPTX has no named physical registers. The result: **there is currently no way in Zig to emit a PTX instruction that LLVM does not expose as an `llvm.nvvm.*` intrinsic.**

## Minimal reproduction

```zig
// zig build-lib -target nvptx64-cuda -mcpu sm_90 -fno-emit-bin -femit-asm=kernel.ptx -O ReleaseFast repro.zig
pub fn readTid() callconv(.kernel) u32 {
    const r: u32 = asm volatile ("mov.u32 $0, %tid.x;"
        : [ret] "=r" (-> u32)
    );
    return r;
}
```

Generated PTX contains the template literally:

```ptx
// begin inline asm
mov.u32 $0, %tid.x;
// end inline asm
```

`ptxas` rejects it:

```
error : Arguments mismatch for instruction 'mov'
error : Unknown symbol '$0'
```

(Confirmed that `$0`, `${0}`, `%0`, and named-operand forms all pass through unsubstituted. The same is true on CPU targets — this is by design, not an NVPTX-specific bug.)

## Why the documented alternatives don't cover NVPTX

- `{reg}` register constraints require named physical registers; PTX registers are virtual (`%r0`, `%rd1`, … assigned by ptxas), so there is nothing to bind to.
- Calling `llvm.nvvm.*` intrinsics as extern functions works today (and we rely on it — thank you), but it is undocumented and only covers the subset of PTX that LLVM models as intrinsics.

## Real-world impact

We maintain [zoxide](https://github.com/zig-ecosystem/zoxide), a CUDA kernel toolchain writing GPU kernels in pure Zig (verified on H20 hardware: SGEMM at 43% of FP32 peak). Reusing NVIDIA's PTX ISA catalog (via [cuda-oxide](https://github.com/NVlabs/cuda-oxide), Apache-2.0), we can map **329** instructions through `llvm.nvvm.*` intrinsics — but **689 catalog entries** (the tensor-core generation: tcgen05, wgmma/register_mma/sparse_mma, most of TMA) are lowered in practice via inline PTX asm, because LLVM has no intrinsics for them. These are entirely unreachable from Zig today, which blocks FP16/BF16 tensor-core kernels on Hopper/Blackwell (the difference between ~44 TFLOPS FP32 and ~148 TFLOPS dense FP16 on an H20).

## Possible resolutions (any one unblocks us)

1. **Officially support calling `llvm.nvvm.*` intrinsics** from Zig (document + stabilize what currently works by accident, cf. #2291) and extend coverage where LLVM has intrinsics.
2. **Implement operand substitution in asm templates** (at minimum `$N`/named operands), bringing NVPTX to parity with the expressiveness other toolchains have via LLVM inline asm.
3. **Zig-native NVPTX builtins** (`@threadIdx()`, `@shuffleSync()`, …) — a bigger commitment, but the most ergonomic.

Option 1 alone covers 329/1025 catalog entries; options 2 or 3 are needed for the tensor-core remainder.

## Environment

Zig 0.16.0, target `nvptx64-cuda`, `-mcpu sm_90`; ptxas 12.8 for validation.

---

## 评审备注（不随 issue 发出）

- 附 zoxide 链接时用 v0.2.0-beta 之后的 tag，SGEMM 数据更有说服力。
- 如果维护者倾向 option 3，我们可以跟进提供常用 builtin 清单（catalog 的 family 分布直接可给）。
- 发出后在 ROADMAP.md v0.4.0 节更新 issue 链接。
