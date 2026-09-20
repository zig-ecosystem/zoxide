# Draft: ziglang/zig issue — NVPTX inline asm operand substitution / NVVM intrinsics

> 状态：**已撤回（前提证伪），仅存档**。评审通过后发到 https://github.com/ziglang/zig/issues 。
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

CUDA kernels can be written in pure Zig today via the NVPTX backend (Zig → PTX → ptxas), and it works well — we have verified correctness and competitive performance (SGEMM at ~43% of FP32 peak) on Hopper hardware. Reusing NVIDIA's PTX ISA catalog data (via the Apache-2.0 [cuda-oxide](https://github.com/NVlabs/cuda-oxide) project), **329** of its 1025 instructions are reachable from Zig through `llvm.nvvm.*` intrinsic calls — but the remaining **689 entries** (the tensor-core generation: tcgen05, wgmma/register_mma/sparse_mma, most of TMA) are lowered in practice via inline PTX asm, because LLVM has no intrinsics for them. These are entirely unreachable from Zig today, which blocks FP16/BF16 tensor-core kernels on Hopper/Blackwell (the difference between ~44 TFLOPS FP32 and ~148 TFLOPS dense FP16 on an H20).

## Possible resolutions (any one unblocks us)

1. **Officially support calling `llvm.nvvm.*` intrinsics** from Zig (document + stabilize what currently works by accident, cf. #2291) and extend coverage where LLVM has intrinsics.
2. **Implement operand substitution in asm templates** (at minimum `$N`/named operands), bringing NVPTX to parity with the expressiveness other toolchains have via LLVM inline asm.
3. **Zig-native NVPTX builtins** (`@threadIdx()`, `@shuffleSync()`, …) — a bigger commitment, but the most ergonomic.

Option 1 alone covers 329/1025 catalog entries; options 2 or 3 are needed for the tensor-core remainder.

## Environment

Zig 0.16.0, target `nvptx64-cuda`, `-mcpu sm_90`; ptxas 12.8 for validation.

---

## 评审备注（不随 issue 发出）

- ~~附 zoxide 链接~~ 应作者要求不引用 zoxide，用抽象描述（"CUDA kernels in pure Zig, verified on Hopper hardware"）。
- 如果维护者倾向 option 3，我们可以跟进提供常用 builtin 清单（catalog 的 family 分布直接可给）。
- 发出后在 ROADMAP.md v0.4.0 节更新 issue 链接。

## 状态更新（2026-09-20）

- 已按作者要求移除 zoxide 公开引用（用抽象用例描述）。
- **发布受阻**：ziglang/zig 仓库当前限制为 collaborator 才能开 issue（gh 报 "Interactions on this repository have been restricted to collaborators only"）。
- 备选渠道：① ziggit.dev 论坛发帖（Zig 官方论坛，维护者活跃）② Zig Discord #compiler 频道 ③ 等限制解除后再发 issue。草稿保持可用。

---

## ⚠️ 已撤回（WITHDRAWN）— 2026-09-20

**前提被证伪，不要发这个 issue。**

对照最新 zig master（codeberg，commit 3bfb299947）源码核查后发现：Zig 的 asm **支持 `%[name]` 具名操作数替换**（`src/codegen/llvm/FuncGen.zig:2812-2878` 的 rendered_template 状态机，`doc/langref/Assembly Syntax Explained.zig` 文档化）。我们此前测试失败是因为试了 `$0`/`%0`/`${0}`/`$[name]` 四种形式，唯独漏了唯一正确的 `%[name]`。

本机验证（zig 0.16.0，nvptx64-cuda sm_90）：

```zig
asm ("mov.u32 \t%[r], %tid.x;" : [r] "=r" (-> u32))
// → PTX: mov.u32 %r1, %tid.x;  ✓ 替换成功，ptxas 可接受
```

连带影响：
- 689 条 asm 降落的 PTX 指令（tcgen05/mma/TMA）**不再被阻塞**——M4d 从"等上游"变为"生成器支持 asm 类条目"
- Zig 上游只剩一个温和诉求：`llvm.nvvm.*` extern 调用的文档化承诺（非阻塞）
- 另有意外收获：被删除的 test/nvptx.zig（2025-10）证明 LLVM 后端这条路上游验证过
