# zoxide announcements

每次发版在此追加一条公告（最新在最上），附相关图片。图片统一放 `docs/assets/`。

---

## v0.0.10-alpha — host API for real pipelines（2026-09-21）

这轮没有性能数字，做的是**别人能不能用起来**。

此前下游用户编出 kernel 就断了：driver 绑定在 CLI 内部，想跑自己的 kernel 得自己写一遍 libcuda FFI。现在有 `zoxide_host` 模块，host 和 device 同包，签名声明一次两边共享。

**类型化 launch。** `cuLaunchKernel` 收 `void**`，参数个数、顺序、宽度都没人管，写错**不崩**——kernel 读到相邻内存，返回看起来合理的错数字。六种以前静默的错误现在是编译错误（每条都拿真编译器验过）：参数多/少、标量宽度、缓冲元素类型、顺序颠倒、传主机指针、device 改了 host 没改。

**stream / pinned / 异步传输 / 设备侧 memset / occupancy 查询。** driver 绑定从 22 个到 36 个。`Pinned(T)` 是和异步拷贝一起做的，不是凑数：`cuMemcpy*Async` 从普通可分页内存发出时是**名义异步**，驱动要经内部 pinned 缓冲转运并阻塞。只给 stream 不给 pinned 会交出一个静默不重叠的 API，比没有更糟。

### 新 API 立刻抓出我自己的一个错

我曾手算「`hgemm_wgmma3` 每 SM 驻留 12 个 block」——228 KB 共享内存除以 18 KB。驱动说是 **4**：98 regs/thread，寄存器远早于共享内存成为约束，真实 occupancy 是 **25% 不是 75%**。算术没错，算的是错的限制。

后果不止是个错数字：我曾用「12 条流、75% occupancy」**排除**过「warpgroup 并发不足」这个假设。4 条流、25% 什么都证明不了，所以那条排除被撤回，三处文档原地改正。

随后用新加的 `--maxrregcount` 把它测清楚了：

| cap | regs | blocks/SM | occupancy | 峰值 | 溢出 |
| --- | --- | --- | --- | --- | --- |
| 无 | 98 | 4 | 25% | **64.3%** | 0 |
| 96 | 94 | 5 | 31% | 62.3% | 0 |
| 88 | 88 | 5 | 31% | 35.3% | 64 B |
| 72 | 72 | 7 | 44% | 18.8% | 192 B |

`cap 96` 那行是关键：**零溢出**，occupancy 提到 31%，吞吐反而掉 3%。所以并发假设这次被真正排除了——而且是拿实测排除，不是拿推算排除。寄存器维度关闭，98 regs / 4 blocks 就是最优点。

**溢出的代价也量化了：16 个寄存器溢出 = 0.55x 吞吐。** 这回过头验证了一件事——写 v0.0.9 时 `afrag[kt % 2]` 的运行时下标曾产生每轮 20 条 `st.local`，我是靠 grep 生成的 PTX 才发现的（源码上双缓冲完全正确）。按这张表，那个 bug 的代价约等于**整个 wgmma 路线的全部收益**。CI 里那条「不许出现 `st.local`」的守卫因此不是装饰。

### 方法教训，同一家族的第二次

上一轮是「排除法指方向，不能代替对机制的分解」。这次是它的镜像：**排除一个假设之前，先确认用来排除它的数字是测出来的而不是推出来的。** 共享内存那个除法单位对、数量级合理，却默默忽略了整个寄存器维度——而能正确回答它的驱动 API 一直都在，我只是没绑。

发布文案（X 单帖）：

> zoxide v0.0.10-alpha: typed CUDA kernel launches in Zig — wrong arg count, order, width or a host pointer where a device one belongs are now compile errors instead of silent memory corruption. Plus streams, pinned memory and occupancy queries, which promptly caught a wrong occupancy figure in my own perf notes. github.com/zig-ecosystem/zoxide

---

## v0.0.9-alpha — A from registers, 64.3% of FP16 peak（2026-09-21）

![hgemm progression](assets/hgemm-progression.svg)

**95.2 TFLOPS（FP16 峰值 64.3%），结果精确，对 mma.sync 基线累计 1.77x。**

这一轮的起点是纠正我自己上一轮的错误结论。当时我把剩余 41.7pp 全归给「n16 shape」，并断言「3.3 倍操作数流量对 tile 形状不变，只有单条指令的 N 变宽才能改变它，所以在 15 输出上限解除前 tile 层面没有可动的东西」。

错了。我整个推理都待在 wgmma 的 **SS 形态**（A、B 都从共享内存取）里，漏掉了 **RS 形态**——A 从寄存器取，而 `m64n16k16` 的累加器仍然只有 8 个，**落在 15 输出限制内**。

| 形态 | 每 K 段共享内存读取 | flops/byte |
| --- | --- | --- |
| SS n16 | 8 × (A 2048 + B 512) = 20480 B | 12.8 |
| **RS n16（本轮）** | A 2048 + 8 × 512 = **6144 B** | **42.7** |
| SS n128（需补丁） | A 2048 + B 4096 = 6144 B | 42.7 |

A 每段只读一次、之后从寄存器喂给全部 8 条 wgmma，读取量与单条 `m64n128k16` 完全相同。实测值 6.0pp。

方法上的教训：排除法把原因收敛到「shape」之后，我把它当成了不可分解的原因。它其实至少含两个可分的成分——操作数流量和单指令成本——而前者还有第二条解法。排除法指方向，不能代替对机制的分解。

有两个 bug 只在读生成的 PTX 时才暴露。一是 `afrag[kt % 2]` 的运行时下标把寄存器数组赶进了 local memory（每轮 20 条 `st.local`，比想省掉的共享内存重读更糟）；二是即便改成双缓冲，寄存器分配器认为每个 fragment 在其最后一条 wgmma 之后即死亡，把物理寄存器回收给下一段，**悄悄把源码表达的双缓冲合并掉了**——而此时上一段的 wgmma 仍在异步读它。后者用 CUTLASS 的 `warpgroup_fence_operand` 手法（空 asm 带 `+r`）钉住。

这同时让编译器补丁的理由变强：操作数流量既已与 n128 持平，却仍距峰值 35.7pp，剩下的只能是单指令成本（8 条指令做 1 条的事），而测量它必须把 N 变宽——正是 15 输出上限禁止的。

发布文案（X 单帖）：

> Zig on Hopper, 95.2 TFLOPS HGEMM — 64.3% of H20 FP16 tensor peak, exact, 1.77x over a tuned mma.sync kernel. This round came from being wrong: I had argued the n16 operand penalty needed a wider wgmma N. It needed A in registers instead, which fits the compiler's asm limits today. github.com/zig-ecosystem/zoxide

---

## v0.0.8-alpha — 3-stage wgmma pipeline, 58.3% of FP16 peak（2026-09-21）

![hgemm progression](assets/hgemm-progression.svg)

**86.3 TFLOPS（FP16 峰值 58.3%），结果精确，对 mma.sync 基线累计 1.60x。**

改动只有流水线深度，shape 一字未动，所以这是一次干净的 A/B。v3 每个 K 段以 `wgmma.wait_group 0` 收尾，把 tensor core 排空——两个 buffer 下没得选：即将被重填的那个正是上一段 wgmma 还在**异步**读的。三个 buffer 破掉这个依赖，`wait_group 1` 退休 `kt-1` 而 `kt` 继续飞，空出的 `(kt-1)%3` 恰好就是 `(kt+2)%3`。

值 4.0pp。但这一步真正的价值是**排除了一个候选**：剩下的 41.7pp 不是流水线。还剩两个，而且规格书算不出谁是主因——n16 让共享内存操作数流量变成 3.3 倍（12.8 vs 42.7 flops/字节），全局流量 3.22 GB/趟 = 2.02 TB/s，占 HBM 标称的 51%。所以下一步是 profiler，不是继续拍脑袋，也不是先去打编译器补丁。

顺带撤销一项：上一轮列的共享内存 swizzle 经分析对我们无效——tile 已是 core-matrix packed，一个 core matrix 是 128 字节连续，共享内存 32 banks × 4B = 128 字节一轮，单次读取已完整跨遍所有 bank，没有冲突可消。swizzle 模式针对的是保持宽行距的布局（TMA 产出那种）。

发布文案（X 单帖）：

> Zig on Hopper: 86.3 TFLOPS HGEMM, 58.3% of H20 FP16 tensor peak, exact — 1.60x over a tuned mma.sync kernel. The win this round was letting wgmma stay async: a third buffer so wait_group 1 retires the previous stage instead of draining the tensor core. github.com/zig-ecosystem/zoxide

---

## v0.0.7-alpha — warpgroup MMA, 54.3% of FP16 peak（2026-09-21）

![hgemm progression](assets/hgemm-progression.svg)

Hopper warpgroup MMA 落地：**80.3 TFLOPS（FP16 峰值 54.3%），结果精确，对 mma.sync 基线 1.49x**（53.8 TF / 36.4%）。

wgmma 和 mma.sync 有两处本质不同。一是 **LLVM 没有 `wgmma.mma_async` intrinsic**——NVVM 只暴露 fence / commit_group / wait_group 三条控制指令，MMA 本身只能手写 inline asm，且门控在 `sm_90a` 而非 `sm_90`（此前记录的「生成器已含 wgmma wrapper」已订正）。二是操作数不是普通二维数组：tensor core 按 **128 字节连续的 8×8 core matrix** 读共享内存，tile 必须按 core matrix 序列打包，再用 64 位描述符（起始地址 + 两个跨 core matrix 字节步长）寻址。

布局一次写对，靠的是不猜：位域与 canonical layout 从 CUTLASS `cute`（BSD-3-Clause）推导，累加器寄存器到 (m,n) 的映射用 `CLayout_64xN`。另配自检 kernel `wgmma_smoke`——发一条 wgmma，1024 个累加元素全部跟闭式重算比对，整数输入所以 f16/f32 都精确，把布局层从 GEMM tiling 里隔离出来单独验。

代价要说清楚：**54.3% 是带着 Zig 15 输出上限跑出来的**。`m64nNk16` 每线程需要 N/2 个累加器寄存器且每个都得是 asm output，`m64n32k16`（16 个）正好超 1 个，所以只能用 `m64n16k16` 铺 8 次覆盖 N=128，A tile 每段重读 8 遍。上限是 ZIR 编码的历史遗留（`outputs_len` 已是 u7），放到 32 就能用 `m64n64k16`——见 `docs/upstream-asm-output-limit.md`。

发布文案（X 单帖）：

> Hopper warpgroup MMA in pure Zig: 80.3 TFLOPS, 54.3% of H20 FP16 tensor peak, exact results — 1.49x over the mma.sync baseline. No LLVM intrinsic exists for wgmma.mma_async, so it is hand-written inline asm + 64-bit smem descriptors. github.com/zig-ecosystem/zoxide

---

## v0.0.4-alpha — tensor-core instructions unlocked（2026-09-20）

![catalog coverage](assets/catalog-coverage.svg)

M4d 前提证伪后重启：Zig asm 的 `%[name]` 具名操作数替换确认可用（此前只试了位置形式）。生成器新增 **618 条 inline-PTX wrapper**（`src/gen/instrinsics_asm.zig`），catalog 覆盖率从 32% 提到 **92%（947/1025）**——mma.sync / TMA 子集与 wgmma 控制指令可达（`wgmma.mma_async` 本身 LLVM 无 intrinsic，须手写 asm），PTX 文本验证通过（无 `$0` 残留，真实寄存器）。上游 issue 计划撤回：只剩 `llvm.nvvm.*` 文档化一个温和诉求。剩余 163 条 unmapped 的主因是 Zig asm 的 15 输出上限（tcgen05.ld 等超宽指令）。

---

## v0.0.3-alpha — verification tooling + gpu_printf（2026-09-20）

![pod-verify](assets/pod-verify.svg)

一键 GPU 回归：`scripts/pod-verify.sh`（doctor → 4 示例 run → bench 冒烟 → ptxas 汇编，结构化报告，无 GPU 自动降级 SKIP）+ `scripts/k8s-gpu-verify.yaml`（kubectl apply 即跑的 Job 模板）。

设备端 printf 落地：`cuda.printf(comptime fmt, args)`——comptime 物化 global format 串 + C varargs 提升规则的 valist 打包（f32→f64）。调查结论：`llvm.nvvm.vprintf` 在 LLVM 19+ 已移除，正确路径是调用字面名 `vprintf` 的函数，NVPTX 后端直接识别。

发布文案（X 单帖）：

> zoxide v0.0.3-alpha: one-command GPU regression for Zig CUDA kernels — a shell script + a k8s Job template. Plus gpu-side printf via comptime-packed varargs. github.com/zig-ecosystem/zoxide

---

## v0.0.2-beta — SGEMM line complete（2026-09-20）

![SGEMM progression on H20](assets/sgemm-progression.svg)

SGEMM 优化线收官：naive 2.8 TF → tiled 4.4 TF → register-blocked 15.3 TF → bank-conflict-free **18.8 TF（H20 FP32 峰值的 42.8%）**。每一步瓶颈都有诊断证据（PTX 审查 + ptxas -v + 对照实验），详见 `docs/verification/`。

关键发现：

- nvptx 后端不会自动融合 `a*b+c`，必须显式 `@mulAdd` 才有 `fma.rn`
- 行内 bank 冲突用行键 XOR swizzle 无解，正解是 fragment 列重映射（零额外指令）
- barrier 紧邻条件分支会被 LLVM 复制 → 真机死锁（与 cuda-oxide 禁用 JumpThreading 同类）

发布文案（X 线程）：

1. NVIDIA's CUDA Rust announcement nails the trend: the kernel should be written in your language, not shipped from somewhere else. We built the Zig answer — the Zig compiler's NVPTX backend emits PTX directly. No custom rustc backend, no pinned nightly, no LLVM plumbing. github.com/zig-ecosystem/zoxide
2. Verified end-to-end on a real H20 pod: 4 kernels PASS, including a GPU-deadlock hunt (LLVM duplicated bar.sync across a branch — the same JumpThreading trap cuda-oxide disables in rustc). SGEMM: 6% → 10% → 35% → 43% of FP32 peak, every step diagnosed.
3. Honest gaps vs cuda-oxide: no proc-macro safety layer yet; tcgen05/mma/TMA intrinsics blocked on Zig's asm-template limitations (689 catalog entries waiting on upstream). But 329 typed intrinsics already generated from cuda-oxide's own catalog (Apache-2.0 data reuse) 🙏

---

## v0.1.0 — first usable release, GPU-verified（2026-09-18）

首个稳定版：纯 Zig 写 CUDA kernel，zig → PTX → ptxas → cubin → GPU 执行，全链路在 NVIDIA H20 pod 真机验收（driver 550.90.07，ptxas 12.8），vector_add / shared_reverse / warp_reduce / atomic_counter 四个示例全部 PASS。

- `src/cuda.zig` 设备端库：索引/同步/warp shuffle/原子/共享内存（`addrspace(.shared)` 原生可用）
- `src/gen/intrinsics.zig`：329 个 NVVM intrinsic 类型安全封装（`zoxide gen` 消费 cuda-oxide catalog 生成）
- `zoxide` CLI：ptx / cubin / run / doctor / gen
- 下游集成：`zig fetch --save git+https://github.com/zig-ecosystem/zoxide#v0.1.0`，`addCudaModule` / `addNvptxKernel`

发布文案（X 单帖）：

> CUDA kernels in pure Zig — no CUDA SDK needed for compilation.
> zig (nvptx backend) → PTX → ptxas → H20 GPU. Verified on real hardware.
> github.com/zig-ecosystem/zoxide
