# cuda-oxide 对照 Port 计划(按优先级)

> 2026-09-24。基准:NVlabs/cuda-oxide(本机 `../cuda-oxide`,catalog.json 1025 条 intrinsic)。
> 现状梳理见 `docs/PORTING.md`;本文只回答"接下来按什么顺序补什么、每个阶段怎么算做完"。
> 判据风格沿用 `docs/tma-plan.md`:每个阶段先立**可在无 GPU 机器上否掉**的门,再立真机门。

## 优先级总览

| 优先级 | 阶段 | 内容 | 为什么是这个位置 |
|---|---|---|---|
| P0 | 收尾在途工作 | TMA S0 真机验证;wgmma4 真机跑 | 代码已提交,不验证就烂尾;tma-plan.md 的 H1–H3 判据已写好 |
| P1 | intrinsics 广度 | 从 catalog 补生成:cp.async、mbarrier 完整版、ldmatrix/stmatrix、redux、packed 类型、原子扩展 | 最大实质缺口(1025 vs 329+618);生成器驱动,机械、低风险、每个条目可独立验证 |
| P1 | 数学库 | libdevice(`__nv_*`)映射 | 没有它,任何超越多项式的数值 kernel 都写不了;依赖 P1 的 asm 生成基建 |
| P2 | TMA 补全 + 加速器面 | s2g、multicast;bf16/int8/fp8 mma、sparse、tcgen05(`.reg` 绕法) | 依赖 P0 的 TMA 验证结论;tcgen05 需要 Blackwell 真机,当前环境没有 |
| P3 | 工程化 | PTX parse/lint、差分 fuzzer、竞态扰动、sanitize/debug 子命令、artifact 嵌入、async 运行时 | 不影响"能不能写 kernel",影响"敢不敢依赖它";v1.0 前再排 |
| — | 明确不做 | MIR/多方言 IR 管线、cluster 系、VAMM/peer access、mdBook | 架构性差异或当前无硬件/无需求,见文末 |

## P0 — 在途收尾(先于一切新能力)

前提:`docs/tma-plan.md` 的 H1–H3 判据已立,这一阶段只执行,不重新论证。

- [ ] **TMA S0 真机验证**(hgemm_tma 或独立 smoke kernel)
  - 无 GPU 门:H1——PTX 中地址运算指令(add.s64/shl/and/or/mul.wide)从 144 降到 70 以下。不满足 → 停止,按 tma-plan.md H1 反向对照处理。
  - 真机门:H2(regs 下降)+ H3(吞吐 ±3% 以内为预期;显著变化要回头查 v0.0.9 排除逻辑)。
  - 证据写入 `docs/verification/`,ROADMAP 追加一行。
- [ ] **hgemm_wgmma4(m64n128k16)真机首跑**
  - PTX 级已验证(`docs/upstream-asm-output-limit.md`),只欠真机。
  - 注意 `.reg` 绕法失去 LLVM 寄存器记账,溢出代价实测 0.55×——首跑第一件事是看 ptxas 报告有无 local spill,有 spill 直接记录为".reg 路线代价数据",不为它调优。
  - 若结果精确且无 spill:与 wgmma3 对比吞吐,判定"n16 单指令成本"假设是否被宽 N 摊薄。

**完成定义**:两条 H20 实测记录进 `docs/verification/`;ROADMAP 的版本表追加 v0.0.13/v0.0.14 行。

## P1 — intrinsics 广度 + 数学库

策略:不手写。cuda-oxide 的 `intrinsics/catalog.json` 已有全部 1025 条的 ABI、effects、PTX ISA floor 和 libNVVM 验证证据;zoxide 的 `src/gen` 已经证明"catalog → Zig 绑定"这条生成路径可行(329 NVVM + 618 asm)。这一阶段是**把生成器的消费面从"已映射子集"扩到剩余条目**,按主题分批,每批独立提交、独立 smoke 验证。

依赖确认(开工第一步):catalog 中剩余 ~470 条(1025 − 已映射)逐条过一遍三分类:
1. 可直接生成(asm 操作数 ≤15 输出 / ≤31 输入)→ 进生成器;
2. 超宽指令(tcgen05 等)→ 走 `.reg` 函数级声明绕法(先例:`hgemm_wgmma4`,注意 spill 代价);
3. 语义已由 Zig 原生覆盖(如 f16 算术,ROADMAP f16 节结论)→ 跳过,记录理由。

分批(每批 = 一个 alpha 版本):

- [ ] **P1a:async copy + mbarrier**——cp.async(g2s,zfill 变体)、mbarrier init/arrive/expect_tx/try_wait/test_wait。理由:TMA 和后续所有 Hopper 流水 kernel 的同步底座,P0 的 TMA 线已在用 mbarrier 子集,补全是顺手的事。
- [ ] **P1b:ldmatrix / movmatrix / stmatrix**——wgmma/mma 的 smem→寄存器装载路径,直接影响 P2 加速器面。
- [ ] **P1c:warp 级补全**——redux.sync(sum/min/max,f32/int)、lanemask 系列、active_mask、shuffle 的 u64/f64 与带 sync 变体。理由:规约 kernel 的性能件,实现成本极低。
- [ ] **P1d:packed 类型与转换**——f16x2/bf16x2/f32x2/i16x2 的 cvt/prmt/clc/dotprod、fp8↔f16 打包转换。理由:bf16/fp8 mma(P2)的前置。
- [ ] **P1e:原子扩展**——atomic 全家(作用域语义、cas、packed atomic add、atomicAdd f16/f64/bf16 等)。理由:独立小批,不阻塞别人。
- [ ] **P1f:libdevice 映射**——`__nv_*` 数学函数到 Zig 的绑定层(对标 cuda-oxide 的 libdevice 映射)。形态建议:生成 wrapper + 链接期解析 libdevice bitcode(若 Zig NVPTX 后端支持)或逐条转 PTX 近似指令。开工先验证哪种形态可行,再定实现。

**每批的完成定义**(统一):
- 无 GPU 门:每条新 wrapper 有 smoke kernel,`zig build kernels` 产出的 PTX 含目标指令(grep 断言,防上游回退,先例:f16_native CI 断言)。
- 真机门:smoke kernel 在 H20 pod 上 run PASS,结果精确。
- catalog 覆盖率的口径更新进 README/PORTING.md。

## P2 — TMA 补全 + 加速器面

前提:P0 的 TMA 结论(尤其 H3——若 TMA 吞吐预期为空结果,P2 的动机就只剩"能力完整性",优先级可降)。

- [ ] **TMA s2g + multicast**:s2g wrapper 已有生成物,补 smoke 验证;multicast 依赖 cluster(tma-plan.md 已注明 cluster 不排期),multicast 随之挂起,除非届时有集群硬件需求。
- [ ] **mma 形状扩展**:bf16、int8/int4、fp8/f6/f4 的 mma.sync 形状。依赖 P1b(ldmatrix)与 P1d(packed/cvt)。每个形状 = 一个 bench 变体进 hgemm 家族,沿用现有"精确结果 + 峰值占比"口径。
- [ ] **sparse mma**:catalog 有条目,等 P2 mma 基建成熟后按同一模式生成。
- [ ] **tcgen05(Blackwell)**:仅当拿到 sm_100 真机才排期。`.reg` 绕法已验证可行,但 spill 代价数据(wgmma4 首跑会产出)决定这条路值不值得走。当前标注 blocked-on-hardware。

## P3 — 工程化(v1.0 稳定化前置)

按对"v1.0 敢不敢承诺 API 冻结"的贡献排序:

- [ ] **PTX parse/lint**(对标 ptx-parse):无损文本视图,先在内部消费——bench 报告直接解析 ptxas 输出而非正则,upstream-asm-output-limit 类分析自动化。
- [ ] **差分验证**:对标 fuzzer 的最小形态——同一 kernel 的 Zig 产 PTX 与 nvcc 参考实现做数值对拍,接入 pod-verify;竞态扰动(ptx-schedule 对应物:插 nanosleep 暴露同步 bug)列为候选,视 mbarrier/cluster 使用密度决定。
- [ ] **`zoxide sanitize` / `zoxide debug` 子命令**:封装 compute-sanitizer / cuda-gdb,doctor 分级模式照搬。
- [ ] **artifact 嵌入 host 二进制**(对标 `#[cuda_module]` 的 oxide-artifacts):消除对外挂 .ptx 文件路径的依赖,是"下游包分发"形态的前提。
- [ ] **async 运行时**(对标 cutile-rs cuda-async):惰性 DeviceOperation + `.sync()`。注意 cuda-oxide 本体已不含这部分(迁去 cutile-rs),对标边界以 crates.io 0.3.1 为准。
- [ ] **`launch_bounds` 等价物**(对标 `#[launch_bounds]`/`#[launch_contract]`):把 bench 的 `--maxrregcount` 从 CLI 参数下沉为 kernel 签名上的 comptime 属性,与类型化 launch 校验合并。
- [ ] **arch 矩阵扩展真机验证**:当前仅 sm_90/H20;v1.0 前至少再覆盖一档(sm_80 或 sm_100,视可及硬件)。

## 明确不做 / 挂起

| 项 | 理由 |
|---|---|
| MIR → 多方言 IR → PTX 的编译器管线 | Zig 自带 NVPTX 后端已替代该路径;这是 Rust 编译器生态的产物,不是能力缺口 |
| cluster 系(cluster barrier/mcast barrier/cluster memory) | 无集群硬件需求,tma-plan.md 已注明不排期;`cluster_launch`/`cooperative_launch` 随之挂起 |
| VAMM / peer access | 单卡工具链阶段无消费者 |
| mdBook 文档工程 | 有 README + PORTING + ROADMAP 三角已够;v1.0 再说 |
| f16/bf16 人性化包装层 | 已测量证伪(ROADMAP v0.0.8 节:Zig 原生降到 packed 指令) |

## 执行方式

每个阶段开工时,把对应 checkbox 组展开成独立执行计划(TDD 粒度:失败测试 → 实现 → 真机验证 → 提交),不一次性展开全文——P1 各批之间、P2 各条目之间相互独立,适合 subagent 并行;P0 两项必须在 P1 之前完成,因为 TMA/mbarrier 的验证结论会改变 P2 的优先级论证。
