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

- [x] **TMA S0/S1 实现 + S2 H1 判定**(2026-09-25,H1 数据:**不成立**,144→135 即 −6.25%,判据要求 <70)
  - 判定依据:tma-plan.md S2 节(地址运算主体在 wgmma descriptor 打包与 C 写回路径,不在载入路径;反向对照条款生效,S3 吞吐测量不再进行)。
  - 遗留:H2(ptxas regs)与 tma_smoke 真机 PASS 需 GPU,挂起至 GPU 环境;TMA 线优先级随之整体下调,P2 的 TMA 补全动机只剩"能力完整性"。
- [ ] **hgemm_wgmma4(m64n128k16)真机首跑**
  - PTX 级已验证(`docs/upstream-asm-output-limit.md`),只欠真机。
  - 注意 `.reg` 绕法失去 LLVM 寄存器记账,溢出代价实测 0.55×——首跑第一件事是看 ptxas 报告有无 local spill,有 spill 直接记录为".reg 路线代价数据",不为它调优。
  - 若结果精确且无 spill:与 wgmma3 对比吞吐,判定"n16 单指令成本"假设是否被宽 N 摊薄。

**完成定义**:两条 H20 实测记录进 `docs/verification/`;ROADMAP 的版本表追加 v0.0.13/v0.0.14 行。

## P1 — intrinsics 广度 + 数学库

策略:不手写。cuda-oxide 的 `intrinsics/catalog.json` 已有全部 1025 条的 ABI、effects、PTX ISA floor 和 libNVVM 验证证据;zoxide 的 `src/gen` 已经证明"catalog → Zig 绑定"这条生成路径可行(329 NVVM + 618 asm)。

**2026-09-25 普查订正**:以 `id` 在 `src/gen/` 两个产出中匹配计,实际已生成 **943/1025**,缺失只有 82:tcgen05 71、tma 10、wgmma_control 1——全部是 asm 输出 >15 的超宽指令,属 asm 上限受限项,不是"没做"。因此 P1 的重点从"扩大生成面"修正为:**验证**(逐 family 的 smoke kernel + PTX grep 断言,防上游回退)、**人性化层**(src/cuda.zig 补高层封装,如 redux/shuffle 更多类型/原子扩展)、以及 P1f libdevice。剩余 82 条归入 P2(tcgen05 需 Blackwell;tma 缺的 10 条需逐条看)。

分批(每批 = 一个 alpha 版本)——**2026-09-25 全部完成**(无 GPU,验收=编译+PTX 断言):

- [x] **P1a:async copy + mbarrier**——35/35 过编译且入 PTX(`cpasync_mbar_smoke.ptx`)。发现:asm 侧 mbarrier wrapper 收 u64 地址,smoke 侧零扩展适配,真机用前生成器应改 u32。
- [x] **P1b:ldmatrix/stmatrix/movmatrix**——ldmatrix 18/18 + movmatrix 1/1;**stmatrix 4 条 LLVM NVPTX 不 lower**,手写 asm 覆盖(`ldmatrix_smoke.ptx`,sm_100a);ldmatrix b8/b8x16 需 sm_100a(sm_90a 上 Cannot select)。
- [x] **P1c:warp 级补全**——36/38(redux 16、vote 4、shuffle 12、match 2/4、activemask、bar.warp.sync,`warpops_smoke.ptx`);**match.all 2 条**因 {i32,i1} 聚合返回 LLVM 不支持不可用;f32 redux 需 sm_100a。
- [x] **P1d:packed 类型与转换**——127/127(packed_alu/conversion/atomic、dotprod、prmt、clc、minmax 全家,`packed_smoke.ptx`);dp2a 的 2 条 LLVM 无选择模式,手写 asm 覆盖。
- [x] **P1e:原子扩展**——cuda.zig 人性化层扩展:atomicAdd f64、atomicMin/Max/And/Or/Xor/Exch/Cas(走 @atomicRmw);red.* 与 inc/dec(PTX 独有)手写 asm,`atomics_smoke.ptx` 30 形态。发现:catalog 中普通 atom/red 条目为 0,gen 无从生成;scoped 变体(@atomicRmw 无法选 scope)未纳入。
- [x] **P1f:libdevice 映射**——三条路线实测(docs/libdevice-path.md):zig 直链 .bc 不可行;llvm-link 手工管线可行但工具链耦合+许可灰色;**采用路线 3**:src/math.zig 三层(fast=单指令 approx、native=Zig 内建、software=多项式),`math_smoke.ptx`。f64 超越函数与大参数精确 sin/cos 为已知边界。

**每批的完成定义**(统一):
- 无 GPU 门:每条新 wrapper 有 smoke kernel,`zig build kernels` 产出的 PTX 含目标指令(grep 断言,防上游回退,先例:f16_native CI 断言)。
- 真机门:smoke kernel 在 H20 pod 上 run PASS,结果精确。
- catalog 覆盖率的口径更新进 README/PORTING.md。

## P2 — TMA 补全 + 加速器面

前提:P0 的 TMA 结论**已出且为否**(H1 不成立,见 tma-plan.md S2)——TMA 对 hgemm 吞吐的动机已被数据否定,P2 的 TMA 项动机只剩"能力完整性",优先级确认下调;TMA s2g/multicast 如无外部需求可继续挂起。

- [ ] **TMA s2g + multicast**:s2g wrapper 已有生成物,补 smoke 验证;multicast 依赖 cluster(tma-plan.md 已注明 cluster 不排期),multicast 随之挂起,除非届时有集群硬件需求。
  - **tma 缺失条目逐条结论**(2026-09-29,对 upstream HEAD catalog 1029 条;注意上游从 1025 涨到 1029,新增的是 cache_hint 变体):缺失 13 条 = g2s 7 + prefetch 6。
    - g2s 7 条(1d/2d/3d/4d/5d + 2d multicast×2):LLVM 无 g2s 方向 intrinsic,生成器无产出(tma-plan S0 已记);其中 2d/3d 已在 `src/tma.zig` 手写 asm 覆盖,1d/4d/5d 同模板可即时补齐;multicast 2 条随 cluster 挂起。
    - prefetch 6 条(1d_l2、5d_l2、gather4_2d_l2、2d/3d/4d 的 cache_hint):fire-and-forget,无 mbarrier 依赖,是 13 条里成本最低的一批;NVVM 已暴露部分非 cache_hint 形式(2d/4d_l2 已生成),补法 = 生成器补 cache_hint 维度或手写 6 条 asm。
  - 同次普查的附带订正:packed_atomic 缺 2 条是**误报**(`packed_atomic_add_{f16x2,bf16x2}` 已生成为 `atom_add_{f16x2,bf16x2}`,fn 名 ≠ catalog id);`wgmma_wait_group` 缺 1 条同理(手写在 `src/wgmma.zig`);sparse_mma 缺 4 条为 fp8(e4m3/e5m2)m16n8k64 非 ordered_metadata 形,归 P2 sparse mma 条目;sreg 12 条为无 probe 的原始寄存器读,常用项已手写在 cuda.zig。
- [ ] **mma 形状扩展**:bf16、int8/int4、fp8/f6/f4 的 mma.sync 形状。依赖 P1b(ldmatrix)与 P1d(packed/cvt)。每个形状 = 一个 bench 变体进 hgemm 家族,沿用现有"精确结果 + 峰值占比"口径。
  - 进展(2026-09-29):**bf16 形状完成(PTX 级)**——`hgemm_bf16`(hgemm_mma2 同构,mma.sync m16n8k16 bf16,Zig 无 bf16 类型故以 u16 位模式传输),bench 变体 + CI PTX grep 断言已就位;真机计时/正确性验证挂起至 GPU 环境。int8/fp8 等其余形状未动。
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
