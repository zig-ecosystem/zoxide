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

- [x] **P1a:async copy + mbarrier**——35/35 过编译且入 PTX(`cpasync_mbar_smoke.ptx`)。发现:asm 侧 mbarrier wrapper 收 u64 地址,smoke 侧零扩展适配,真机用前生成器应改 u32。**已修(2026-09-30)**:根因是 probes 对 asm 地址操作数一律用 "l"(64 位)约束,而 PTX shared 窗口是 32 位;gen.zig 加了规则(`.shared` 指令里作 `[addr]` 的 "l" 输入 → u32/"r"),6 条 mbarrier wrapper + test_wait/try_wait 的 hint 操作数已在提交文件中按同一规则收窄(目录数据不在仓库内,完整 regen 属另一决策,头部有例外注释);PTX 同一指令、地址寄存器由 %rd 变 %r,cpasync smoke 的零扩展 shim 已删,tma.zig 的过时注释已订正。
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
  - 进展(2026-09-30):**s2g smoke 完成(PTX 级)**——`tma_s2g_smoke`,按 one-smoke-one-claim 独立成 kernel,g2s 进 shared(复用 tma_smoke 已验证路径,只使用不重复断言)→ 生成物 `gen.tma.cp_async_bulk_tensor_2d_s2g` 写回**另一个** tensor,host 全量比对(瓦片精确 + 哨兵区完好,坐标仍取非原点 (64,16))。**完成机制差异已写入 kernel 注释**:s2g 无 mbarrier 操作数,走 `cp.async.bulk.commit_group` + `cp.async.bulk.wait_group.read 0`(`.read` 只保证 shared 可读复用,全局可见性由 host 侧 cuCtxSynchronize 兜底)。PTX 双向指令、commit/wait_group、无残留,CI 断言已加;pod-verify.sh 2g 段已登记(降级模式 SKIP,arch 拒绝按 SKIP 处理);真机运行挂起至 GPU 环境。multicast 仍随 cluster 挂起。
  - **tma 缺失条目逐条结论**(2026-09-29,对 upstream HEAD catalog 1029 条;注意上游从 1025 涨到 1029,新增的是 cache_hint 变体):缺失 13 条 = g2s 7 + prefetch 6。
    - g2s 7 条(1d/2d/3d/4d/5d + 2d multicast×2):LLVM 无 g2s 方向 intrinsic,生成器无产出(tma-plan S0 已记);其中 2d/3d 已在 `src/tma.zig` 手写 asm 覆盖,1d/4d/5d 同模板可即时补齐;multicast 2 条随 cluster 挂起。
    - prefetch 6 条(1d_l2、5d_l2、gather4_2d_l2、2d/3d/4d 的 cache_hint):fire-and-forget,无 mbarrier 依赖,是 13 条里成本最低的一批;NVVM 已暴露部分非 cache_hint 形式(2d/4d_l2 已生成),补法 = 生成器补 cache_hint 维度或手写 6 条 asm。**已补(2026-09-30)**,实际根因与预案不同:NVVM 每个维度只暴露**一个**带 (cache_hint, use_hint) 尾参的 intrinsic,plain 与 cache_hint 同体,生成器按 catalog id 命名只能中一个——故 1d/2d/3d/4d/5d 的 plain+cache_hint 在 `src/tma.zig` 以薄 wrapper 按 flag 选择(PTX 实测两种形态均正确发射);**gather4 的 intrinsic 不 lower**(PTX 里残留未解析 extern 调用,实测),2 条 gather4 为手写 asm(同 g2s 先例)。验证:tma_s2g_smoke 加 prefetch-then-load(2d 真实用法 + gather4 两形态编译覆盖),CI 断言三条指令均在。
  - 同次普查的附带订正:packed_atomic 缺 2 条是**误报**(`packed_atomic_add_{f16x2,bf16x2}` 已生成为 `atom_add_{f16x2,bf16x2}`,fn 名 ≠ catalog id);`wgmma_wait_group` 缺 1 条同理(手写在 `src/wgmma.zig`);sparse_mma 缺 4 条为 fp8(e4m3/e5m2)m16n8k64 非 ordered_metadata 形,归 P2 sparse mma 条目;sreg 12 条为无 probe 的原始寄存器读,常用项已手写在 cuda.zig。
- [ ] **mma 形状扩展**:bf16、int8/int4、fp8/f6/f4 的 mma.sync 形状。依赖 P1b(ldmatrix)与 P1d(packed/cvt)。每个形状 = 一个 bench 变体进 hgemm 家族,沿用现有"精确结果 + 峰值占比"口径。
  - 进展(2026-09-29):**bf16 形状完成(PTX 级)**——`hgemm_bf16`(hgemm_mma2 同构,mma.sync m16n8k16 bf16,Zig 无 bf16 类型故以 u16 位模式传输),bench 变体 + CI PTX grep 断言已就位;真机计时/正确性验证挂起至 GPU 环境。
  - 进展(2026-09-29):**int8 形状完成(PTX 级)**——`imma_s8`,`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`,i8 输入 / i32 累加器,128x128 tile + k_slice 32,复用 hgemm_bf16 的 cp.async 双缓冲与 ldmatrix.x4 A 路径。
    - **架构门是 sm_80,不是 sm_75**(catalog 谓词:`getSmVersion() >= 80` + `getPTXVersion() >= 70`)。sm_75 只支持 `m8n8k16` 那个小形状(A 1 reg / B 1 reg / C 2 reg)。建计划时记的 sm_75 口径是错的,已订正。
    - **B fragment 无法走 ldmatrix**,这是指令层面的固有不匹配而非疏漏:s8 的 B 寄存器要的是"同一列、k 方向 4 个连续字节",而 ldmatrix 每 lane 只发 4 个**内存连续**字节,`.trans` 换的是哪个轴连续,仍只给每个 8x8 矩阵 2 个 k 行。CUTLASS 靠 store 时把 B 置换成 crosswise 布局绕开,而 cp.async 的 16B chunk 只能搬全局连续字节,这条路在此不可用。因此 B 以 128 条 `ld.shared.b8` 手工打包——**先正确,置换优化列为后续**。代价在 PTX 里可见(无 `ldmatrix.*.trans`),预期吞吐显著低于 int8 峰值,CI 已把"无 trans"和"128 条 b8"一起断言,将来任一侧改动会强制同时复核。
    - 验证:`zig build kernels` 产出 64 条目标 mma、8 条 ldmatrix.x4、128 条 ld.shared.b8、2 条 bar.sync、无 `$N`/`%[` 残留、无 `__local_depot`(即无显式 spill),CI 断言已加。bench 变体 `runImmaS8` 走**精确整数相等**校验(s8×s8→s32 无舍入),输入取满 i8 范围以覆盖符号扩展与小端打包;峰值口径 H20 INT8 296 TOPS(与仓库既有 148/44 同一份规格表)。
    - **未验证项**:本机无 CUDA 工具链,`ptxas` 不可用,故寄存器分配/spill 与 occupancy 未经检查(`.reg %r<2351>` 是虚拟寄存器数,不代表实际压力);真机计时与 PASS 同 bf16 一并挂起至 GPU 环境。
  - 进展(2026-09-30):**int4 形状完成(PTX 级)**——`imma_s4`,`mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`(选 k64 而非 k32,每 mma 的 K 更深)。s4 无字节类型,A/B 按"每字节两个值、低半字节 = 偶数 k"打包(`examples_abi.packS4` 一处定义,host/device 不会漂移);A 行 64 个 s4 = 32B,imma_s8 的 ldmatrix.x4 寻址原样复用,B 同样无法走 ldmatrix(每寄存器要 8 个 k 连续值 + 半字节提取),以 8 条 `ld.shared.b8`/寄存器手工打包(全 kernel 256 条),trans 守卫与 s8 同款。
    - 验证:64 条目标 mma、8 条 ldmatrix.x4、256 条 ld.shared.b8、无 trans、无 `$N`/`%[`、无 `st.local` spill,CI 断言已加;bench 变体 `runImmaS4` 精确整数相等校验,输入取满 [-8,7](最坏 |sum| = n·8·8 = 262144 @ n=4096,深在 i32 内)。
    - **峰值口径是假设**:仓库规格来源里没有 H20 INT4 数字(Hopper 官方不宣传 INT4),按惯例取 INT8 的 2 倍 = 592 TOPS,注释与输出里均标注为假设而非实测规格。
    - fp8(e4m3/e5m2)未做:catalog 标 minimum_sm 89,sm_90a 是否接受有争议(CUTLASS 在 Hopper 上 FP8 走 wgmma),留待单独评估。
  - fp8 形状:`mma.sync m16n8k16` 的 fp8 变体 catalog 已有 8 条(e4m3/e5m2 × f16/f32 累加),但 **sm_90 上的可用性有架构疑问**,留后处理,不与 int8 合并推进。int4/f6/f4 未动。
- [ ] **sparse mma**:catalog 有条目,等 P2 mma 基建成熟后按同一模式生成。
  - 进展(2026-09-30):**f16 sparse 完成(PTX 级)**——`hgemm_sp`,`mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`。选型说明:src/gen 中**没有** plain `mma.sp.sync` 的 f16 封装(32 条 plain 条目全是 u4/s4/u8/s8 整数形),f16 sparse 只有 ordered_metadata 形;sm_80 起步,H20 可验证。
    - 关键事实(A fragment 与 metadata 读取,按 PTX ISA 解读,已写入 kernel 注释,真机首跑由精确比对证伪):A 以"已剪枝"形式存储(每 4 个 k 保 2,行宽减半 = 8 f16/16B),A fragment 每 lane 2 个 .b32(行 g 的 k-group t 两个保留值 + 行 g+8 同位),plain `ldmatrix.x2` 可直接加载(无需 dense 的 x4);B 保持 dense,沿用 `.x2.trans` 路径;metadata 每 mma 一个 32 位寄存器,由 selector 立即数选中的 lane 提供(取 0),低 16 位 = 行 g、高 16 位 = 行 g+8,行内 k-group j 占半字节 [4j+3:4j](低 2 位 = 第一个保留下标,高 2 位 = 第二个)。
    - 正确性口径:host 自行剪枝 A(伪随机 6 选 2 组合),参考为**剪枝后矩阵**的 dense CPU matmul,输入小整数精确 → 实际等价于零容差;metadata 位序/lane 读错会在首跑硬 FAIL。峰值按"sparse = 2× dense = 296 TFLOPS"假设标注(同 imma_s4 的处理)。
    - 验证:64 条 mma.sp、8 条 plain ldmatrix.x2、16 条 x2.trans、16 条 metadata 的 ld.shared.b16、2 条 bar.sync、无残留、无 st.local;CI 断言已加。int8/int4 sparse 形状未做,同模式可续。
- [ ] **tcgen05(Blackwell)**:仅当拿到 sm_100 真机才排期。`.reg` 绕法已验证可行,但 spill 代价数据(wgmma4 首跑会产出)决定这条路值不值得走。当前标注 blocked-on-hardware。

## P3 — 工程化(v1.0 稳定化前置)

按对"v1.0 敢不敢承诺 API 冻结"的贡献排序:

- [x] **PTX parse/lint**(对标 ptx-parse):无损文本视图,先在内部消费——bench 报告直接解析 ptxas 输出而非正则,upstream-asm-output-limit 类分析自动化。(2026-09-30 第一刀落地:`src/ptx.zig` 语句级无损视图 + `zoxide lint` / `--census`,CI 全量 lint,8 个单测含 38 份 kernel PTX 往返。范围修正:bench 资源占用读的是 driver attr,本无正则可换,首个内部消费者改为 census;操作数保持 span,无完整文法。)
- [ ] **差分验证**:对标 fuzzer 的最小形态——同一 kernel 的 Zig 产 PTX 与 nvcc 参考实现做数值对拍,接入 pod-verify;竞态扰动(ptx-schedule 对应物:插 nanosleep 暴露同步 bug)列为候选,视 mbarrier/cluster 使用密度决定。**blocked-on-hardware**:需要 nvcc(CUDA 工具链)+ GPU 做数值对拍,本机两者皆无;不做无硬件的空壳 harness。
- [x] **`zoxide sanitize` / `zoxide debug` 子命令**:封装 compute-sanitizer / cuda-gdb,doctor 分级模式照搬。(2026-09-30 落地,`src/toolwrap.zig` 两个子命令共享一条 probe+exec 路径:probe 顺序 PATH → $CUDA_HOME/bin → /usr/local/cuda/bin,与 ptxas 同源(findPtxas 已重构为调用共享 probe);参数逐字透传,`--` 可选;exec 用 `std.process.replace`,exit code 即工具的,cuda-gdb 保持交互。缺席行为照搬 cubin(报错点名探测位置,exit 1)而非 doctor 的 warn——子命令的全部职责就是这个工具。范围说明:未做"缺省命令推导"之类的糖。验证:probe 顺序与 argv 组装 4 个单测;exec 路径用假 binary 验证(找到、透传、exit 42 透传);真工具本机不存在,真实 exec 未验证。)
- [x] **artifact 嵌入 host 二进制**(对标 `#[cuda_module]` 的 oxide-artifacts):消除对外挂 .ptx 文件路径的依赖,是"下游包分发"形态的前提。(2026-09-30 落地。差距分析:下游包形态本已由 `@embedFile("kernel_ptx")` + `moduleFromPtx` 覆盖(tests/downstream);真正缺口在 **CLI 自身**——run/bench/lint 只收文件路径,pod 部署须带 kernels/ 目录。落地:`-Dembed-kernels`(默认开;38 个 kernel 共 ~700KB PTX,二进制 +0.7MB),build.zig 生成 `embedded_kernels` 注册表模块(匿名 import 各 kernel 的 emitted asm),CLI 解析顺序 = 存在的显式路径 > 裸 stem 命中内嵌表 > 报错并列出可用 stem;带路径分隔符的输入永不回退内嵌(打错路径不能静默跑别的 kernel)。内嵌 PTX 落到 /tmp 临时文件复用既有 ptxas 流水线,stem 保留为 basename 以便 run/bench 的 kernel 匹配。验证:无 kernels/ 目录下 `run vector_add` 抵达 ptxas 阶段(而非 file-not-found)、`bench hgemm_bf16`/`lint vector_add` 同;临时文件用后清除。)
- [x] **async 运行时**(对标 cutile-rs cuda-async):惰性 DeviceOperation + `.sync()`。注意 cuda-oxide 本体已不含这部分(迁去 cutile-rs),对标边界以 crates.io 0.3.1 为准。(2026-09-30 落地为 `src/async.zig`(宿主侧 `gpu.async_ops`):**刻意做薄**——Operation = (stream, payload, runFn) 值,Builder 收集入 arena,`.sync()` = 按序 issue + 每条流 sync 一次;无 futures/DAG/调度器。**未做跨流等待**:`cuStreamWaitEvent` 绑定尚不存在,跨流依赖不能用 sync 造假,留待补绑定(需真机验证)。eager API 不变,单流代码应继续用它。测试 4 个(构造顺序/流归属、custom seam 的运行顺序与错误即停、launch payload 在 arena 中的存活、元素类型保持);真实执行路径挂起至 GPU。)
- [x] **`launch_bounds` 等价物**(对标 `#[launch_bounds]`/`#[launch_contract]`):把 bench 的 `--maxrregcount` 从 CLI 参数下沉为 kernel 签名上的 comptime 属性,与类型化 launch 校验合并。(2026-09-30 落地。
  - **机制实验**(命令:`zig build-lib scratch.zig -target nvptx64-cuda -mcpu sm_90 -femit-asm`):Zig 无任何函数属性可达 PTX(grep std 无 maxntid/maxnreg);但 kernel 体内的 `asm volatile` 原样进 `.entry` body——`.maxntid 128;`/`.maxnreg 64;` 实测落入 body,位于 ld.param 前导之后、计算指令之前(合法位置;ptxas 接受性属真机挂起项)。module-scope asm 不可行(`.maxntid` 在 `.entry` 外不合法)。
  - 落地:`kernel_abi.LaunchBounds`(max_threads / min_blocks_per_sm / max_registers / grid_multiple_of)+ 结构化读取(abi 模块声明普通字段即可,不 import kernel_abi——避免同一文件进两个 module 的编译冲突,此坑实测踩过);设备侧 `cuda.launchBounds(...)`(freestanding,自带 comptime itoa);宿主侧 `Module.kernel` 接受 WithBounds 形,`checkGeometry` 违例时报"declared launch bound"。示范:`hgemm_bf16`(max_threads=128),CI 断言 `.maxntid 128;` 且位于 `.entry` 之后。`--maxrregcount` CLI 保留为 ptxas 期实验旋钮。
  - 未做:minnctapersm 无宿主侧校验(occupancy 是驱动实测,声明值只进 PTX);ptxas 对指令位置的接受性待真机。)
- [ ] **arch 矩阵扩展真机验证**:当前仅 sm_90/H20;v1.0 前至少再覆盖一档(sm_80 或 sm_100,视可及硬件)。**blocked-on-hardware**:需要非 sm_90 硬件;本机(无 GPU)与 pod(H20)都不满足,无可替代的部分验证路径。

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
