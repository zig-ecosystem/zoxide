# zoxide roadmap

> 2026-09-20 起生效。版本节奏：探索性迭代用 alpha，阶段性成果用 beta，稳定版不带后缀。
> 每次发版在 `docs/announcement.md` 追加公告 + 配图。
> 历史与详细能力对照见 `docs/PORTING.md`；真机证据见 `docs/verification/`。

## 已完成

| 版本 | 内容 |
|---|---|
| v0.1.0 | M0–M3：工具链、设备端库、host runner、intrinsics 生成器；H20 真机 4/4 PASS |
| v0.0.2-beta | SGEMM 线：6 个变体，6.3% → 42.8% FP32 峰值，全链诊断证据 |
| v0.0.3-alpha | 验证基建：pod-verify.sh、k8s Job 模板、gpu_printf |
| v0.0.4-alpha | asm 类 intrinsics 生成：618 条 inline-PTX wrapper，catalog 覆盖 92% |
| v0.0.6-alpha | FP16 tensor core（mma.sync）：hgemm_mma2 53.8 TF / 36.4% 峰值，结果精确 |
| v0.0.7-alpha | Hopper warpgroup MMA：hgemm_wgmma 80.3 TF / 54.3% 峰值，结果精确，1.49x |
| v0.0.8-alpha | wgmma 三级流水：hgemm_wgmma2 86.3 TF / 58.3% 峰值，结果精确，累计 1.60x |

## 规划

### v0.0.3 — 验证基建与开发体验（alpha → beta）

目标：让"改代码 → 验证"成为一键动作，降低外部贡献门槛。

- [x] **pod 端到端回归脚本**（M4b）：`scripts/pod-verify.sh`——doctor + 全部示例 run PASS + bench 冒烟，输出结构化报告（JUnit 或纯文本表格），失败定位到具体 kernel；README 附 k8s Job yaml 示例
- [ ] **CI GPU 扩展**：支持 self-hosted GPU runner 时自动跑 pod-verify（workflow 里 `runs-on: [self-hosted, gpu]` 的可选 job，无 runner 时跳过）
- [x] **gpu_printf**：`cuda.printf(comptime fmt, args)`——注意 `llvm.nvvm.vprintf` 已被 LLVM 移除，正确路径是调用字面名为 `vprintf` 的函数；valist 为 8 字节小端 slot 打包
- [ ] `zoxide ptx` 支持 examples 风格的多文件/模块输入（当前只支持单文件）

### v0.0.4 — asm 类 intrinsics 生成（已解锁，M4d）

**前提修正（v0.0.3-alpha）**：Zig asm 支持 `%[name]` 具名操作数替换——之前只试了 `$0`/`%0`/`${0}`/`$[name]` 四种不支持的形式。生成器已升级：`zoxide gen` 把 probe 里的 LLVM 位置模板（`$N`）重写为具名操作数，asm 类条目从"689 条不可映射"变为 **618 条已生成**（`src/gen/instrinsics_asm.zig`），含 mma.sync 多输出形式；剩余 163 条未映射的主因是 zig asm 操作数上限（输出 ≤15、输入 ≤31，AstGen 硬性限制）挡住 tcgen05.ld 等超宽指令。

- [x] 模板转换器（$N → %[name]，约束 r/l/f/d/h/n 映射，immarg 用 comptime "n" 约束）
- [x] 多输出 asm（≤15 个输出，struct 返回）与 smoke kernel（asm_smoke.ptx 含 abs.bf16x2/mul.rn.f32x2/mma.sync）
- [x] **真机验证**：fp16 mma kernel 在 H20 上跑通，结果精确（hgemm_mma 36.8 TF → hgemm_mma2 53.8 TF → hgemm_wgmma 80.3 TF / 54.3% FP16 峰值）
- [ ] tcgen05 等超宽指令（>15 输出）的替代路线——**已确认 15 输出上限也卡住 wgmma 宽 N 形态**（`m64n32k16` 需 16 个累加器寄存器，正好超 1 个），且无法靠拆调用绕开：一条 wgmma 的累加器必须在同一操作数列表。上游补丁建议（`>= 16` → `> 32`，ZIR 侧无需改动）见 `docs/upstream-asm-output-limit.md`

### v0.0.8 — wgmma 流水深化 ✅
- [x] 三级缓冲 + `wait_group 1`：wgmma 与下一段 cp.async 重叠，58.3% 峰值（+4.0pp）
- [x] ~~共享内存 swizzle~~ —— **撤销，经分析无效**：tile 已是 core-matrix packed，一个 core matrix
  是 128 字节连续，共享内存 32 banks × 4B = 128 字节一轮，单次读取已完整跨遍所有 bank，
  无冲突可消。swizzle 模式针对的是保持宽行距的布局（TMA 产出那种）

### v0.0.9 — 定位剩余 41.7pp（先测量，再优化）
流水线已排除（只值 4pp）。剩下两个候选无法用规格书算出谁是主因：
- n16 的 shape 代价：共享内存操作数流量 3.3 倍（12.8 vs 42.7 flops/字节）
- 全局流量：3.22 GB/趟 = 2.02 TB/s，占 HBM 标称 ~4 TB/s 的 51%

- [ ] **ncu profile 分流**（最便宜且决定性）：`sm__pipe_tensor_op_hmma_cycles_active`
  高但吞吐上不去 → 瓶颈在喂数据（指向 n16）；`gpu__dram_throughput` 接近饱和 → 先降流量
- [ ] M=128 / 双 warpgroup（全局流量 3.22 GB → 约 2 GB）。不依赖上游，且同样能分流：
  大涨说明受制于流量，微涨说明 n16 是墙
- [ ] 打补丁的 zig（output 上限 32）编 `m64n64k16`，测出上限到底值多少性能
- [ ] 上游提交（ziglang/zig issue 创建受限于 collaborator，走 `docs/drafts/` 里记录的 fallback 渠道）
- [ ] TMA 替代 cp.async

### v0.0.5 — 类型化启动与单文件体验

- [ ] `@embedFile` cubin + comptime 生成类型化 launch（对标 cuda-oxide `#[cuda_module]`）：kernel 参数在编译期检查类型/数量
- [ ] host+device 同文件/同包的标准项目模板（`zoxide new` scaffold）
- [ ] launch 参数校验（block/grid vs device 限制）

### v1.0.0 — 稳定化

- [ ] API 冻结（cuda.zig / driver / gen 的公共接口）
- [ ] arch 矩阵扩展验证（sm_80/sm_86/sm_100，取决于可用硬件）
- [ ] 文档站或完整 mdBook；错误信息全面审查
- [ ] 依赖 zig 版本策略明确（跟随 stable 还是钉版本）

## 风险与开放问题

| 项 | 状态 | 影响 |
|---|---|---|
| ~~Zig asm 无模板替换~~ → 已证伪（`%[name]` 具名替换可用）；遗留：asm 操作数上限（15 出/31 入） | 已解决/残余跟踪 | tcgen05 等超宽指令待替代路线 |
| `llvm.nvvm.*` 调用是意外暴露能力（ziglang/zig#2291） | 跟踪上游 | zig 升级可能破坏 |
| zig 0.16 std API 不稳定 | 已钉 0.16.0 | 升级成本 |
| 单 arch（sm_90）单平台（H20 pod）验证 | 开放 | 泛化性待证 |
| ncu 不可用（pod 权限） | 已知限制 | 深度调优靠 PTX 审查 + 对照实验 |

## 决策记录

- 2026-09-17：走"Zig 实现"路线（非代码移植）；arch 基线 sm_90
- 2026-09-18：版本号 alpha/beta 节奏；host 绑定手写 extern（非 @cImport）
- 2026-09-20：pod 上 `zoxide run` 用 gnu 动态构建（musl 静态 dlopen 不可靠）；发版必配 announcement.md 条目 + 图片
