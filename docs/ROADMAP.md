# zoxide roadmap

> 2026-09-20 起生效。版本节奏：探索性迭代用 alpha，阶段性成果用 beta，稳定版不带后缀。
> 每次发版在 `docs/announcement.md` 追加公告 + 配图。
> 历史与详细能力对照见 `docs/PORTING.md`；真机证据见 `docs/verification/`。

## 已完成

| 版本 | 内容 |
|---|---|
| v0.1.0 | M0–M3：工具链、设备端库、host runner、intrinsics 生成器；H20 真机 4/4 PASS |
| v0.2.0-beta | SGEMM 线：6 个变体，6.3% → 42.8% FP32 峰值，全链诊断证据 |

## 规划

### v0.3.0 — 验证基建与开发体验（alpha → beta）

目标：让"改代码 → 验证"成为一键动作，降低外部贡献门槛。

- [ ] **pod 端到端回归脚本**（M4b）：`scripts/pod-verify.sh`——doctor + 全部示例 run PASS + bench 冒烟，输出结构化报告（JUnit 或纯文本表格），失败定位到具体 kernel；README 附 k8s Job yaml 示例
- [ ] **CI GPU 扩展**：支持 self-hosted GPU runner 时自动跑 pod-verify（workflow 里 `runs-on: [self-hosted, gpu]` 的可选 job，无 runner 时跳过）
- [ ] **gpu_printf**：封装 `vprintf`（生成器已有映射），kernel 内可直接打印调试
- [ ] `zoxide ptx` 支持 examples 风格的多文件/模块输入（当前只支持单文件）

### v0.4.0 — asm 类 intrinsics 破局（M4d）

目标：解决 689 条 tcgen05/mma/TMA 条目的 Zig 不可用问题——这是当前最大的能力缺口（H20 的 FP16 tensor core ~148 TFLOPS 依赖它们）。

- [ ] **上游沟通**：向 ziglang/zig 提 issue（附最小复现：asm 模板无操作数替换 + NVPTX 无具名寄存器 + NVVM intrinsic 覆盖不足），请求：扩充 NVPTX 内建 / 官方支持 LLVM intrinsic 调用 / asm 模板替换。issue 草稿先在仓库 `docs/` 里评审再发出
- [ ] **备选路线评估**（与上游并行）：
  - PTX 级整段生成：kernel 的关键段落直接生成为完整 PTX 函数（绕过 Zig codegen，用 zoxide 工具链拼接）——评估维护成本
  - 预编译 cubin 嵌入：mma 微内核用 CUDA C/PTX 预先编译成 cubin，`@embedFile` + driver API 加载——作为过渡方案
- [ ] **里程碑验证**：一个 fp16 mma 微 kernel 在 H20 上跑通（哪怕走备选路线），bench 对比 FP32 的 42.8%

### v0.5.0 — 类型化启动与单文件体验

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
| Zig asm 无模板替换（asm 类 intrinsics 不可映射） | 开放，v0.4.0 主攻 | tensor core 能力缺口 |
| `llvm.nvvm.*` 调用是意外暴露能力（ziglang/zig#2291） | 跟踪上游 | zig 升级可能破坏 |
| zig 0.16 std API 不稳定 | 已钉 0.16.0 | 升级成本 |
| 单 arch（sm_90）单平台（H20 pod）验证 | 开放 | 泛化性待证 |
| ncu 不可用（pod 权限） | 已知限制 | 深度调优靠 PTX 审查 + 对照实验 |

## 决策记录

- 2026-09-17：走"Zig 实现"路线（非代码移植）；arch 基线 sm_90
- 2026-09-18：版本号 alpha/beta 节奏；host 绑定手写 extern（非 @cImport）
- 2026-09-20：pod 上 `zoxide run` 用 gnu 动态构建（musl 静态 dlopen 不可靠）；发版必配 announcement.md 条目 + 图片
