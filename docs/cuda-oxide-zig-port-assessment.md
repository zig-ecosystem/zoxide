# cuda-oxide Zig 移植可行性评估

> 评估日期：2026-09-17 · 源仓库：`~/work/code/cuda-oxide`（NVlabs, Apache-2.0, alpha）
> 本目录 `zig-ecosystem` 用于存放面向 Zig 生态的移植评估与后续移植产物。

## 一、项目是什么

cuda-oxide 是一个**自定义 rustc 代码生成后端**，让开发者用纯 Rust 编写 CUDA SIMT kernel（非 DSL、kernel 本身无 FFI）。单源编译：host 与 device 代码在同一个 crate 里，通过 `cargo oxide build` 构建。

编译流水线：

```
Rust 源码 → rustc MIR → Pliron（Rust 实现的类 MLIR IR）方言
         → 文本 LLVM IR (.ll) → llc（NVPTX 后端）→ PTX
         → ptxas / libNVVM+nvJitLink → cubin / LTOIR
```

## 二、代码结构（27 个 crate，约 59 万行 src，其中约 36 万行为机器生成）

| 层 | crate | 作用 |
|---|---|---|
| rustc 集成 | `rustc-codegen-cuda` | `-Zcodegen-backend` 动态库，`#![feature(rustc_private)]`，kernel 发现、MIR 收集，host 代码委托给 `rustc_codegen_llvm` |
| | `mir-importer` | 通过 `rustc_public`（stable MIR）把 Rust MIR 翻译成 `dialect-mir`（手写 ~4.2 万行） |
| 编译后端 | `cuda-oxide-codegen` | **刻意不依赖 rustc** 的 PTX 后端：prep → lower → export → 外部 LLVM 工具 |
| IR 方言 | `dialect-mir/-nvvm/-ptx/-iket` | 基于 Pliron（固定 git rev）的 IR 方言 |
| 降级/变换 | `mir-lower`、`llvm-export`、`mir-transforms`、`nvvm-transforms`、`iket-lower` | MIR 方言 → LLVM 方言 → 文本 .ll |
| 设备端库 | `cuda-device`、`cuda-intrinsics`(-gen) | no_std 设备内建函数，由 `intrinsics/catalog.json`（36.5 万行 JSON）驱动生成 |
| host/产物 | `cuda-host`、`cuda-macros`、`cuda-artifact-finalizer`、`oxide-artifacts`、`libnvvm-sys`、`nvjitlink-sys` | proc macro、模块加载、PTX→cubin 终结 |
| 工具 | `cargo-oxide`、`ptx-parse`、`ptx-schedule`、`fuzzer`、`cuda-target-spec` | CLI 驱动、PTX 解析、调度扰动 |

关键事实：

- 编译器本体**没有任何 C/C++/汇编代码**；外部依赖全是进程级工具：`llc`/`opt`/`llvm-link`（LLVM ≥ 21）、`ptxas`、`libNVVM`/`nvJitLink`（dlopen）。
- 没有自定义 rustc target JSON —— 尽管有 `cuda-target-spec` crate，编译始终在 host target 上进行，GPU 定向完全在后端内部完成。
- 强绑定：`rust-toolchain.toml` 钉死 `nightly-2026-08-28` + `rustc-dev` + `llvm-tools`；`-Z mir-enable-passes=-JumpThreading` 等标志是正确性所必需的（防止 barrier 复制导致死锁）。

## 三、能否用 Zig 移植？分层评估

结论先行：**整体移植不可行/无意义；约 20–30% 的组件可以低成本移植或作为 Zig 生态的对应物重建。**

### 3.1 无法移植的部分（核心价值所在）

1. **rustc driver 集成就是产品本体。** 后端是 rustc 加载的 dylib，消费 `TyCtxt`、MIR body、布局/ABI 查询（`rustc_abi`）和 stable MIR，同时把 host 代码交还给 `rustc_codegen_llvm`。Zig 无法复用其中任何一部分 —— stable MIR 没有稳定的序列化格式，`-Z always-encode-mir` 的 rlib 元数据与 rustc 版本锁死。要绕过它等于重新发明一个 Rust 前端。
2. **整个 Rust 语言前端都是依赖**：proc macro（`#[kernel]`、`#[cuda_module]`，还用了 nightly 的 `proc_macro_def_site`）、单态化、泛型、trait 求解、闭包捕获 —— 后端消费的是单态化之后的 MIR。在 Zig 里你只能：写一门新语言的前端，或者仍用 Rust 做 MIR 提取器（那 Rust 就还在链路里，移植失去意义）。
3. **Pliron 中间端**：所有方言与 pass 都建立在 Rust 的类 MLIR 框架上（含 `pliron-derive` 宏），移植等于移植/重写 Pliron 本体。
4. **布局/ABI 位级一致性**：`mir-lower/src/convert/types/*`（enum niche 布局、union、函数 ABI）必须逐位复现 rustc 的布局决策，因为 kernel 与 host 共享结构体。这是一台"rustc 布局预言机"，在 Zig 中重建成本极高且永远追赶 rustc 演进。

### 3.2 适合移植/重建的部分（Zig 友好）

这些组件与 Rust 前端解耦，是常规系统编程，Zig 表达力足够甚至更好：

| 组件 | 规模 | Zig 移植难度 | 说明 |
|---|---|---|---|
| 外部工具编排（llc/opt/ptxas 解析与调用） | 小 | ★☆☆☆☆ | 纯子进程管理，`cuda-oxide-codegen` 已刻意做成 rustc 无关 |
| libNVVM / nvJitLink dlopen 绑定 | ~2k | ★☆☆☆☆ | Zig 的 `dlopen`/`@cImport` 体验良好 |
| PTX 文本解析/打印（`ptx-parse`，基于 combine） | 3.4k | ★★☆☆☆ | Zig 写解析器很顺手 |
| 产物嵌入格式（`oxide-artifacts`） | 小 | ★☆☆☆☆ | 已发布 crates.io 的独立格式 |
| `cargo-oxide` CLI 驱动 | 17.6k | ★★☆☆☆ | 作为独立构建工具重建（Zig build system 集成） |
| PTX → cubin 终结器 | 3.8k | ★☆☆☆☆ | 纯 dlopen + 缓冲区管理 |
| intrinsics 目录管線（catalog.json → 声明） | 数据驱动 | ★★☆☆☆ | 36 万行 JSON 是数据而非代码，可用 Zig comptime/代码生成消费 |

### 3.3 根本性问题：这门"语言"是 Rust

即使把 `cuda-oxide-codegen` 之后的所有东西都搬到 Zig，仍然缺一块：**谁产出 `dialect-mir` 等价的 IR？** 目前答案是 rustc。而该 IR 明确声明"不是跨版本稳定的交换格式"。所以 Zig 路线的真实选项是：

- **方案 A（推荐）：做 Zig 版的"CUDA 单源编译器"，而非移植。** 用 Zig 语言本身作为 kernel 语言：Zig 已有 NVPTX 后端目标（`nvptx64-nvidia-cuda`），可以基于 `intrinsics/catalog.json` 数据生成 Zig 的 PTX 内建函数绑定，加上 Zig 版的 PTX 解析/产物嵌入/ptxas 编排工具。这是"精神移植"，产出物对 Zig 生态真正有用。
- **方案 B：只移植外围工具链。** 把 3.2 中的组件搬过来，链路里的 Rust 部分照旧（cuda-oxide 继续负责 MIR→.ll，Zig 工具负责 .ll 之后的编排与终结）。价值有限，但工作量小、风险低。
- **方案 C：完整移植。** 需要在 Zig 中重建 Rust 前端到 MIR 等价物的通路 + rustc 布局预言机 + 类 MLIR IR 栈。工作量与"重写一个 Rust 编译器后端生态"相当，**不建议**。

## 四、工作量与风险量化

- 手写编译器核心约 **12–15 万行** Rust（不含生成代码与示例）；其中与 rustc/Pliron 深度耦合的约占 70%。
- 方案 A 可行范围的首个里程碑（Zig kernel → PTX → cubin hello-world + 基础内建函数）估计 **2–4 千行** Zig 即可达成，因为 Zig 自带 NVPTX 后端，省掉了整条 rustc→MIR→Pliron→LLVM IR 链路。
- 主要风险：Zig 的 NVPTX 后端成熟度（GPU 特性覆盖如 TMA/tcgen05/WGMMA/集群等需要逐个验证）；sm_XX 与 PTX ISA 下限数据可从 `cuda-target-spec` 与 `intrinsics/catalog.json` 直接复用。

## 五、建议

1. 以**方案 A** 立项：在本目录（zig-ecosystem）下新建 Zig 包，复用 cuda-oxide 的 intrinsics 数据与产物格式设计，但用 Zig 原生能力（comptime、build system、NVPTX target）重建。
2. 短期可先做**方案 B 的落地件**作为练手与基础设施：PTX 解析器、产物嵌入格式、libNVVM/nvJitLink 绑定、ptxas/llc 编排。
3. 放弃方案 C。

## 六、参考位置（源仓库）

- 流水线全景图：`crates/rustc-codegen-cuda/src/lib.rs` 顶部文档注释
- rustc 无关后端入口：`crates/cuda-oxide-codegen/src/pipeline.rs`
- LLVM 工具解析：`crates/cuda-oxide-codegen/src/llvm_tools.rs`
- 内建函数数据：`intrinsics/catalog.json`、`intrinsics/overlay.toml`
- 钉死的工具链：`rust-toolchain.toml`
