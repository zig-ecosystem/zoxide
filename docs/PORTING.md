# zoxide：cuda-oxide → Zig 能力对照与移植规划

> 2026-09-17 · 配套文档：`cuda-oxide-zig-port-assessment.md`（可行性评估）
> 前提结论：不做代码级整体移植，走"精神移植"——Zig 原生 NVPTX 后端 + 复用 cuda-oxide 的数据与设计。
> 已完成里程碑 M0：`zoxide/` 骨架（kernel→PTX→ptxas 编排→doctor），见文末。

## 一、能力对照总表

对照维度：cuda-oxide 现有能力 → Zig 侧实现方式 → 移植难度（★越少越易）→ 阶段。

### 1. 编译链路（kernel 源码 → 可加载产物）

| cuda-oxide 能力 | 对应组件 | Zig 侧方案 | 难度 | 阶段 |
|---|---|---|---|---|
| Rust 源码 → MIR（rustc 前端） | rustc + rustc-codegen-cuda | **不需要**——Zig 语言即 kernel 语言，Zig 编译器自带 NVPTX 后端 | ✅ 已替代 | M0 ✅ |
| MIR → Pliron 方言 → LLVM IR | mir-importer / mir-lower / llvm-export（~10 万行） | **不需要**——Zig 直接经自家 LLVM 后端发 PTX | ✅ 已替代 | M0 ✅ |
| LLVM IR → PTX | 外部 `llc`（NVPTX） | Zig 内建（`-femit-asm`），无需外部 llc | ✅ 已替代 | M0 ✅ |
| PTX → cubin | cuda-artifact-finalizer（ptxas） | `zoxide cubin`（已实现探测与调用封装） | ★ | M0 ✅（待真机验证） |
| NVVM IR / LTOIR 产物 | libNVVM + nvJitLink（dlopen） | Zig dlopen 绑定 libNVVM/nvJitLink | ★★ | M3 |
| 工具链探测与编排 | llvm_tools.rs / cargo-oxide | `zoxide doctor` + 子进程封装（已实现） | ★ | M0 ✅ |
| 构建系统集成 | cargo-oxide（cargo 子命令） | build.zig 原生 step（已示范 `zig build kernel`） | ★ | M1 完善 |

### 2. Kernel 表达能力

| cuda-oxide 能力 | Zig 侧方案 | 难度 | 阶段 |
|---|---|---|---|
| threadIdx/blockIdx/blockDim/gridDim | inline asm 读特殊寄存器（已示范）→ 封装为 `std.cuda` 风格 API | ★ | M1 |
| 共享内存 / 全局内存 / 常量内存地址空间 | Zig 地址空间支持需验证；退路：inline asm ld/st.shared | ★★ | M1 |
| `__syncthreads` / barrier / warp 原语 | inline asm 或 Zig 内建 asm 封装 | ★ | M1 |
| 原子操作（含 scope/ordering、Hopper/Blackwell 新原子） | 常用子集手写 asm；全量从 catalog.json 生成 | ★★ / ★★★ | M1 / M3 |
| 完整 PTX intrinsics（365k 行 catalog） | **数据驱动**：写 Zig 生成器消费 `intrinsics/catalog.json` + `overlay.toml`，comptime/构建期生成 Zig 绑定 | ★★★ | M3 |
| WGMMA / TMA / tcgen05 / cluster / Hopper+Blackwell 特性 | 同上生成器覆盖；**风险**：Zig NVPTX 后端对这些 target feature 的暴露程度需逐个验证 | ★★★★ | M3–M4 |
| gpu_printf | PTX `vprintf` 封装 | ★ | M2 |
| 内联 PTX asm（用户级） | Zig `asm` 原生支持 | ✅ 已有 | M0 |

### 3. Host 侧能力

| cuda-oxide 能力 | Zig 侧方案 | 难度 | 阶段 |
|---|---|---|---|
| cubin 模块加载、kernel 启动 | `cuda.h` 驱动 API：Zig `@cImport` 或手写 extern + dlopen libcuda | ★★ | M2 |
| `#[cuda_module]` 产物嵌入 + 类型化启动 | Zig comptime：`@embedFile` cubin + 编译期生成类型化 launch wrapper | ★★ | M2 |
| launch contract（网格/块参数校验） | comptime 参数检查 | ★★ | M2 |
| 异步 / stream / event | extern 绑定 driver API | ★★ | M2 |
| cuda-bindings（bindgen over cuda.h） | Zig `@cImport` 直接替代（无需 libclang 外部进程） | ★ | M2 |

### 4. 不移植 / 无需移植

| cuda-oxide 能力 | 原因 |
|---|---|
| rustc_codegen_cuda / mir-importer（~15 万行） | Rust 前端专用，Zig 路线整体跳过 |
| Pliron 方言栈（dialect-mir/nvvm/ptx、mir-transforms 等） | 同上 |
| cuda-macros（12k 行 proc macro） | Zig comptime + `@embedFile` 天然覆盖其职责 |
| rustlantis fuzzer、ptx-schedule | 测试基建，后期按需重建 |
| `-Z mir-enable-passes=-JumpThreading` 等 rustc 正确性开关 | Zig 后端无此问题类（不同 IR 通路），但需自己的正确性测试 |

### 5. 可直接复用的资产（从源仓库取，无需移植）

| 资产 | 路径 | 用途 |
|---|---|---|
| PTX intrinsics 目录（365k 行 JSON） | `intrinsics/catalog.json` | 生成 Zig 绑定的数据源 |
| 人工覆盖/修正层 | `intrinsics/overlay.toml` | 同上 |
| sm_XX 与 PTX ISA 下限数据 | `crates/cuda-target-spec` | Zig 侧 arch 校验逻辑的数据来源 |
| 产物嵌入格式设计 | `crates/oxide-artifacts` | zoxide 嵌入格式参考（Apache-2.0，可借鉴格式而非代码） |
|  correctness 陷阱清单 | rustc-codegen-cuda 文档（JumpThreading-barrier 等） | Zig 侧回归测试用例来源 |

## 二、分阶段规划

### M0 —— 骨架验证 ✅（已完成）

kernel.zig → PTX（NVPTX 后端）→ ptxas 编排 → doctor。**结论：路线可行，Zig 0.16 NVPTX 后端可用（需显式 `-mcpu sm_75+`）。**

### M1 —— 最小可用 kernel 开发体验（预计 3–6 天工作量）

目标：用户能在 Zig 里写一个非平凡 kernel 并在真机上跑起来。

1. `src/cuda/*.zig` 设备端库：`threadIdx()`/`blockIdx()`/`syncThreads()`/warp shuffle/基础原子，inline asm 封装
2. 共享内存支持验证（Zig addrspace 或 asm 退路）——**本阶段最大技术验证点**
3. `zoxide ptx` 暴露 `--arch sm_XX`；arch 数据接入（复用 cuda-target-spec 数据）
4. build.zig 集成：`kernel` step 参数化（多 kernel、arch、优化级别）
5. 真机验证：在有 NVIDIA GPU + CUDA Toolkit 的机器上 `cubin` + 加载运行（需要借用/申请环境）

验收：`examples/vector_add` 与 `examples/sgemm_naive`（含共享内存 tile）端到端通过。

### M2 —— Host 侧闭环（预计 4–7 天）

1. libcuda dlopen 绑定（cuInit/cuModuleLoad/cuLaunchKernel/…，约 30–50 个函数起步）
2. `@embedFile` cubin + comptime 生成类型化 launch API（对标 `#[cuda_module]`）
3. gpu_printf 封装
4. `zoxide run` 一条命令：编译 kernel → cubin → 加载 → 启动 → 校验输出

验收：host+device 单文件（或单包）体验，vector_add 全自动端到端。

### M3 —— intrinsics 全量生成（预计 1–2 周）

1. 写 `intrinsics-gen`（Zig）：解析 `catalog.json` + `overlay.toml` → 生成 Zig 绑定文件
2. 原子操作全集（scope/ordering 矩阵）、cp.async、TMA、WGMMA、cluster 等按 arch 门控
3. 与 cuda-oxide 生成的声明做 diff 对拍（数据源相同，输出应语义等价）

验收：生成覆盖率 ≥ catalog 的 90%；抽样 100 个 intrinsics 编译通过。

### M4 —— 健壮性与生态（持续）

1. 正确性回归套件（参考 cuda-oxide 记录的陷阱：barrier 分歧、mem2reg、循环展开）
2. PTX 解析/检查器（对标 ptx-parse，可用于 lint 与测试断言）
3. sm_90/sm_100/sm_120 特性逐项验证，维护 arch 能力矩阵
4. 文档 + 示例 + 包发布（Zig package manager）

## 三、风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| Zig NVPTX 后端不成熟（target feature 缺失、LLVM 版本滞后） | M1/M3 阻塞 | M1 先做共享内存与特性探针；缺特性时走 inline asm 兜底；跟踪 ziglang/zig 的 nvptx issue |
| Zig 0.16 std/Build API 不稳定，升级即破碎 | 维护成本 | 钉住 zig 版本（rust-toolchain.toml 等价物：`zig-version` 声明 + CI 检查），封装所有 std 触点在少量模块 |
| 本机无 GPU，验证依赖外部环境 | 验收延迟 | M1 起准备 CI runner 或云 GPU 环境；本机只能验证到 PTX 层 |
| catalog.json 与上游许可证 | 法律 | 数据源自 PTX ISA 文档（NVIDIA），cuda-oxide 为 Apache-2.0；生成器自己写，仅引用数据，保留 THIRD_PARTY 声明 |
| Zig 地址空间模型与 PTX generic/shared/global 语义不完全对齐 | 性能或正确性 | 语义以 inline asm 为准，类型系统只做薄封装 |

## 四、决策点（下一步开工前需要确认）

1. **arch 基线**：✅ 已定为 **sm_90**（验证环境为 H20 pod，Hopper 架构）。zoxide 默认 `-mcpu sm_90`，白名单 75/80/86/89/90/100/120 可通过 `--arch` 覆盖。
2. **host 绑定方式**：手写最小 extern 集（零依赖）+ 按需扩充；pod 内 dlopen 顺序 `libcuda.so.1` → `libcuda.so`（runtime 镜像常缺 dev 符号链接）。
3. **真机验证环境**：✅ k8s pod（H20）。zoxide 已提供 musl 静态交叉编译产物（`zig build -Dtarget=x86_64-linux-musl`），scp 进 pod 即可运行；PTX→cubin 用 pod 内 ptxas（需 ≥ CUDA 12.4）。

## 五、已知 Zig 0.16 NVPTX 陷阱（M0 踩坑记录）

1. `export fn` + kernel callconv 会触发 LLVM alias bug（"NVPTX aliasee must be a non-kernel function definition"）。变通：`pub fn ... callconv(.kernel)` + 有函数体的 dummy export 物化 kernel 指针；PTX 中符号名带 `kernel_$_` 前缀，host 端 `cuModuleGetFunction` 需用该名。
2. 需 `.strip = true` / `-fstrip`，否则 PTX `.target` 行带 `, debug` 后缀。
3. `bundle_ubsan_rt = false`（UBSan runtime 同样触发 alias bug）。
4. target 写法：`nvptx64-cuda`（三段式 `nvptx64-nvidia-cuda` 报 UnknownOperatingSystem）。
