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
| v0.0.9-alpha | wgmma RS（A 进寄存器）：hgemm_wgmma3 95.2 TF / 64.3% 峰值，结果精确，累计 1.77x |
| v0.0.10-alpha | Host API：类型化 launch、stream、pinned、设备侧 memset、occupancy 查询 |
| v0.0.11-alpha | `zoxide new` 脚手架；hgemm 签名改 f16；host API 真机验证全通 |
| v0.0.12-alpha | 订正 v0.0.8 归因（1 字节溢出值 7.3% 吞吐，三级流水值 0）；bench 改用类型化 API；GPU CI job |

## 规划

### v0.0.3 — 验证基建与开发体验（alpha → beta）

目标：让"改代码 → 验证"成为一键动作，降低外部贡献门槛。

- [x] **pod 端到端回归脚本**（M4b）：`scripts/pod-verify.sh`——doctor + 全部示例 run PASS + bench 冒烟，输出结构化报告（JUnit 或纯文本表格），失败定位到具体 kernel；README 附 k8s Job yaml 示例
- [ ] **CI GPU 扩展**：支持 self-hosted GPU runner 时自动跑 pod-verify（workflow 里 `runs-on: [self-hosted, gpu]` 的可选 job，无 runner 时跳过）。
  本轮的教训给这条加了权重：`tests/downstream` 的符号不匹配在无 GPU 的机器上**完全测不出来**，
  只有真正启动一次 kernel 才会暴露，而它活了两个提交
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

### cuda-oxide 对照：f16/bf16 人性化层 —— 经测量，不需要
对照 cuda-oxide 时我把「设备侧缺 f16/bf16 人性化层」列为真实缺口，依据是它的
`cuda-device` 里有手写的 `f16.rs` / `f16x2.rs` / `bf16x2.rs`。**查生成的 PTX 后这个判断不成立**：

| Zig 写法 | 生成的 PTX |
| --- | --- |
| `a * a`（f16） | `mul.rn.f16` |
| `@mulAdd(f16, ...)` | `fma.rn.f16` |
| `v * v`（`@Vector(2, f16)`） | `mul.rn.f16x2` |
| `@mulAdd(@Vector(2, f16), ...)` | `fma.rn.f16x2` |
| `@max(v, w)` | `max.f16x2` |
| `@Vector(2, f16)` 的 load | `ld.global.b32`（正确合并） |

语言本身就降到 packed 指令，写包装层等于重复语言能力。`src/gen/` 里那 66 条
f16/f16x2/bf16/bf16x2 wrapper 仍有价值——它们覆盖 `ftz`/`nan`/`xorsign_abs`/`relu`/`sat`
这些 Zig 没有语法的变体。

见 `src/examples/f16_native.zig`。这条是 Zig NVPTX 后端的性质而不是本仓库的性质，
所以 CI 断言那些指令仍然出现——上游回退否则我们看不见。

真正的缺口是另一件事，已在本轮修掉：hgemm 全系列此前收 `[*]const u8` 手工算字节偏移，
导致类型化 launch 对它们检查不到元素类型（`Slice(u8)` 什么都能塞）。现在收 `[*]const f16`。

### v0.0.9 — 定位剩余 41.7pp：已收敛到 n16
四个候选，三个已用实验排除（详见 `docs/verification/`）：

| 假设 | 状态 |
| --- | --- |
| 流水线排空 tensor core | 已排除 —— 且比原以为的更彻底：那 4pp 实为一个 1 字节溢出，三级流水本身值 −0.0pp |
| 全局流量 / DRAM 带宽 | 已排除 —— L2 边界 sweep 跨界持平（57.9% → 58.3% → 57.8%） |
| warpgroup 并发 / occupancy | 已排除（实测）—— 驱动实测 4 blocks/SM、25%（寄存器约束，非共享内存）。用 `--maxrregcount 96` 零溢出地提到 5 blocks/31%，吞吐反而 0.97x |
| **n16 单指令 tensor core 效率** | **剩下的唯一候选** |

- [x] ~~ncu profile 分流~~ —— pod 上不可用（`ERR_NVGPUCTRPERM`，需节点级放开计数器权限）；
  改用 L2 边界 sweep 替代，结论同样明确
- [x] ~~M=128 / 双 warpgroup~~ —— **撤销**。立项理由是降低全局流量，流量既已排除则理由不成立；
  也不能靠提高并发救场（并发同样已排除）。根本原因：**3.3 倍操作数流量比例对 tile 形状不变**，
  放宽 M/N 只改变 block 数量，不改变每条 wgmma 重读 A 的次数——只有单条指令的 N 变宽才改变它
- [x] **A 进寄存器（wgmma RS）** —— +6.0pp，64.3%。这条推翻了上一轮「只有宽 N 能降操作数流量」
  的断言：RS 形态累加器仍是 8 个，落在 15 输出限制内，而操作数流量已与 n128 持平（42.7 flops/byte）
- [x] **`--maxrregcount` 扫描** —— occupancy 不是约束：零溢出下 5 blocks/31% 比 4 blocks/25% 慢 3%。
  寄存器维度关闭（98 regs / 4 blocks 即最优）。顺带量化了溢出代价：16 个寄存器溢出 = 0.55x 吞吐
- [ ] **打补丁的 zig（output 上限 32）编 `m64n64k16`** —— 四条排除现已全部有实测支撑，
  n16 单指令效率是唯一剩下的候选，而测它必须把 N 变宽。按你的要求，这条留到最后再谈
- [ ] 上游提交（ziglang/zig issue 创建受限于 collaborator，走 `docs/drafts/` 里记录的 fallback 渠道）
- [ ] TMA 替代 cp.async（与 n16 无关的独立方向）

### v0.0.5 — 类型化启动与单文件体验

- [ ] `@embedFile` cubin + comptime 生成类型化 launch（对标 cuda-oxide `#[cuda_module]`）：kernel 参数在编译期检查类型/数量
- [x] host+device 同包的标准项目模板（`zoxide new`）—— 生成的包与 `tests/downstream` 同构，
  fingerprint 从编译器的错误消息取回；CI 断言生成物的 PTX 入口点 == host 查找的符号
- [x] launch 参数校验（block/grid vs device 限制）—— 启动前用缓存的设备限制做纯算术校验，
  消息点明维度与上限；最有价值的一条是「设备允许但本 kernel 不允许」（寄存器压低了上限，
  源码里毫无线索）。另把高价值 CUresult 映射为独立错误（ArchMismatch / InvalidPtx /
  LaunchOutOfResources / IllegalAddress / CudaOutOfMemory），因为 `try` 会丢掉 lastError()

### v0.0.13 — 设备全局变量（host 写、device 按名读）
- [x] 实测确认 Zig 无法表达这个模式，六种写法全部失败：`addrspace(.constant)` 拒绝可变值；
  不可变 `.constant` 落到 `.global` + `ld.global.nc`（非 `.const` 存储体，且 host 不可写）；
  无 `export` 的 `.global` 符号无 `.visible`；`export var`(带/不带 addrspace)与 `@export`
  全部撞 LLVM alias 限制，其中裸 `export var` **直接 abort 编译器**
- [x] 根因定位：Zig 把变量的 `export` 降成 LLVM alias，NVPTX 只接受 aliasee 是非 kernel 函数的 alias
- [x] ~~`src/ptx.zig` PTX 后处理加 `.visible`、`zoxide ptx-export` 子命令、
  `tools/ptx-promote.zig`~~ —— **全部已删，实测证明针对的是不存在的问题**：
  反向对照（把 `.visible` 剥掉再跑）显示 ptxas 本来就把 module-scope `.global`
  暴露给 `cuModuleGetGlobal`。**教训：先做反向对照，再写绕法**
- [x] **第二个更隐蔽的坑**：LLVM 假设模块外无写者，普通读取被常量折叠、符号被丢弃，
  且**取决于初始值**——`.{10,20,30,40}` 保留，`.{0,0,0,0}`（最自然的占位）折成常量 0 并
  删除符号，host 上传被静默忽略。`cuda.ldg()`（inline asm `ld.global.nc`，对优化器不透明）
  解决，顺带拿到只读缓存路径（= CUDA `__ldg`）
- [x] host 侧：绑定 `cuModuleGetGlobal_v2`，`Module.global()` 把 NOT_FOUND 映射为
  `GlobalNotFound` 并点出两个长得一样的成因（名字错 / 符号被常量折叠掉了）
- [x] **H20 实测通过**（`devglobal-20260924`）：
  `cuModuleGetGlobal` 解析到符号且大小正确（16 B），两轮不同的表
  （`{1,2,3,4}` 与 `{-100.5, 0.25, 7, 65536}`）各 1024/1024 精确 ——
  证明 device 每次 launch 重读，而非把值烤进代码
- [x] 反向对照（`devglobal-neg2-20260924`）：**否定结果** —— 剥掉 `.visible` 照样能解析，
  pass 多余，已删。做这个对照的价值就在这里：它否掉的是我自己加的复杂度
  - 第一版对照（`devglobal-neg-20260924`）**无效**：`zoxide run` 按文件名 stem 选
    example，剥离后的副本写成 `dev_global_unpromoted.ptx`，于是在 unknown-example
    检查处就退出了，根本没走到 `cuModuleGetGlobal`。而脚本把「没 PASS」当成了确认。
    教训：反向对照必须要求**那条具体错误**，否则它会因为无关原因「通过」

### v0.0.14 — 真正的 constant memory（`.const` 存储体）
- [x] 推翻「constant memory 不可达、只能递上游」的结论。**模块级 inline asm 可以直接
  发出 `.const` 声明**，Zig 逐字透传，配合 `ld.const` 就是真正的 constant 存储体
  ——不是 `addrspace(.constant)` 那条落到 `.global` + `ld.global.nc` 的假路
- [x] `cuda.ConstBank(name, T, len)`：`declaration` 暴露成字符串常量而非 `declare()`
  函数——模块级 asm 必须直接出现在 file-scope `comptime` 块里，包成函数调用会报
  "unable to evaluate comptime expression"；`get(i)` 按 5 个寄存器类分派
- [x] 寻址必须**相对符号**：`mov.u64 %b, sym; add; ld.const.T [%b]`。直接把裸字节偏移
  喂给 `ld.const` 会从 const 窗口起点读，当 bank 是窗口里唯一对象时会「碰巧正确」
  ——example 里放了 `pad_before[64]` 把 table 推离起点，让寻址错误暴露成错值而不是通过
- [x] 符号名**不经 mangling**（asm 逐字透传），host 查 `const_scales` 而非
  `const_bank_$_const_scales`
- [x] **H20 实测通过**（`constmem-20260924`）：ptxas 接受 asm 发出的 `.const` 声明，
  `cuModuleGetGlobal` 解析到 `.const` 符号（256 B），两轮不同表各 4096/4096 精确。
  `pad_before` 在场，说明寻址是真的相对符号而非碰巧从窗口起点读对
- [x] `getAt(comptime i)`：编译期下标把偏移折进指令，`ld.const.f32 [sym+N]` 一条，
  而 `get(i)` 的运行时下标需要 mov/cvt/add/ld 四条（偏移是 asm 输入操作数，折不进去）
- [x] **H20 实测：uniform 下 `.const` 快 1.11x**（0.030 vs 0.033 ms，两边结果精确）。
  但**这个数不能支持「广播缓存」这个解释**：`.const` 版语句数少 15%（241 vs 285），
  11% 的加速与之量级一致，同样符合「单纯指令少」
- [ ] **待测：分离机制**（`constbench-div-20260924`）。判据：constant memory 对 warp 内
  发散访问按不同地址串行化（最高 32×），而只读缓存会合并。加发散变体后语句数反转
  （const 603 vs ldg 476，多 27%），于是「纯指令数」假说有了定量预言 = 发散比 0.79x：
  - 发散比 ≈ 0.79x → 优势只是代码更短，删掉文档里的广播断言
  - 发散比 ≪ 0.79x（低于 0.59x）→ 串行化真实存在，广播机制成立，uniform 的 1.11x 是缓存
  判据写进了 `runConstVsLdg`，避免事后看数编解释
- [ ] ~~待测：`.const` 到底比只读缓存快多少~~（`constbench-20260924`，已出数见上）。
  三处 doc comment 一直在断言「广播缓存所以更快」，那是 CUDA 文档而非本机测量。
  `const_vs_ldg` 两个 kernel 同表、同访问模式、**手工展开对齐**（交给 LLVM 会得到
  4× vs 8× 两种展开因子，那测的是展开器不是内存路径）。PTX 级已可见一个事实：
  `.const` 版语句数反而更少（237 vs 285），因为 `ld.const [sym+N]` 不需要地址运算。
  三种结果都是真答案——持平或更慢就删掉文档里的断言

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
| 设备全局变量的 `.visible` 靠 PTX 后处理，依赖 LLVM 输出形状而非语言保证 | 已缓解，失败显式 | ptxas 是否认这个提升待 GPU 验证；诉求见 `docs/upstream-device-globals.md` |
| 设备全局变量的普通下标读会被常量折叠、符号消失（**取决于初始值**，全零时静默丢失） | 已缓解（`cuda.ldg()`） | 正确性依赖用户不用普通下标读，非语言级保证 |
| 单 arch（sm_90）单平台（H20 pod）验证 | 开放 | 泛化性待证 |
| ncu 不可用（pod 权限） | 已知限制 | 深度调优靠 PTX 审查 + 对照实验 |

## 决策记录

- 2026-09-17：走"Zig 实现"路线（非代码移植）；arch 基线 sm_90
- 2026-09-18：版本号 alpha/beta 节奏；host 绑定手写 extern（非 @cImport）
- 2026-09-20：pod 上 `zoxide run` 用 gnu 动态构建（musl 静态 dlopen 不可靠）；发版必配 announcement.md 条目 + 图片
