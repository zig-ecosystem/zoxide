# libdevice 落地形态调研(P1f 可行性验证)

日期:2026-09-25。环境:macOS arm64,zig 0.16.0(homebrew),LLVM 21.1.8,无 GPU。
结论先行:**走"纯 Zig 手写小数学库"(路线 3),已落地为 `src/math.zig` + `cuda.math`;
"LLVM 手工管线链接 libdevice"(路线 2)实测可行但代价大,留作需要 bit-exact
CUDA 语义时的备案;"zig 直接链 .bc"(路线 1)在 0.16.0 上不可行。**

## 1. Zig NVPTX 编译单元能否直接链接 libdevice bitcode?——不能

实测(zig 0.16.0):

- **zig 不自带 libdevice。** `zig env` 的 lib_dir
  (`.../lib/zig`) 下没有任何 `libdevice*.bc`,`std/` 里也没有 libdevice 字样;
  不存在 clang 那种 `--cuda-path` 自动发现机制。
- **nvptx 目标只能产 PTX 文本。** `zig build-obj -target nvptx64-cuda-none` 加任何
  二进制输出都报:`error: cannot emit nvptx64 binary with the LLVM backend; only
  '-femit-asm' is supported`。没有传统的"链接"阶段可供外部 .bc 注入。
- **把 .bc 作为输入喂给 zig 会走 clang 路径并失败。** CLI 帮助里 `.bc` 确实被列为
  可接受输入("LLVM IR Module, requires LLVM extensions"),但实测
  `zig build-obj k.zig libdevice.10.bc -target nvptx64-cuda-none -mcpu=sm_90`
  报两条 clang 驱动诊断:`'-gdwarf32' is not supported for target
  'nvptx64-nvidia-cuda11.0.1-unknown'` 和 `must pass in an explicit nvptx64 gpu
  architecture to 'ptxas'`——即 .bc 被当作 clang 翻译单元处理,nvptx 工具链还要
  调 ptxas,此路不通。

## 2. 备选:emit-llvm-bc + llvm-link + llc 手工管线——可行,已端到端验证

实验(/tmp/libdev-exp,可复现):

1. libdevice 来源:PyPI 轮子 `nvidia-cuda-nvcc-cu12`(12.9.86)里的
   `nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc`(486 KB,CUDA EULA 许可)。
2. `zig build-obj k.zig -target nvptx64-cuda-none -mcpu=sm_90 -femit-llvm-bc=k.bc
   -fno-emit-bin -O ReleaseFast` 能产出 LLVM bitcode(zig 对 nvptx 禁产二进制但
   不禁 `-femit-llvm-bc`)。homebrew 的 zig 0.16.0 动态链接
   `/opt/homebrew/opt/llvm@21/lib/libLLVM.dylib`(21.1.8),与 brew `llvm@21` 的
   `llvm-link/opt/llc` 逐版本一致;官方 zig 0.16 同样基于 LLVM 21。**工具链大版本
   必须匹配,这是该路线最脆的一点。**
3. 完整管线(全部实测通过):
   ```
   llvm-link k.bc libdevice.10.bc -o linked.bc
   opt --internalize-public-api-list='<kernel_mangled>,__zoxide_keep_kernels' \
       -passes='internalize,globaldce' linked.bc -o int.bc
   opt -O3 int.bc -o opt.bc
   llc -mcpu=sm_90 opt.bc -o out.ptx
   ```
   产物中 `.entry k_$_kSin` 保留,`__nv_sinf/__nv_atanf/__nv_cbrtf` 全部从
   libdevice 内联进来(9945 行 PTX),无未解析 `__nv_*`。
4. 两个坑(都已实测确认):
   - **必须 internalize + globaldce。** 裸 `opt -O3` 会把 libdevice 的
     `__nv_fast_tanhf` 等保留为 `.visible .func`,其体内的
     `call llvm.nvvm.tanh.approx.f32` 不会被 llc 降级(即使 `-mattr=+ptx83`,
     实测恒为 extern call),ptxas 会拒绝。internalize 后 DCE 掉即干净。
   - **kernel 要被 Keep 保活。** 没有 `cuda.Keep` 时 zig 端就把未引用的
     `callconv(.kernel)` 函数 DCE 掉,.bc 里根本没有 kernel。

代价:build.zig 要从"zig 一步出 PTX"改成两段式;CI(ubuntu-latest,
setup-zig)要额外 pin LLVM 21 工具链 + 下载/缓存 CUDA 轮子;libdevice 的
CUDA EULA 与项目的 Apache/MIT 生态不完全相容(可再分发条款仅限于 CUDA 应用
开发语境)。

## 3. 备选:PTX 近似指令 + 手写软件实现——零新依赖

**Zig 内建函数在 nvptx 上的实测行为**(0.16.0,sm_90):

| Zig 内建 | nvptx 结果 |
|---|---|
| `@sqrt` | `sqrt.rn.f32` ✓ |
| `@max` / `@min` | `max.f32` / `min.f32` ✓ |
| `@abs` | `abs.f32` ✓ |
| `@floor` / `@round` / `@trunc` | `cvt.rmi` / `cvt.rzi` ✓ |
| `@sin` / `@cos` | **LLVM ERROR: Cannot select: fsin**(直接崩) |
| `@exp` / `@log` | **error: no libcall available for fexp**(编译失败) |

即超越函数在 nvptx 上既没有 libcall 也没有内联展开——这正是 libdevice
存在的原因,也是 zoxide 必须自己补的层。

**PTX 单指令覆盖**(生成绑定已存在于 `src/gen/intrinsics_asm.zig` 的
`float` 组):

| libdevice 函数 | PTX 单指令 | 备注 |
|---|---|---|
| `__nv_fast_sinf/cosf` | `sin.approx.f32` / `cos.approx.f32` | 同一条指令,精度即 fast math |
| `__nv_fast_log2f` | `lg2.approx.f32` | |
| `__nv_fast_expf` | `ex2.approx.f32`(乘 log2e) | |
| `__nv_fast_logf` | `lg2.approx.f32`(乘 ln2) | |
| `__nv_fast_tanhf` | `tanh.approx.f32` | sm_75+ |
| `__nv_frsqrt_rn` 近似版 | `rsqrt.approx.f32` | |
| `__nv_fsqrt_rn` | `sqrt.rn.f32` | IEEE 正确舍入 |
| `__nv_fmaxf/fminf/fabsf/floorf` | `max/min/abs/cvt.rmi` | Zig 内建直达 |
| `__nv_sinf/cosf`(精确) | **无** | libdevice 是软件实现(象限归约+多项式+慢速路径) |
| `__nv_atanf/atan2f` | **无** | 多项式 |
| `__nv_cbrtf` | **无** | 位技巧种子 + Newton |
| `__nv_expf/logf`(精确) | **无** | 指数注入 + 多项式 |
| `__nv_powf/erff/lgammaf/…` | **无** | 更长尾 |

结论:`__nv_fast_*` 全家 = 单指令包装,零成本;精确超越函数是一小组有成熟
参考实现(musl/cephes 系数)的多项式例程,规模可控(本次落地 6 个约 100 行)。

## 4. 推荐形态与落地

**推荐路线 3**,理由:

1. 路线 1 在 zig 0.16 上客观不可行;
2. 路线 2 引入"LLVM 大版本必须匹配 zig 内置 LLVM"的硬耦合、CUDA 轮子下载和
   许可问题,换来的只是与 libdevice 的 bit-exact——zoxide 的 GEMM/ML kernel
   场景不需要;
3. 路线 3 零新依赖,CI 形态不变(继续 grep PTX),且 `fast` 层与 libdevice
   `__nv_fast_*` 是同一条硬件指令,语义天然一致。

已落地:

- `src/math.zig`:三层结构——fast 层(`sinFast/cosFast/expFast/logFast/
  log2Fast/tanhFast/rsqrtFast`,单 PTX approx 指令,复用 `asm_gen.float`
  绑定)、native 层(`sqrt/fmax/fmin/fabs/floor`,Zig 内建直达)、software 层
  (`sin/cos/exp/log/atan/cbrt`,Cody–Waite f64 象限归约 + musl/cephes
  多项式,few-ulp 目标,不承诺与 libdevice bit-exact)。
- `src/cuda.zig`:`pub const math = @import("math.zig");`,kernel 侧
  `cuda.math.sin(x)` 即用。
- `src/examples/math_smoke.zig` + build.zig 注册:三层全部进一个 kernel。
- CI(`ci.yml`)新增断言:`sin/cos/lg2/ex2/tanh/rsqrt.approx.f32`、
  `sqrt.rn.f32`、`max.f32` 必须出现在 `math_smoke.ptx`,且不得残留
  `__nv_*` / `llvm.*` 调用。

若未来需要 bit-exact libdevice(例如对齐 cuda-oxide 数值结果做交叉验证),
按第 2 节的管线接入即可,所有坑已在此记录。

### 已知边界

- 软件层是 few-ulp 精度;`sin/cos` 的归约在 f64 完成,|x| → 2^24 量级后精度
  退化(libdevice 在此切慢速路径,我们没有);`exp` 的指数注入对 |n| 做了
  截断,溢出/下溢端不产 IEEE 精确 inf/0。
- f64 超越函数未实现(fast 层同样只有 f32 指令)。
- 无 GPU 本机验证只到 PTX 层;数值精度需在 GPU runner(scripts/pod-verify.sh)
  上补。
