# nvptx 上的设备全局变量与 constant memory:已在仓库内解决

状态:**两条都已解决,不依赖上游**。本文保留为实测记录 + 一条纯 bug 报告。

原先的判断是「没有语言内的绕法、只能递上游」。这个判断错了两次:

1. 设备全局变量:以为需要 `.visible`,写了 PTX 后处理 pass。H20 反向对照证明
   **ptxas 本来就暴露 module-scope `.global`**,pass 是多余的,已删。
2. constant memory:以为不可达。实际上**模块级 inline asm 可以直接发出 `.const`
   声明**,配合 `ld.const` 就是真正的 constant 存储体。见 `cuda.ConstBank`。

剩下真正该递上游的只有一条:`export` 作用在变量上会让编译器 abort(诉求一)。
而它现在**不阻塞任何东西**——我们根本不需要 `export`。

环境:zig 0.16.0 / LLVM 21.1.8,target `nvptx64-cuda`,`-mcpu sm_90`。

## 诉求一(bug):`export` 作用在变量上会让 NVPTX 后端崩溃

```zig
export var dev_one: f32 = 1.0;
pub fn k(out: [*]f32) callconv(.kernel) void {
    out[0] = dev_one;
}
```

```
LLVM ERROR: NVPTX aliasee must be a non-kernel function definition
zsh: abort
```

编译器 abort,不是诊断。带地址空间的两个变体报的是另一条:

```zig
export var dev_scale: [4]f32 addrspace(.global) = .{ 0, 0, 0, 0 };
// error: Alias and aliasee types don't match (Producer: 'zig 0.16.0' Reader: 'LLVM 21.1.8')

comptime { @export(&dev_scale, .{ .name = "dev_scale", .linkage = .strong }); }
// 同上
```

根因:Zig 把变量的 `export` 降级成 LLVM alias,而 NVPTX 后端只接受
aliasee 是**非 kernel 函数**的 alias。所以 `export` 作用在任何变量上,和 NVPTX
在结构上不兼容。

期望:NVPTX target 下把导出的全局变量直接以 external linkage 发出,不经 alias。
退一步的最低要求是**给出诊断而不是 abort**。

注:同一个 alias 限制也影响 kernel 保活 —— 这就是本仓库需要
`cuda.Keep()` 的原因,且 UBSan 运行时必须关掉(`bundle_ubsan_rt = false`),
因为它生成的 alias 指向 kernel 函数。

## 已解决:constant memory —— 用模块级 inline asm

原以为不可达。实际做法:

```zig
var buf: [64]f32 addrspace(.constant) = undefined;
// error: mutable values with address space 'constant' are not supported on nvptx
```

`.constant` 只接受不可变值,而不可变意味着编译期定死,那就不是 constant memory
——CUDA 的 `__constant__` 的定义就是 host 在 launch 前写、device 读。

而且不可变版本也没进 PTX 的 `.const` 存储体:

```zig
const cbuf: [4]f32 addrspace(.constant) = .{ 1, 2, 3, 4 };
```

生成 `.global .align 4 .b8 __anon_815` + `ld.global.nc.b32`。是只读缓存
(等价 `__ldg`),不是 `.const` 的 64KB 广播通路。

期望:`addrspace(.constant)` 的可变全局变量在 nvptx 上映射到 PTX `.const`
存储体,并以 external linkage 发出,使 `cuModuleGetGlobal` 可解析。

## 连带问题:没有 `export` 的全局变量不可见

不用 `export` 能编译,但符号是模块局部的:

```zig
var dev_scale: [4]f32 addrspace(.global) = .{ 1, 2, 3, 4 };
```

```
.global .align 4 .b8 g_$_dev_scale[16] = {0, 0, 128, 63, ...};
```

没有 `.visible`。

H20 实测(2026-09-24):加上 `.visible` 后 ptxas 接受,`cuModuleGetGlobal` 解析到
符号且大小正确(16 字节),两轮不同的表(含 `{-100.5, 0.25, 7, 65536}`)结果全精确
——说明 device 每次 launch 重读,不是把值烤进了代码。

仍未实测:**不加 `.visible` 是否真的解析不到**。「promoted 能用」同时也符合
「ptxas 本来就暴露 module-scope global」这个解释,那样后处理就是多余的复杂度。
反向对照见 `scripts/devglobal-probe.sh`(release tag `devglobal-neg-20260924`)。

## 实际做法

**设备全局变量**:直接声明 `var x addrspace(.global)`,host 用
`Module.global()` + `copyHtoD` 写。不需要 `export`,不需要 `.visible`。

**constant memory**:`cuda.ConstBank(name, T, len)` —— 声明走模块级 asm,
读取走 `ld.const`。符号名不经 mangling(asm 逐字透传),所以 host 查的是
`const_scales` 而不是 `const_bank_$_const_scales`。

两者唯一需要小心的是 **LLVM 常量折叠**,而这才是真正的坑:

LLVM 假设模块外没人写 module-scope global,所以普通读取会被折叠、符号被丢弃。
**而这取决于初始值**:

| 初始值 | 结果 |
|---|---|
| `.{ 10, 20, 30, 40 }` | 符号保留,发出真实 load |
| `.{ 0, 0, 0, 0 }` | 折成常量 0,符号消失,host 上传被静默忽略 |

全零恰恰是最自然的占位写法。所以 `cuda.ldg()`(inline asm 的 `ld.global.nc`,
对优化器不透明)不是便利函数而是正确性前提。`ConstBank.get()` 同理走 asm,
天然免疫。

这一条值得上游关注的程度高于可见性:即使 Zig 把符号发成 visible,只要 LLVM
仍假设外部无写者,普通下标读就还是会静默出错。不过既然 asm 路线已经完全可用,
这更像是「文档该写清楚」而不是「阻塞」。
