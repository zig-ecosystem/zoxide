# Draft: ziglang/zig issue — compiler SEGV building NVPTX kernel at any Release opt level (x86_64-linux host only)

> 草稿,未提交上游。证据来自本仓库 CI 的系统性最小化(2026-10-01,13 轮二分)。

**Title:** compiler SIGSEGV compiling NVPTX kernel at ReleaseFast/Safe/Small (x86_64-linux host; macOS arm64 unaffected)

## Reproduction

`sgemm_opt2_segv_repro.zig`(同目录,144 行,自包含依赖只有 repo 内的 cuda.zig/examples_abi.zig——上游提交前需再抽掉这两个依赖):

```sh
zig build-lib -OReleaseFast -fstrip -target nvptx64-cuda -mcpu sm_90 \
  -femit-asm -fno-emit-bin repro.zig
# exit 139 (SIGSEGV), deterministic
```

## Evidence matrix(全部确定性复现,各跑 2 次以上)

| 变量 | 结果 |
|---|---|
| host: macOS arm64 zig 0.16.0 | **不崩**(同一源码同一参数) |
| host: x86_64-linux zig 0.16.0(GitHub ubuntu-latest) | 崩 |
| host: x86_64-linux zig master(2026-10-01) | 仍崩 |
| -ODebug | 不崩 |
| -OReleaseFast / ReleaseSafe / ReleaseSmall | 全崩 |
| -mcpu sm_90 / sm_80 / baseline;-target nvptx32-cuda | 全崩 |
| -fstrip 与否;ZIG_GLOBAL_CACHE_DIR 与 LOCAL 同目录与否 | 无关 |
| noinline 任意 helper | 无关 |

## 最小化结论(从 162 行 kernel 二分)

崩溃要求同时存在(缺一不崩):

1. 从全局内存的 16 字节向量 load(A 或 B 单侧即可),经寄存器(含按值/指针传递的 struct,两种方式都崩);
2. 向 `addrspace(.shared)` 数组的**批量 store**——A 半区(2 条 v4 store)或 B 半区单独不崩,两半区并存(4 条 v4 store)崩;两侧各 8 条标量 store(16 条)也崩;
3. 同一 kernel 内对 shared 数组的后续 read(8 条标量或 2 条向量,任一)。

空化 store(`storeShared` 体置空)→ 存活;空化 load → 存活;空化 shared read → 存活。
缩小 tile(128→32)与累加器(8x8→2x2)后仍崩,见 repro 文件。

## 推测(标注为推测)

IR 跨平台一致、崩溃随 host 变化 → 疑似编译器自身的 UB(未初始化读/悬垂指针),由 LLVM 优化管线的某个 NVPTX 相关 pass 触发。未符号化(官方二进制 strip)。

## 仓库侧的处置

在 ziglang/zig 修复前,CI(Linux)对 sgemm_opt2 单独豁免:`zig build kernels` 失败后复查,
若唯一缺失的 PTX 是 sgemm_opt2 则以 loud warning 继续,其 PTX 断言随之跳过。
见 ci.yml 注释与本文件互为索引。豁免范围被断言钉死:任何*其他* kernel 缺失 = CI 红。
