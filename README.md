# zoxide

A Zig-native CUDA kernel development skeleton — the first milestone of a
Zig-native reimplementation inspired by [cuda-oxide](https://github.com/Rust-GPU/cuda-oxide)
(a Rust CUDA compiler). Target GPU: NVIDIA H20 (Hopper,
compute capability 9.0, `sm_90`). No GPU or CUDA toolkit is required to
produce PTX; only `ptxas` (for cubin assembly) comes from CUDA.

## Requirements

- Zig 0.16.0 (with the LLVM NVPTX backend, included in standard builds)
- Optional: `ptxas` from the CUDA toolkit, for `zoxide cubin`

## Usage

```sh
zig build           # builds the `zoxide` CLI into zig-out/bin/
zig build kernels   # compiles every kernel in src/examples/ to zig-out/kernels/<name>.ptx (sm_90)
zig build kernel    # single default kernel -> zig-out/kernels/kernel.ptx (kept for compat)

zig-out/bin/zoxide doctor [--arch sm_90]                       # probe toolchain + GPU
zig-out/bin/zoxide ptx src/kernel.zig -o out.ptx [--arch sm_90]  # Zig source -> PTX
zig-out/bin/zoxide cubin out.ptx -o out.cubin [--arch sm_90]     # PTX -> cubin (needs ptxas)
```

Supported arch values: `sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120`
(default `sm_90` everywhere).

`zoxide ptx` shells out to
`zig build-lib -target nvptx64-cuda -mcpu <arch> -fstrip -fno-ubsan-rt -fno-emit-bin -femit-asm=... -O ReleaseFast`.

`zoxide doctor` probes zig, the nvptx64 backend, ptxas, libNVVM and the GPU
(via `nvidia-smi --query-gpu=name,compute_cap,driver_version
--format=csv,noheader`). Every probe is tagged by severity:

- `[ok]` — present and working
- `[info]` — absent but only relevant to a future feature (libNVVM / LTOIR)
- `[warn]` — absent; only blocks part of the workflow (zig → can't compile
  new PTX; ptxas → can't assemble cubins; no GPU → fine on a dev machine)
- `[fail]` — a real error (e.g. `--arch sm_90` passed while the visible GPU
  is CC 8.6: the cubin would not load)

The exit code is 1 only when a `[fail]` item is present; `--arch` with an
invalid value is a usage error and also exits 1. The closing `summary:`
block reports readiness per workflow — `compile (zig -> PTX)`,
`assemble (ptxas -> cubin)`, `run (GPU)` — so a GPU pod without zig
correctly reports compile-unavailable but assemble/run-ready, exit 0.

## Cross-compiling for a Linux GPU pod

```sh
zig build -Dtarget=x86_64-linux-musl  --prefix zig-out/x86_64-linux-musl   # static, for ptx/cubin/doctor
zig build -Dtarget=aarch64-linux-musl --prefix zig-out/aarch64-linux-musl
zig build -Dtarget=x86_64-linux-gnu.2.28 -Doptimize=ReleaseSmall --prefix zig-out/x86_64-linux-gnu  # dynamic, for `run`
```

The musl builds are statically linked ELF executables (verified with
`file`). **For `zoxide run` prefer the gnu dynamic build**: the runner
dlopens libcuda, and a statically linked musl binary falls back to zig's
own ElfDynLib loader, which may not handle libcuda's dependency chain. The
gnu build links against glibc ≤ 2.28 symbols, so it runs on any pod image
with glibc ≥ 2.28 (Ubuntu 20.04+).

## Running kernels on the GPU (`zoxide run`)

`src/cuda_driver.zig` binds the CUDA driver API by dlopening
`libcuda.so.1` → `libcuda.so` (no cuda.h, no @cImport). On the pod:

```sh
# build PTX on the dev machine, copy to pod, then:
./zoxide run vector_add.ptx                    # assembles via ptxas, runs, verifies
./zoxide run vector_add.cubin                  # skip ptxas, load cubin directly
./zoxide run warp_reduce.ptx --grid 64
./zoxide run vector_add.ptx --kernel vector_add_$_vectorAdd   # override mangled name
```

Known examples are matched by file basename (`vector_add`, `shared_reverse`,
`warp_reduce`, `atomic_counter`); each has a hardcoded host harness that
fills inputs, launches, copies back and verifies against a CPU reference.
Expected output on success, e.g.:

```
device: NVIDIA H20
kernel: vector_add_$_vectorAdd
vector_add: n=1048576 grid=4096 block=256
PASS: vector_add n=1048576, max err 0
```

Any FAIL exits 1. Without libcuda/GPU the command prints a clear error and
exits 1 (no crash).

Driver bindings cover: cuInit, cuDeviceGetCount, cuDeviceGet,
cuDeviceGetName, cuCtxCreate_v2, cuCtxSetCurrent, cuModuleLoadData,
cuModuleGetFunction, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2,
cuMemcpyDtoH_v2, cuLaunchKernel, cuCtxSynchronize, cuGetErrorString,
cuGetErrorName. ABI notes: CUdeviceptr is u64; `_v2` are the real symbol
names; kernelParams is an array of pointers to argument values.

## Generating intrinsics bindings (`zoxide gen`)

`zoxide gen <cuda-oxide/intrinsics> [-o src/gen/intrinsics.zig]` consumes
cuda-oxide's catalog.json + probes/*.ll and emits `src/gen/intrinsics.zig`
(committed; regenerate only when the catalog changes). probes/*.ll are the
authoritative source for LLVM signatures (concrete `declare` lines); the
catalog contributes family/module/name metadata. Get the data by pinning a
cuda-oxide commit (the `intrinsics/` directory is self-contained; do not
vendor it here — catalog.json alone is ~365k lines). Catalog provenance:
NVIDIA PTX ISA documentation data, Apache-2.0.

Current coverage: 329 wrappers in 17 groups (sreg, warp, float, async_copy,
convert, barrier, fence, matrix, ...). Unmapped ~1387 entries:

- 689 probes lower via inline PTX asm — **unmappable in Zig** (no asm
  template substitution). These need per-family NVVM or builtin alternatives.
- 690 catalog entries have no probe (tcgen05 233, register_mma 154,
  sparse_mma 122, extended_minmax 52, packed_alu 30, ...; mostly Hopper/
  Blackwell matrix/TMA features, deferred to a later milestone).

Type mapping: `void`→`void`, `i1`→`bool`, `iN`→`iN`, `float/double/half`→
`f32/f64/f16`, `ptr`→`?*anyopaque`, `ptr addrspace(1)`→`[*]addrspace(.global)
const u8`, `ptr addrspace(3)`→`[*]addrspace(.shared) u8`, aggregates →
generated `Agg*` extern structs. Vectors/bfloat/immarg annotations are
currently unmapped.

Generated bindings are re-exported as `cuda.gen` — e.g.
`cuda.gen.warp.ballot_sync(mask, pred)`.

## Device-side library (`src/cuda.zig`)

Freestanding (no host std); all device operations go through LLVM NVVM
intrinsics or Zig builtins. API overview:

| API | Lowers to |
|---|---|
| `threadIdx() / blockIdx() / blockDim() / gridDim()` → `Idx3{x,y,z}` | `llvm.nvvm.read.ptx.sreg.{tid,ctaid,ntid,nctaid}.{x,y,z}` |
| `laneId()`, `warpId()` (in-block), `globalThreadId()` | `...sreg.laneid`, `tid.x/32`, `ctaid.x*ntid.x+tid.x` |
| `syncThreads()` | `llvm.nvvm.barrier0` → `bar.sync 0` |
| `syncWarp(mask)` | `llvm.nvvm.bar.warp.sync` |
| `shflDownSync / shflUpSync / shflXorSync / shflIdxSync(T, mask, val, off)` for `i32/u32/f32` | `llvm.nvvm.shfl.sync.{down,up,bfly,idx}.i32` (f32 via bitcast) |
| `atomicAdd(T, ptr, val)` for `u32/u64/f32`, global or shared pointers | Zig `@atomicRmw` → single `atom.{global,shared}.add.*` |
| `Keep(.{ &kernelA, &kernelB })` | dummy-export DCE workaround, see below |

Example kernel skeleton:

```zig
const cuda = @import("cuda");

pub fn myKernel(out: [*]f32) callconv(.kernel) void {
    const gid = cuda.globalThreadId();
    out[gid] = 1.0;
}

comptime {
    _ = cuda.Keep(.{&myKernel}).__zoxide_keep_kernels;
}
```

The kernel's PTX symbol is mangled (`mykernel_$_myKernel`); the host must
look it up under that name.

## Example kernels (`src/examples/`)

- `vector_add.zig` — c = a + b, one f32 per thread (`globalThreadId`)
- `shared_reverse.zig` — per-block array reversal via shared memory +
  `syncThreads` (PTX shows `.shared` decl + `st.shared`/`ld.shared`)
- `warp_reduce.zig` — block-wide sum: `shflDownSync` within warps, shared
  memory + `syncThreads` across warps (`shfl.sync.down.b32`, `bar.sync 0`)
- `atomic_counter.zig` — `atomicAdd` on global u32/f32, histogram bins, and a
  shared-memory counter (`atom.global.add.*`, `atom.shared.add.u32`)

## Shared memory: verified approach

Declare a container-level variable with `addrspace(.shared)`:

```zig
var tile: [256]f32 addrspace(.shared) = undefined;
```

Zig 0.16 accepts this and the NVPTX backend emits a per-block
`.shared .align 4 .b8 <mangled>[1024];` declaration inside the `.entry`;
loads/stores lower to `ld.shared.b32` / `st.shared.b32`. No intrinsic or
`@ptrFromInt` fallback is needed.

## Notes on the kernel (`src/kernel.zig`)

- Freestanding vector-add kernel; reads `%tid.x` / `%ctaid.x` / `%ntid.x` to
  compute the global thread id.
- **Do not use inline asm with `$0`/`%0` operand placeholders.** Zig emits
  asm templates verbatim (no LLVM template substitution — this is zig-wide,
  not NVPTX-specific), so `mov.u32 $0, %tid.x;` leaks a literal `$0` into the
  PTX and ptxas rejects it (`Arguments mismatch for instruction 'mov'` /
  `Unknown symbol '$0'`). Instead, call the LLVM NVVM intrinsics directly:
  `extern fn @"llvm.nvvm.read.ptx.sreg.tid.x"() i32;` etc. They lower to
  proper `mov.u32 %rN, %tid.x;` instructions that ptxas accepts. (The
  `@"llvm.*"` intrinsic access is technically an accident of the compiler —
  ziglang/zig#2291 — and may be restricted in a future release.)
- The kernel is `pub fn ... callconv(.kernel)` (not `export fn`): zig 0.16 +
  LLVM's NVPTX backend rejects `export` on kernel functions ("NVPTX aliasee
  must be a non-kernel function definition"). A dummy `export fn
  __zoxide_keep_kernels()` returning a pointer to the kernel keeps it alive
  across dead-code elimination. The kernel's PTX symbol is therefore mangled
  to `kernel_$_vectorAdd`.
- `bundle_ubsan_rt = false` / `-fno-ubsan-rt` for the same alias reason.
- **i1 in extern fns is not extern-compatible on nvptx**; map LLVM `i1`
  params/returns to Zig `bool` (verified: `vote.sync.ballot` gets a proper
  predicate register). Declaring them as `i32` crashes LLVM ("Copy one
  register into another with a different width").
- **`@ptrCast` cannot change pointer address spaces.** Declare kernel
  parameters with the target address space directly (e.g.
  `g: [*]addrspace(.global) const u8`) or use `@addrSpaceCast`.
- **`addrspace(.shared)` variables must be container-level**; a local
  `var x: [N]T addrspace(.shared)` is rejected.
- Pointer address-space syntax is `[*]addrspace(.shared) u8` (qualifier
  between `[*]` and the element type).
- `llvm.nvvm.shfl.sync.*` in current LLVM returns plain `i32` (4 args), not
  the legacy `{i32, i1}` pair — the generated bindings follow the probe
  signatures from cuda-oxide's rust-llvm-23.1.
