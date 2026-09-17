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
zig build kernel    # compiles src/kernel.zig to PTX at zig-out/kernel.ptx (sm_90)

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
zig build -Dtarget=x86_64-linux-musl  --prefix zig-out/x86_64-linux-musl
zig build -Dtarget=aarch64-linux-musl --prefix zig-out/aarch64-linux-musl
```

Both produce statically linked ELF executables (verified with `file`).

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
