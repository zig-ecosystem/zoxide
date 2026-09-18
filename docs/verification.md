# Verification log

## H20 pod acceptance (2026-09)

Environment: NVIDIA H20 (Hopper, compute capability 9.0), driver 550.90.07,
ptxas 12.8, x86_64 Linux pod (no zig on pod; binary cross-built with
`-Dtarget=x86_64-linux-gnu.2.28 -Doptimize=ReleaseSmall`).

- `zoxide doctor --arch sm_90`: zig/ptxas/libNVVM absence reported as
  warn/info, GPU detected, arch check ok, exit 0.
- All four example kernels PASS end-to-end via `zoxide run <name>.ptx`
  (ptxas assembly + cuModuleLoadData + launch + host-side verification):
  - `vector_add`: n=1048576, max err 0
  - `shared_reverse`: exact per-block reversal
  - `warp_reduce`: block sums match CPU reference
  - `atomic_counter`: counters and histogram match expected values

Note: `atomic_counter` initially deadlocked the GPU because LLVM duplicated
`bar.sync` onto divergent program points (conditional write before the
barrier). Fixed by restructuring the kernel (unconditional same-value
write); see README "known issues". The bar.sync single-point check is now
enforced in CI.
