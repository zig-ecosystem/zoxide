#!/bin/sh
# pod-verify.sh — end-to-end verification of zoxide on a GPU pod.
#
# Usage: pod-verify.sh [zoxide-binary] [ptx-dir] [--quick]
#   zoxide-binary  default: ./zoxide
#   ptx-dir        default: ./kernels (fallback: .)
#   --quick        skip the sgemm bench smoke
#
# Exit 0 iff nothing FAILed (SKIP is allowed for missing PTX files).
# A timestamped text report is written to ./zoxide-verify-<ts>.txt.

set -u

ZOXIDE="${1:-./zoxide}"
PTXDIR="${2:-./kernels}"
QUICK=0
for a in "$@"; do [ "$a" = "--quick" ] && QUICK=1; done
[ -d "$PTXDIR" ] || PTXDIR=.

TS="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || date +%s)"
REPORT="zoxide-verify-$TS.txt"

PASS=0; FAIL=0; SKIP=0
LINES=""

record() { # status name detail
    LINES="$LINES
$1  $2  $3"
    case "$1" in
        PASS) PASS=$((PASS+1));;
        FAIL) FAIL=$((FAIL+1));;
        SKIP) SKIP=$((SKIP+1));;
    esac
    echo "$1  $2  $3"
}

have_ptx() { [ -f "$PTXDIR/$1.ptx" ]; }

echo "== zoxide pod verification ($TS)"
echo "binary: $ZOXIDE   ptx dir: $PTXDIR"

if [ ! -x "$ZOXIDE" ]; then
    echo "FAIL  binary  $ZOXIDE not executable"
    exit 1
fi

# 1. doctor
if "$ZOXIDE" doctor --arch sm_90 > /tmp/zoxide-doctor.$$.txt 2>&1; then
    record PASS doctor "exit 0"
else
    record FAIL doctor "exit $?"
fi
GPU_LINE=$(grep -m1 '^\[ok  \] gpu:' /tmp/zoxide-doctor.$$.txt | sed 's/.*gpu: //')
PTXAS_LINE=$(grep -m1 'ptxas' /tmp/zoxide-doctor.$$.txt | sed 's/^[^ ]* ptxas: //')

# Degraded mode: without a visible GPU (or ptxas), skip run/bench/cubin
# checks instead of failing — used by CI runners.
DEGRADED=0
if [ -z "$GPU_LINE" ]; then DEGRADED=1; echo "note: no GPU visible — degraded mode (run/bench checks become SKIP)"; fi
if [ "$DEGRADED" = 1 ]; then
    for ex in vector_add shared_reverse warp_reduce atomic_counter; do
        if have_ptx "$ex"; then record SKIP "run/$ex" "no GPU"; else record SKIP "run/$ex" "PTX missing"; fi
    done
    if have_ptx wgmma_smoke; then record SKIP "run/wgmma_smoke" "no GPU"; else record SKIP "run/wgmma_smoke" "PTX missing"; fi
    if have_ptx dev_global; then record SKIP "run/dev_global" "no GPU"; else record SKIP "run/dev_global" "PTX missing"; fi
    if have_ptx const_bank; then record SKIP "run/const_bank" "no GPU"; else record SKIP "run/const_bank" "PTX missing"; fi
    if have_ptx const_vs_ldg; then record SKIP "run/const_vs_ldg" "no GPU"; else record SKIP "run/const_vs_ldg" "PTX missing"; fi
    if have_ptx tma_smoke; then record SKIP "run/tma_smoke" "no GPU"; else record SKIP "run/tma_smoke" "PTX missing"; fi
    if have_ptx sgemm_swz; then record SKIP "bench/sgemm_swz" "no GPU"; else record SKIP "bench/sgemm_swz" "PTX missing"; fi
    for k in hgemm_wgmma hgemm_wgmma2 hgemm_wgmma3; do
        if have_ptx "$k"; then record SKIP "bench/$k" "no GPU"; else record SKIP "bench/$k" "PTX missing"; fi
    done
    if have_ptx intrinsics_smoke; then record SKIP "cubin/intrinsics_smoke" "no GPU/ptxas"; else record SKIP "cubin/intrinsics_smoke" "PTX missing"; fi
else
# 2. functional examples
for ex in vector_add shared_reverse warp_reduce atomic_counter; do
    if ! have_ptx "$ex"; then
        record SKIP "run/$ex" "PTX missing"
        continue
    fi
    out=$("$ZOXIDE" run "$PTXDIR/$ex.ptx" 2>&1)
    if echo "$out" | grep -q '^PASS'; then
        record PASS "run/$ex" "$(echo "$out" | grep -m1 '^PASS')"
    else
        record FAIL "run/$ex" "$(echo "$out" | tail -1)"
    fi
done

# 2b. wgmma layout smoke. sm_90a-only, so a Hopper-specific failure here
#     (ptxas rejecting the arch, or a non-Hopper GPU) is reported as SKIP
#     rather than FAIL; a wrong *result* is a real FAIL.
if ! have_ptx wgmma_smoke; then
    record SKIP "run/wgmma_smoke" "PTX missing"
else
    out=$("$ZOXIDE" run "$PTXDIR/wgmma_smoke.ptx" --arch sm_90a 2>&1)
    if echo "$out" | grep -q '^PASS'; then
        record PASS "run/wgmma_smoke" "$(echo "$out" | grep -m1 '^PASS')"
    elif echo "$out" | grep -qi 'failed to assemble\|not supported\|invalid arch'; then
        record SKIP "run/wgmma_smoke" "sm_90a unavailable: $(echo "$out" | tail -1)"
    else
        record FAIL "run/wgmma_smoke" "$(echo "$out" | tail -1)"
    fi
fi

# 2c. host-written device global read by name on the device. Two failure modes,
#     worth keeping apart in the report:
#       - cannot resolve the symbol -> ptxas did not expose it
#       - resolves but round 1 differs -> the kernel is not re-reading it, i.e.
#         the read was constant-folded against the initialiser
if ! have_ptx dev_global; then
    record SKIP "run/dev_global" "PTX missing"
elif ! grep -q '^\.global.*dev_bias' "$PTXDIR/dev_global.ptx"; then
    record FAIL "run/dev_global" "PTX has no 'dev_bias' global — the read was folded away; it must go through cuda.ldg()"
else
    out=$("$ZOXIDE" run "$PTXDIR/dev_global.ptx" 2>&1)
    if echo "$out" | grep -q '^PASS'; then
        record PASS "run/dev_global" "$(echo "$out" | grep -m1 '^PASS')"
    elif echo "$out" | grep -q 'cannot resolve device global'; then
        record FAIL "run/dev_global" "ptxas did not expose the symbol: $(echo "$out" | tail -1)"
    else
        record FAIL "run/dev_global" "$(echo "$out" | tail -2 | tr '\n' ' ')"
    fi
fi

# 2d. real CUDA constant memory (PTX .const), declared via module-scope asm
#     because Zig has no way to express it. The symbol is unmangled, so a
#     resolution failure here usually means the declaration did not reach the
#     PTX at all rather than a naming slip.
if ! have_ptx const_bank; then
    record SKIP "run/const_bank" "PTX missing"
elif ! grep -q '^\.const .*const_scales' "$PTXDIR/const_bank.ptx"; then
    record FAIL "run/const_bank" "PTX has no '.const ... const_scales' declaration"
else
    out=$("$ZOXIDE" run "$PTXDIR/const_bank.ptx" 2>&1)
    if echo "$out" | grep -q '^PASS'; then
        record PASS "run/const_bank" "$(echo "$out" | grep -m1 '^PASS')"
    else
        record FAIL "run/const_bank" "$(echo "$out" | tail -2 | tr '\n' ' ')"
    fi
fi

# 2e. is .const actually faster than the read-only cache for a warp-uniform
#     read? Measured rather than asserted. A null result is a PASS: it would
#     mean ConstBank is only worth it for parameter-space or layout reasons, and
#     the broadcast claim in the docs is wrong.
if ! have_ptx const_vs_ldg; then
    record SKIP "run/const_vs_ldg" "PTX missing"
else
    out=$("$ZOXIDE" run "$PTXDIR/const_vs_ldg.ptx" 2>&1)
    if echo "$out" | grep -q '^PASS'; then
        record PASS "run/const_vs_ldg" "$(echo "$out" | grep -m1 'x the throughput')"
    else
        record FAIL "run/const_vs_ldg" "$(echo "$out" | tail -2 | tr '\n' ' ')"
    fi
fi

# 2f. TMA. The g2s direction is hand-written asm because LLVM has no intrinsic
#     for it, so this is the first time ptxas sees the construction. sm_90a, and
#     an arch rejection is a SKIP; a wrong *result* is a FAIL.
if ! have_ptx tma_smoke; then
    record SKIP "run/tma_smoke" "PTX missing"
else
    out=$("$ZOXIDE" run "$PTXDIR/tma_smoke.ptx" --arch sm_90a 2>&1)
    if echo "$out" | grep -q '^PASS'; then
        record PASS "run/tma_smoke" "$(echo "$out" | grep -m1 '^PASS')"
    elif echo "$out" | grep -qi 'failed to assemble\|not supported\|invalid arch\|CUDA 12'; then
        record SKIP "run/tma_smoke" "TMA unavailable: $(echo "$out" | tail -1)"
    else
        record FAIL "run/tma_smoke" "$(echo "$out" | tail -2 | tr '\n' ' ')"
    fi
fi

# 3. bench smoke (small n, PASS check only) unless --quick
if [ "$QUICK" = 1 ]; then
    record SKIP "bench/sgemm_swz" "--quick"
elif have_ptx sgemm_swz; then
    # n=512 on purpose: this is a correctness smoke test, not a measurement. At
    # that size only 16 blocks exist for 78 SMs, so the GFLOPS figure is far below
    # what the same kernel reaches at n=4096 and should not be read as a
    # regression. The label says so.
    out=$("$ZOXIDE" bench "$PTXDIR/sgemm_swz.ptx" --n 512 --iters 2 2>&1)
    if echo "$out" | grep -q '^PASS'; then
        record PASS "bench/sgemm_swz" "smoke n=512 (not a perf number): $(echo "$out" | grep -m1 'GFLOPS')"
    else
        record FAIL "bench/sgemm_swz" "$(echo "$out" | tail -1)"
    fi
else
    record SKIP "bench/sgemm_swz" "PTX missing"
fi

# 3b. hgemm_wgmma bench (correctness + TFLOPS). Same sm_90a caveat.
if [ "$QUICK" = 1 ]; then
    record SKIP "bench/hgemm_wgmma" "--quick"
else
    for k in hgemm_wgmma hgemm_wgmma2 hgemm_wgmma3; do
        if ! have_ptx "$k"; then
            record SKIP "bench/$k" "PTX missing"
            continue
        fi
        out=$("$ZOXIDE" bench "$PTXDIR/$k.ptx" --n 4096 --iters 5 --arch sm_90a 2>&1)
        if echo "$out" | grep -q '^PASS'; then
            record PASS "bench/$k" "$(echo "$out" | grep -m1 'GFLOPS')"
        elif echo "$out" | grep -qi 'failed to assemble\|not supported\|invalid arch'; then
            record SKIP "bench/$k" "sm_90a unavailable: $(echo "$out" | tail -1)"
        else
            record FAIL "bench/$k" "$(echo "$out" | tail -1)"
        fi
    done
fi

# 4. intrinsics_smoke assembles via ptxas (run inside `run` path is not
#    defined for it; use cubin assembly as the check)
if have_ptx intrinsics_smoke; then
    if "$ZOXIDE" cubin "$PTXDIR/intrinsics_smoke.ptx" -o /tmp/zoxide-smoke.$$.cubin --arch sm_90 >/dev/null 2>&1 \
       && [ -s /tmp/zoxide-smoke.$$.cubin ]; then
        record PASS "cubin/intrinsics_smoke" "ptxas ok"
    else
        record FAIL "cubin/intrinsics_smoke" "ptxas failed"
    fi
    rm -f /tmp/zoxide-smoke.$$.cubin
else
    record SKIP "cubin/intrinsics_smoke" "PTX missing"
fi
fi

{
    echo "zoxide pod verification report"
    echo "timestamp (UTC): $TS"
    echo "binary: $ZOXIDE"
    echo "ptx dir: $PTXDIR"
    echo "gpu: ${GPU_LINE:-not detected}"
    echo "ptxas: ${PTXAS_LINE:-not detected}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/nvidia-smi: /'
    fi
    echo "---"
    echo "$LINES" | sed '/^$/d'
    echo "---"
    echo "totals: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
} | tee "$REPORT"

[ "$FAIL" -eq 0 ]
