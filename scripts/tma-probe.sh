#!/bin/sh
# First TMA run. The global->shared direction of cp.async.bulk.tensor has no LLVM
# intrinsic, so it is hand-written asm here and ptxas has never assembled this
# construction. Nothing about a wrong descriptor faults, which is why there is a
# control rather than just a comparison.
set -eu
echo "== emitted TMA instructions =="
grep -nE 'cp\.async\.bulk|mbarrier\.(init|arrive|try_wait)' kernels/tma_smoke.ptx
echo
echo "== run =="
# Two descriptors over the same tensor and tile:
#   .none  -> shared memory holds the row-major tile, linear readout must match
#   .b128  -> hardware permutes 16-byte chunks, the same readout must differ
# The second is the control. If both agreed, "matches the source" could not be
# told apart from "the buffer happened to hold the right bytes".
./zoxide run kernels/tma_smoke.ptx --arch sm_90a
echo
echo "Reading it:"
echo "  swizzle .none exact   -> the descriptor is driving the copy"
echo "  .b128 differs         -> the swizzle field reaches the hardware"
echo "  .b128 identical       -> FAIL, and it would mean the first line proves little"
echo "  chunk^row diagnostic  -> my model of the permutation, not a claim under test"
