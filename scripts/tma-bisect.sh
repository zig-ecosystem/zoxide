#!/bin/sh
# Which half is broken: the mbarrier code, or the TMA copy?
#
# tma_smoke has hung four times. Three explanations were tried and all three were
# wrong, the last one included -- a 1024-poll kernel still failed to return, so the
# budget was not it either. Guessing which part of a two-part mechanism is at fault
# has not worked, so this tests the parts separately.
#
# Each step is a separate launch with an unbuffered marker before it, so a hang
# names the step instead of going silent.
set -eu

echo "== 1. mbarrier alone, no TMA (30s ceiling) =="
# barOnly        init + arrive + wait
# barExpectZero  + arrive.expect_tx(0): completion then rests on arrivals only, so
#                this separates a mis-encoded expect_tx from a copy that never
#                delivered
# barTwoPhase    + a second phase on parity 1. A wrong parity passes a
#                single-phase test and deadlocks a pipelined one, which is what S2
#                will be.
if timeout 30 ./zoxide run kernels/mbar_smoke.ptx --arch sm_90a 2>&1; then
    mbar_ok=1
else
    rc=$?
    mbar_ok=0
    [ "$rc" = 124 ] && echo "
TIMED OUT in the barrier layer. The last marker above is the step. TMA is not
involved in any of these three kernels, so the fault is in src/tma.zig's barrier
helpers."
fi

echo
echo "== 2. full TMA copy (30s ceiling) =="
if [ "$mbar_ok" = 0 ]; then
    echo "skipped: the barrier layer has to work before a TMA hang can be attributed"
    echo "to the copy."
else
    timeout 30 ./zoxide run kernels/tma_smoke.ptx --arch sm_90a 2>&1 || {
        rc=$?
        [ "$rc" = 124 ] && echo "
TIMED OUT, but step 1 passed. So the barrier helpers are fine and the stall is in
cp.async.bulk.tensor or the descriptor driving it."
    }
fi
