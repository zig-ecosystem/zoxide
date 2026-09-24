#!/bin/sh
# TMA descriptor-driven copy, fourth attempt.
#
# Last run localised the stall: the descriptor encoded, the launch went through,
# and cuCtxSynchronize never returned. So the kernel itself was not terminating --
# and the poll budget alone explains that. A failing mbarrier.try_wait suspends
# the thread for an implementation-defined interval, and the budget was 1<<20.
# At even 60us per poll that is 63 seconds, past the 60s ceiling, with no TMA bug
# required.
#
# Budget is now 1024. The kernel returns either way, so this is the first run that
# actually tests the copy rather than my patience.
#
# The kernel also samples the raw mbarrier state word at three points. That word
# separates the two failures that look identical from outside:
#   unchanged      -> the copy engine never touched the barrier; no transfer at all
#   moved, short   -> a transfer started; expect_tx and delivered bytes disagree
set -eu
echo "== run (60s ceiling) =="
if timeout 60 ./zoxide run kernels/tma_smoke.ptx --arch sm_90a 2>&1; then
    :
else
    rc=$?
    [ "$rc" = 124 ] && echo "
STILL TIMED OUT. With a 1024-poll budget the kernel cannot be spinning, so the
stall is elsewhere -- the last stderr marker above is the place."
    [ "$rc" != 124 ] && echo "(exit $rc)"
fi
