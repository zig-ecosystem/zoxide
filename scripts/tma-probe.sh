#!/bin/sh
# TMA descriptor-driven copy.
#
# The previous run hung. Two changes: the kernel now issues
# fence.proxy.async.shared::cta after mbarrier.init (the async proxy is a separate
# memory consumer and is not guaranteed to observe an initialised barrier without
# it -- CUTLASS orders it the same way), and the wait is bounded so a barrier that
# never completes is reported instead of hanging.
#
# Being straight about it: the fence is the most likely cause, not a confirmed
# one. The bounded wait is the part that matters either way, because it turns an
# undiagnosable hang into a message.
set -eu
echo "== emitted TMA instructions =="
grep -nE 'fence\.proxy|cp\.async\.bulk|mbarrier\.(init|arrive|try_wait)' kernels/tma_smoke.ptx
echo
echo "== run (60s ceiling; a hang here means the bounded wait did not help) =="
if timeout 60 ./zoxide run kernels/tma_smoke.ptx --arch sm_90a; then
    :
else
    rc=$?
    if [ "$rc" = 124 ]; then
        echo
        echo "TIMED OUT at the shell. The in-kernel poll budget did not expire, so"
        echo "the launch itself is not returning -- a different problem from a"
        echo "barrier that never completes."
    else
        echo "(exit $rc)"
    fi
fi
echo
echo "Reading it:"
echo "  'mbarrier never completed'  -> expect_tx byte count wrong, or copy never started"
echo "  swizzle .none exact         -> the descriptor is driving the copy"
echo "  .b128 differs from .none    -> the swizzle field reaches the hardware"
echo "  .b128 identical             -> FAIL; the .none pass would prove little"
