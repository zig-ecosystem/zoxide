#!/bin/sh
# TMA descriptor-driven copy, third attempt.
#
# The first two runs hung and returned no output at all. That absence was a tool
# problem, not a clue: the report writer is buffered and flushed at exit, so
# killing the process discarded every line. Fixed three ways:
#
#   - progress markers go to stderr, which is unbuffered, so they survive a kill
#   - the kernel writes stage markers to device memory, so a device-side stall
#     reports which instruction it got past
#   - the retry loop moved out of inline asm into Zig. The asm version wrote its
#     output register before re-reading the barrier address on later iterations,
#     and nothing stops the compiler from giving an output and an input the same
#     register -- it needed "=&r". Not hand-rolling the loop removes the class.
set -eu
echo "== emitted TMA instructions =="
grep -nE 'fence\.proxy|cp\.async\.bulk|mbarrier\.(init|arrive|try_wait)' kernels/tma_smoke.ptx
echo
echo "== run (60s ceiling) =="
if timeout 60 ./zoxide run kernels/tma_smoke.ptx --arch sm_90a 2>&1; then
    :
else
    rc=$?
    [ "$rc" = 124 ] && echo "
TIMED OUT. The last stderr marker above is where it stopped."
    [ "$rc" != 124 ] && echo "(exit $rc)"
fi
echo
echo "Markers localise a stall, in order:"
echo "  encoding descriptor / launching / launched / synchronised"
echo "  then per-stage: entered, initialised, issued, barrier completed"
echo
echo "Then the actual claim:"
echo "  .none exact + .b128 differs  -> the descriptor drives the copy and swizzle reaches HW"
