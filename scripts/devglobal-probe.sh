#!/bin/sh
# Host-written device global, read by name on the device.
#
# This is the first check of a path Zig cannot express by itself. The PTX symbol
# is made visible by a post-pass at build time; whether ptxas honours that, and
# whether the device actually re-reads the global after a host write, can only be
# answered on a GPU.
#
# What to look at, in order:
#   1. "resolved device global ... 16 bytes" -> ptxas exposed the promoted symbol
#      and cuModuleGetGlobal found it. If instead you see "cannot resolve device
#      global", the .visible promotion did not survive assembly.
#   2. round 0 exact -> the upload is being read at all
#   3. round 1 exact -> it is re-read per launch. Round 0 passing while round 1
#      fails would mean the value was baked into the code.
set -eu
echo "== declaration in the shipped PTX =="
grep -n 'dev_bias' kernels/dev_global.ptx || echo "  (symbol absent — build did not promote it)"
echo
echo "== run =="
./zoxide run kernels/dev_global.ptx
echo
echo "== negative control: same PTX with .visible removed =="
# Confirms the post-pass is actually load-bearing. Without this, "promoted PTX
# works" is consistent with ptxas exposing module-scope globals anyway, which
# would make the whole pass unnecessary complexity.
#
# Expected: 'cannot resolve device global'. If it PASSES, the promotion is not
# needed and should be deleted.
sed 's/^\.visible \.global/.global/' kernels/dev_global.ptx > /tmp/dev_global_unpromoted.ptx
grep -n 'dev_bias\[16\]' /tmp/dev_global_unpromoted.ptx
if ./zoxide run /tmp/dev_global_unpromoted.ptx 2>&1 | tee /tmp/unpromoted.log | grep -q '^PASS'; then
    echo "UNEXPECTED: unpromoted PTX also works — the .visible pass is not needed"
else
    echo "as expected, unpromoted fails:"
    grep -m1 'cannot resolve\|FAIL\|error' /tmp/unpromoted.log || tail -1 /tmp/unpromoted.log
fi
