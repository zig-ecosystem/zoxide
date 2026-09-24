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
# The basename has to stay dev_global.ptx: 'zoxide run' picks the example by
# filename stem, so renaming the file makes the run fail for an unrelated reason
# and the control proves nothing.
rm -rf /tmp/unpromoted && mkdir -p /tmp/unpromoted
sed 's/^\.visible \.global/.global/' kernels/dev_global.ptx > /tmp/unpromoted/dev_global.ptx
grep -n 'dev_bias\[16\]' /tmp/unpromoted/dev_global.ptx
./zoxide run /tmp/unpromoted/dev_global.ptx >/tmp/unpromoted.log 2>&1 || true
# Demand the specific error. Accepting any failure is how the first version of
# this control reported success while actually failing on an unknown-example
# check, before it ever reached cuModuleGetGlobal.
if grep -q '^PASS' /tmp/unpromoted.log; then
    echo "UNEXPECTED: unpromoted PTX also works — the .visible pass is not needed, delete it"
elif grep -q 'cannot resolve device global' /tmp/unpromoted.log; then
    echo "as expected: symbol is not resolvable without .visible — the pass is load-bearing"
    grep -m1 'cannot resolve device global' /tmp/unpromoted.log
else
    echo "INCONCLUSIVE: failed for some other reason, control proves nothing:"
    cat /tmp/unpromoted.log
fi
