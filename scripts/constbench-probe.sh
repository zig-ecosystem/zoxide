#!/bin/sh
# Is .const actually faster than the read-only data cache for a warp-uniform
# read? The repo has been claiming so in three doc comments without ever
# measuring it on this device.
#
# Both kernels read the same 64-entry table 64 times with an index uniform across
# every thread, hand-unrolled identically. Left to LLVM the two loops got
# different unroll factors (4x vs 8x), which would have measured the unroller
# instead of the memory path.
set -eu
echo "== emitted instruction mix (should be 64 table reads each) =="
python3 - <<'PY' 2>/dev/null || grep -c 'ld.const.f32\|ld.global.nc.f32' kernels/const_vs_ldg.ptx
import re
t=open('kernels/const_vs_ldg.ptx').read().split('\n')
b=[i for i,l in enumerate(t) if '.entry' in l and '(' in l]
for j,i in enumerate(b):
    nm=t[i].split('(')[0].replace('.entry','').strip()
    nx=b[j+1] if j+1<len(b) else len(t)
    body='\n'.join(t[i:nx]); f=lambda p: len(re.findall(p,body))
    n=len([x for x in t[i:nx] if x.strip().endswith(';')])
    print(f"  {nm:32s} ld.const={f(r'ld.const.f32'):3d} ld.global.nc={f(r'ld.global.nc.f32'):3d} statements={n}")
PY
echo
echo "== measurement =="
# Best of 10 per kernel. Correctness is checked too, so a kernel reading the
# wrong table cannot win by being fast.
./zoxide run kernels/const_vs_ldg.ptx
echo
echo "Any of the three outcomes is a real answer:"
echo "  faster  -> ConstBank earns its complexity"
echo "  equal   -> read-only cache already handles it; the docs are wrong"
echo "  slower  -> ConstBank should be marked not recommended for this pattern"
