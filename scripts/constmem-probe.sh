#!/bin/sh
# Real CUDA constant memory (PTX .const) plus the device-global path, both
# host-written.
#
# These are two different mechanisms and the PTX shows which is which:
#   dev_global  -> .global + ld.global.nc   (read-only data cache, __ldg)
#   const_bank  -> .const  + ld.const      (64 KB constant window, broadcast)
set -eu
echo "== const_bank: declarations and access =="
grep -nE '^\.const|ld\.const|mov\.u64.*const_scales' kernels/const_bank.ptx
echo
echo "== const_bank: run =="
# pad_before displaces the table from offset 0, so wrong (non symbol-relative)
# addressing gives wrong values instead of passing by accident.
./zoxide run kernels/const_bank.ptx
echo
echo "== dev_global: run (unchanged, for contrast) =="
./zoxide run kernels/dev_global.ptx
