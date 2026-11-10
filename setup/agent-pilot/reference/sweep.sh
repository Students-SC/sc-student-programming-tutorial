#!/bin/bash
set -euo pipefail

SIZES="32 64 128 256 512 1024 2048"
for bs in $SIZES; do
    hipcc -O2 -DBLOCK_SIZE=$bs -o vector_scale_$bs vector_scale.cpp
done

srun --partition=mi2101x --nodes=1 --ntasks=1 --time=5:00 bash -c '
printf "%-10s %-16s %s\n" "Block" "Bandwidth(GB/s)" "Status"
for bs in '"$SIZES"'; do
    if out=$(./vector_scale_$bs 2>&1); then
        bw=$(echo "$out" | awk "/Bandwidth/ {print \$2}")
        st=$(echo "$out" | awk "/Status/ {print \$2}")
    else
        bw="-"
        st="ERROR: $(echo "$out" | tail -1)"
    fi
    printf "%-10s %-16s %s\n" "$bs" "$bw" "$st"
done'
