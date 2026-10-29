#!/usr/bin/env bash
OUT=${1:-$HOME/bags/cpu_$(date +%Y%m%d_%H%M%S).log}
echo "registro CPU per processo, RAM e swap in $OUT (Ctrl-C per fermare)"
while true; do
  t=$(date +%H:%M:%S)
  top -b -n1 -w 220 | awk -v t=$t 'NR>7 && $9>2.0 {print t, $1, $9, $12}'
  free -m | awk -v t=$t '/^Mem:/ {u=$3} /^Swap:/ {s=$3} END {print t, "MEM", u, s}'
  sleep 2
done > $OUT
