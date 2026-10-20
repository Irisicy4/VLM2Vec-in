#!/bin/bash
M=/raid/icy/mmrag_data/runs/emaidx-1M-s3000/metrics.jsonl
N0=$(wc -l < "$M"); T0=$(date +%s)
sleep "${1:-900}"
N1=$(wc -l < "$M"); T1=$(date +%s)
DT=$((T1-T0)); D=$((N1-N0))
echo "CLEAN WINDOW: steps $N0 -> $N1 = $D steps in ${DT}s"
if [ "$D" -gt 0 ]; then
  python3 -c "
s=$DT/$D
print(f'  {s:.1f} s/step')
print(f'  full 3000 steps: {3000*s/3600:.1f} h')
print(f'  remaining from step $N1: {(3000-$N1)*s/3600:.1f} h')
print(f'  ck750 in {(750-$N1)*s/3600:.1f} h | ck1500 in {(1500-$N1)*s/3600:.1f} h')
print(f'  vs baseline 6.64 s/step -> {s/6.64:.1f}x slower')
"
else echo "  NO PROGRESS in window"; fi
