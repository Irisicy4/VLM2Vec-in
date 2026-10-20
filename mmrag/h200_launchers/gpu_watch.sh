#!/bin/bash
while true; do
  echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk '{printf "g%s:%s/%sMB,%s%% ", $1,$2,$3,$4}' ORS='')"
  sleep 60
done
