#!/bin/bash
# Wait for the vLLM reward server, then start the 3-day mixed-domain scaling run against it.
# Falls back to the in-process HF reader if the server never comes up, so the experiment starts either
# way -- the deliverable is the curve, not the serving stack.
set -uo pipefail
D=/raid/icy/mmrag_data
URL=http://127.0.0.1:8377
for i in $(seq 1 60); do
  curl -s --max-time 3 $URL/health >/dev/null 2>&1 && break
  sleep 20
done
if curl -s --max-time 3 $URL/health >/dev/null 2>&1; then
  echo "$(date +%T) reward server healthy -> training with vLLM rewards"
  exec bash $D/run_mix2m_scale3d.sh 6 mix2m-scale3d "$URL"
else
  echo "$(date +%T) reward server never came up -> falling back to in-process HF reader"
  exec bash $D/run_mix2m_scale3d.sh 6 mix2m-scale3d ""
fi
