#!/usr/bin/env bash
# Submit the mm-RAG data-scaling ladder as MLX jobs — one 1-GPU job per cell, so all cells run
# in parallel instead of queueing behind each other on the shared local node (the first local
# attempt was SIGKILLed after ~50 min when the box was reclaimed).
#
# Usage: bash mmrag/mlx_submit_scale.sh              # all cells
#        bash mmrag/mlx_submit_scale.sh scale-rows50k:50000:500   # just one
set -eu
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-in
MLX=/opt/tiger/mlx_deploy/bin/mlx
CFGDIR=$ROOT/mmrag/mlx_configs
D=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
mkdir -p "$CFGDIR"

CELLS=(
  "scale-rows12k:12500:500"
  "scale-rows25k:25000:500"
  "scale-rows50k:50000:500"
  "scale-rows100k:100000:500"
  "scale-rows200k:200000:500"
  "scale-rows200k-s1500:200000:1500"
)
[ $# -gt 0 ] && CELLS=("$@")

submit_one() {
  local cell="$1"
  local name="${cell%%:*}"
  if [ -f "$D/results/$name.vqa_top5.json" ]; then
    echo "[scale] $name already has results, skipping"; return 0
  fi
  local cfg="$CFGDIR/$name.yaml"
  cat > "$cfg" <<EOF
caption: '[mm-RAG] data-scaling ${name}'
notifyRequest:
  silent: true
jobDefVersion:
  entrypointMode: FULL_SCRIPT
  imageMeta:
    imageVid: d6lj4vjc77u7t7t31hp0
  lazyDownloadCode: true
  name: 'mmrag-${name}'
  resource:
    arnoldConfig:
      clusterId: 11
      groupIds:
        - 665
      quotaPool: default
      preemptible: false
      bytenasVolumes:
        - name: tns-algo-video-public-my2
          accessMode: RW
      roles:
        - cpu: 24
          gpu: 1
          gpuv: 19
          memory: 220160
          name: worker
          num: 1
          queueName: compute-543-my2-cloudnative-aigcp-tns.algo.public.pool.minipod.gcpsd.c-guarantee
    backend: ARNOLD
jobRunParams:
  entrypointFullScript: |
    bash ${ROOT}/mmrag/mlx_entry_scale.sh
  envsList:
    SCALE_CELL: ${cell}
namespace: /user/yliang.26
EOF
  local jid
  jid=$("$MLX" job submitv2 --path "$cfg" 2>&1 | grep -vE "自动升级|新版本|TTY" | grep -oE "[a-f0-9]{16}" | head -1)
  if [ -n "$jid" ]; then
    echo "[scale] submitted $name  job=$jid"
    echo "$(date +%F_%H:%M) $name $jid" >> "$D/mlx_jobs.txt"
  else
    echo "[scale] SUBMIT FAILED for $name — see $cfg"
    "$MLX" job submitv2 --path "$cfg" 2>&1 | tail -5
  fi
}

for c in "${CELLS[@]}"; do submit_one "$c"; sleep 2; done
echo "--- submitted jobs ---"; tail -10 "$D/mlx_jobs.txt" 2>/dev/null || true
