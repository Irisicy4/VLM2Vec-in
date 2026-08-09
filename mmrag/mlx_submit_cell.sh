#!/usr/bin/env bash
# Submit a generic mm-RAG cell as a 1-GPU MLX job.
#   usage: mlx_submit_cell.sh "<name>" "<profile>" "<extra train_rl.py args>"
set -eu
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-in
MLX=/opt/tiger/mlx_deploy/bin/mlx
D=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
CFGDIR=$ROOT/mmrag/mlx_configs; mkdir -p "$CFGDIR"

NAME="${1:?name}"; PROFILE="${2:-gme2b}"; ARGS="${3:-}"
if [ -f "$D/results/$NAME.vqa_top5.json" ]; then echo "[cell] $NAME already done"; exit 0; fi
cfg="$CFGDIR/$NAME.yaml"
cat > "$cfg" <<EOF
caption: '[mm-RAG] ${NAME}'
notifyRequest:
  silent: true
jobDefVersion:
  entrypointMode: FULL_SCRIPT
  imageMeta:
    imageVid: d6lj4vjc77u7t7t31hp0
  lazyDownloadCode: true
  name: 'mmrag-${NAME}'
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
    bash ${ROOT}/mmrag/mlx_entry_cell.sh
  envsList:
    CELL_NAME: ${NAME}
    CELL_PROFILE: ${PROFILE}
    CELL_TRAIN_ARGS: ${ARGS}
namespace: /user/yliang.26
EOF
jid=$("$MLX" job submitv2 --path "$cfg" 2>&1 | grep -vE "自动升级|新版本|TTY" | grep -oE "[a-f0-9]{16}" | head -1)
if [ -n "$jid" ]; then
  echo "[cell] submitted $NAME  job=$jid"
  echo "$(date +%F_%H:%M) $NAME $jid" >> "$D/mlx_jobs.txt"
else
  echo "[cell] SUBMIT FAILED $NAME"; "$MLX" job submitv2 --path "$cfg" 2>&1 | tail -5
fi
