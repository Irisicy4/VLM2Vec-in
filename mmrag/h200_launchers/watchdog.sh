#!/bin/bash
# Emits a line ONLY on a collapse signature, trainer death, or a periodic trend snapshot.
# Collapse signature is the VLM2Vec-4B failure: reward group_std pinned at 0 -> zero advantage.
D=/raid/icy/mmrag_data
M=$D/runs/emaidx-1M-s3000/metrics.jsonl
SNAP=${SNAP:-1800}   # seconds between routine snapshots
last_snap=0
while true; do
  if ! pgrep -f "train_rl.py --profile gme2b" >/dev/null; then
    echo "TRAINER GONE at $(date +%T) — last step $(wc -l < $M)"; exit 2
  fi
  python3 - "$M" <<'PY'
import json, sys, time
rows=[json.loads(l) for l in open(sys.argv[1])]
if len(rows) < 60: sys.exit(0)
tail = rows[-50:]
gs = [r["reward/group_std_mean"] for r in tail]
zf = sum(1 for g in gs if g == 0)/len(gs)
base = rows[:100]
rw_b = sum(r["reward/raw_mean"] for r in base)/len(base)
rw_t = sum(r["reward/raw_mean"] for r in tail)/len(tail)
gp_b = sum(r["retrieval/gold_in_topN"] for r in base)/len(base)
gp_t = sum(r["retrieval/gold_in_topN"] for r in tail)/len(tail)
step = rows[-1]["step"]
if zf >= 0.5:
    print(f"COLLAPSE-SIGNAL step {step}: group_std==0 in {zf:.0%} of last 50 "
          f"(VLM2Vec-4B signature; >=50% is the alert threshold)", flush=True)
if rw_b > 0 and rw_t < 0.6*rw_b:
    print(f"COLLAPSE-SIGNAL step {step}: judge reward {rw_t:.3f} < 60% of base {rw_b:.3f}", flush=True)
if gp_b > 0 and gp_t < 0.6*gp_b:
    print(f"COLLAPSE-SIGNAL step {step}: gold-in-pool {gp_t:.3f} < 60% of base {gp_b:.3f}", flush=True)
PY
  now=$(date +%s)
  if [ $((now-last_snap)) -ge $SNAP ]; then
    last_snap=$now
    python3 - "$M" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
if len(rows) < 30: sys.exit(0)
tail=rows[-50:]; base=rows[:100]
f=lambda k,rs: sum(r[k] for r in rs)/len(rs)
print(f"TREND step {rows[-1]['step']}/3000  reward {f('reward/raw_mean',base):.3f}->{f('reward/raw_mean',tail):.3f}"
      f"  goldpool {f('retrieval/gold_in_topN',base):.3f}->{f('retrieval/gold_in_topN',tail):.3f}"
      f"  grpstd {f('reward/group_std_mean',tail):.4f}"
      f"  stale {rows[-1].get('index/stale_mean',0):.0f}", flush=True)
PY
  fi
  sleep 300
done
