# EMA experiments — handoff for a fresh agent

Branch: `mmrag-ema` of https://github.com/Irisicy4/VLM2Vec-in.git (on-disk checkout name here
is `VLM2Vec-rl`; the remote repo name is `VLM2Vec-in`). Start:

```bash
git clone -b mmrag-ema https://github.com/Irisicy4/VLM2Vec-in.git VLM2Vec-rl
cd VLM2Vec-rl
```

## Environment you must have

- The bytenas volume `tns-algo-video-public-my2` mounted (all data lives there, not in git):
  `D=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data` — pools/corpora in `$D/built/`,
  runs in `$D/runs/`, results in `$D/results/`. `HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home`.
- MLX CLI at `/opt/tiger/mlx_deploy/bin/mlx` (submit: `mlx job submitv2 --path <yaml>`; status:
  `mlx job get <id>` and grep `arnold_trial_status`; there is NO `mlx job stop` — web UI only).
- **Path caveat**: `mmrag/mlx_entry_cell.sh` and `mmrag/mlx_submit_cell.sh` hardcode
  `ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl`. Either work from that
  existing checkout, or after cloning elsewhere edit `ROOT=` in both scripts (and the
  `namespace:` in the submit script if you are not `yliang.26`). The MLX worker executes the
  entry script *from the bytenas path baked into the yaml*, so your clone must be on the mount.

## Recipe (fixed — do not vary alongside the EMA knob)

v3-pure = PL-GRPO on GME-Qwen2-VL-2B, frozen Qwen2.5-VL-3B reward reader:
`--algo plgrpo --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 --no_force_gold
--reward judge --learning_rate 2e-5 --seed 0` (batch 4, LoRA r32 from the entry-script defaults).
Eval readers: 3B for reward, **7B for eval** (`eval_vqa.py --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500`).
Metrics: entR@5 / ansR@5 from `$D/results/<name>.retrieval.metrics.json`
(`entity_recall["5"]`, `answer_recall["5"]`), acc from `<name>.vqa_top5.json`. Name the metric in every claim.

EMA knobs in `train_rl.py`: `--index_ema <m>` (index blend, **m is retention**, fresh weight = 1−m)
with `--index_refresh_extra 256`; `--model_ema <m>` (EMA doc-tower weights).

## What is already known (all 1 seed, 500 steps, div45k_v3 pool, ledgered in RL_REDESIGN.md)

| cell | args delta | entR@5 | ansR@5 | acc |
|---|---|---|---|---|
| div45kv3-v3pure | (none) | 76.83 | 64.57 | 33.00 |
| div45kv3-emaidx | `--index_ema 0.9 --index_refresh_extra 256` | 75.77 | 63.14 | 32.93 |
| div45kv3-emaenc | + `--model_ema 0.99` | 72.47 | 60.49 | 32.20 |

- emaidx = parity at 500 steps / 45k pool. Expected: index staleness barely binds there.
- emaenc's number is **confounded** (eval ran against live weights, not the EMA-tracked ones —
  see RL_REDESIGN.md 2026-08-27/28 entries); treat it as unusable, re-run before citing.
- Geometry (geom_probe_q.py): RL spread lives on the **query tower**; doc side barely moves;
  emaidx damping was doc-local. So model-EMA on the doc tower is aimed at the wrong tower a priori.

## The experiments to run, in priority order

### 1. EMA × scaling — the informative cell (unrun)

Long training on the big pool is where the index goes stale and retrieval declines:
`v3-rows1M-s3000` (810k-row pool, 3000 steps, constant LR) ends at entR@5 **72.63** / ansR@5 61.59 /
acc 33.8, vs the 500-step point ~78.7 entR@5 (seed 0; that peak failed seed validation — s1 gave
74.37 — so compare full curves, not the peak). Question: does index-EMA prevent the decline?

```bash
bash mmrag/mlx_submit_cell.sh "emaidx-1M-s3000" gme2b \
  "--pool built/pool_train_1M.jsonl --image_dataset infoseek --max_train_rows 0 \
   --max_steps 3000 --lr_schedule constant --no_force_gold --reward judge \
   --algo plgrpo --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 \
   --seed 0 --learning_rate 2e-5 --index_ema 0.9 --index_refresh_extra 256"
```

(That is exactly `$D/runs/v3-rows1M-s3000/args.json` + the two EMA flags; corpus defaults to
`built/corpus_small.jsonl` in the entry script.) Read it against v3-rows1M-s3000 AND its interim
checkpoints (ck2250 acc 35.0 exists) — evaluate interim checkpoints of the new run the same way
(`CELL_CKPT`-style configs, see `mmrag/mlx_configs/v3-rows1M-s3000-ck2250.yaml`).

### 2. Clean emaenc duplicate

Re-run `--index_ema 0.9 --index_refresh_extra 256 --model_ema 0.99` on
`built/pool_div45k_v3.jsonl` (500 steps, otherwise v3-pure args) as a fresh MLX cell under a NEW
name (e.g. `div45kv3-emaenc2`), and make sure eval consumes the saved checkpoint, so the number is
clean. Only then compare against div45kv3-v3pure.

### 3. Blend-weight cell (unrun live knob)

`--index_ema` retention sweep on div45kv3, 500 steps: m ∈ {0.5, 0.99} bracketing the 0.9 point.
Convention pinned in RL_REDESIGN.md: **m is retention** (0.99 = very stale index, 0.5 = half-fresh).

### 4. Seeds

Any cell that looks like a win gets seeds 1–2 (`--seed N`, suffix `-s1`/`-s2`) before any claim.
No margins without matched seeds and matched estimator.

## Mechanics / house rules

- `mlx_submit_cell.sh "<name>" gme2b "<train args>"` writes `mmrag/mlx_configs/<name>.yaml`,
  submits, appends the job id to `$D/mlx_jobs.txt`. The entry script is **idempotent**: it skips
  everything if `$D/results/<name>.vqa_top5.json` exists, and skips training if
  `$D/runs/<name>/adapter_model.safetensors` exists — safe to resubmit after a kill; never
  launch a second job on the same run dir while one is alive.
- One cell = train → encode corpus → retrieval eval (max-q 3000) → VQA eval (7B, k=5, max-q 1500).
- Trends: least-squares slope with 95% CI over the series, never eyeballed; CI spanning zero =
  not distinguishable.
- Eval artifacts stay in `$D/results` / repo-gitignored dirs, never `/tmp`. Never `pkill keep_gpu`
  on a dev box (the entry script guards this; leave the guard alone).
- Ledger every result (including nulls and confounds) in `mmrag/RL_REDESIGN.md`; the results page
  with per-row reproduce commands is `mmrag/docs/vqa_retriever_loop.html` — add rows there too.
