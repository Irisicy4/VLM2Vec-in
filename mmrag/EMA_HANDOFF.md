# EMA experiments — handoff for a fresh agent

Branch: `mmrag-ema` of https://github.com/Irisicy4/VLM2Vec-in.git (on-disk checkout name here
is `VLM2Vec-rl`; the remote repo name is `VLM2Vec-in`). Start:

```bash
git clone -b mmrag-ema https://github.com/Irisicy4/VLM2Vec-in.git VLM2Vec-rl
cd VLM2Vec-rl
```

## Environment

Nothing but code is in git. Every script resolves data through one env var:
`MMRAG_DATA=<your data dir>` (call it `$D`). Layout: `$D/built/` pools/corpora/queries,
`$D/images/` image archives, `$D/runs/` training output, `$D/results/` eval output,
`$D/cache/` corpus-embedding caches. Set `HF_HOME` somewhere with ~50 GB free.

On the original cluster `$D=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data`
(bytenas volume `tns-algo-video-public-my2`) and jobs go through the MLX CLI
(`/opt/tiger/mlx_deploy/bin/mlx`; submit `mlx job submitv2 --path <yaml>`, status
`mlx job get <id>` grep `arnold_trial_status`, NO stop subcommand — web UI only).
**On any other cluster: skip MLX entirely and run the entry script directly** (below);
the yaml resource block (bytenas volume, queue, namespace `/user/yliang.26`) is site-specific.

Hardware per cell: **one GPU, ~40 GB VRAM minimum, 48 GB comfortable.** Measured per stage
(stages run sequentially, so the training peak is the ceiling):

| stage | resident | VRAM |
|---|---|---|
| RL training | GME-2B bf16 (~4.4 GB) + Qwen2.5-VL-3B reward reader (7.1 GB) co-resident | 25.9 GiB measured peak alloc, ~35 GiB working set |
| corpus encode + retrieval eval | GME-2B | 8.7–10.3 GiB measured |
| VQA eval | Qwen2.5-VL-7B (16 GB) + batch-8 image activations | ~25–30 GiB (estimated) |

24 GB cards are not viable (the 7B eval reader alone is 16 GB of weights). To fit a tighter
card, prefer `--encode_bs 64→32` and the eval-stage `--bs 192` / `--batch_size 8` in
`mlx_entry_cell.sh`: encoding is pure forward passes, so shrinking it is deterministic and
cannot change which data the run sees. **Change `--reader_batch_size` only as a last resort**
— it gates sampled generation (`reader_temperature 0.7`), so a different batch size can
reorder RNG consumption and change which rollouts are drawn.

On a card shared with another tenant, `train_rl.py --mem_frac <f>` caps the training process
(`torch.cuda.set_per_process_memory_fraction`). Note it reaches **training only**:
`CELL_TRAIN_ARGS` is passed to `train_rl.py` alone, so the encode/retrieval/VQA stages that
follow run uncapped. An eval-stage OOM is cheap to recover — the entry script skips training
when `$RUN/adapter_model.safetensors` exists, so a rerun resumes at the eval stage. Historical OOMs in the logs show PyTorch holding only ~8.7 GiB on
a full card; those were the `keep_gpu` daemon, not real demand — don't size against them.

**On a shared GPU box, two hazards:**
1. `--device cuda:0` is hardcoded throughout the entry scripts. To use any other physical
   GPU, launch with `CUDA_VISIBLE_DEVICES=<n>` so `cuda:0` maps onto it.
2. `mlx_entry_cell.sh` runs `pkill -x keep_gpu` on any host whose name doesn't match
   `$MMRAG_DATA/local_hostname.txt` — if that file is missing or stale **the pkill fires and
   may kill another team's VRAM-holding daemon**. Run `hostname > $D/local_hostname.txt`
   before the first cell, or delete that line outright on a shared host.

Host side: ~24 CPU cores and ~220 GB RAM per job. `train_rl.py` has NO resume — a kill
restarts at step 0 — so use a non-preemptible partition and ask for ≥24 h walltime
(~12 h train for the 3000-step cell, plus a few hours of eval).
Deps are checked/installed by the entry script: torch 2.8.0, transformers 4.57.0, peft 0.17.1,
accelerate 1.13.0, qwen-vl-utils, torchvision, flash-attn 2.8.1 (`--no-build-isolation`).
`torchvision` is required but **not declared** by qwen-vl-utils (`vision_process.py` imports
it; the package's own deps are av/packaging/pillow/requests), so a fresh venv fails without
it — it is now in the entry scripts' pip line, but check yours if you cloned before that.
transformers 4.57.0 is yanked on PyPI; the exact pin still installs and imports fine, but a
new site may prefer 4.57.1.
flash-attn 2.8.1 is what this cluster installed, not a requirement: upstream ships no 2.8.1
wheel for torch 2.8, and 2.8.3 has precedent in this project (`README.md:73`, `RESUME.md:13`,
the Isambard env, there paired with torch 2.9). Use the 2.8.3 build matching your torch —
it is an attention kernel, not an estimator change.

## Staging the data (fresh site — nothing pre-mounted)

Two classes of data. **Class 1 must be copied** from the original cluster — these are
locally-built artifacts; rebuilding them (build_div_mix*.py) would give a *different* pool and
your numbers would not be comparable to the baselines below. **Class 2 can be re-downloaded**
from public sources if a direct copy is impossible.

### Class 1 — download from Hugging Face (easiest, ~2.1 GB)

Published public at **https://huggingface.co/datasets/Icey444/mmrag-class1-built** — all five
built files plus the baseline `results/*.json` bundle and a README carrying sha256+bytes for
each. Verify against that manifest before training.

```bash
mkdir -p $D/built $D/results $D/runs $D/cache $D/images/{Infoseek,EVQA,OKVQA,OVEN}
hf download Icey444/mmrag-class1-built --repo-type dataset --local-dir $D/hf_pull
mv $D/hf_pull/built/*.jsonl $D/built/
cp -n $D/hf_pull/results/*.json $D/results/   # -n: never clobber your own results
```

**Never rebuild `pool_train_1M.jsonl`.** Its exact build flags are not recorded (the ledger
says only "full InfoSeek train, target ~1M rows"; actual 810,276 rows), and `train_rl.py`
draws training rows with `rng.choice(rows)` — by INDEX into the file's row order. A 3000-step
cell at batch 4 touches only ~12k of those 810k rows, so row order alone decides the training
set: anything not byte-identical trains on different data and cannot be compared to the
baselines. `corpus_small.jsonl` / `queries_test.jsonl` *are* deterministically rebuildable
(`build_infoseek_data.py --train_per_entity 10 --max_train 60000 --distractor_articles 20000`,
verified by an exact-count rebuild in `REPRO_NEWCLUSTER.md` and a `cmp` check in
`build_big_pool.sh`) — but downloading is still simpler.

### Class 1, alternate — copy verbatim from `SRC=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data` (~2.2 GB)

```bash
mkdir -p $D/built $D/results $D/runs $D/cache $D/images/{Infoseek,EVQA,OKVQA,OVEN}
rsync -avP $SRC/built/pool_train_1M.jsonl   $D/built/   # 962M  cell 1 train pool
rsync -avP $SRC/built/corpus_small.jsonl    $D/built/   # 277M  cell 1 corpus + eval corpus
rsync -avP $SRC/built/pool_div45k_v3.jsonl  $D/built/   #  48M  cells 2-3 train pool
rsync -avP $SRC/built/corpus_div_v2.jsonl   $D/built/   # 797M  cells 2-3 corpus
rsync -avP $SRC/built/queries_test.jsonl    $D/built/   #  52M  eval queries (all cells)
rsync -avP $SRC/results/*.json              $D/results/ # small: baseline rows for comparison
```

Do NOT copy `$D/results/<your-new-cell-name>.*` patterns you plan to create — the entry
script is idempotent and will skip a cell whose `results/<name>.vqa_top5.json` already exists.

### Class 2 — image archives (copy if you can, re-download if you can't)

`images/Infoseek/` and `images/EVQA/` on the source are **symlinks** into `$SRC/images_dl/` —
copy with `rsync -avLP` (follow links) or you get dangling links. The `*.tar.index.json`
files regenerate automatically on first use; don't worry if they're missing.

| archive | size | needed by | public source |
|---|---|---|---|
| `images/Infoseek/infoseek_train_images.tar` | 45G | ALL cells | HF dataset `BByrneLab/M2KR_Images` |
| `images/Infoseek/infoseek_val_images.tar` | 8.4G | ALL cells (eval queries) | HF `BByrneLab/M2KR_Images` |
| `images/EVQA/inat.zip` | 8.4G | cells 2-3 (div pool) | HF `BByrneLab/M2KR_Images` |
| `images/EVQA/inat_val.tar` | 8.6G | cells 2-3 | iNat-2021: `ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz` (retar) |
| `images/EVQA/google-landmark.tar` | 2.7G | cells 2-3 | HF `BByrneLab/M2KR_Images` |
| `images/OKVQA/train2014.zip` | 13G | cells 2-3 | COCO `images.cocodataset.org/zips/train2014.zip` |
| `images/OVEN/shard01..04.tar` | 94G | cells 2-3 | HF `BByrneLab/M2KR_Images` (OVEN dirs 01–04) |

Re-download route: `python -c "from huggingface_hub import hf_hub_download; hf_hub_download('BByrneLab/M2KR_Images', '<file>', repo_type='dataset', local_dir='$D/images_dl')"`,
then place/symlink into the `images/<Dataset>/` names above (`mmrag/image_store.py:59` is the
authority on expected paths). **Cell 1 only needs the two Infoseek tars (~53 GB)** — if you
stage nothing else, you can still run the highest-priority experiment.

### Class 3 — models (plain HF downloads, ~25 GB into `$HF_HOME`)

```bash
huggingface-cli download Alibaba-NLP/gme-Qwen2-VL-2B-Instruct   # retriever base
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct            # reward reader (train)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct            # eval reader
```

### Wire it up

```bash
export MMRAG_DATA=$D  HF_HOME=<your hf cache>  PYTHONPATH=<clone root>
hostname > $D/local_hostname.txt   # guard: entry script pkills keep_gpu ONLY on hosts != this
# 14 shell scripts under mmrag/ hardcode original-cluster paths, in FIVE patterns: the repo
# path appears both as project/VLM2Vec-rl (checkout name) AND project/VLM2Vec-in (remote name —
# mlx_entry_scale.sh, mlx_submit_scale.sh, scripts/run_repro.sh), plus the data dir, the
# hf_home, and a data/tmp TMPDIR fallback the data-dir pattern does NOT match. Sweep them all:
grep -rl "/mnt/bn" mmrag --include="*.sh" | xargs sed -i "\
        s#/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl#$(pwd)#; \
        s#/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-in#$(pwd)#; \
        s#/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data#$D#; \
        s#/mnt/bn/tns-algo-video-public-my2/yijiangli/data/tmp#$D/tmp#; \
        s#/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home#$HF_HOME#" \
  && grep -rn "/mnt/bn" mmrag --include="*.sh" && echo "REMNANTS ABOVE — fix by hand" \
  || echo "clean: 0 remnants"
# Leave mmrag/mlx_configs/*.yaml and results manifests untouched — they are the provenance
# record of prior runs, and a non-MLX site never executes them. python defaults
# (push_ckpts.py, mmeb_subset.py) are overridden by the exported MMRAG_DATA.
```

Smoke-test the staging before burning GPU-days (each should print, not throw):

```bash
python3 - <<'PY'
import os, json
from mmrag.image_store import open_stores
D = os.environ["MMRAG_DATA"]
q = [json.loads(l) for _, l in zip(range(5), open(f"{D}/built/queries_test.jsonl"))]
s = open_stores(D, "infoseek")               # add open_stores(D, "div") if staging cells 2-3
print([s.get(r["image_id"]).size for r in q])
PY
```

### Running a cell without MLX (any 1-GPU box)

```bash
CELL_NAME=emaidx-1M-s3000 CELL_PROFILE=gme2b \
CELL_TRAIN_ARGS="--pool built/pool_train_1M.jsonl --image_dataset infoseek --max_train_rows 0 \
 --max_steps 3000 --lr_schedule constant --no_force_gold --reward judge --algo plgrpo \
 --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 --seed 0 --learning_rate 2e-5 \
 --index_ema 0.9 --index_refresh_extra 256" \
nohup bash mmrag/mlx_entry_cell.sh > $D/local_emaidx-1M-s3000.log 2>&1 &
```

Same env-triple with your cell name/args for any other cell; the script does
train → corpus encode → retrieval eval → VQA eval and drops
`$D/results/<name>.{retrieval.metrics,vqa_top5}.json`. It is safe to rerun after a crash
(skips completed stages). On the *original* cluster keep using
`bash mmrag/mlx_submit_cell.sh "<name>" gme2b "<args>"` instead.

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
checkpoints (ck2250 = entR@5 74.67 / ansR@5 62.82 / acc 35.0 exists) — evaluate interim
checkpoints of the new run through `mmrag/mlx_entry_eval.sh` (env: `CELL_NAME`, `CELL_CKPT`
relative to `$D`, `CELL_PROFILE=gme2b`; same estimator as the in-cell eval — retrieval
`--max-q 3000`, 7B top-5 VQA `--max-q 1500`). It hardcodes the original paths like the other
two scripts — apply the same sed. MLX users: `mmrag/mlx_configs/v3-rows1M-s3000-ck2250.yaml`
is the submit-side template.

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
