---
name: h200-training-setup
description: Best practices for setting up GPU training on the iris H200 box (8× H200, shared multi-user). Covers conda env layout, TMPDIR placement, surgical Ray cleanup, library version pairings, and pitfalls learned the hard way.
---

# H200 training setup playbook

This box has **8× NVIDIA H200 (143 GB each)**, shared between multiple users (seen so far: `iris`, `hm`, `william` — vLLM serving fleets + jax training, often 100–135 GB/card — and `ztw`). What follows is the recipe for setting up a new training run without stepping on other users, getting bitten by `/tmp` filling, or killing your run with a stray `pkill`.

---

## 1. Filesystem layout — read this first

```
/dev/md0   → /             438 GB ext4   ← root SSD; /tmp lives here. SMALL. Don't store training data or Ray sessions.
/dev/md1   → /raid          14 TB raid   ← shared big pool, slow random I/O, ~700 GB free typical.
/dev/shm   → /dev/shm     1008 GB tmpfs  ← RAM-backed, fast, ephemeral. Best place for Ray TMPDIR.
/dev/nvme2n1p1 → /home    880 GB nvme    ← per-user, fast, but smaller than /raid.
```

**Where things should go:**

| Thing | Right path | Why |
|---|---|---|
| Conda env | `/raid/icy/iris/envs/<name>` (use `conda create -p`) | /raid has space; envs can be 5–15 GB |
| Repo / source code | `/raid/icy/iris/<project>/` | /raid is fine, IO is rare for source |
| Datasets | `/raid/icy/iris/data/<project>/` | /raid; tens to hundreds of GB |
| HF model cache | `/raid/icy/iris/.cache/huggingface/` (`HF_HOME=...`) | /raid; a single 7B Qwen ~15 GB |
| Run logs (text) | `/raid/icy/iris/<project>/logs/` | /raid; tqdm-heavy logs can hit GB/hour |
| Checkpoints | `/raid/icy/iris/data/<project>/checkpoints/` | /raid; 30 GB+ per FSDP shard set |
| **Ray sessions / TMPDIR / Triton cache** | `<repo>/tmp/` on /raid OR `/dev/shm/<user>-tmp/` | **NEVER /tmp.** See §3 below. |

The default behavior of Ray + vLLM is to dump everything to `/tmp`, which is on the small root partition. A multi-turn run can fill 90 GB in 6 hours and kill itself. Always override.

---

## 2. Environment setup template

Use `conda create -p` with an absolute path under /raid so the env doesn't go to a /home quota.

```bash
# Accept the conda channel ToS once if needed:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -p /raid/icy/iris/envs/<name> python=3.10 pip -y
conda activate /raid/icy/iris/envs/<name>
pip install uv
```

### Library version pairings that work on this box (CUDA 12.8 driver, H200 sm_90)

For projects pinned to **vllm 0.8.x / EasyR1-style** (e.g., grounded-rl, older verl forks):

| Package | Version |
|---|---|
| python | 3.10 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0 |
| vllm | 0.8.3 |
| transformers | 4.51.0 (must be `<4.53` for vendored verl) |
| flash-attn | **2.7.4.post1** (NOT 2.5.x — fails to compile on Hopper sm_90 with torch 2.6) |
| xformers | 0.0.29.post2 |
| ray | 2.55.1 |
| tensordict | **0.6.2** (newer 0.10+ adds strict TensorDict locking that breaks EasyR1 0.3.0's `union_tensor_dict`) |
| triton | 3.2.0 |
| deepspeed | `<=0.16.9` |

For projects on **upstream verl 0.8+ / vllm 0.11+** (newer):

| Package | Version |
|---|---|
| python | 3.12 |
| torch | 2.8.0+cu128 |
| vllm | 0.11.0 |
| transformers | 4.56.1 |
| flash-attn | 2.8.1 |
| tensordict | 0.10.0 |

(See `/raid/miniconda3/envs/verl/` for a working example of the upstream combo.)

**Verification one-liner after install:**

```bash
python -c "
import torch, vllm, transformers, flash_attn, ray, tensordict
print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())
print('vllm', vllm.__version__, 'transformers', transformers.__version__)
print('flash_attn', flash_attn.__version__, 'ray', ray.__version__, 'tensordict', tensordict.__version__)
"
```

### HF token

For gated checkpoints / data downloads, export at session start:

```bash
export HF_TOKEN="hf_..."
export HF_HUB_TOKEN="$HF_TOKEN"
export HF_HOME="${HF_HOME:-/raid/icy/iris/.cache/huggingface}"
```

---

## 3. TMPDIR — the most important hygiene item

**Symptom of getting this wrong:** after several hours, the run dies with `[ray_log_sink] Failed flush to file /tmp/ray/.../python-core-worker-*.log: No space left on device` and the whole `/` partition is at 100%. Bash itself stops working until someone clears `/tmp`.

**Fix:** redirect TMPDIR + RAY_TMPDIR + TRITON_CACHE_DIR before launching anything. Both options below work; pick based on how long the run lasts.

```bash
# Option A: repo-local on /raid (persistent across reboots, slower but big)
REPO_TMP="$REPO_ROOT/tmp"
export TMPDIR="$REPO_TMP"
export RAY_TMPDIR="$REPO_TMP/ray"
export TRITON_CACHE_DIR="$REPO_TMP/triton"
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$TRITON_CACHE_DIR"

# Option B: tmpfs on /dev/shm (RAM, ~1 TB available, fastest, ephemeral)
export TMPDIR="/dev/shm/$USER-tmp"
export RAY_TMPDIR="$TMPDIR/ray"
export TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p "$RAY_TMPDIR" "$TRITON_CACHE_DIR"
```

`mktemp` honors `TMPDIR`, so any scratch script or wandb media files also land in the right place.

I/O speed comparison from a multi-turn web_grounding GRPO run (3B model, 4 GPUs, identical config):
- /tmp on /dev/md0 SSD: **~27 min/step**
- /raid TMPDIR on /dev/md1 RAID: **~25 min/step** (close enough — disk I/O is not the dominant cost for this workload)

So performance-wise either option is fine; the choice is about *not filling the small root partition*.

---

## 4. GPU coordination (multi-user box)

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

**Always check OWNERSHIP, not just free memory.** A 0-utilization GPU with 80 GB allocated could mean:

1. Another user's training in a between-rollout pause → leave alone.
2. Orphaned root-owned VLLM/FSDP zombies from someone's killed job → can't kill as `iris`, ask for sudo.
3. Your own orphan (rare if you use the kill recipe in §5).

To trace a process to its owner / launcher:
```bash
ps -p <pid> -o pid,user,ppid,etime,cmd
ps -o pid,user,etime,cmd -p $(ps -o ppid= -p <pid>)   # parent
ls -l /proc/<pid>/cwd                                  # working directory
```

User `hm` runs SkyRL training out of `/raid/hm/SkyRL/.venv` and `/raid/hm/bloomswe/skyrl/`. Their Ray workers spawn as **root** (some setuid/container thing) — don't try to `pkill -u $USER` and assume that's enough.

---

## 5. Launching and killing runs cleanly

**The single biggest mistake: `pkill -f 'ray::'`.** Ray names every actor `ray::WorkerDict` or `ray::FSDPPolicyWorkerBase` etc., and these names are *identical across all training jobs on the box*. A blanket `pkill -f` will catch your neighbor's run too. **Don't.**

### Right way to launch

Use `setsid` so the wrapper has its own process group (PGID), independent of the shell:

```bash
TS=$(date +%Y%m%d_%H%M%S)
LOG="$REPO_ROOT/logs/run_${TS}.log"
mkdir -p "$REPO_ROOT/logs"

CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 setsid nohup \
    bash my_train_wrapper.sh > "$LOG" 2>&1 < /dev/null &
disown
echo "PID=$!  PGID=$(ps -o pgid= -p $! | tr -d ' ')"
```

The shell prints both PID and PGID. Save them — that's how you'll kill cleanly later.

### Right way to kill

```bash
# All children of the wrapper share its PGID. One signal takes down the whole tree:
kill -- -<PGID>
sleep 3
# fallback if anything survived:
kill -9 <PID>
```

The `--` is required so kill doesn't interpret `-PGID` as a flag.

### What to verify after kill

```bash
# Are any of MY processes still alive?
pgrep -af -u $USER 'verl|train|ray::|VLLM' | grep -v 'bin/bash\|pgrep'

# Do my GPUs show >100 MB residual VRAM?
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

If memory is still allocated but no process shows up, the process is in zombie state (`Z` in `ps` STAT). It'll get reaped by init shortly. Don't panic.

---

## 6. Monitor / observe a running training

Don't try to `tail -F` a verl/EasyR1 log directly — tqdm progress bars produce hundreds of MB/hour of nearly-identical output. Filter to *events* (rewards, errors, checkpoints):

```bash
# example filter for verl/EasyR1 reward emissions + errors:
tail -F "$LOG" | python3 -u <<'PY'
import re, sys, time
reward = re.compile(r"reward metrics:\s*(\{.*?\})")
err    = re.compile(r"(Traceback|OutOfMemory|RuntimeError|FAILED|NCCL error|Killed|No space left)", re.I)
ckpt   = re.compile(r"(checkpoint saved|saving checkpoint|saved checkpoint to)", re.I)
for line in sys.stdin:
    s = line.rstrip()
    if err.search(s):    print("[ERR]", s[-400:], flush=True)
    elif ckpt.search(s): print("[CKPT]", s.strip()[-300:], flush=True)
    elif (m := reward.search(s)): print("[RWD]", m.group(1), flush=True)
PY
```

(The grounded-rl project has a ready-made version at `/raid/icy/iris/grounded-rl/demistify_project/_tail_monitor.py`.)

For a Claude Code session, use the `Monitor` tool with this filter as the script — events arrive as chat notifications.

---

## 7. Common pitfalls and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Run dies after hours, log mentions `No space left on device` on `/tmp/ray/...` | Default Ray TMPDIR fills `/dev/md0` | §3 — redirect TMPDIR before launch |
| `pkill -f 'ray::'` killed an unrelated job | Ray actor names collide across jobs | §5 — use `setsid` + PGID kill |
| `flash-attn` build fails with sm_90 + torch 2.6 | flash-attn 2.5.x doesn't support Hopper on torch 2.6 | Pin `flash-attn==2.7.4.post1` |
| `RuntimeError: Cannot modify locked TensorDict` in `union_tensor_dict` | tensordict ≥ 0.10 added strict locking; older verl/EasyR1 doesn't `unlock_()` | Either pin `tensordict==0.6.2`, or patch `union_tensor_dict` to `unlock_()` / `lock_()` around the in-place merge |
| `KeyError: 'visual.patch_embed.proj.weight'` loading Qwen2.5-VL | transformers version mismatch with checkpoint | `pip install transformers==4.51.0` |
| Conda refuses to create env (channel ToS) | First time on this box | `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/{main,r}` |
| GPU-utilization ≈ 0% but memory full | Run is between rollout phases; or it's stalled. Check log mtime — if not updating, it's hung. | If hung: PGID-kill and inspect the last log lines for traceback |

---

## 8. Quick-start template for a new project

```bash
PROJECT=mything
ENV_DIR=/raid/icy/iris/envs/$PROJECT
REPO=/raid/icy/iris/$PROJECT

# 1. Env
conda create -p $ENV_DIR python=3.10 pip -y
conda activate $ENV_DIR
pip install uv
# (install project deps here, see §2 for working version pairings)

# 2. Repo + data
git clone <repo-url> $REPO
mkdir -p $REPO/tmp/{ray,triton}  $REPO/logs

# 3. Launcher (write this to $REPO/run.sh)
cat > $REPO/run.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
source /raid/miniconda3/etc/profile.d/conda.sh
conda activate /raid/icy/iris/envs/mything

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export TMPDIR="$REPO/tmp"
export RAY_TMPDIR="$TMPDIR/ray"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export HF_HOME="${HF_HOME:-/raid/icy/iris/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before launch}"
mkdir -p "$RAY_TMPDIR" "$TRITON_CACHE_DIR"

# (your actual python/training command here)
python -m my.trainer "$@"
SH
chmod +x $REPO/run.sh

# 4. Launch (with PGID-clean detach)
TS=$(date +%Y%m%d_%H%M%S)
LOG=$REPO/logs/run_$TS.log
CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup bash $REPO/run.sh > $LOG 2>&1 < /dev/null &
disown
echo "PID=$!  PGID=$(ps -o pgid= -p $! | tr -d ' ')  log=$LOG"
```

---

## 9. Reference: working examples on this box

- `/raid/icy/iris/grounded-rl/demistify_project/cmds.sh` — full `vigorl_*` function library (env activate, train baseline/ablation, surgical kill, plot, monitor). Source to inherit all of this. Good template for a new project.
- `/raid/icy/iris/grounded-rl/demistify_project/_tail_monitor.py` — production-ready event filter for verl/EasyR1 logs.
- `/raid/icy/iris/grounded-rl/demistify_project/train_4gpu_no2nocrop.sh` — example wrapper showing TMPDIR redirection, env override, scratch-script pattern.
- `/raid/miniconda3/envs/verl/` — reference upstream-verl env (the modern combo: vllm 0.11, torch 2.8).
- `/raid/miniconda3/envs/vigorl` (= `/raid/icy/iris/envs/vigorl`) — reference EasyR1-vendored env (the legacy combo: vllm 0.8.3, torch 2.6).

---

## 10. Hosting inference/tool servers for testing (spagent lanes) — low-VRAM etiquette

Learned during the spagent tool-CI work (Aug 2026). Applies whenever you bring up
a model server (GroundingDINO, SAM2, depth, OneFormer, CountGD, LingBot-Map, …)
to smoke-test a tool PR while other users are training.

**Rules**
- **One backend at a time, one GPU.** Pick the card with the most free memory
  (`nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader`),
  pin with `CUDA_VISIBLE_DEVICES=<n>`, run the lane, **tear down immediately**.
  Never leave a server resident "for later".
- Typical footprints measured here: depth-anything vitb 2 GB · OneFormer swin-L 3 GB ·
  CountGD 1.5 GB · GroundingDINO SwinB 9 GB · SAM2.1-b 7 GB · Pi3/Pi3X 7–11 GB ·
  VGGT 13 GB · Sana 13 GB · Molmo2-4B 30 GB · LingBot-Map 9 GB peak. Budget ~10 GB
  for anything unknown; 40 GB free on a card is plenty.
- **Launch pattern** (the spagent servers are Flask; health endpoint is `/health`):
  ```bash
  CUDA_VISIBLE_DEVICES=3 nohup $PY spagent/external_experts/<X>/<x>_server.py --port <P> > <log> 2>&1 &
  until curl -s -m 2 http://localhost:<P>/health >/dev/null; do sleep 5; done   # in a background Bash, with a deadline
  ```
- **A healthy `/health` is not a working server.** Some servers answer health while
  every real call errors (wrong checkpoint name); and several spagent tools convert a
  *dead* backend into `success=True, 0 detections`. Always probe reachability
  out-of-band and measure the returned artifact, never the label.

**Teardown traps**
- `pkill -f "<x>_server.py"` **matches your own wrapping bash** (its command line
  contains the pattern) and kills the compound command mid-way → exit 144 and
  half-finished cleanup. Kill by PID instead, and find the PID via the port:
  `ss -tlnp | grep ":<P>"` → `users:(("python",pid=N,...))` → `kill N`.
- Servers that use `multiprocessing` leave a **child holding the socket** after the
  parent dies (depth server did). Re-check `ss -tln | grep ":<P>"` after every kill.
- `/health` responding on a port you killed = a *different* process (another
  project's server, e.g. REVPT's uvicorn on 9991/9993, or someone's vLLM on 30000).
  Check ownership with `ss -tlnp` + `ps -o user= -p <pid>` before assuming it's yours.
- Never `pkill -u <other-user>`; other users' processes (vLLM::Worker, jax, ray)
  need their owner or root. Enumerate with
  `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} ps -o user= -p {} | sort | uniq -c`.

**Where weights live (so you don't re-download)**
- Shared (root/spshare, rw): `/raid/william/project/spagent/checkpoints/{grounding_dino,sam2,depth_anything,pi3,pi3x}`
  — note the trap names: `sam2/sam2.1_b.pt` (not `sam2.1_hiera_base_plus.pt`) and
  directory `depth_anything/` (not `depth_anything_v2/`).
- HF cache (`~/.cache/huggingface/hub`): Molmo2-4B, Sana-Sprint-1.6B, VGGT-1B,
  PaddleOCR-VL-1.5, OneFormer ade20k swin-L, map-anything.
- Downloaded by iris into `/raid/icy/spagent-pr230/checkpoints/`: `flowseek/flowseek_M_TartanCT_TSKH.pth`
  (gdown `1gbZ-6NE3muAnGqvypiS2s_BADHrI4ySf`), `countgd/checkpoint_fsc147_best.pth`
  (gdown `1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI`), `lingbot_map/lingbot-map-long.pt` (HF `robbyant/lingbot-map`, 4.6 GB),
  plus `third_party/lingbot-map` clone. Orient-Anything-V2 HF Space clone: `/raid/icy/spagent/third_party/orient_anything_v2`.
- Pinned-torch repos (WildDet3D: `torch==2.5.1` + `vis4d_cuda_ops` from source) must go in
  their **own** conda env — never install into the shared `spagent_icy` env.

**Shell gotcha in this harness:** the Bash tool's cwd resets to `/raid/icy/spagent`
after every call — always `cd /raid/icy/spagent-pr230 &&` or use absolute paths.
