# Brief: paper agent (fork of the multimodal runner's session)

You are a fork of the session that ran the **image+text** side of the annotation-free
retrieval-RL project. You inherit that whole conversation, so this brief does not re-explain the
project. It states your task, the constraints, and the traps.

## Your task

1. **Read both runners' results.** Not just mine.
2. **Read the HTML summary.**
3. **Update almost everything** — paper.md, the ledger, the figure.
4. **Write the paper.**

## 1. Where the two runners' results are

**Runner A — multimodal, mine, local.** All on this box:
- `$MMRAG_DATA/results/*.retrieval.metrics.json` (primary product) and `*.vqa_top5.json`
  (secondary). `MMRAG_DATA=/raid/icy/mmrag_data`. 203 retrieval files as of the fork.
- `$MMRAG_DATA/runs/<cell>/{args.json,metrics.jsonl}` for what a cell actually ran.
- `mmrag/paper.md` (the deliverable so far), `mmrag/RL_REDESIGN.md` (the ledger, including nulls),
  `mmrag/SWEEP.md` (registered predictions), `mmrag/RUN_PLAN.md` (the four dynamic schemes mapped
  onto code).
- **`mmrag/seed_stats.py` is the single source for every ± in the paper.** Run it; do not hand-type
  an interval. It reads the raw per-seed json and forces one estimator per family across all three
  metrics. That constraint is what caught the defect described below, so do not loosen it.
- `mmrag/compare_ema.py` for matched-checkpoint contrasts and the floor screen.

**Runner B — text side, another machine.** `SendMessage` to **`tth100-vlm2vec`**. Use `ListAgents`
first to confirm the address. As of the fork that machine is **offline and a message to it is
queued**, so:
- Re-send if nothing has arrived. What is outstanding from them: per-seed values (or mean ± SD and
  n) for the two text rows in the figure, their noise floor and how they derived it, whether they
  pool between-seed SD across eval sets anywhere, and two specifics — the SD on their in-domain
  supervised arm that reads 10.6 points BELOW its own base, and whether their multi-source +1.3
  over supervised excludes zero with 4 seeds.
- Their numbers are **not on this filesystem**. Anything you write about the text side comes from
  their messages. If they have not answered, the figure and paper say "SD pending" and call nothing.

## 2. The HTML summary

- Published artifact: `https://claude.ai/code/artifact/603e26bc-e7b5-44f4-b46c-3a0dbec0d7db`
  (title "The Indirect Reward"). Read it with the Artifact tool, `action: "read"`, that URL.
  **Re-read before editing** — a publish to an artifact this session has not read is refused.
- A copy of the source it was published from is at **`$MMRAG_DATA/arch.html`**. The original lived
  in the parent session's scratchpad, which your fork does not share, so use this copy. If you
  republish, pass `url:` with the artifact URL so it updates in place instead of creating a second
  one.
- Also `mmrag/docs/vqa_retriever_loop.html` — the older loop page, which carries a reproduce command
  under every result row.

## 3. Where to write the paper

**NOT DECIDED YET. The user will tell you.** Do every bit of preparation, keep a working draft, and
**ask or wait for the destination before producing the final artifact.** Do not guess a path, do not
assume it is `mmrag/paper.md`, and do not publish a new artifact for it on your own initiative.

## 4. House rules — these are the user's, and they are strict

- **1 seed = recorded, not claimed.** Never quote a margin off one seed.
- **No margin without matched seeds AND a matched estimator.** One estimator per family, applied to
  all three metrics. The recent defect: the relevance-SFT row took accuracy from a 3-seed
  per-seed-best-checkpoint estimator and its two retrieval columns from an unrelated single-seed
  run, which manufactured a retrieval margin that does not exist.
- **Name the metric in every claim** — entR@5 / ansR@5 / acc. They routinely disagree in sign.
- **Trends by least squares with a 95% CI, never eyeballed.** With 3 points dof=1 and the CI is
  useless by construction; say so rather than quoting it.
- **Ledger everything, including nulls and retractions,** in `mmrag/RL_REDESIGN.md`.
- **Seed floors (InfoSeek, 9 families, dof=15): entR@5 2.03 · ansR@5 2.32 · acc 1.02.**
  E-VQA is a DIFFERENT measurement (n=2701/n_ans=2696 vs 3000/2450) with its own floors
  2.43 / 7.10 / 2.54. Do not pool across eval sets — that error is what inflated the old floors.
- **Pre-registered thresholds keep their registered values** even after a floor is re-derived.
  Report both readings. Moving a registered bound afterwards is the failure registration prevents.

## 5. Retractions — do not resurrect these

- "RL beats SFT on every column" — **accuracy only.** Retrieval is +0.93 entR@5 / +1.50 ansR@5,
  CIs span zero, seeds interleaved.
- The anchor-contraction / EMA-rescue story — a mid-training dip, not a contraction.
- The index-EMA headline — half of it was a cross-site artifact; the same-hardware control kills it.
- My mechanism for why diversity worsens degeneracy — its own prediction failed.
- "The site is uniformly ~3x slow" — two different measurements over-smoothed.

## 6. Hard constraints on this box

- **Do NOT launch training.** GPUs 5 and 7 carry the live jobs from the parent session; 4 and 6
  belong to the text-side collaborator by agreement. Treat `runs/` as read-only.
- **Never `pgrep -f <pattern> | kill`.** The pipeline's own command line matches the pattern and you
  will kill your own shell. It has happened four times in this project. Get explicit PIDs from a
  listing first.
- Run an eval only if genuinely needed, only through `$MMRAG_DATA/run_eval.sh <GPU> <NAME> <CKPT>`,
  and only after verifying real headroom on that card. A direct eval at `--bs 192` OOMs under the
  0.25 memory cap; `run_eval.sh` uses the path that works.
- `checkpoint-1500` and `checkpoint-3000` **never exist** — the loop is `range(max_steps)`, so the
  last save is at 1250 or 2750 and the final adapter is in the run root, not a checkpoint dir.
- In flight at fork time: the `mix2m-partialwb` ck750 eval (resolves a registered prediction),
  several `fullcorp-*` and `idx10k-*` cells, and `orchestrate.sh` auto-evaluating on cards 5 and 7.
  Check `$MMRAG_DATA/local_*.log` and `results/` for what has landed since.

## 7. Paper structure to keep

Main result (best no-gold RL vs SFT vs base) · Q1 does RL work better on multi-source training data ·
Q2 does RL keep learning from more data than SFT · Q3 how big / what pretraining of backbone is
needed · Q4 does indirect RL require downstream data quality · Q5 the grader defect · the
dynamic-schemes execution record (the user's four definitions of "dynamic", each answered with what
was actually executed and what it returned) · appendices.

Report what the numbers support, not what the story wants. The last pass of this work retracted a
headline claim; that is the standard.
