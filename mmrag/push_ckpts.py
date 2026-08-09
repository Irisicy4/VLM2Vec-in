"""Push every completed mm-RAG adapter into ONE HF repo, one subfolder per run, named from the
run's own args.json so the folder name IS the training spec.

Folder name:
    <profile>_<algo>-<reward>_<gold|nogold>_lr<lr>_bs<bs>_N<cand>_cc<coef>_steps<n>_rows<n>_seed<n>

Each subfolder gets the LoRA adapter + tokenizer/processor files + args.json + metrics.jsonl and a
README carrying that run's eval numbers. The repo root README is a table over all runs.

    python3 mmrag/push_ckpts.py --repo Icey444/mmrag-ckpts [--dry-run] [--runs A B ...]
"""
import argparse
import json
import os

D = os.environ.get("MMRAG_DATA", "/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data")
UPLOAD = ["adapter_config.json", "adapter_model.safetensors", "args.json", "metrics.jsonl",
          "added_tokens.json", "chat_template.jinja", "merges.txt", "preprocessor_config.json",
          "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json"]


def fmt_lr(x):
    return f"{float(x):g}".replace("-0", "-")


def spec_name(a):
    rows = a.get("max_train_rows") or 0
    rows_s = "all" if not rows else (f"{rows // 1000}k" if rows % 1000 == 0 else str(rows))
    return "_".join([
        a.get("profile", "gme2b"),
        f"{a.get('algo','grpo')}-{a.get('reward','judge')}",
        "nogold" if a.get("no_force_gold") else "gold",
        f"lr{fmt_lr(a.get('learning_rate', 2e-5))}",
        f"bs{a.get('batch_size', 4)}",
        f"N{a.get('num_candidates', 8)}",
        f"cc{a.get('contrastive_coef', 0.3)}",
        f"steps{a.get('max_steps', 500)}",
        f"rows{rows_s}",
        f"seed{a.get('seed', 0)}",
    ])


def metrics_for(run):
    out = {}
    p = f"{D}/results/{run}.vqa_top5.json"
    if os.path.exists(p):
        v = json.load(open(p))
        out.update(acc=v["acc"], strict_em=v["strict_em"], f1=v["f1"], n=v["n"])
    p = f"{D}/results/{run}.retrieval.metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))
        out.update(entity_r5=m["entity_recall"]["5"], entity_r1=m["entity_recall"]["1"],
                   answer_r5=m["answer_recall"]["5"], n_retrieval=m["n"])
    return out


def card(run, a, met, folder):
    pool = "pool_train_big.jsonl (250k rows, <=200 q/entity)" if "big" in (a.get("pool") or "") \
        else "pool_train.jsonl (45,248 rows, <=10 q/entity)"
    rows = a.get("max_train_rows") or "all"
    m = "\n".join(f"| {k} | {v} |" for k, v in met.items()) or "| — | not yet evaluated |"
    return f"""---
base_model: Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
library_name: peft
tags: [retrieval, multimodal-rag, grpo, infoseek, lora]
---

# {folder}

LoRA adapter for **GME-Qwen2-VL-2B**, trained by RL on a frozen VLM reader's answer success —
no relevance labels. Part of the mm-RAG data-scaling ladder (`{run}`).

## Training spec

| field | value |
|---|---|
| algorithm | {a.get('algo')} |
| reward | {a.get('reward')} — sampled answer accuracy from {a.get('reader_model')}, {a.get('reader_rollouts')} rollouts @ T={a.get('reader_temperature')}, z-scored per pool |
| grounding | {'**[no-gold]** annotation-free — pure top-N policy pools, InfoNCE positive = self-labeled top-1' if a.get('no_force_gold') else '[gold] gold evidence forced into pool'} |
| pool sampling | {'stochastic' if a.get('pool_sampling') else 'deterministic top-N'} |
| learning rate | {a.get('learning_rate')} |
| batch size | {a.get('batch_size')} |
| candidates per query (N) | {a.get('num_candidates')} |
| policy temperature | {a.get('temperature')} |
| InfoNCE anchor coef | {a.get('contrastive_coef')} |
| max steps | {a.get('max_steps')} |
| **train rows** | **{rows}** |
| train pool | {pool} |
| LoRA r / alpha | {a.get('lora_r')} / {a.get('lora_alpha')} |
| index refresh | every {a.get('refresh_steps')} steps, {a.get('n_distractor_articles')} distractor articles |
| seed | {a.get('seed')} |

## Eval (InfoSeek, corpus_small 422,378 passages)

L1 = 3,000 seed-0 test queries. L2 = 1,500-query subsample, top-5 context, reader
Qwen2.5-VL-7B-Instruct, v2 token-boundary cover-EM.

| metric | value |
|---|---|
{m}

Reference on the same protocol: GME-2B zero-shot **0.3080**, relevance-SFT **0.3100**.

## Use

```python
from peft import PeftModel
# load the gme2b base through mmrag/encoder.py, then:
#   enc = load_encoder("gme2b", checkpoint_path="<this folder>")
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Icey444/mmrag-ckpts")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = args.runs or sorted(
        r for r in os.listdir(f"{D}/runs")
        if os.path.exists(f"{D}/runs/{r}/adapter_model.safetensors")
        and os.path.exists(f"{D}/runs/{r}/args.json"))

    rows = []
    for run in runs:
        a = json.load(open(f"{D}/runs/{run}/args.json"))
        rows.append((run, a, spec_name(a), metrics_for(run)))

    print(f"{len(rows)} checkpoint(s) -> {args.repo}\n")
    for run, a, folder, met in rows:
        acc = met.get("acc")
        print(f"  {run:24s} -> {folder}" + (f"   acc={acc:.4f}" if acc else "   (unevaluated)"))
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)

    for run, a, folder, met in rows:
        src = f"{D}/runs/{run}"
        tmp = f"/tmp/ckpt_readme_{folder}.md"
        open(tmp, "w").write(card(run, a, met, folder))
        api.upload_file(path_or_fileobj=tmp, path_in_repo=f"{folder}/README.md",
                        repo_id=args.repo, repo_type="model")
        os.unlink(tmp)
        for f in UPLOAD:
            p = os.path.join(src, f)
            if os.path.exists(p):
                api.upload_file(path_or_fileobj=p, path_in_repo=f"{folder}/{f}",
                                repo_id=args.repo, repo_type="model")
        print(f"  uploaded {folder}", flush=True)

    # ---- root card
    hdr = ("| folder (= training spec) | train rows | steps | seed | VQA acc | strict EM | "
           "entity R@5 | ans R@5 |\n|---|---|---|---|---|---|---|---|\n")
    body = ""
    for run, a, folder, met in sorted(rows, key=lambda x: (x[1].get("max_train_rows") or 0)):
        g = lambda k: f"{met[k]:.4f}" if k in met else "—"  # noqa: E731
        body += (f"| [`{folder}`]({folder}) | {a.get('max_train_rows') or 'all'} | "
                 f"{a.get('max_steps')} | {a.get('seed')} | {g('acc')} | {g('strict_em')} | "
                 f"{g('entity_r5')} | {g('answer_r5')} |\n")
    root = f"""---
library_name: peft
tags: [retrieval, multimodal-rag, grpo, infoseek, lora, model-collection]
base_model: Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
---

# mm-RAG checkpoints — indirect-feedback retriever RL

LoRA adapters for **GME-Qwen2-VL-2B** trained by RL on a frozen VLM reader's answer success,
with **no relevance labels** ([no-gold]: pure top-N policy pools, the only supervision being the
gold *answer* inside the reward). One subfolder per run; the folder name is the full training spec.

Naming: `<profile>_<algo>-<reward>_<gold|nogold>_lr_bs_N_cc_steps_rows_seed`

## Data-scaling ladder

Fixed compute (500 steps x batch 4 => ~2,000 queries visited) with the training **pool** size
varied 12.5k -> 200k. Evaluated on InfoSeek: L1 over 422,378 passages (3,000 queries),
L2 top-5 with a Qwen2.5-VL-7B reader on 1,500 queries, v2 token-boundary cover-EM.

{hdr}{body}
**Reference points on the identical protocol** — GME-2B zero-shot **0.3080**, tuned
relevance-SFT **0.3100**, the released 45k no-gold headline checkpoint **0.3393**.

**Finding:** 16x more training data buys nothing. Four of five cells sit in 0.3433-0.3487; the
whole ladder spans 0.0200 while the gap from SFT to RL is +0.029-0.039. At 500 steps the run only
ever visits ~2,000 queries, so pool size past ~10k is never sampled. Caveat: one seed per cell,
and cell-to-cell spread is far larger than the headline config's 5-seed sd (~0.0015) — replication
and pool-composition controls were still running when this was pushed.

Full write-up: [Irisicy4/VLM2Vec-in @ mmrag](https://github.com/Irisicy4/VLM2Vec-in/tree/mmrag/mmrag)
"""
    tmp = "/tmp/mmrag_ckpts_root.md"
    open(tmp, "w").write(root)
    api.upload_file(path_or_fileobj=tmp, path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    os.unlink(tmp)
    print(f"\nhttps://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
