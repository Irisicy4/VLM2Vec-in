"""Collect every run's numbers into results_summary.json + RESULTS.md (idempotent — rerun after
each job wave). Sources:
  results/<name>.retrieval.metrics.json   entity/answer R@k + eval counts
  results/<name>.vqa_*.json               downstream VQA accuracy (+ per-split)
  runs/<name>/args.json                   training config axes (algo, reward, lr, gold-in-pool, ...)

Every RL row states explicitly whether the gold passage was FORCED into the candidate pools
(`gold_in_pool`) and the exact reward. All numbers are marked provisional pending the main-session
review (SFT tuning fairness, gold-seeding, advantage shapes, cover-EM, split/seed noise).

    python3 mmrag/collect_results.py --data_dir $MMRAG_DATA --repo_out mmrag
"""

import argparse
import glob
import json
import os
import re


def load_args(runs_dir, name):
    p = os.path.join(runs_dir, name, "args.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {}


def axes_for(name, targs):
    """Config axes from args.json, name-parsing as fallback for legacy runs."""
    ax = {}
    if targs:
        if "algo" in targs:  # RL run
            ax = {
                "arm": "rl",
                "algo": targs["algo"],
                "reward": targs["reward"],
                "reward_detail": ("teacher-forced mean logP(answer|image,question,passage), z-scored per pool"
                                  if targs["reward"] == "logit" else
                                  f"sampled answer accuracy (cover-EM vs aliases), {targs.get('reader_rollouts')} rollouts @T={targs.get('reader_temperature')}, z-scored per pool"),
                "reward_model": targs.get("reader_model"),
                "lr": targs["learning_rate"],
                "gold_in_pool": not targs.get("no_force_gold", False),
                "pool_sampling": targs.get("pool_sampling", False),
                "num_candidates": targs.get("num_candidates"),
                "max_steps": targs.get("max_steps"),
                "contrastive_coef": targs.get("contrastive_coef"),
                "max_train_rows": targs.get("max_train_rows") or "all(~41k)",
                "profile": targs.get("profile", "gme2b"),
                "seed": targs.get("seed", 0),
            }
        else:  # SFT run
            ax = {
                "arm": "sft",
                "lr": targs.get("learning_rate"),
                "num_hard_negs": targs.get("num_hard_negs"),
                "max_steps": targs.get("max_steps"),
                "batch_size": targs.get("batch_size"),
                "max_train_rows": targs.get("max_train_rows") or "all(~41k)",
                "profile": targs.get("profile", "gme2b"),
                "seed": targs.get("seed", 0),
            }
    elif "zeroshot" in name:
        ax = {"arm": "zero-shot", "profile": name.split("-")[0]}
    elif name.startswith("sft-"):  # legacy SFT without args.json
        m = re.match(r"sft-lr(\S+?)-h(\d+)", name)
        ax = {"arm": "sft", "lr": m.group(1) if m else "?",
              "num_hard_negs": int(m.group(2)) if m else "?", "profile": "gme2b",
              "note": "axes parsed from name (pre-args.json run)"}
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA", "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data"))
    ap.add_argument("--repo_out", default="mmrag")
    args = ap.parse_args()
    R = os.path.join(args.data_dir, "results")
    runs_dir = os.path.join(args.data_dir, "runs")

    rows = []
    for mf in sorted(glob.glob(os.path.join(R, "*.retrieval.metrics.json"))):
        name = os.path.basename(mf).replace(".retrieval.metrics.json", "")
        m = json.load(open(mf))
        row = {
            "name": name,
            "dataset": "evqa" if "evqa" in name else "infoseek",
            "axes": axes_for(name.replace("-evqa", ""), load_args(runs_dir, name)
                             or load_args(runs_dir, name.replace("-evqa", ""))),
            "n_eval_retrieval": m["n"],
            "n_eval_answer_level": m["n_ans"],
            "entity_R": {k: round(v, 4) for k, v in m["entity_recall"].items() if k in ("1", "5", 1, 5, "10", 10)},
            "answer_R": {k: round(v, 4) for k, v in m["answer_recall"].items() if k in ("1", "5", 1, 5, "10", 10)},
        }
        if row["axes"].get("arm") == "rl":
            row["label"] = "[gold]" if row["axes"].get("gold_in_pool") else "[no-gold]"
            row["supervision_note"] = (
                "gold passage force-inserted at pool slot 0 AND used as the InfoNCE-anchor positive "
                "-> same supervision level as SFT" if row["axes"].get("gold_in_pool")
                else "annotation-free: pure top-N pools, InfoNCE positive = self-labeled top-1")
        for vf in glob.glob(os.path.join(R, f"{name}.vqa_*.json")):
            v = json.load(open(vf))
            row.setdefault("vqa", {})[v["mode"]] = {
                "acc": round(v["acc"], 4), "f1": round(v["f1"], 4), "n": v["n"],
                "strict_em": round(v["strict_em"], 4) if "strict_em" in v else None,
                "metric_version": "v2-token-boundary" if "strict_em" in v else "v1-substring (over-credits; rescore pending)",
                "reader": os.path.basename(v["reader"]),
                "acc_by_split": {k: round(x, 4) for k, x in v.get("acc_by_split", {}).items()},
            }
        rows.append(row)

    summary = {
        "status": "PROVISIONAL — pending main-session review (SFT tuning fairness, gold-seeded "
                  "pools vs annotation-free claims, PPO/GRPO advantage shapes, cover-EM crediting, "
                  "split/seed noise)",
        "eval_protocol": {
            "corpus": "corpus_small.jsonl (422,378 passages / 26,454 articles: all gold entities of "
                      "train+test + 20k random distractor articles) unless name says 'full'",
            "retrieval_eval": "5000 (zero-shot) or 3000 (trained) test queries, seed-0 sample of 71,335",
            "vqa_eval": "1500-query seed-0 subsample; reader Qwen2.5-VL-7B; acc = cover-EM vs "
                        "official InfoSeek answer_eval alias list (E-VQA: '|'-split answers)",
            "gold_definition": "entity-level = any passage of the question's Wikipedia article; "
                               "answer-level = passages containing an answer alias verbatim",
            "exposure_asymmetry": "RL runs visit ~2,000 queries (500 steps x batch 4; ~16k "
                                  "(query,passage) reward calls); SFT visits ~19,200 (600 x 32). "
                                  "SFT sees ~10x more queries — asymmetry favors SFT on exposure, "
                                  "RL on per-query signal; m6d data-scaling cells control this.",
            "vqa_metric": "cover-EM v2: token-boundary containment of a gold alias (v1 raw substring "
                          "over-credited, e.g. 'us' in 'museum'); strict-EM reported alongside; "
                          "per-question predictions stored in vqa jsons for rescoring",
        },
        "runs": rows,
    }
    out_json = os.path.join(args.repo_out, "results_summary.json")
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"wrote {out_json} ({len(rows)} runs)")

    # ---- scaling section (RL vs SFT vs feedback-data size; zero-shot = 0 point) ----
    def find(name):
        for r in rows:
            if r["name"] == name:
                return r
        return None

    scale_points = []
    zs = find("gme2b-zeroshot")
    if zs:
        scale_points.append(("0 (zero-shot)", "—", zs))
    for arm, pat in (("rl", "rl-j2e5-rows{n}"), ("sft", "sft-lr1e4-h0-rows{n}")):
        for n, label in ((2000, "2k"), (8000, "8k")):
            r = find(pat.format(n=label))
            if r:
                scale_points.append((label, arm, r))
    for arm, full in (("rl", "rl-grpo-judge-lr2e5"), ("sft", "sft-lr1e4-h0")):
        r = find(full)
        if r:
            scale_points.append(("41k (full)", arm, r))

    # ---- markdown report ----
    md = ["# mm-RAG results (PROVISIONAL — under review, do not cite yet)", "",
          summary["status"], "",
          "Reward reader is IN-PROCESS (no server): Qwen2.5-VL-3B-Instruct on the trainer GPU.",
          "Each RL row lists `gold_in_pool` (was the gold passage force-inserted at slot 0 of every "
          "candidate pool — i.e. the run is NOT annotation-free) and the exact reward.", "",
          "| run | arm | label | axes | entity R@1/R@5 | answer R@1/R@5 | VQA top5 acc / strictEM (n) |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["dataset"], x["axes"].get("arm", ""), x["name"])):
        a = r["axes"]
        ax_s = ", ".join(f"{k}={v}" for k, v in a.items() if k not in ("arm", "reward_detail", "reward_model"))
        er = r["entity_R"]; ar = r["answer_R"]
        v = r.get("vqa", {}).get("top5")
        vs = (f"{v['acc']} / {v.get('strict_em', '?')} ({v['n']})"
              + ("" if v.get("strict_em") is not None else " ⚠v1-metric") if v else "—")
        md.append(f"| {r['name']} | {a.get('arm','?')} | {r.get('label','')} | {ax_s} | "
                  f"{er.get('1','?')}/{er.get('5','?')} | {ar.get('1','?')}/{ar.get('5','?')} | {vs} |")
    md += ["", "## Feedback-data scaling (required deliverable; auto-fills as m6d lands)", "",
           "| feedback size | arm | entity R@5 | answer R@5 | VQA top5 acc | n_retr / n_vqa |",
           "|---|---|---|---|---|---|"]
    for label, arm, r in scale_points:
        v = r.get("vqa", {}).get("top5")
        md.append(f"| {label} | {arm} | {r['entity_R'].get('5','?')} | {r['answer_R'].get('5','?')} | "
                  f"{v['acc'] if v else '—'} | {r['n_eval_retrieval']} / {v['n'] if v else '—'} |")
    md += ["", "RL runs above with `gold_in_pool=True` use the gold-evidence passage as a forced pool "
           "member (grounding); the annotation-free variants are the `nogold` runs of m6b.", ""]
    out_md = os.path.join(args.repo_out, "RESULTS.md")
    open(out_md, "w").write("\n".join(md))
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
