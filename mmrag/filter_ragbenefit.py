"""Q4 OFFLINE POOL FILTER: keep only training queries that retrieval can actually help.

This is the experiment Q4 asked for — "filter the training data to only those QA who benefit from
RAG, using the gold sample" — and it is NOT the same thing as `train_rl.py --reward_gate_noctx`.
That flag is an ONLINE advantage mask: it pays rollout and reader cost for every query and then
zeroes the gradient of ~60% of them, so the effective batch drops to ~0.4x nominal. This script
selects the pool ONCE, before training, so every query in every batch contributes and the compute
per useful query is 1.0x. The text-side project's +2.75 R@5 comes from this form, not the mask.

Decision per query, both passes with the SAME reader the reward uses (not the eval reader — the
filter must reflect what the reward model can do, or it selects for the wrong ability):

    gold pass   : reader sees the image + the gold passage   -> gold_ok
    no-ctx pass : reader sees the image only                 -> noctx_ok

    KEEP iff gold_ok AND NOT noctx_ok      ("RAG-BENEFITS": retrieval is the whole game)

The three discarded classes, from this project's Q4 taxonomy on the eval set:
    gold_ok AND noctx_ok          RAG-UNNEEDED  — reader already knows it, reward is saturated
    NOT gold_ok AND NOT noctx_ok  RAG-HOPELESS  — wrong even WITH the gold passage, dead gradient
    NOT gold_ok AND noctx_ok      RAG-HARMS     — context breaks a correct answer

Resumable by design: every decision is appended to <out>.decisions.jsonl as it is made, and a
restart skips qids already decided. A 4,000-query pass was lost earlier in this campaign to a
SIGKILL with no resume path; do not reintroduce that.

    python3 mmrag/filter_ragbenefit.py --n 40000 --out built/pool_train_q4filt.jsonl
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.image_store import open_stores  # noqa: E402
from mmrag.reader_vlm import LiveVLMReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="built/pool_train_1M.jsonl")
    ap.add_argument("--out", default="built/pool_train_q4filt.jsonl")
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data"))
    ap.add_argument("--dataset", default="infoseek", choices=["infoseek", "evqa"])
    # The REWARD reader, deliberately, not the 7B eval reader.
    ap.add_argument("--reader", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--n", type=int, default=40000, help="candidate rows to score (2 reader passes each)")
    ap.add_argument("--chunk", type=int, default=256, help="rows per scored+flushed chunk")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_ctx_chars", type=int, default=900)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    D = args.data_dir
    out_path = os.path.join(D, args.out)
    dec_path = out_path + ".decisions.jsonl"

    rows = [json.loads(l) for l in open(os.path.join(D, args.pool))]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    done = {}
    if os.path.exists(dec_path):
        for l in open(dec_path):
            d = json.loads(l)
            done[d["qid"]] = d
        print(f"resuming: {len(done)} decisions already on disk", flush=True)

    store = open_stores(D, args.dataset)
    todo = [r for r in rows if r["qid"] not in done and r["image_id"] in store
            and (r.get("pos") or [""])[0]]
    print(f"pool {len(rows)} sampled | {len(todo)} left to score | reader {args.reader}", flush=True)

    reader = None
    dec_f = open(dec_path, "a")
    t0 = time.time()
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i : i + args.chunk]
        if reader is None:      # built lazily so a fully-resumed run costs no GPU
            reader = LiveVLMReader(args.reader, device=args.device, batch_size=args.batch_size,
                                   max_ctx_chars=args.max_ctx_chars)
        imgs = [store.get(r["image_id"]) for r in chunk]
        ans = [r.get("answer_aliases") or r["answer"] for r in chunk]
        # `pos` is a LIST of passage strings (always length 1 in this pool, median 615 chars).
        # Passing the list straight through would stringify a Python list into the prompt -- the
        # same defect class as the grader bug that scored 21.9% of queries 0 by construction.
        # Index it exactly the way train_rl.py:404 does, so the filter sees what training sees.
        gold = [{"image": im, "question": r["question"],
                 "context": (r.get("pos") or [""])[0], "answer": a}
                for im, r, a in zip(imgs, chunk, ans)]
        noctx = [{"image": im, "question": r["question"], "context": "", "answer": a}
                 for im, r, a in zip(imgs, chunk, ans)]
        g = reader.score_judge(gold, n_rollouts=1, temperature=0.0)
        z = reader.score_judge(noctx, n_rollouts=1, temperature=0.0)
        for r, gi, zi in zip(chunk, g, z):
            d = {"qid": r["qid"], "gold_ok": int(gi > 0.5), "noctx_ok": int(zi > 0.5)}
            d["keep"] = int(d["gold_ok"] == 1 and d["noctx_ok"] == 0)
            done[r["qid"]] = d
            dec_f.write(json.dumps(d) + "\n")
        dec_f.flush()
        n = len(done)
        el = time.time() - t0
        kept = sum(d["keep"] for d in done.values())
        print(f"  {n}/{len(rows)} scored  kept {kept} ({100*kept/max(n,1):.1f}%)  "
              f"{el/max(i+len(chunk),1):.2f}s/row  eta {(len(todo)-i-len(chunk))*el/max(i+len(chunk),1)/3600:.1f}h",
              flush=True)
    dec_f.close()

    # taxonomy over everything decided, then write the kept pool in the SAME schema as the input
    tax = {"benefits": 0, "unneeded": 0, "hopeless": 0, "harms": 0}
    for d in done.values():
        if d["gold_ok"] and not d["noctx_ok"]:
            tax["benefits"] += 1
        elif d["gold_ok"] and d["noctx_ok"]:
            tax["unneeded"] += 1
        elif not d["gold_ok"] and not d["noctx_ok"]:
            tax["hopeless"] += 1
        else:
            tax["harms"] += 1
    n = sum(tax.values())
    keep_ids = {q for q, d in done.items() if d["keep"]}
    with open(out_path, "w") as f:
        for r in rows:
            if r["qid"] in keep_ids:
                f.write(json.dumps(r) + "\n")
    stats = {"pool": args.pool, "out": args.out, "reader": args.reader, "n_scored": n,
             "n_kept": len(keep_ids), "keep_rate": len(keep_ids) / max(n, 1),
             "taxonomy": tax, "taxonomy_share": {k: round(v / max(n, 1), 4) for k, v in tax.items()}}
    json.dump(stats, open(out_path + ".stats.json", "w"), indent=1)
    print(f"\nwrote {len(keep_ids)} rows -> {out_path}")
    print(f"taxonomy over {n} scored TRAINING rows: "
          + "  ".join(f"{k} {100*v/max(n,1):.1f}%" for k, v in tax.items()))
    print("compare the project's EVAL-set taxonomy (n=1500): benefits 32.3  unneeded 18.1  "
          "hopeless 48.5  harms 1.1 — a training-stream taxonomy is a DIFFERENT population and the "
          "two should not be reported as one measurement.")


main()
