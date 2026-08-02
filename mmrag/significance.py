"""Paired per-question significance tests over stored VQA predictions (v2 eval jsons).

For each (A, B) run pair: join per-question correctness by qid (same fixed 1500-q eval set),
report McNemar exact test (binomial on discordant pairs) and a paired bootstrap 95% CI of the
accuracy difference. Only v2 runs (with `predictions`) qualify.

    python3 mmrag/significance.py --pairs rl-j2e5-nogold-det:sft-lr2e4-h0 ...
    python3 mmrag/significance.py --headline   # canonical comparisons
"""

import argparse
import json
import math
import os
import random


def load_preds(results_dir, name):
    p = os.path.join(results_dir, f"{name}.vqa_top5.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if "predictions" not in d:
        return None
    return {q["qid"]: q["acc"] for q in d["predictions"]}


def mcnemar_p(b, c):
    """Exact two-sided binomial test on discordant counts b (A right, B wrong) vs c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap(diffs, iters=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    stats = []
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        stats.append(s)
    stats.sort()
    return stats[int(0.025 * iters)], stats[int(0.975 * iters)]


def compare(results_dir, a, b):
    pa, pb = load_preds(results_dir, a), load_preds(results_dir, b)
    if pa is None or pb is None:
        return f"{a} vs {b}: SKIP (missing v2 predictions: " \
               f"{'A ' if pa is None else ''}{'B' if pb is None else ''})"
    qids = sorted(set(pa) & set(pb))
    if len(qids) < 100:
        return f"{a} vs {b}: SKIP (only {len(qids)} shared qids)"
    da = [pa[q] for q in qids]
    db = [pb[q] for q in qids]
    acc_a, acc_b = sum(da) / len(qids), sum(db) / len(qids)
    bcnt = sum(1 for x, y in zip(da, db) if x > y)
    ccnt = sum(1 for x, y in zip(da, db) if y > x)
    p = mcnemar_p(bcnt, ccnt)
    lo, hi = paired_bootstrap([x - y for x, y in zip(da, db)])
    sig = "SIG" if p < 0.05 else "n.s."
    return (f"{a} ({acc_a:.4f}) vs {b} ({acc_b:.4f}): diff={acc_a-acc_b:+.4f} "
            f"[{lo:+.4f},{hi:+.4f}] McNemar p={p:.2g} ({bcnt}/{ccnt} discordant) n={len(qids)} {sig}")


HEADLINE = [
    ("rl-j2e5-nogold-det", "sft-lr2e4-h0"),
    ("rl-j2e5-nogold-det", "sft-lr1e4-h0"),
    ("rl-j2e5-nogold-det", "gme2b-zeroshot"),
    ("rl-j2e5-nogold-det-s2", "sft-lr2e4-h0"),
    ("rl-j2e5-nogold-det-s2", "gme2b-zeroshot"),
    ("rl-j2e5-gold-det-s1", "sft-lr2e4-h0"),
    ("rl-grpo-judge-lr2e5", "sft-lr2e4-h0"),
    ("rl7b-j-lr1e5", "sft7b-lr5e5-h0"),
    ("rl7b-j-nogold-lr1e5", "sft7b-lr5e5-h0"),
    ("rl7b-j-nogold-lr1e5", "gme7b-zeroshot"),
    ("rl-j2e5-nogold-det", "rl-j2e5-gold-det-s1"),
    ("rl-ansmatch-nogold", "rl-j2e5-nogold-det"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(
        os.environ.get("MMRAG_DATA", "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data"), "results"))
    ap.add_argument("--pairs", nargs="*", default=None, help="A:B pairs")
    ap.add_argument("--headline", action="store_true")
    args = ap.parse_args()
    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else HEADLINE
    for a, b in pairs:
        print(compare(args.results_dir, a, b))


if __name__ == "__main__":
    main()
