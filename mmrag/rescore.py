"""Re-score every stored eval with a tolerance-aware grader. No regeneration required.

THE DEFECT. 21.9% of InfoSeek's held-out queries carry a numeric gold with an explicit tolerance,
stored as a dict: {'wikidata': 176.0, 'range': [158.4, 193.6]}. `answer_correct` does
`_normalize(str(g))`, so the gold becomes the literal text "{'wikidata': 176.0, 'range': ...}" and
is then looked for as a token sequence inside the reader's answer. No answer can ever match. Worked
example from `v3-pure`, scored 0.0:

    gold  {'wikidata': 176.0, 'range': [158.4, 193.60000000000002]}
    pred  "176 metres"                                              <- exactly right, scored ZERO

So roughly a fifth of every accuracy number in this project is zero by construction, and correct
answers on that fifth are counted wrong. Every `*.vqa_top5.json` stores its per-question predictions
(a deliberate choice recorded in eval_vqa.py: "so metric fixes can rescore without regeneration"),
which means the fix is arithmetic, not compute.

THE GRADER. String golds keep cover-EM exactly as before, so nothing already correct changes.
Dict golds are scored numerically: pull every number out of the prediction and accept if one lands
inside the tolerance range. Two variants are always reported side by side because the choice is a
judgement call and the reader should see its size:

    any   — accept if ANY number in the prediction falls in range. Credits "50-105 g" against
            [88.02, 107.58] because 105 is inside. Lenient on hedged range answers.
    first — accept only if the FIRST number falls in range. Stricter; "10 to 12 years" against
            [14.58, 17.82] fails under both, "176 metres" passes under both.

Neither variant can lower a score: a dict-gold query scored 0 by construction can only stay 0 or
become 1. So the corrected numbers are upper-adjusted for every arm alike, and the interesting
question is whether the MARGINS between arms move, not whether the levels do.

    python3 mmrag/rescore.py --arms v3-pure sft-curve-ck750 gme2b-zeroshot-3k-repro
    python3 mmrag/rescore.py --all --write        # rewrite corrected acc into *.vqa_rescored.json
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.reader_vlm import answer_correct  # noqa: E402  (string golds keep the original grader)

D = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data")
R = os.path.join(D, "results")
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def numbers(s):
    return [float(x) for x in NUM.findall(s.replace(",", ""))]


def score_dict_gold(pred, golds, mode="any"):
    """golds is a list of {'wikidata': v, 'range': [lo, hi]} dicts."""
    vals = numbers(pred)
    if not vals:
        return 0.0
    cand = vals if mode == "any" else vals[:1]
    for g in golds:
        rng = g.get("range")
        lo, hi = (min(rng), max(rng)) if rng else (g.get("wikidata"), g.get("wikidata"))
        if lo is None:
            continue
        if any(lo <= v <= hi for v in cand):
            return 1.0
    return 0.0


def load_gold():
    gold = {}
    with open(os.path.join(D, "built", "queries_test.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            gold[r["qid"]] = r.get("answer_aliases") or r["answer"]
    return gold


def rescore(name, gold, write=False):
    p = os.path.join(R, f"{name}.vqa_top5.json")
    if not os.path.exists(p):
        return None
    doc = json.load(open(p))
    preds = doc["predictions"]
    out = {"n": len(preds), "orig": 0.0, "any": 0.0, "first": 0.0,
           "n_dict": 0, "fixed_any": 0, "fixed_first": 0, "dict_orig_right": 0}
    per_q = []
    for d in preds:
        g = gold.get(d["qid"])
        o = d["acc"]
        out["orig"] += o
        if isinstance(g, list) and g and isinstance(g[0], dict):
            out["n_dict"] += 1
            if o > 0:
                out["dict_orig_right"] += 1      # should be impossible; counted to prove it
            a = score_dict_gold(d["pred"], g, "any")
            fst = score_dict_gold(d["pred"], g, "first")
            out["any"] += a
            out["first"] += fst
            out["fixed_any"] += int(a > o)
            out["fixed_first"] += int(fst > o)
            per_q.append({"qid": d["qid"], "acc": a, "acc_first": fst, "orig": o})
        else:
            s = answer_correct(d["pred"], g) if g is not None else o
            out["any"] += s
            out["first"] += s
            per_q.append({"qid": d["qid"], "acc": s, "acc_first": s, "orig": o})
    for k in ("orig", "any", "first"):
        out[k] = 100.0 * out[k] / out["n"]
    if write:
        json.dump({"name": name, "acc_rescored_any": out["any"] / 100,
                   "acc_rescored_first": out["first"] / 100, "acc_orig": out["orig"] / 100,
                   "n_dict_gold": out["n_dict"], "predictions": per_q},
                  open(os.path.join(R, f"{name}.vqa_rescored.json"), "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    gold = load_gold()
    arms = args.arms
    if args.all:
        arms = sorted({f.replace(".vqa_top5.json", "") for f in os.listdir(R)
                       if f.endswith(".vqa_top5.json")})
    print(f"  gold loaded for {len(gold)} queries; "
          f"{sum(1 for g in gold.values() if isinstance(g, list) and g and isinstance(g[0], dict))}"
          f" carry a dict (tolerance) gold\n")
    print(f"  {'arm':32s} {'n':>5s} {'dict':>5s} {'orig':>7s} {'any':>7s} {'first':>7s}"
          f" {'+any':>6s} {'+1st':>6s} {'fixed':>6s}")
    rows = []
    for a in arms:
        r = rescore(a, gold, write=args.write)
        if not r:
            continue
        rows.append((a, r))
        print(f"  {a:32s} {r['n']:5d} {r['n_dict']:5d} {r['orig']:7.2f} {r['any']:7.2f}"
              f" {r['first']:7.2f} {r['any']-r['orig']:+6.2f} {r['first']-r['orig']:+6.2f}"
              f" {r['fixed_any']:6d}")
    bad = [a for a, r in rows if r["dict_orig_right"]]
    print(f"\n  dict-gold queries the ORIGINAL grader scored correct: "
          f"{sum(r['dict_orig_right'] for _, r in rows)} across {len(rows)} arms"
          + ("  (expected 0 — the stringified dict can never match)" if not bad else "  <- UNEXPECTED"))
    print("  Neither variant can lower a score, so every arm moves up; the question is whether the")
    print("  MARGINS move. Compare deltas across arms, not levels.")


main()
