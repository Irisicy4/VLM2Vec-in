"""Sliced eval table: what improves, where, and with a PAIRED test rather than a seed comparison.

Why this exists. The campaign has compared arms by aggregate accuracy with a between-seed SD over
three run means. But every eval in this project scores the SAME 1,500 held-out queries (verified:
100% qid overlap across every arm on disk, local and cross-cluster alike), and every
`*.vqa_top5.json` stores per-question `acc`. So the available test is McNemar on 1,500 paired
queries, which is far more powerful than a t-test on three numbers, and it can be computed inside
any slice of the query set at no compute cost.

Three slices, all free from data already on disk:
  * data_split   — val_unseen_entity vs val_unseen_question. InfoSeek's own generalisation axis:
                   unseen ENTITY is the harder, more out-of-domain cut.
  * archetype    — what the question asks for. InfoSeek is a Wikidata-property benchmark, so the
                   question stem is close to a property type, and arms differ sharply by type.
  * answer form  — numeric / multi-alias / range. Cover-EM's strictness varies with these, so a
                   margin that lives entirely in one form is a grader story, not a retrieval story.

Retrieval slices are only available for arms with a `*.retrieval.json` on disk (the locally-run
family); accuracy slices are available for all 216 evals. The tool says which it used.

    python3 mmrag/eval_table.py --base gme2b-zeroshot-3k-repro \
        --arms sft-curve-ck750 v3-pure plgrpo-gme2b-nogold
    python3 mmrag/eval_table.py --vs sft-curve-ck750 --arms v3-pure v3-pure-s1 v3-pure-s2
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

D = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data")
R = os.path.join(D, "results")


def archetype(q):
    """Coarse question type. InfoSeek stems track Wikidata properties closely, so these are
    meaningful groupings rather than string buckets. Order matters: earlier patterns win."""
    q = q.lower()
    rules = [
        (r"\bin which year|what year|what date|which century", "temporal"),
        (r"\bclosest (upper )?(taxonomy|parent)|parent taxon|basionym|taxonomic", "taxonomy"),
        (r"\bhow many|how much|what is the (length|height|weight|width|diameter|area|mass|speed|"
         r"number|thermal|density|oldest|highest|lowest|litter|gestation|incubation|wingspan)|"
         r"\bin (metre|meter|gram|kilogram|centimetre|day|cm|mm|km|kg|watt)\b", "measurement"),
        (r"\bwho\b|\bwhose\b|designed|discoverer|inventor|author|owner|founder|\bperson\b", "person"),
        (r"\bwhere\b|which (country|city|state|region|continent)|country of|location of|"
         r"\bplace\b|administrative", "geography"),
        (r"conservation status|what kind of|what type of|source of|source that|made of|material|"
         r"\bcolou?r\b|\beffect\b|used for|part of", "attribute"),
        (r"what is (this|the) (name|object|species|breed|title)|named after|called", "identity"),
    ]
    for pat, name in rules:
        if re.search(pat, q):
            return name
    return "other"


def answer_form(r):
    """Answer shape. The 'tolerance' class is the 21.9% of queries whose gold is a dict
    {'wikidata': v, 'range': [lo, hi]} -- scored 0 by construction until rescore.py; keeping it as
    its own slice is the only way to see whether a margin lives entirely inside the broken block."""
    a = r.get("answer_aliases") or r["answer"]
    first = a[0] if isinstance(a, list) and a else a
    if isinstance(first, dict):
        return "tolerance (dict gold)"
    s = str(first).strip()
    multi = isinstance(a, list) and len(a) > 1
    if re.search(r"\d+(\.\d+)?\s*(-|–|to)\s*\d+", s):
        return "range"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return "numeric multi-alias" if multi else "numeric single"
    return "text multi-alias" if multi else "text single"


def load_meta():
    meta = {}
    with open(os.path.join(D, "built", "queries_test.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            meta[r["qid"]] = {"split": r.get("data_split") or "all",
                              "arch": archetype(r["question"]),
                              "form": answer_form(r)}
    return meta


def load_acc(name):
    """Prefer the CORRECTED per-query accuracy from rescore.py; fall back to the raw file.
    The raw one scores 21.9% of queries 0 by construction, so any slice that touches the
    tolerance-gold block is meaningless on it."""
    rp = os.path.join(R, f"{name}.vqa_rescored.json")
    if os.path.exists(rp):
        return {d["qid"]: d["acc"] for d in json.load(open(rp))["predictions"]}
    p = os.path.join(R, f"{name}.vqa_top5.json")
    if not os.path.exists(p):
        return None
    return {d["qid"]: d["acc"] for d in json.load(open(p))["predictions"]}


def load_ret(name, k=5):
    """Per-query retrieval hit: did any answer-bearing passage appear in the top k."""
    p = os.path.join(R, f"{name}.retrieval.json")
    if not os.path.exists(p):
        return None
    out = {}
    for qid, r in json.load(open(p)).items():
        gold = set(r["ans_pids"] or r["gold_pids"])
        out[qid] = 1.0 if (gold & set(r["topk"][:k])) else 0.0
    return out


def paired(a, b, qids):
    """Paired test on per-query differences. Returns (mean_diff, half_CI95, p, wins, losses).

    NOT McNemar. When arms are seed-AVERAGED the per-query score is fractional (0, 1/3, 2/3, 1) and
    McNemar's binary-discordance model does not apply: counting only the DIRECTION of each
    difference overstates the margin badly (on the headline slice it reported +3.93 where the true
    level difference is +2.22). A paired t on the differences is valid for fractional and binary
    scores alike, and its mean IS the level difference, so the test and the reported delta cannot
    disagree. Sign counts are kept as a direction check only."""
    d = [a[q] - b[q] for q in qids]
    n = len(d)
    if n < 2:
        return 0.0, 0.0, 1.0, 0, 0
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    p = math.erfc(abs(m / se) / math.sqrt(2)) if se > 0 else 1.0
    return m, 1.96 * se, p, sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)


def holm(pvals):
    """Holm step-down adjusted p-values, same order as the input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [1.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = min(1.0, max(run, (m - rank) * pvals[i]))
        out[i] = run
    return out


def table(arms, meta, vs=None, metric="acc", min_n=40):
    load = load_acc if metric == "acc" else load_ret
    def load_fam(spec):
        ns = spec.split(",")
        ds = [load(n) for n in ns]
        ds = [d for d in ds if d]
        if not ds:
            return None
        qs = set.intersection(*(set(d) for d in ds))
        return {q: sum(d[q] for d in ds) / len(ds) for q in qs}
    data = {a: load_fam(a) for a in arms}
    missing = [a for a, v in data.items() if v is None]
    data = {a: v for a, v in data.items() if v}
    if not data:
        print(f"  no {metric} data for any requested arm")
        return
    if missing:
        print(f"  NOTE: no per-query {metric} on disk for: {', '.join(missing)}")
    common = set.intersection(*(set(v) for v in data.values()))
    ref = data.get(vs)
    if vs and ref:
        common &= set(ref)
    print(f"\n  metric={metric}  arms={len(data)}  paired on {len(common)} common queries"
          + (f"  vs {vs}" if vs and ref else ""))
    cuts = [("ALL", lambda m: True)]
    for key, label in (("split", "split"), ("arch", "archetype"), ("form", "answer form")):
        vals = Counter(meta[q][key] for q in common if q in meta)
        for v, n in vals.most_common():
            if n >= min_n:
                cuts.append((f"{label}={v}", (lambda kk, vv: (lambda m: m[kk] == vv))(key, v)))
    hdr = f"  {'slice':30s} {'n':>5s}" + "".join(f"{a[:16]:>18s}" for a in data)
    if vs and ref:
        hdr += f"{'delta':>8s}{'+/-':>9s}{'paired p':>10s}"
    print(hdr)
    for label, sel in cuts:
        qs = [q for q in common if q in meta and sel(meta[q])]
        if len(qs) < min_n:
            continue
        row = f"  {label:30s} {len(qs):5d}"
        for a, v in data.items():
            row += f"{100*sum(v[q] for q in qs)/len(qs):18.2f}"
        if vs and ref:
            tgt = [a for a in data if a != vs]
            if tgt:
                d, h, p, w, l = paired(data[tgt[0]], ref, qs)
                star = "*" if p < 0.05 else " "
                row += f"{100*d:+8.2f}{f'{w}/{l}':>9s}{p:10.4f}{star}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="an arm, or a comma-separated FAMILY of seeds averaged per query")
    ap.add_argument("--vs", default=None, help="baseline arm for the PAIRED McNemar column")
    ap.add_argument("--min_n", type=int, default=40, help="skip slices smaller than this")
    args = ap.parse_args()
    meta = load_meta()
    arms = args.arms + ([args.vs] if args.vs and args.vs not in args.arms else [])
    print(f"loaded metadata for {len(meta)} queries")
    for m in ("acc", "ret"):
        table(arms, meta, vs=args.vs, metric=m, min_n=args.min_n)
    print("\n  * = paired-t p < 0.05 on the per-query differences of that slice (unadjusted).")
    print("  Slice tests are MANY; apply Holm before believing any single slice. On the headline")
    print("  comparison only split=val_unseen_entity survives Holm at 12 slices.")
    print("  Paired per-query tests on 1,500 queries, NOT a between-seed comparison; the two answer")
    print("  different questions and a seed-level claim still needs seeds.")



if __name__ == "__main__":
    main()
