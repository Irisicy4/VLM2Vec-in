#!/usr/bin/env python3
"""Per-seed SD / 95% CI for every multi-seed family, and arm-vs-arm contrasts.

Why this file exists: the +/- numbers in paper.md were previously typed in by hand,
one column at a time, and that let a row mix estimators across its own columns (the
relevance-SFT row took accuracy from a 3-seed per-seed-best-checkpoint estimator and
its two retrieval columns from a different single-seed run). Every interval in the
paper is now recomputed here from the raw per-seed json, and a family declares its
estimator once for all three metrics.

Rules encoded:
  * n=2 prints the range, not a CI: at dof=1 the t multiplier is 12.71 and the
    interval is wider than the metric's own scale, so it states nothing.
  * families are pooled into a noise floor only within one eval set. E-VQA (n=2701,
    n_ans=2696) is a different measurement from InfoSeek (n=3000, n_ans=2450) and
    gets its own floor.
  * no seed is dropped from a mean. If a seed is on disk it is in the family.
"""
import json, os, math

R = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data") + "/results"
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
MET = ("entR@5", "ansR@5", "acc")

# eval set -> family -> (per-seed stems, estimator note)
FAM = {
 "InfoSeek": {
  "base (frozen, zero-shot)":   (["gme2b-zeroshot-3k-repro"], "no training"),
  "relevance-SFT, per-seed best ck": (["sft-curve-ck750", "sft-curve-lr1e4-s1-ck1050",
                                      "sft-curve-lr1e4-s2-ck1050"], "acc-argmax over ck450-1200 within each seed"),
  "v3-pure (headline)":         (["v3-pure", "v3-pure-s1", "v3-pure-s2"], "endpoint, step 500"),
  "v2 +anchor":                 (["plgrpo-gme2b-nogold", "plgrpo-gme2b-nogold-s1",
                                  "plgrpo-s2", "plgrpo-s3"], "endpoint; all 4 seeds on disk"),
  "scale-rows25k":              (["scale-rows25k", "scale-rows25k-s1", "scale-rows25k-s2"], "endpoint"),
  "v3-rows200k":                (["v3-rows200k", "v3-rows200k-s1", "v3-rows200k-s2"], "endpoint"),
  "plgrpo-t005":                (["plgrpo-t005", "plgrpo-t005-s1"], "endpoint"),
  "v3-rows1M @2k draws":        (["v3-rows1M", "v3-rows1M-s1"], "endpoint"),
  "mix2m-v3pure @ck750":        (["mix2m-v3pure-ck750", "mix2m-v3pure-s1-ck750"], "ck750"),
  "mix2m-v3pure-ema @ck750":    (["mix2m-v3pure-ema-ck750", "mix2m-v3pure-ema-s1-ck750"], "ck750"),
 },
 "E-VQA (out of domain)": {
  "v3-pure":                    (["evqa-v3pure", "evqa-v3pure-s1", "evqa-v3pure-s2"], "endpoint"),
 },
}

CONTRASTS = [("InfoSeek", "v3-pure (headline)", "relevance-SFT, per-seed best ck"),
             ("InfoSeek", "v3-pure (headline)", "base (frozen, zero-shot)"),
             ("InfoSeek", "v2 +anchor", "relevance-SFT, per-seed best ck")]

def read(stem):
    rp, vp = f"{R}/{stem}.retrieval.metrics.json", f"{R}/{stem}.vqa_top5.json"
    if not os.path.exists(rp):
        return None
    r = json.load(open(rp))
    o = {"entR@5": r["entity_recall"]["5"]*100, "ansR@5": r["answer_recall"]["5"]*100, "acc": None}
    if os.path.exists(vp):
        o["acc"] = json.load(open(vp))["acc"]*100
    return o

def stat(xs):
    n = len(xs); mu = sum(xs)/n
    if n < 2: return mu, None, None, n
    sd = math.sqrt(sum((x-mu)**2 for x in xs)/(n-1))
    return mu, sd, TCRIT.get(n-1, 1.96)*sd/math.sqrt(n), n

CELLS = {}
for ev, fams in FAM.items():
    print(f"\n########## {ev} ##########")
    pooled = {m: [] for m in MET}
    for fam, (stems, est) in fams.items():
        got = {s: v for s in stems if (v := read(s))}
        miss = [s for s in stems if s not in got]
        print(f"\n=== {fam} ===  n={len(got)}  estimator: {est}")
        if miss: print("    NOT ON DISK: " + ", ".join(miss))
        for m in MET:
            xs = [v[m] for v in got.values() if v[m] is not None]
            if not xs:
                print(f"    {m:7s} no data"); continue
            mu, sd, half, n = stat(xs)
            CELLS[(ev, fam, m)] = (mu, sd, n, xs)
            per = " / ".join(f"{x:.2f}" for x in xs)
            if sd is None:
                print(f"    {m:7s} {mu:6.2f}   1 seed — recorded, not claimed   [{per}]")
            else:
                pooled[m].append((sd, n-1))
                tag = f"95% CI +/-{half:.2f}" if n >= 3 else f"range +/-{(max(xs)-min(xs))/2:.2f} (dof=1, CI unusable)"
                print(f"    {m:7s} {mu:6.2f} +/- {sd:.2f} SD  n={n}  {tag}   [{per}]")
    print(f"\n--- {ev}: pooled between-seed SD (dof-weighted) ---")
    for m in MET:
        den = sum(d for _, d in pooled[m])
        if den:
            sd = math.sqrt(sum(s*s*d for s, d in pooled[m])/den)
            print(f"    {m:7s} pooled SD {sd:.3f}   2 SD floor {2*sd:.2f}   "
                  f"({len(pooled[m])} families, dof={den})")

print("\n########## arm vs arm (Welch, matched estimator required) ##########")
for ev, a, b in CONTRASTS:
    print(f"\n=== {a}  MINUS  {b}   [{ev}] ===")
    for m in MET:
        A, B = CELLS.get((ev, a, m)), CELLS.get((ev, b, m))
        if not A or not B: print(f"    {m:7s} missing a leg"); continue
        (ma, sa, na, _), (mb, sb, nb, _) = A, B
        d = ma - mb
        if sa is None or sb is None:
            print(f"    {m:7s} {d:+6.2f}   one leg is 1 seed — no interval"); continue
        se = math.sqrt(sa*sa/na + sb*sb/nb)
        dof = (sa*sa/na + sb*sb/nb)**2 / ((sa*sa/na)**2/(na-1) + (sb*sb/nb)**2/(nb-1))
        t = TCRIT.get(int(round(dof)), 2.0); half = t*se
        sig = "CI excludes 0" if abs(d) > half else "CI SPANS 0 — not callable"
        print(f"    {m:7s} {d:+6.2f}  SE {se:.2f}  = {abs(d)/se:.1f} SE  "
              f"95% CI [{d-half:+.2f}, {d+half:+.2f}] (dof~{dof:.1f})  {sig}")

# ---- assumption-light checks for the headline, where n=3 makes the t-test fragile ----
from itertools import combinations
print("\n########## headline: two conventions + exact rank test ##########")
A = CELLS[("InfoSeek", "v3-pure (headline)", "acc")]
B = CELLS[("InfoSeek", "relevance-SFT, per-seed best ck", "acc")]
for m in MET:
    (ma, sa, na, xa) = CELLS[("InfoSeek", "v3-pure (headline)", m)]
    (mb, sb, nb, xb) = CELLS[("InfoSeek", "relevance-SFT, per-seed best ck", m)]
    d = ma - mb
    # Student, equal variance (dof = na+nb-2)
    sp = math.sqrt(((na-1)*sa*sa + (nb-1)*sb*sb)/(na+nb-2))
    se_s = sp*math.sqrt(1/na + 1/nb); half_s = TCRIT[na+nb-2]*se_s
    # Welch
    se_w = math.sqrt(sa*sa/na + sb*sb/nb)
    dofw = (sa*sa/na + sb*sb/nb)**2/((sa*sa/na)**2/(na-1) + (sb*sb/nb)**2/(nb-1))
    half_w = TCRIT.get(int(round(dofw)), 2.0)*se_w
    # exact one-sided rank-sum: fraction of the C(6,3) splits at least as extreme
    pool = xa + xb; n = len(pool)
    obs = sum(sorted(pool).index(v)+1 for v in xa)
    ge = sum(1 for c in combinations(range(n), na)
             if sum(sorted(pool).index(pool[i])+1 for i in c) >= obs)
    p = ge/len(list(combinations(range(n), na)))
    sep = "complete separation" if min(xa) > max(xb) else "overlapping"
    print(f"  {m:7s} delta {d:+.2f}"
          f"\n          Student dof={na+nb-2}: CI [{d-half_s:+.2f}, {d+half_s:+.2f}]"
          f"{'  excludes 0' if abs(d)>half_s else '  spans 0'}"
          f"\n          Welch  dof={dofw:.1f}: CI [{d-half_w:+.2f}, {d+half_w:+.2f}]"
          f"{'  excludes 0' if abs(d)>half_w else '  spans 0'}"
          f"\n          exact rank-sum one-sided p={p:.3f}   seeds are {sep}")
