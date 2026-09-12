"""emaidx-1M-s3000 vs v3-rows1M-s3000 — matched-checkpoint table + trend-vs-trend contrast.

Applies the reading registered in RL_REDESIGN.md BEFORE any EMA number existed:
  primary   ansR@5  (only metric whose baseline decline survives every seed anchor)
  secondary entR@5  (baseline trend anchor-fragile), acc (no baseline trend at all)
A single-seed gap inside the observed between-seed spread is reported as NOT CALLABLE.

    python3 compare_ema.py [--arm emaidx-1M-s3000] [--base v3-rows1M-s3000]
"""
import argparse, json, math, os

R = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data") + "/results"
METRICS = ["entR@5", "ansR@5", "acc"]
PRIMARY = "ansR@5"
# 2x pooled between-seed SD over every InfoSeek family on disk with >=2 matched seeds.
# Re-derived 2026-09-12 by seed_stats.py, which is now the single source for these numbers:
#   9 families, dof=15 -> entR@5 sd 1.017 | ansR@5 sd 1.158 | acc sd 0.511
# Two fixes over the previous {2.40, 1.91, 0.61}:
#   (1) that version pooled only 4 families;
#   (2) it pooled ACROSS EVAL SETS. E-VQA (n=2701, n_ans=2696) is a different measurement from
#       InfoSeek (n=3000, n_ans=2450) and has its own, much wider floors (2.43 / 7.10 / 2.54).
# Net effect: the acc floor nearly doubles (0.61 -> 1.02) and entR@5 tightens (2.40 -> 2.03), so
# some previously-callable acc deltas are no longer callable and some entR@5 deltas now are.
# Both earlier versions {4.33, 1.51, 0.33} and {2.40, 1.91, 0.61} are superseded.
SEED_FLOOR = {"entR@5": 2.03, "ansR@5": 2.32, "acc": 1.02}
# InfoSeek only. For an E-VQA comparison use these instead:
EVQA_FLOOR = {"entR@5": 2.43, "ansR@5": 7.10, "acc": 2.54}
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def read(name):
    """RETRIEVAL is the primary product and is written BEFORE the VQA stage runs.

    acc is optional on purpose: the pipeline order is encode -> retrieval -> VQA, and VQA is
    both the slowest stage and the most contention-sensitive, so there is a real window where
    entR@5/ansR@5 exist and acc does not. Requiring both used to make this function return None
    in that window, hiding the PRIMARY metric (ansR@5) behind the secondary one.
    """
    rp, vp = f"{R}/{name}.retrieval.metrics.json", f"{R}/{name}.vqa_top5.json"
    if not os.path.exists(rp):
        return None
    r = json.load(open(rp))
    out = {"entR@5": r["entity_recall"]["5"] * 100, "ansR@5": r["answer_recall"]["5"] * 100,
           "entR@1": r["entity_recall"]["1"] * 100, "entR@10": r["entity_recall"]["10"] * 100,
           "acc": None, "n_ret": r["n"], "n_ans": r.get("n_ans"), "n_vqa": None}
    if os.path.exists(vp):
        v = json.load(open(vp))
        out["acc"], out["n_vqa"] = v["acc"] * 100, v["n"]
    return out


def curve(stem):
    """(step, metrics) for ck750/1500/2250 and the final, skipping what is not yet evaluated."""
    out = []
    for step, nm in [(750, f"{stem}-ck750"), (1500, f"{stem}-ck1500"),
                     (2250, f"{stem}-ck2250"), (3000, stem)]:
        d = read(nm)
        if d:
            out.append((step, d))
    return out


def fit(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    dof = n - 2
    s2 = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / dof
    se = math.sqrt(s2 / sxx)
    t = TCRIT[dof]
    return b * 1000, (b - t * se) * 1000, (b + t * se) * 1000, dof


def show_trend(pts, label):
    xs = [p[0] for p in pts]
    for m in METRICS:
        ys = [p[1][m] for p in pts if p[1].get(m) is not None]
        xs2 = [p[0] for p in pts if p[1].get(m) is not None]
        f = fit(xs2, ys) if len(xs2) == len(xs) else fit(xs2, ys)
        xs_used = xs2
        tag = " (PRIMARY)" if m == PRIMARY else ""
        if not f:
            print(f"  {label:<22} {m:<7} n={len(xs_used)} — need >=3 points for a CI"); continue
        b, lo, hi, dof = f
        v = "DECLINE" if hi < 0 else ("RISE" if lo > 0 else "spans zero — not distinguishable")
        print(f"  {label:<22} {m:<7}{tag:<10} {b:+7.3f} pts/1k [{lo:+7.3f},{hi:+7.3f}] dof={dof}  {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="emaidx-1M-s3000")
    ap.add_argument("--base", default="v3-rows1M-s3000")
    a = ap.parse_args()
    A, B = curve(a.arm), curve(a.base)
    if not A:
        print(f"no evaluated checkpoints for {a.arm} yet — nothing to compare"); return
    have = {s for s, _ in A} & {s for s, _ in B}
    print(f"MATCHED CHECKPOINTS ({a.arm} vs {a.base}); 1 seed each = recorded, not claimed")
    print(f"{'step':>5} " + "".join(f"{m+' arm':>13}{m+' base':>14}{'delta':>9}{'':>4}" for m in METRICS))
    da = {m: [] for m in METRICS}
    for s in sorted(have):
        ma = dict(A)[s]; mb = dict(B)[s]
        line = f"{s:>5} "
        for m in METRICS:
            if ma.get(m) is None or mb.get(m) is None:
                line += f"{'-':>13}{'-':>14}{'pending':>9}{'':>4}"
                continue
            d = ma[m] - mb[m]; da[m].append(d)
            flag = "" if abs(d) >= SEED_FLOOR[m] else " ~"      # ~ = inside seed noise
            line += f"{ma[m]:>13.2f}{mb[m]:>14.2f}{d:>+9.2f}{flag:>4}"
        print(line)
    print(f"  (~ = |delta| below the observed between-seed spread "
          f"{SEED_FLOOR} -> NOT CALLABLE on one seed)")
    if da[PRIMARY]:
        print(f"\nMEAN DELTA over {len(da[PRIMARY])} matched checkpoints:")
        for m in METRICS:
            if not da[m]:
                print(f"  {m:<7}{' (PRIMARY)' if m==PRIMARY else '':<10}   pending (VQA stage not finished)")
                continue
            mu = sum(da[m]) / len(da[m])
            call = "callable" if abs(mu) >= SEED_FLOOR[m] else "INSIDE SEED NOISE — not callable"
            print(f"  {m:<7}{' (PRIMARY)' if m==PRIMARY else '':<10} {mu:+7.2f}  {call}")
    print("\nTREND-VS-TREND (least squares, 95% CI):")
    show_trend(A, a.arm); show_trend(B, a.base)
    print("\nRegistered scope: ansR@5 is primary. entR@5 secondary (baseline trend is")
    print("anchor-fragile: present under seed-0/seed-avg anchors, absent under seed-1).")
    print("acc secondary (no baseline trend under any anchor — nothing to prevent).")


if __name__ == "__main__":
    main()
