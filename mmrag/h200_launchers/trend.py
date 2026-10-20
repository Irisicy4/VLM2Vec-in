"""Is emaidx-1M-s3000 collapsing or improving relative to its own start?

TRAIN-SIDE PROXIES ONLY. The registered comparison is entR@5/ansR@5/acc at matched checkpoints;
these signals cannot substitute for it. They exist to catch a COLLAPSE early -- especially the
group_std -> 0 failure that killed the VLM2Vec-4B cells at step 249 -- long before ck750.

House rule applied: least-squares slope with 95% CI. CI spanning zero = not distinguishable.
Per-step rows are 4 queries each, so raw points are extremely noisy; we bin before fitting.
"""
import json, math, os, sys

D = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data")
RUN = sys.argv[1] if len(sys.argv) > 1 else "emaidx-1M-s3000"
BIN = int(os.environ.get("BIN", "25"))
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980}


def tcrit(dof):
    for k in sorted(TCRIT):
        if dof <= k:
            return TCRIT[k]
    return 1.96


def fit(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    dof = n - 2
    s2 = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / dof
    se = math.sqrt(s2 / sxx)
    t = tcrit(dof)
    return b, b - t * se, b + t * se, dof


rows = [json.loads(l) for l in open(f"{D}/runs/{RUN}/metrics.jsonl")]
if not rows:
    print("no metrics yet"); sys.exit(0)
last = rows[-1]["step"]
print(f"{RUN}: {len(rows)} rows, through step {last}  (target 3000)")

SIGNALS = [("reward/raw_mean", "judge reward", 1.0),
           ("retrieval/gold_in_topN", "gold in pool", 1.0),
           ("reward/group_std_mean", "group std  ", 1.0),
           ("reward/group_degenerate_frac", "degenerate%", 1.0),
           ("index/stale_mean", "index stale", 1.0)]

# bins
bins = {}
for r in rows:
    bins.setdefault(r["step"] // BIN, []).append(r)
keys = sorted(bins)
print(f"\nbinned means (bin = {BIN} steps):")
hdr = "  bin  steps      " + "".join(f"{lbl:>13}" for _, lbl, _ in SIGNALS)
print(hdr)
for k in keys:
    b = bins[k]
    line = f"  {k:>3}  {k*BIN:>4}-{k*BIN+BIN-1:<4}  "
    for key, _, sc in SIGNALS:
        vals = [x[key] for x in b if key in x and x[key] == x[key]]
        line += f"{(sum(vals)/len(vals)*sc if vals else float('nan')):>13.4f}"
    print(line)

print(f"\nleast-squares trend over binned means, per 1000 steps, 95% CI:")
for key, lbl, sc in SIGNALS:
    xs, ys = [], []
    for k in keys:
        vals = [x[key] for x in bins[k] if key in x and x[key] == x[key]]
        if vals:
            xs.append(k * BIN + BIN / 2); ys.append(sum(vals) / len(vals) * sc)
    f = fit(xs, ys)
    if not f:
        print(f"  {lbl:<12} n={len(xs)} bins — need >=3 for a CI"); continue
    b, lo, hi, dof = f
    v = "RISING" if lo > 0 else ("FALLING" if hi < 0 else "flat (CI spans 0)")
    print(f"  {lbl:<12} {b*1000:+10.4f}  [{lo*1000:+9.4f},{hi*1000:+9.4f}]  dof={dof}  {v}")

# collapse guards
print("\ncollapse guards:")
gs = [r["reward/group_std_mean"] for r in rows]
zero = sum(1 for g in gs if g == 0)
tail = gs[-25:]
tail_zero = sum(1 for g in tail if g == 0)
print(f"  group_std exactly 0: {zero}/{len(gs)} all-time, {tail_zero}/{len(tail)} in last {len(tail)}"
      f"   [VLM2Vec-4B died at 100% here]")
dg = [r.get("reward/group_degenerate_frac", float('nan')) for r in rows[-25:]]
dg = [d for d in dg if d == d]
if dg:
    print(f"  degenerate group frac, last {len(dg)}: mean {sum(dg)/len(dg):.3f}  max {max(dg):.3f}")
rw = [r["reward/raw_mean"] for r in rows]
first, tailr = rw[: max(5, len(rw)//10)], rw[-max(5, len(rw)//10):]
print(f"  judge reward: first {len(first)} mean {sum(first)/len(first):.4f}"
      f"  ->  last {len(tailr)} mean {sum(tailr)/len(tailr):.4f}")
gp = [r["retrieval/gold_in_topN"] for r in rows]
print(f"  gold in pool: first {len(first)} mean {sum(gp[:len(first)])/len(first):.4f}"
      f"  ->  last {len(tailr)} mean {sum(gp[-len(tailr):])/len(tailr):.4f}")
st = [r.get("index/stale_mean") for r in rows if r.get("index/stale_mean") is not None]
if st:
    print(f"  index staleness: {st[0]:.1f} -> {st[-1]:.1f} (expect a rise toward ~472 = 173755/368)")
print("\nNOTE: train-side proxies on 4 queries/step. NOT the registered metrics; a null here is")
print("      not a null on entR@5/ansR@5/acc. Use only to catch collapse before ck750.")
