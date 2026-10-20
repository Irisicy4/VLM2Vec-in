#!/usr/bin/env python3
"""Pool the Q4 gate diagnostics over many steps.

Per step the denominators are only 3-4 queries, so the per-step keep-given-gold vs
keep-given-no-gold numbers state nothing. Two questions worth answering once a few hundred
steps have accumulated:

  (1) Is the gate a POOL-QUALITY detector or only a query-difficulty filter? If it keeps
      gold-bearing and gold-free pools at the same rate it is the latter, however well it
      separates rewards. Tested as a pooled two-proportion z.
  (2) Does the gate remove the DEGENERATE groups, i.e. the ones whose reward variance is
      already zero and which therefore contribute no gradient anyway? If it does, the filter
      costs nothing real and only sheds dead weight. If it drops live groups instead, the
      50% keep rate is a genuine loss of signal. Tested as the correlation between the
      per-step keep rate and the per-step degenerate fraction, plus the implied overlap.
"""
import json, sys, math

def load(run):
    return [json.loads(l) for l in open(f"/raid/icy/mmrag_data/runs/{run}/metrics.jsonl")]

def z2p(k1, n1, k2, n2):
    if min(n1, n2) == 0: return None
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2); se = math.sqrt(p*(1-p)*(1/n1+1/n2))
    return (p1, p2, p1-p2, (p1-p2)/se if se > 0 else float('nan'))

for run in sys.argv[1:]:
    try: rows = load(run)
    except FileNotFoundError: print(f"{run}: no metrics yet"); continue
    rows = [r for r in rows if "gate/noctx_keep_rate" in r]
    if not rows: print(f"{run}: gate never fired — FLAG INERT"); continue
    B = 4   # batch_size default; the per-step denominators are this small
    kg = sum(r["gate/noctx_keep_given_gold"] for r in rows)
    kn = sum(r["gate/noctx_keep_given_no_gold"] for r in rows)
    n = len(rows)
    keep = [r["gate/noctx_keep_rate"] for r in rows]
    deg  = [r.get("reward/group_degenerate_frac", float('nan')) for r in rows]
    mk, md = sum(keep)/n, sum(deg)/n
    cov = sum((keep[i]-mk)*(deg[i]-md) for i in range(n))/n
    sk = math.sqrt(sum((v-mk)**2 for v in keep)/n); sd = math.sqrt(sum((v-md)**2 for v in deg)/n)
    r_kd = cov/(sk*sd) if sk > 0 and sd > 0 else float('nan')
    print(f"\n=== {run} — {n} gated steps ===")
    print(f"  keep rate               {mk:.3f}   (the filter drops {100*(1-mk):.0f}% of queries)")
    print(f"  degenerate group frac   {md:.3f}   (zero reward variance, no gradient regardless)")
    print(f"  mean keep | gold in pool      {kg/n:.3f}")
    print(f"  mean keep | NO gold in pool   {kn/n:.3f}")
    print(f"  difference                    {kg/n - kn/n:+.3f}"
          f"  <- if ~0 the gate reads query difficulty, not pool quality")
    print(f"  corr(keep rate, degenerate frac)  {r_kd:+.3f}")
    if not math.isnan(r_kd):
        print(f"    {'NEGATIVE: the gate sheds the dead groups (filter is close to free)' if r_kd < -0.2 else 'not clearly negative: the gate is dropping LIVE groups too, so the keep rate is a real cost'}")
    # Frechet bounds from the aggregates alone — kept so the join can be compared against
    # what the aggregates could have told us. The earlier "at best 2pp" line was wrong: it
    # computed the overlap of KEPT with LIVE, not of DROPPED with DEAD.
    drop = 1 - mk
    lo, hi = max(0.0, drop + md - 1), min(drop, md)
    print(f"  from aggregates only: dropped {drop:.3f}, degenerate {md:.3f} -> overlap in "
          f"[{lo:.3f}, {hi:.3f}], i.e. [{100*lo/drop:.0f}%, {100*hi/drop:.0f}%] of the drop "
          f"was already dead. Too wide to conclude.")
    if "gate/drop_and_live" in rows[0]:
        dl = sum(r["gate/drop_and_live"] for r in rows)/n
        dd = sum(r["gate/drop_and_dead"] for r in rows)/n
        kl = sum(r["gate/keep_given_live"] for r in rows)/n
        kd = sum(r["gate/keep_given_dead"] for r in rows)/n
        print(f"  THE JOIN (exact, not inferred):")
        print(f"    dropped AND already dead  {dd:.3f} of all queries = {100*dd/drop:.0f}% of the drop  <- free")
        print(f"    dropped AND live          {dl:.3f} of all queries = {100*dl/drop:.0f}% of the drop  <- the real cost")
        print(f"    keep | live group  {kl:.3f}     keep | dead group  {kd:.3f}")
        if kl > kd + 0.05:
            print(f"    -> the gate PREFERENTIALLY KEEPS live groups ({kl:.3f} vs {kd:.3f}). It is not "
                  f"merely shedding waste, it is selecting for reward variance.")
        elif kd > kl + 0.05:
            print(f"    -> INVERTED: the gate keeps DEAD groups more often than live ones. It is "
                  f"removing usable signal and the +R@5 on the text side cannot be this mechanism.")
        else:
            print(f"    -> the gate is blind to degeneracy ({kl:.3f} vs {kd:.3f}): it drops live and "
                  f"dead groups alike, so ~{100*dl/drop:.0f}% of the drop is pure signal loss.")
    else:
        print(f"  THE JOIN: not logged in this run (pre-patch). Restart needed to resolve.")
