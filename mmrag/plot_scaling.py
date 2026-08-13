"""Publication scaling figure: downstream VQA accuracy vs. training-pool size, per recipe.

EVERYTHING IS DISCOVERED FROM DISK. There are no hand-typed accuracies and no hand-typed
seed lists in this file (the previous version had both and went a wave stale). Sources:

  fresh wave    $MMRAG_DATA/results/<run>.vqa_top5.json   (accuracy)
                $MMRAG_DATA/runs/<run>/args.json          (recipe, pool size, seed)
  recorded wave mmrag/results_summary.json                (the pre-cluster-move runs, which
                                                           have no args.json on this machine)

Which runs land on the figure is decided by CONFIG EQUIVALENCE, not by name matching: a run
joins a series iff its args.json differs from that series' reference run only in the keys
listed in FREE_KEYS (seed / pool size / output paths). Anything that moved a real
hyperparameter -- temperature, group size, support, LR, step budget -- is excluded
automatically, so an ablation dropped into results/ can never silently become a scaling point.

Recipes are read off the args, not off the run name:
    v1  algo in {grpo, ppo}         (deterministic top-N listwise update, tag v1-listwise-top8)
    v2  algo plgrpo, contrastive_coef > 0    (PL policy gradient + InfoNCE anchor)
    v3  algo plgrpo, contrastive_coef == 0   (pure PL policy gradient)

Pending cells are simply absent: if a run has no results json yet, nothing is drawn for it.
Never add a point by hand -- if it is not in a json, it does not go on the figure.

    python3 mmrag/plot_scaling.py --out mmrag/fig_mm_scaling
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Args keys a run may differ in and still belong to the same series. Everything else must
# match the reference run exactly. NOTE `pool` is free here but constrained per series by
# POOLS below -- pool_train and pool_train_big are different query DISTRIBUTIONS
# (train_per_entity 10 vs 200), so a curve must not silently mix them.
FREE_KEYS = {"seed", "max_train_rows", "pool", "output_dir", "data_dir", "n_distractor_articles"}

# Reference runs defining each fresh-wave series (must exist in $MMRAG_DATA/runs/).
# `pools` = the pool files admissible for that series; anything else is a different experiment.
#
# The excluded case is scale-small12k/25k: pool-COMPOSITION controls that draw the same row
# counts from pool_train (<=10 queries/entity) rather than pool_train_big, i.e. a different
# query distribution at the same pool size. They are deliberately OFF the figure -- their
# divergence from the ladder is an open finding with an inconsistent sign at one seed each
# (12.5k: small 0.3467 vs big 0.3433; 25k: small 0.3340 vs big 0.3462), so plotting them would
# invite reading a composition effect that the data does not yet support. If they are ever
# added, they must be faint unconnected markers labelled as controls, never ladder points.
SERIES = [
    # key, reference run,   label,                            colour,    marker, pools
    ("v1", "scale-rows12k", "v1 listwise $+$ anchor (fresh)", "#0072B2", "o",
     {"built/pool_train_big.jsonl"}),
    # v3 curve = big pool only, matching the consumption axis. The 45,248-row pool_train headline
    # cell is a DIFFERENT query distribution (<=10 vs 200 queries/entity), so it is drawn as an
    # off-curve marker rather than joined into the line -- same rule, both axes.
    ("v3", "v3-pure", "v3-pure (fresh)", "#009E73", "D", {"built/pool_train_big.jsonl"}),
]

# Consumption axis: which AVAILABLE pool sizes each series may draw its points from. The 200k
# pool is the no-reuse guarantee. v3 additionally admits 45,248 because the headline 3-seed cell
# (500 steps on all of pool_train) is a legitimate 2k-consumed point -- but it is a DIFFERENT
# query distribution from its own 6k/12k continuations, which the prose flags as provisional.
# Each CURVE is pinned to ONE available-pool size so it never mixes query distributions.
CONSUMPTION_AVAIL = {"v1": {200000}, "v3": {200000}}

# Off-curve markers: same consumption axis, different query distribution, so they are drawn as
# hollow unconnected points rather than joined into a curve. The v3 headline cell (500 steps on
# all 45,248 rows of pool_train, <=10 q/entity) is a legitimate 2k-draw result but belongs to a
# different distribution from its own 6k/12k continuations on pool_train_big (200 q/entity).
# Once v3-rows200k lands, the matched big-pool cell joins the curve and this marker becomes the
# composition comparison rather than a substitute for it.
# Explicit off-curve runs on the consumption axis: same draw count, DIFFERENT experiment.
# v3-b16-s750 draws 12k at batch 16, i.e. 750 updates against the 3000 a batch-4 cell would take
# for the same draws -- so it cannot join a curve whose other points hold batch fixed.
CONSUMPTION_EXPLICIT = [("v3-b16-s750", "v3-pure (batch 16, off-curve)", "#009E73", "s")]

CONSUMPTION_OFFCURVE = [("v3", "v3-pure", {45248}, "v3-pure (45k pool, off-curve)", "#009E73", "D")]

# Recorded-wave points come from results_summary.json; same config-equivalence idea, but the
# axes dict is all we have. Reference = the recorded full-pool no-gold cell.
#
# Axes equivalence is NOT sufficient on the recorded wave: that dict has no field for the
# training pool file, index liveness, or a served reward model, so on axes alone the
# InfoSeek+EVQA data-mix cells, the frozen-index cell and the 7B-reward-server cell all look
# identical to the reference and pollute the 41k point (measured: 9 runs instead of 4).
# RECORDED_FAMILY restricts the series to the rows-ladder family and its seed replicates.
# It is a structural rule, not a value list -- new seeds are picked up automatically.
# Membership was checked by hand: the regex admits exactly {det, det-s2/s3/s4, rows2k, rows8k};
# config equivalence alone additionally admits mix x3 + frozen + 7B-reward (9 runs at 41k
# averaging 0.3640 rather than 4 averaging 0.3415).
#
# Why not fix this by adding the missing axes to results_summary.json instead? Because the
# recorded runs' args.json files died with the old scratch, so any axes added now would be
# derived from the run NAMES -- the same epistemics as this regex, but laundered into the
# results file where it would read as harvested provenance. A name-based rule that says it is
# name-based, in the plotting script, is the honest version.
RECORDED_REF = "rl-j2e5-nogold-det"
RECORDED_FREE = {"max_train_rows", "seed"}
RECORDED_FAMILY = re.compile(r"^rl-j2e5-nogold-(det|rows\d+k)(-s\d+)?$")
C_RECORDED = "#56B4E9"

# Horizontal reference arms (fresh repro), drawn as dashed lines.
REFERENCE_ARMS = [
    ("gme2b-zeroshot-3k-repro", "zero-shot base", "#888888", (0, (4, 3))),
    ("sft-lr1e4-h0-repro", "relevance-SFT", "#D55E00", (0, (1, 2))),
]

# The gold-context oracle (0.5040) is ~15 accuracy points above every scaling cell. Drawing it as
# a line would stretch the accuracy panel from 0.30 to 0.51 and flatten the curves the figure
# exists to show, so it is ANNOTATED off-scale instead of plotted. Retrieval has no oracle
# analogue, so the note appears on the accuracy panel only.
ORACLE_RUN = "gme2b-zeroshot-3k-repro"

# Flags added to train_rl.py's argparser AFTER some runs were launched, with their defaults.
# An older run's args.json simply lacks these keys; a newer run carries them at their default.
# Without this map, config equivalence treats "key absent" and "key present at default" as a
# real difference and silently drops the newer runs -- it did exactly that to
# scale-rows200k-s1500-s1 and scale-rows200k-s3000 (the 6k second seed and the whole 12k point)
# before this was caught. Absence matches presence IFF the present value is the default here;
# a non-default value for a flag the old code lacked is still a genuine difference.
LATER_FLAG_DEFAULTS = {
    "pl_k": 4, "pl_group": 4, "pl_support": 24, "inner_epochs": 1,
    "reward_gate_noctx": False, "reward_gate_std": 0.0,
    "kl_beta": 0.0, "entropy_coef": 0.0, "baseline": "group_z",
    "pl_behavior_temperature": None, "pool_sample_temperature": None,
    # added 2026-08-13; three COSINE runs exist on disk, so this must stay a default-match rather
    # than a blanket allow -- a cosine cell is a genuinely different config and must not merge
    # into the constant-LR curves.
    "lr_schedule": "constant", "warmup_steps": 20,
}
_MISSING = object()

# Rows in the pool files, used when max_train_rows == 0 ("use everything").
POOL_ROWS_CACHE = {}


def data_dir():
    d = os.environ.get("MMRAG_DATA")
    if not d:
        sys.exit("MMRAG_DATA is not set — point it at the data dir (see mmrag/README.md)")
    return d


def acc_of(D, run):
    """Fresh-wave accuracy, or None if the cell has not landed yet."""
    p = os.path.join(D, "results", f"{run}.vqa_top5.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["acc"]


def r5_of(D, run):
    """Fresh-wave in-domain entity R@5, or None. Paired with acc_of so both panels of a figure
    are drawn from the SAME run — the join is on run name, which the config-equivalence
    machinery already gives us."""
    p = os.path.join(D, "results", f"{run}.retrieval.metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["entity_recall"]["5"]


# Which metric each panel shows: (key, getter, axis label, reference-arm index into the tuple)
PANELS = [("acc", acc_of, "VQA accuracy (7B reader, top-5)"),
          ("r5", r5_of, "in-domain entity R@5")]


def args_of(D, run):
    p = os.path.join(D, "runs", run, "args.json")
    return json.load(open(p)) if os.path.exists(p) else None


def pool_rows(D, pool):
    """Effective training queries when max_train_rows == 0: the pool file's row count."""
    if pool not in POOL_ROWS_CACHE:
        p = os.path.join(D, pool)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            POOL_ROWS_CACHE[pool] = sum(1 for _ in f)
    return POOL_ROWS_CACHE[pool]


def n_queries(D, a):
    return a["max_train_rows"] or pool_rows(D, a["pool"])


def recipe_of(a):
    if a.get("algo") == "plgrpo":
        return "v3" if not a.get("contrastive_coef") else "v2"
    if a.get("algo") in ("grpo", "ppo"):
        return "v1"
    return None


def _agree(a, ref, k):
    """Compare one key, tolerating flags that postdate one of the two runs (see
    LATER_FLAG_DEFAULTS). Absent on one side matches the default on the other."""
    va, vr = a.get(k, _MISSING), ref.get(k, _MISSING)
    if va is _MISSING or vr is _MISSING:
        if k not in LATER_FLAG_DEFAULTS:
            return False
        present = vr if va is _MISSING else va
        return present == LATER_FLAG_DEFAULTS[k]
    return va == vr


def same_config(a, ref, free=None):
    """True iff a differs from ref only in `free` (default FREE_KEYS)."""
    free = FREE_KEYS if free is None else free
    return all(_agree(a, ref, k) for k in (set(a) | set(ref)) - free)


def collect_fresh(D, ref_run, recipe, pools):
    """-> {n_queries: [acc, ...]} over every fresh run config-equivalent to ref_run."""
    ref = args_of(D, ref_run)
    if ref is None:
        return {}, None
    out = {}
    for p in sorted(glob.glob(os.path.join(D, "results", "*.vqa_top5.json"))):
        run = os.path.basename(p)[: -len(".vqa_top5.json")]
        a = args_of(D, run)
        if a is None or recipe_of(a) != recipe or not same_config(a, ref):
            continue
        if a.get("pool") not in pools:
            continue
        n = n_queries(D, a)
        acc, r5 = acc_of(D, run), r5_of(D, run)
        if n is None or acc is None:
            continue
        out.setdefault(n, []).append((run, acc, r5))
    return out, ref


def collect_recorded(repo_root):
    """-> {n_queries: [(run, acc), ...]} from the pre-cluster-move summary."""
    p = os.path.join(repo_root, "mmrag", "results_summary.json")
    if not os.path.exists(p):
        return {}
    runs = {r["name"]: r for r in json.load(open(p))["runs"]}
    ref = runs.get(RECORDED_REF)
    if ref is None:
        return {}
    out = {}
    for name, r in runs.items():
        if not RECORDED_FAMILY.match(name):
            continue
        if r.get("dataset") != ref.get("dataset") or r.get("eval_of") != ref.get("eval_of"):
            continue  # E-VQA evals carry identical axes but are a different benchmark
        ax, rax = r.get("axes") or {}, ref["axes"]
        keys = (set(ax) | set(rax)) - RECORDED_FREE
        if not all(ax.get(k) == rax.get(k) for k in keys):
            continue
        top5 = (r.get("vqa") or {}).get("top5") or {}
        if "acc" not in top5:
            continue
        rows = ax.get("max_train_rows")
        # recorded axes store either an int or the string "all(~41k)"
        n = rows if isinstance(rows, int) else 41000
        out.setdefault(n, []).append((name, top5["acc"], (r.get("entity_R") or {}).get("5")))
    return out


def draw(ax, points, colour, marker, label, mi, filled=True, lw=2.0):
    """Draw one series on one panel. `mi` selects the metric: 1 = acc, 2 = entity R@5."""
    xs = sorted(x for x in points if any(p[mi] is not None for p in points[x]))
    if not xs:
        return 0
    ys = [statistics.mean(p[mi] for p in points[x] if p[mi] is not None) for x in xs]
    ax.plot(xs, ys, color=colour, lw=lw, marker=marker, ms=6, zorder=3, label=label,
            markerfacecolor=colour if filled else "white",
            markeredgecolor=colour, markeredgewidth=1.4)
    for x in xs:
        v = [p[mi] for p in points[x] if p[mi] is not None]
        if len(v) > 1:
            ax.vlines(x, min(v), max(v), color=colour, lw=2, alpha=0.55, zorder=2)
    return len(xs)


def style_panel(ax, ylabel, xticks=None, xlabels=None, xlabel=None):
    ax.set_xscale("log")
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.minorticks_off()
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)


def ref_lines(D, ax, mi):
    """Reference arms on a panel, plus the off-scale oracle note on the accuracy panel."""
    getter = acc_of if mi == 1 else r5_of
    for run, label, colour, dash in REFERENCE_ARMS:
        v = getter(D, run)
        if v is None:
            continue
        ax.axhline(v, color=colour, lw=1.2, ls=dash, zorder=1)
        ax.annotate(f"{label} ({v:.3f})", xy=(1.02, v), xycoords=("axes fraction", "data"),
                    fontsize=7, color=colour, va="center")
    if mi == 1:
        p = os.path.join(D, "results", f"{ORACLE_RUN}.vqa_gold.json")
        if os.path.exists(p):
            o = json.load(open(p))["acc"]
            ax.annotate(f"gold-context oracle {o:.3f} — off scale", xy=(0.015, 0.90),
                        xycoords="axes fraction", fontsize=6.6, color="#555555", style="italic")


def consumed_of(D, run, a):
    """Examples DRAWN = batch_size x steps (near-distinct, not exactly unique -- see
    draw_consumption). A checkpoint eval is named "<parent>-ck<STEP>": it inherits the parent's
    config and its step count is the checkpoint's, not the parent's final budget."""
    m = re.match(r"^(?P<parent>.+)-ck(?P<step>\d+)$", run)
    if m:
        pa = args_of(D, m.group("parent"))
        if pa is None:
            return None, None
        return int(m.group("step")) * pa["batch_size"], pa
    if a is None:
        return None, None
    return a["max_steps"] * a["batch_size"], a


def collect_consumption(D, ref_run, recipe, avail_rows):
    """-> {consumed: [(run, acc), ...]} for the consumption axis.

    Two axis-specific rules, both deliberate:
      * `max_steps` is FREE here -- it IS the axis. (On the pool-size axis it is fixed at 500
        and must stay pinned, which is why the two collectors do not share a free-key set.)
      * the series is pinned to specific AVAILABLE pool sizes via `avail_rows`, so that every
        point is guaranteed no-reuse (consumed << available) and comes from one query
        distribution. Without this every 500-step cell in the pool-size ladder piles onto the
        2k point, averaging nine differently-pooled runs into one meaningless marker.
    """
    ref = args_of(D, ref_run)
    if ref is None:
        return {}
    free = FREE_KEYS | {"max_steps"}
    out = {}
    for p in sorted(glob.glob(os.path.join(D, "results", "*.vqa_top5.json"))):
        run = os.path.basename(p)[: -len(".vqa_top5.json")]
        a0 = args_of(D, run)
        n, a = consumed_of(D, run, a0)
        if a is None or n is None or recipe_of(a) != recipe:
            continue
        if not same_config(a, ref, free):
            continue
        if n_queries(D, a) not in avail_rows:
            continue
        acc = acc_of(D, run)
        if acc is not None:
            out.setdefault(n, []).append((run, acc, r5_of(D, run)))
    return out


def draw_consumption(D, out_path, min_points=3):
    """Second figure: accuracy AND retrieval vs examples DRAWN (4 x steps), two stacked panels.

    "Drawn", not "unique": train_rl.py re-samples each batch, so examples recur across steps at
    the birthday rate (0.5% at 2k draws over the 200k pool, 2.94% at 12k).

    Two panels rather than one because the metrics DISAGREE along this axis (v3 rises in accuracy
    from 2k to 6k while its entity R@5 falls), and a single-metric plot silently picks a side of
    that dissociation. Same series, colours and markers in both panels so one arm reads
    vertically. Emits nothing until a series has `min_points` points.
    """
    series, offcurve = [], []
    for recipe, ref_run, label, colour, marker, _pools in SERIES:
        pts = collect_consumption(D, ref_run, recipe, CONSUMPTION_AVAIL[recipe])
        series.append((recipe, label, colour, marker, pts))
        print(f"  consumption {recipe}: " + (", ".join(
            f"{x}:{[p[0] for p in pts[x]]}" for x in sorted(pts)) or "no points yet"))
    for recipe, ref_run, avail, label, colour, marker in CONSUMPTION_OFFCURVE:
        pts = collect_consumption(D, ref_run, recipe, avail)
        if pts:
            offcurve.append((label, colour, marker, pts))
            print(f"  consumption {recipe} OFF-CURVE ({label}): " + ", ".join(
                f"{x}:{[p[0] for p in pts[x]]}" for x in sorted(pts)))
    for run, label, colour, marker in CONSUMPTION_EXPLICIT:
        a = args_of(D, run)
        n, _ = consumed_of(D, run, a)
        acc, r5 = acc_of(D, run), r5_of(D, run)
        if n is not None and acc is not None:
            offcurve.append((label, colour, marker, {n: [(run, acc, r5)]}))
            print(f"  consumption EXPLICIT off-curve: {run} at {n} draws "
                  f"(batch {a['batch_size']}, {a['max_steps']} updates)")
    if not any(len(pts) >= min_points for *_, pts in series):
        print(f"  consumption figure NOT written — no series has {min_points}+ points yet.")
        return False

    fig, axes = plt.subplots(2, 1, figsize=(5.4, 5.0), dpi=200, sharex=True,
                             gridspec_kw={"hspace": 0.13})
    seen = sorted({x for *_, pts in series for x in pts}
                  | {x for _l, _c, _m, pts in offcurve for x in pts})
    for ax, (key, _getter, ylab) in zip(axes, PANELS):
        mi = 1 if key == "acc" else 2
        ref_lines(D, ax, mi)
        for _recipe, label, colour, marker, pts in series:
            if draw(ax, pts, colour, marker, label if mi == 1 else None, mi, lw=1.6):
                xs = sorted(x for x in pts if any(p[mi] is not None for p in pts[x]))
                ys = [statistics.mean(p[mi] for p in pts[x] if p[mi] is not None) for x in xs]
                env, best = [], float("-inf")
                for y in ys:
                    best = max(best, y); env.append(best)
                if env != ys:
                    ax.plot(xs, env, color=colour, lw=1.0, ls=(0, (2, 2)), alpha=0.8, zorder=2)
        for label, colour, marker, pts in offcurve:
            lab = label if mi == 1 else None
            for x in sorted(pts):
                v = [p[mi] for p in pts[x] if p[mi] is not None]
                if not v:
                    continue
                ax.plot([x], [statistics.mean(v)], marker=marker, ms=6, zorder=3,
                        markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.4,
                        linestyle="none", label=lab)
                if len(v) > 1:
                    ax.vlines(x, min(v), max(v), color=colour, lw=2, alpha=0.4, zorder=2)
                lab = None
        style_panel(ax, ylab)
    axes[-1].set_xticks(seen)
    axes[-1].set_xticklabels([f"{x//1000}k" if x >= 1000 else str(x) for x in seen], fontsize=7.5)
    axes[-1].set_xlabel("VQA examples drawn ($4 \\times$ steps; $\\leq$3% resampled)", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    for ext in (".pdf", ".png"):
        fig.savefig(out_path + ext, bbox_inches="tight")
    print("wrote", out_path + ".pdf/.png  (2 panels; dashed = best-so-far envelope)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mmrag/fig_mm_scaling")
    ap.add_argument("--consumption-out", default="mmrag/fig_mm_consumption",
                    help="second figure: accuracy vs examples drawn (4 x steps)")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()
    D = data_dir()

    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.0), dpi=200, sharex=True,
                             gridspec_kw={"hspace": 0.13})
    provenance = []
    rec = collect_recorded(args.repo)
    if rec:
        provenance.append("recorded v1: " + ", ".join(
            f"{x}={statistics.mean(p[1] for p in rec[x]):.4f}(n={len(rec[x])})" for x in sorted(rec)))
    fresh = []
    for recipe, ref_run, label, colour, marker, pools in SERIES:
        pts, ref = collect_fresh(D, ref_run, recipe, pools)
        if ref is None:
            print(f"  series {recipe}: reference {ref_run} has no args.json, skipped"); continue
        fresh.append((recipe, label, colour, marker, pts))
        print(f"  series {recipe} (ref {ref_run}): "
              + ", ".join(f"{x}:{[p[0] for p in pts[x]]}" for x in sorted(pts)))
        if pts:
            provenance.append(f"{label}: " + ", ".join(
                f"{x}={statistics.mean(p[1] for p in pts[x]):.4f}(n={len(pts[x])})" for x in sorted(pts)))

    for ax, (key, getter, ylab) in zip(axes, PANELS):
        mi = 1 if key == "acc" else 2
        ref_lines(D, ax, mi)
        draw(ax, rec, C_RECORDED, "s", "v1 listwise (recorded wave)" if mi == 1 else None,
             mi, filled=False, lw=1.4)
        for _recipe, label, colour, marker, pts in fresh:
            draw(ax, pts, colour, marker, label if mi == 1 else None, mi)
        for _recipe, ref_run, avail, olabel, ocolour, omarker in CONSUMPTION_OFFCURVE:
            opts, _ = collect_fresh(D, ref_run, _recipe, {"built/pool_train.jsonl"})
            lab = olabel if mi == 1 else None
            for x in sorted(opts):
                v = [q[mi] for q in opts[x] if q[mi] is not None]
                if not v:
                    continue
                ax.plot([x], [statistics.mean(v)], marker=omarker, ms=6, zorder=3,
                        markerfacecolor="white", markeredgecolor=ocolour, markeredgewidth=1.4,
                        linestyle="none", label=lab)
                if len(v) > 1:
                    ax.vlines(x, min(v), max(v), color=ocolour, lw=2, alpha=0.4, zorder=2)
                lab = None
        style_panel(ax, ylab)
    # 12.5k dropped from the ticks only (its point is still plotted) -- it collides with 8k on a
    # log axis at this width.
    axes[-1].set_xticks([2000, 8000, 25000, 50000, 100000, 200000])
    axes[-1].set_xticklabels(["2k", "8k", "25k", "50k", "100k", "200k"], fontsize=7.5)
    axes[-1].set_xlabel("training-pool size (queries available; 500 updates throughout)", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    for ext in (".pdf", ".png"):
        fig.savefig(args.out + ext, bbox_inches="tight")
    print("wrote", args.out + ".pdf/.png  (2 panels)")

    print("\n--- consumption axis (examples drawn = 4 x steps) ---")
    draw_consumption(D, args.consumption_out)

    print("\nPROVENANCE (paste into the figure's REPRO comment):")
    for line in provenance:
        print("  " + line)


if __name__ == "__main__":
    main()
