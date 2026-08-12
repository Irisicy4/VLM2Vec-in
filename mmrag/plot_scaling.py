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
    # v3-pure (all of pool_train) is the ~45k point of the v3 curve; the v3 scaling cells use
    # the big pool, so this series legitimately spans both files. Flagged in the caption.
    ("v3", "v3-pure", "v3-pure (fresh)", "#009E73", "D",
     {"built/pool_train.jsonl", "built/pool_train_big.jsonl"}),
]

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


def same_config(a, ref):
    """True iff a differs from ref only in FREE_KEYS."""
    keys = (set(a) | set(ref)) - FREE_KEYS
    return all(a.get(k) == ref.get(k) for k in keys)


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
        acc = acc_of(D, run)
        if n is None or acc is None:
            continue
        out.setdefault(n, []).append((run, acc))
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
        out.setdefault(n, []).append((name, top5["acc"]))
    return out


def draw(ax, points, colour, marker, label, filled=True, lw=2.0):
    xs = sorted(points)
    if not xs:
        return 0
    ys = [statistics.mean(a for _, a in points[x]) for x in xs]
    ax.plot(xs, ys, color=colour, lw=lw, marker=marker, ms=6, zorder=3, label=label,
            markerfacecolor=colour if filled else "white",
            markeredgecolor=colour, markeredgewidth=1.4)
    for x in xs:
        accs = [a for _, a in points[x]]
        if len(accs) > 1:
            ax.vlines(x, min(accs), max(accs), color=colour, lw=2, alpha=0.55, zorder=2)
    return len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mmrag/fig_mm_scaling")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()
    D = data_dir()

    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=200)
    provenance = []

    # horizontal reference arms
    for run, label, colour, dash in REFERENCE_ARMS:
        acc = acc_of(D, run)
        if acc is None:
            print(f"  reference {run}: NOT FOUND, skipped")
            continue
        ax.axhline(acc, color=colour, lw=1.2, ls=dash, zorder=1)
        ax.annotate(f"{label} ({acc:.3f})", xy=(1.02, acc), xycoords=("axes fraction", "data"),
                    fontsize=7, color=colour, va="center")
        provenance.append(f"{label}: {run} = {acc:.4f}")

    # recorded wave first (background), then the fresh series
    rec = collect_recorded(args.repo)
    n = draw(ax, rec, C_RECORDED, "s", "v1 listwise (recorded wave)", filled=False, lw=1.4)
    if n:
        provenance.append("recorded v1: " + ", ".join(
            f"{x}={statistics.mean(a for _, a in rec[x]):.4f}(n={len(rec[x])})" for x in sorted(rec)))

    for recipe, ref_run, label, colour, marker, pools in SERIES:
        pts, ref = collect_fresh(D, ref_run, recipe, pools)
        if ref is None:
            print(f"  series {recipe}: reference run {ref_run} has no args.json, skipped")
            continue
        n = draw(ax, pts, colour, marker, label)
        if recipe == "v3" and len(pts) == 1:
            x = next(iter(pts))
            y = statistics.mean(a for _, a in pts[x])
            ax.annotate(f"v3-pure, {x/1000:.0f}k pool\n(v3 scaling cells in flight)",
                        xy=(x, y), xytext=(-4, 14), textcoords="offset points",
                        fontsize=6.5, color=colour, ha="right")
        print(f"  series {recipe} (ref {ref_run}): {n} pool sizes — "
              + ", ".join(f"{x}:{[r for r, _ in pts[x]]}" for x in sorted(pts)))
        if n:
            provenance.append(f"{label}: " + ", ".join(
                f"{x}={statistics.mean(a for _, a in pts[x]):.4f}(n={len(pts[x])})" for x in sorted(pts)))

    ax.set_xscale("log")
    ax.set_xlabel("training-pool size (queries available; 500 updates throughout)", fontsize=9)
    ax.set_ylabel("VQA accuracy (7B reader, top-5)", fontsize=9)
    ax.tick_params(labelsize=8)
    # Ticks are the v1 ladder's own pool sizes. v3-pure sits at 45,248 (all of pool_train),
    # close to the ladder's 50k point but a DIFFERENT pool file, so it is annotated rather
    # than given a colliding tick.
    ax.set_xticks([2000, 8000, 12500, 25000, 50000, 100000, 200000])
    ax.set_xticklabels(["2k", "8k", "12.5k", "25k", "50k", "100k", "200k"], fontsize=7.5)
    ax.minorticks_off()
    ax.tick_params(axis="x", pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=1)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(args.out + ext, bbox_inches="tight")
    print("wrote", args.out + ".pdf/.png")
    print("\nPROVENANCE (paste into the figure's REPRO comment):")
    for line in provenance:
        print("  " + line)


if __name__ == "__main__":
    main()
