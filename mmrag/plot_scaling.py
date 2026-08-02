"""Publication scaling figure: downstream VQA accuracy vs feedback-data size, per arm.

Colorblind-safe (Okabe-Ito subset, user-specified): no-gold RL #0072B2, SFT #D55E00,
gold RL #009E73. Zero-shot is the dashed reference. Seed spread shown as vertical
whiskers at the full-pool point (min-max over seeds). Outputs PDF (paper) + PNG (preview).

    python3 mmrag/plot_scaling.py --out mmrag/fig_mm_scaling
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# v2-metric VQA top-5 accuracy (n=1500), InfoSeek val, corpus_small. Sources: results/*.vqa_top5.json
X = [2000, 8000, 41000]
NOGOLD = [0.3333, 0.3407, 0.3413]      # rl-j2e5-nogold-rows{2k,8k}, rl-j2e5-nogold-det
NOGOLD_SEEDS_FULL = [0.3413, 0.3393]   # s0, s2 (s1, s3, s4 pending)
GOLD = [0.3287, 0.3367, 0.3433]        # rl-j2e5-rows{2k,8k}, rl-grpo-judge-lr2e5
GOLD_SEEDS_FULL = [0.3433, 0.3427, 0.3500, 0.3393]
SFT = [0.3340, 0.3007, 0.3100]         # sft-lr1e4-h0-rows{2k,8k}, sft-lr1e4-h0
SFT_SEEDS_FULL = [0.3100, 0.3080, 0.3033]
ZS = 0.3033

C_NOGOLD, C_SFT, C_GOLD = "#0072B2", "#D55E00", "#009E73"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mmrag/fig_mm_scaling")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=200)
    ax.axhline(ZS, color="#888888", lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("zero-shot", xy=(2050, ZS + 0.0012), fontsize=8, color="#666666")

    for xs, ys, seeds, c, label, mk in [
        (X, NOGOLD, NOGOLD_SEEDS_FULL, C_NOGOLD, "RL, indirect reward [no-gold]", "o"),
        (X, GOLD, GOLD_SEEDS_FULL, C_GOLD, "RL, indirect reward [gold-seeded]", "s"),
        (X, SFT, SFT_SEEDS_FULL, C_SFT, "Relevance-SFT (direct labels)", "^"),
    ]:
        ax.plot(xs, ys, color=c, lw=2, marker=mk, ms=6, zorder=3, label=label)
        ax.vlines(xs[-1], min(seeds), max(seeds), color=c, lw=2, alpha=0.55, zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels(["2k", "8k", "41k (full)"])
    ax.minorticks_off()
    ax.set_xlabel("feedback-data size (training queries)", fontsize=9)
    ax.set_ylabel("VQA accuracy (7B reader, top-5)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0.295, 0.356)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png")
    print("wrote", args.out + ".pdf/.png")


if __name__ == "__main__":
    main()
