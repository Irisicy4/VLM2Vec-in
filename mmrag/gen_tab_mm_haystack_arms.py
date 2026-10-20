"""Regenerate tab:mm_haystack_arms (tab_mm_haystack_arms.tex): every arm scored on the 2M index.

WHY THIS TABLE. tab:mm_ood_full carries one indirect row per panel, which is the right shape for the
headline comparison but throws away the fact that a dozen recipes were scored on the same 2M index. This
is the full ledger: it shows the direction is not carried by one lucky configuration, and it prices the
one arm whose in-domain effect was null but whose haystack effect is not (the Q4 pool filter).

    python3 mmrag/gen_tab_mm_haystack_arms.py

Cells fill themselves as run_bench2m.sh lands them; a missing arm prints as a dash rather than being
dropped, so the table's row set stays stable across regenerations.

Accuracy is acc_rescored_any (never vqa_top5's broken "acc"), matching every other table.
"""
import json, os

D = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data")
OUT = os.environ.get("MMRAG_PAPER", "/raid/icy/Overleaf_Indirect") + "/tab_mm_haystack_arms.tex"
MET = ["acc", "eR@1", "eR@5", "eR@10", "eR@20", "aR@5", "aR@10"]

# Pooled between-seed noise floors on this query set (mmrag/seed_stats.py, 9 families, dof=15).
FLOOR = {"acc": 1.83, "eR@1": 1.97, "eR@5": 2.03, "eR@10": 1.78, "eR@20": 1.43, "aR@5": 2.32}

# (label, result stem, group). Group "ref" rows are not indirect arms.
ROWS = [
    (r"base (frozen GME-2B)",                         "zeroshot",            "ref"),
    (r"relevance-SFT, matched to $422$K",             "sft-local-s0",        "ref"),
    (r"periodic full index refresh",                  "idx10k-full",         "422k"),
    (r"index EMA write-back",                         "emaidx-1M-s3000",     "422k"),
    (r"frozen index (no refresh)",                    "frozen-idx",          "2m"),
    (r"default recipe, seed $0$",                     "mix2m-v3pure",        "2m"),
    (r"default recipe, seed $1$",                     "mix2m-v3pure-s1",     "2m"),
    (r"RAG-benefit pool filter",                      "q4pool-filt",         "2m"),
    (r"\quad matched random-pool control",            "q4pool-rand",         "2m"),
    (r"diverse training mixture",                     "divmix-v3pure",       "2m"),
    (r"full-corpus draw, $10\%$ per step",            "fullcorp-rand10p50",  "2m"),
    (r"reader-gated advantages",                      "q4gate-s1",           "2m"),
]
HEAD = {"422k": r"\emph{trained on the $422$K in-domain index} --- never saw a $2$M passage",
        "2m":   r"\emph{trained on the $2$M index}"}


def load(stem):
    rp = f"{D}/results/{stem}-on-mix2M.retrieval.metrics.json"
    vp = f"{D}/results/{stem}-on-mix2M.vqa_rescored.json"
    if not (os.path.exists(rp) and os.path.exists(vp)):
        return None
    b, a = json.load(open(rp)), json.load(open(vp))
    return {"acc": 100 * a["acc_rescored_any"],
            **{f"eR@{k}": 100 * b["entity_recall"][k] for k in ("1", "5", "10", "20")},
            "aR@5": 100 * b["answer_recall"]["5"], "aR@10": 100 * b["answer_recall"]["10"]}


CAPTION = r"""\caption{\textbf{Every configuration scored on the $2$M mixed-domain index.} The queries are the InfoSeek
test set, unchanged; only the index grows, from the $422$K in-domain passages the models were tuned against
to $2$M passages carrying nine BEIR domains, HotpotQA, OVEN and OK-VQA alongside the Wikipedia articles that
hold every gold. Each corpus is re-encoded by the model being scored. Accuracy is tolerance-aware cover-EM
from a frozen reader over the top $5$ on a fixed $1{,}500$-query subset; $R@k$ are retrieval recalls over all
$3{,}000$ queries, all in \%. The first group is trained on the $422$K index alone and has therefore never
seen a passage of the index it is scored on; the second is trained against that index directly. The two
reference rows are not indirect arms: the frozen backbone, and a relevance-SFT model matched on the same
question pool, which needs no index in training because it takes its positives from the pool's gold
passages. All rows are single seed, so differences should be read against the pooled between-seed noise
floors for this query set ($1.83$ accuracy; $1.97$, $2.03$, $1.78$, $1.43$ for entity $R@1$, $5$, $10$, $20$;
$2.32$ answer $R@5$). Bold marks the best entry per column among the indirect arms. Dashes are reads still
running.
\par\smallskip
\emph{The RAG-benefit pool filter} selects the training questions \emph{before} training, keeping only those
for which retrieval is both necessary and sufficient. Each candidate question is scored by two passes of the
\emph{reward} reader --- the same frozen Qwen$2.5$-VL-$3$B that grades the reward during training, not the
evaluation reader, since the filter must reflect what the reward model can actually do --- once with the gold
passage in context and once with the image alone. A question is kept iff the gold pass is correct
\emph{and} the no-context pass is wrong. Of $20{,}000$ candidates drawn from the $810$K-question pool,
$33.9\%$ qualify; the discards are $52.8\%$ wrong even with the gold passage (no reward signal to earn),
$11.1\%$ already correct with no context (reward saturated, so the group carries no advantage), and $2.2\%$
made worse by context. This selects the pool once, so every question in every batch contributes gradient,
which is what distinguishes it from an online advantage mask that pays full rollout cost and then zeroes
most of the batch. \emph{The matched random-pool control} trains on a random $6{,}789$-question draw from the
same pool, so pool size, step count, seed and index are identical to the filtered arm and only the selection
rule differs.}"""


def main():
    data = {stem: load(stem) for _, stem, _ in ROWS}
    ind = [d for lab, stem, g in ROWS if g != "ref" and (d := data[stem])]
    best = {m: max(d[m] for d in ind) for m in MET} if ind else {}

    L = [r"\begin{table}[H]", r"\centering\small", CAPTION, r"\label{tab:mm_haystack_arms}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccc}", r"\toprule",
         r"configuration & VQA accuracy & entity $R@1$ & entity $R@5$ & entity $R@10$ & entity $R@20$ & answer $R@5$ & answer $R@10$ \\",
         r"\midrule"]
    seen = set()
    for lab, stem, g in ROWS:
        if g != "ref" and g not in seen:
            if L[-1] != r"\midrule":
                L.append(r"\midrule")
            L += [r"\multicolumn{8}{l}{" + HEAD[g] + r"}\\", r"\midrule"]
            seen.add(g)
        d = data[stem]
        if d is None:
            L.append(lab + " & " + " & ".join("---" for _ in MET) + r" \\")
        else:
            L.append(lab + " & " + " & ".join(
                (r"$\mathbf{%.2f}$" % d[m]) if (g != "ref" and abs(d[m] - best.get(m, 1e9)) < 1e-9)
                else ("$%.2f$" % d[m]) for m in MET) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(OUT, "w").write("\n".join(L) + "\n")
    miss = [s for _, s, _ in ROWS if data[s] is None]
    print(f"wrote {OUT}\n  pending: {miss if miss else 'none'}")


if __name__ == "__main__":
    main()
