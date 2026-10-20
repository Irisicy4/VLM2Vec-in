"""Regenerate tab:mm_ood_full (tab_mm_ood.tex) from the result files.

WHY A GENERATOR. Every cell in this table is a mean over a declared seed family, and the one error this
project has made repeatedly is mixing estimators inside a column -- taking accuracy from one family and
retrieval from another. Generating the whole table from one loop makes that impossible, and it means the
dashed relevance-SFT haystack cell fills itself in the moment sft-local-s0-on-mix2M lands:

    python3 mmrag/gen_tab_mm_ood.py && cd /raid/icy/Overleaf_Indirect && git add tab_mm_ood.tex

Accuracy is acc_rescored_any, NOT acc_orig (the broken stringified-dict-gold metric) and NOT
acc_rescored_first. That is the convention in tab:mm_main and every cell here matches it.
"""
import json, os, statistics as st

D = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data")
OUT = os.environ.get("MMRAG_PAPER", "/raid/icy/Overleaf_Indirect") + "/tab_mm_ood.tex"
MET = ["acc", "eR1", "eR5", "eR10", "eR20", "aR5", "aR10"]


def load(n, suf=""):
    b = json.load(open(f"{D}/results/{n}{suf}.retrieval.metrics.json"))
    a = json.load(open(f"{D}/results/{n}{suf}.vqa_rescored.json"))
    return {"acc": 100 * a["acc_rescored_any"],
            **{f"eR{k}": 100 * b["entity_recall"][k] for k in ("1", "5", "10", "20")},
            **{f"aR{k}": 100 * b["answer_recall"][k] for k in ("5", "10")}}


def c(v, sd=None, b=False):
    if sd is None:
        return (r"$\mathbf{%.2f}$" % v) if b else ("$%.2f$" % v)
    return (r"$\mathbf{%.2f} \pm \mathbf{%.2f}$" % (v, sd)) if b else (r"$%.2f \pm %.2f$" % (v, sd))


CAPTION = r"""\caption{\textbf{Out-of-domain behaviour of the multimodal retrievers, full metric series.} Every model is
trained on InfoSeek only and no row saw a target-domain query. The two panels are two \emph{different} shifts
and should not be compared to each other. \textsc{Domain shift} re-evaluates on Encyclopedic-VQA and its own
corpus ($n=2{,}701$ for retrieval), so queries, images and corpus all change. \textsc{Haystack shift} keeps
the InfoSeek queries ($n=3{,}000$ for retrieval) and enlarges the index from the $422$K in-domain passages to $2$M mixed-domain ones, so
only the distractor pool changes. That panel separates the two indirect training conditions: the $422$K row
has never seen a passage from the larger index and is therefore transfer, while the $2$M row trained against
the index it is scored on and is the matched-condition ceiling. The relevance-SFT row there is matched to the
$422$K condition, on the same question pool; supervised training uses gold passages directly and never builds
an index, so it has no $2$M counterpart. Every corpus is re-encoded by the evaluated model. Accuracy is
tolerance-aware cover-EM from a frozen reader over the top $5$, scored on a fixed $1{,}500$-query subset in
both panels (reader decoding, not retrieval, is the cost that forces the subset); $R@k$ are retrieval recalls
over the full query set, all in \%.
``ours'' is the indirect objective, which uses no relevance labels and no gold passage anywhere in training;
each panel reports its best such configuration, $\pm$ one sample SD where seeds are available. Two
relevance-SFT rows appear under domain shift because checkpoint choice dominates that baseline out of domain:
step $750$ is the checkpoint an \emph{in-domain} criterion selects (it also contributes seed $0$ to
Table~\ref{tab:mm_main}'s SFT row, whose other two seeds select step $1050$), so it is the comparator
consistent with the in-domain table, while the endpoint is where the supervised run lands if no checkpoint is
selected at all. Bold marks the best entry per column within each panel; because the haystack rows are
single seed, differences there should be read against the pooled between-seed noise floors measured on this
query set ($1.83$ accuracy, $1.97$/$2.03$/$1.78$/$1.43$ entity $R@1$/$5$/$10$/$20$, $2.32$ answer $R@5$), so the
two entity-recall gaps at depth $5$ and $10$ in that panel are ties rather than leads.}"""

SFT2M = "sft-local-s0"


def main():
    ev = {n: load(n) for n in ("evqa-zs", "evqa-sft", "evqa-sft-ck750")}
    s3 = [load(n) for n in ("evqa-v3pure", "evqa-v3pure-s1", "evqa-v3pure-s2")]
    oev = {m: (st.mean([s[m] for s in s3]), st.stdev([s[m] for s in s3])) for m in MET}
    tm = {n: load(n, "-on-mix2M") for n in ("zeroshot", "emaidx-1M-s3000", "mix2m-v3pure")}
    have = os.path.exists(f"{D}/results/{SFT2M}-on-mix2M.retrieval.metrics.json")
    if have:
        tm[SFT2M] = load(SFT2M, "-on-mix2M")

    bev = {m: max(ev["evqa-zs"][m], ev["evqa-sft"][m], ev["evqa-sft-ck750"][m], oev[m][0]) for m in MET}
    pool = [tm["zeroshot"], tm["emaidx-1M-s3000"], tm["mix2m-v3pure"]] + ([tm[SFT2M]] if have else [])
    b2m = {m: max(d[m] for d in pool) for m in MET}
    eq = lambda a, b: abs(a - b) < 1e-9
    row = lambda lab, d, best: lab + " & " + " & ".join(c(d[m], b=eq(d[m], best[m])) for m in MET) + r" \\"

    L = [r"\begin{table}[H]", r"\centering\small", CAPTION, r"\label{tab:mm_ood_full}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccc}", r"\toprule",
         r"model & VQA accuracy & entity $R@1$ & entity $R@5$ & entity $R@10$ & entity $R@20$ & answer $R@5$ & answer $R@10$ \\",
         r"\midrule",
         r"\multicolumn{8}{l}{\emph{domain shift}: InfoSeek $\rightarrow$ Encyclopedic-VQA, own corpus ($n=2{,}701$)}\\",
         r"\midrule",
         row("base (frozen GME-2B)", ev["evqa-zs"], bev),
         row(r"relevance-SFT, step $750$ (selected in domain)", ev["evqa-sft-ck750"], bev),
         row("relevance-SFT, endpoint", ev["evqa-sft"], bev),
         r"\textbf{ours}, indirect (3 seeds) & " + " & ".join(
             c(oev[m][0], oev[m][1], b=eq(oev[m][0], bev[m])) for m in MET) + r" \\",
         r"\midrule",
         r"\multicolumn{8}{l}{\emph{haystack shift}: same InfoSeek queries ($n=3{,}000$), index $422$K $\rightarrow$ $2$M mixed-domain}\\",
         r"\midrule",
         row("base (frozen GME-2B)", tm["zeroshot"], b2m),
         (row(r"relevance-SFT, matched to the $422$K condition", tm[SFT2M], b2m) if have else
          r"relevance-SFT, matched to the $422$K condition & " + " & ".join("---" for _ in MET) + r" \\"),
         row(r"\textbf{ours}, indirect, trained on $422$K \emph{(transfer)}", tm["emaidx-1M-s3000"], b2m),
         row(r"\textbf{ours}, indirect, trained on $2$M \emph{(matched)}", tm["mix2m-v3pure"], b2m),
         r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}  (relevance-SFT haystack cell: {'FILLED' if have else 'dashed, waiting on ' + SFT2M})")


if __name__ == "__main__":
    main()
