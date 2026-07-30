"""L2 downstream eval: VLM reader answers each visual question given the retriever's top-k passages;
score = answer accuracy (cover-EM) + F1. This is the objective the RL optimizes — the metric where
indirect fine-tuning should beat zero-shot retrieval.

    python3 mmrag/eval_vqa.py --retrieval results/vlm2vec2b-zeroshot.retrieval.json \
        --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 2000
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.image_store import open_stores  # noqa: E402
from mmrag.reader_vlm import LiveVLMReader, answer_correct, squad_f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True, help="<name>.retrieval.json from eval_retrieval.py")
    ap.add_argument("--reader", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA", "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data"))
    ap.add_argument("--dataset", default="infoseek", choices=["infoseek", "evqa"])
    ap.add_argument("--corpus", default=None, help="default per dataset")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-q", type=int, default=2000)
    ap.add_argument("--no-context", action="store_true", help="reader-only baseline (no retrieval)")
    ap.add_argument("--gold-context", action="store_true", help="oracle: gold passages as context")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    D = args.data_dir
    args.corpus = args.corpus or {"infoseek": "built/corpus_small.jsonl",
                                  "evqa": "built/corpus_evqa.jsonl"}[args.dataset]
    retr = json.load(open(args.retrieval))
    qids = sorted(retr)
    if args.max_q and len(qids) > args.max_q:
        qids = random.Random(args.seed).sample(qids, args.max_q)

    pid2text = {}
    need = set()
    for qid in qids:
        r = retr[qid]
        need.update(r["topk"][: args.k])
        need.update(r["ans_pids"] or r["gold_pids"][:2])
    with open(os.path.join(D, args.corpus)) as f:
        for line in f:
            p = json.loads(line)
            if p["pid"] in need:
                pid2text[p["pid"]] = p["text"]

    store = open_stores(D, args.dataset)
    reader = LiveVLMReader(args.reader, device=args.device, batch_size=args.batch_size,
                           max_ctx_chars=args.k * 900)

    triples, kept = [], []
    for qid in qids:
        r = retr[qid]
        if r["image_id"] not in store:
            continue
        if args.no_context:
            ctx = ""
        elif args.gold_context:
            gp = (r["ans_pids"] or r["gold_pids"][:2])[: args.k]
            ctx = "\n\n".join(pid2text[p] for p in gp if p in pid2text)
        else:
            ctx = "\n\n".join(pid2text[p] for p in r["topk"][: args.k] if p in pid2text)
        triples.append({"image": store.get(r["image_id"]), "question": r["question"],
                        "context": ctx, "answer": r["answer"]})
        kept.append(qid)

    accs, preds = reader.score_judge(triples, n_rollouts=1, temperature=0.0, return_preds=True)
    f1s = [squad_f1(p, t["answer"]) for p, t in zip(preds, triples)]
    n = len(kept)
    acc, f1 = sum(accs) / n, sum(f1s) / n
    by_split = {}
    for qid, a in zip(kept, accs):
        s = retr[qid].get("data_split") or "all"
        by_split.setdefault(s, []).append(a)
    split_acc = {s: sum(v) / len(v) for s, v in by_split.items()}
    name = os.path.basename(args.retrieval).replace(".retrieval.json", "")
    mode = "noctx" if args.no_context else ("gold" if args.gold_context else f"top{args.k}")
    print(f"[{name}] L2 VQA  reader={os.path.basename(args.reader)} mode={mode} n={n}  "
          f"acc={acc:.4f}  F1={f1:.4f}  by_split={ {k: round(v, 4) for k, v in split_acc.items()} }")
    out = {"name": name, "reader": args.reader, "mode": mode, "n": n, "acc": acc, "f1": f1,
           "acc_by_split": split_acc}
    fp = args.retrieval.replace(".retrieval.json", f".vqa_{mode}.json")
    json.dump(out, open(fp, "w"), indent=2)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
