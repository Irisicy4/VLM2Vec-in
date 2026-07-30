"""L1 retrieval eval: encode corpus + image-question queries, exact inner-product top-k, R@k.

Gold is entity-level (any passage of the gold article counts — the InfoSeek/EchoSight convention);
we also report answer-level recall on the subset of queries with an answer-bearing passage.

Writes <name>.retrieval.json {qid: {topk: [pid...], gold_pids, ans_pids}} for eval_vqa.py.

    python3 mmrag/eval_retrieval.py --model Qwen/Qwen2-VL-2B-Instruct \
        --checkpoint TIGER-Lab/VLM2Vec-Qwen2VL-2B --name vlm2vec2b-zeroshot \
        --corpus built/corpus_small.jsonl --queries built/queries_test.jsonl --max-q 5000
"""

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.encoder import ENCODER_PROFILES, MMRagEncoder  # noqa: E402
from mmrag.image_store import TarImageStore  # noqa: E402

KS = (1, 5, 10, 20, 50)


def load_jsonl(path, max_rows=0, seed=0):
    rows = [json.loads(l) for l in open(path)]
    if max_rows and len(rows) > max_rows:
        rows = random.Random(seed).sample(rows, max_rows)
    return rows


def encode_corpus(enc, corpus, bs, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"corpus embeddings from cache: {cache_path}", flush=True)
        return torch.load(cache_path)
    embs = []
    for i in range(0, len(corpus), bs):
        e = enc.encode_docs([p["text"] for p in corpus[i : i + bs]], batch_size=bs, to_cpu=True)
        embs.append(e)
        if (i // bs) % 50 == 0:
            print(f"  corpus {i + len(e)}/{len(corpus)}", flush=True)
    embs = torch.cat(embs)
    if cache_path:
        torch.save(embs, cache_path)
    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="gme2b", choices=list(ENCODER_PROFILES))
    ap.add_argument("--checkpoint", default="__profile__",
                    help="LoRA adapter dir to eval; default = the profile's own adapter (zero-shot)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA", "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data"))
    ap.add_argument("--corpus", default="built/corpus_small.jsonl")
    ap.add_argument("--queries", default="built/queries_test.jsonl")
    ap.add_argument("--image_tars", nargs="+", default=None)
    ap.add_argument("--max-q", type=int, default=5000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--corpus_bs", type=int, default=int(os.environ.get("ENCODE_BS", 64)))
    ap.add_argument("--query_bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512, help="doc-side max tokens")
    ap.add_argument("--query_instruction", default=None, help="default: profile style's own")
    ap.add_argument("--cache_corpus", default=None, help="path to cache corpus embeddings (.pt)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    D = args.data_dir
    corpus = load_jsonl(os.path.join(D, args.corpus))
    queries = load_jsonl(os.path.join(D, args.queries), args.max_q)
    tars = args.image_tars or [os.path.join(D, "images/Infoseek/infoseek_val_images.tar"),
                               os.path.join(D, "images/Infoseek/infoseek_train_images.tar")]
    store = TarImageStore([t for t in tars if os.path.exists(t)])
    n0 = len(queries)
    queries = [q for q in queries if q["image_id"] in store]
    print(f"queries with image available: {len(queries)}/{n0}; corpus: {len(corpus)}", flush=True)

    enc = MMRagEncoder.from_profile(args.profile, checkpoint_path=args.checkpoint,
                                    device=args.device, max_len=args.max_len,
                                    query_instruction=args.query_instruction)

    p_emb = encode_corpus(enc, corpus, args.corpus_bs, args.cache_corpus)  # (C, d) cpu fp32
    pids = [p["pid"] for p in corpus]
    p_emb_gpu = p_emb.to(args.device, dtype=torch.float16)

    out = {}
    r_at = {k: 0 for k in KS}
    ans_r_at = {k: 0 for k in KS}
    n_ans = 0
    for i in range(0, len(queries), args.query_bs):
        chunk = queries[i : i + args.query_bs]
        imgs = [store.get(q["image_id"]) for q in chunk]
        q_emb = enc.encode_queries([q["question"] for q in chunk], imgs, batch_size=args.query_bs)
        sims = q_emb.half() @ p_emb_gpu.T  # (b, C)
        top = sims.topk(min(args.topk, sims.shape[1]), dim=1).indices.cpu()
        for j, q in enumerate(chunk):
            top_pids = [pids[r] for r in top[j].tolist()]
            gold = set(q["gold_pids"])
            ans = set(q["ans_pids"])
            out[q["qid"]] = {"topk": top_pids, "gold_pids": q["gold_pids"], "ans_pids": q["ans_pids"],
                             "image_id": q["image_id"], "question": q["question"],
                             "answer": q.get("answer_aliases", q["answer"]),
                             "data_split": q.get("data_split", "")}
            for k in KS:
                if gold & set(top_pids[:k]):
                    r_at[k] += 1
            if ans:
                n_ans += 1
                for k in KS:
                    if ans & set(top_pids[:k]):
                        ans_r_at[k] += 1
        done = i + len(chunk)
        if (i // args.query_bs) % 20 == 0:
            print(f"  queries {done}/{len(queries)}  entityR@5={r_at[5]/done:.3f}", flush=True)

    n = len(queries)
    line_e = "  ".join(f"R@{k}={r_at[k]/n:.4f}" for k in KS)
    line_a = "  ".join(f"R@{k}={ans_r_at[k]/max(n_ans,1):.4f}" for k in KS)
    print(f"[{args.name}] entity-level ({n} q): {line_e}")
    print(f"[{args.name}] answer-level ({n_ans} q): {line_a}")

    out_dir = args.out_dir or os.path.join(D, "results")
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"{args.name}.retrieval.json")
    json.dump(out, open(fp, "w"))
    metrics = {"name": args.name, "n": n, "n_ans": n_ans,
               "entity_recall": {k: r_at[k] / n for k in KS},
               "answer_recall": {k: ans_r_at[k] / max(n_ans, 1) for k in KS}}
    json.dump(metrics, open(os.path.join(out_dir, f"{args.name}.retrieval.metrics.json"), "w"), indent=2)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
