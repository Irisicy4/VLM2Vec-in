"""Mine hard negatives for the relevance-SFT baseline: encode the corpus + train image-queries with
the base retriever, take the top-ranked passages NOT from the gold entity as `neg`.

Writes pool file back with a `neg` field (list of passage texts, hardest first).

    python3 mmrag/mine_hard_negs.py --pool built/pool_train.jsonl \
        --out built/pool_train_negs.jsonl --num_negs 8 --cache_corpus results/corpus_small_2b.pt
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.encoder import ENCODER_PROFILES, MMRagEncoder  # noqa: E402
from mmrag.eval_retrieval import encode_corpus, load_jsonl  # noqa: E402
from mmrag.image_store import open_stores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="gme2b", choices=list(ENCODER_PROFILES))
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA", "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data"))
    ap.add_argument("--corpus", default="built/corpus_small.jsonl")
    ap.add_argument("--pool", default="built/pool_train.jsonl")
    ap.add_argument("--out", default="built/pool_train_negs.jsonl")
    ap.add_argument("--num_negs", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--corpus_bs", type=int, default=int(os.environ.get("ENCODE_BS", 64)))
    ap.add_argument("--query_bs", type=int, default=16)
    ap.add_argument("--cache_corpus", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    D = args.data_dir

    corpus = load_jsonl(os.path.join(D, args.corpus))
    rows = load_jsonl(os.path.join(D, args.pool), args.max_rows)
    store = open_stores(D, "infoseek")
    rows = [r for r in rows if r["image_id"] in store]
    print(f"rows: {len(rows)}, corpus: {len(corpus)}", flush=True)

    enc = MMRagEncoder.from_profile(args.profile, device=args.device, max_len=512)
    cache = args.cache_corpus and os.path.join(D, args.cache_corpus)
    p_emb = encode_corpus(enc, corpus, args.corpus_bs, cache)
    p_emb_gpu = p_emb.to(args.device, dtype=torch.float16)

    out_path = os.path.join(D, args.out)
    with open(out_path, "w") as f:
        for i in range(0, len(rows), args.query_bs):
            chunk = rows[i : i + args.query_bs]
            imgs = [store.get(r["image_id"]) for r in chunk]
            q_emb = enc.encode_queries([r["question"] for r in chunk], imgs, batch_size=args.query_bs)
            sims = q_emb.half() @ p_emb_gpu.T
            top = sims.topk(min(args.num_negs * 4 + 20, sims.shape[1]), dim=1).indices.cpu()
            for j, r in enumerate(chunk):
                gold_url = r["entity_url"]
                negs = []
                for t in top[j].tolist():
                    p = corpus[t]
                    if p["url"] != gold_url:
                        negs.append(p["text"])
                    if len(negs) >= args.num_negs:
                        break
                r["neg"] = negs
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i // args.query_bs) % 50 == 0:
                print(f"  {i + len(chunk)}/{len(rows)}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
