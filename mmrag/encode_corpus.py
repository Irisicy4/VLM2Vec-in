"""Pre-encode corpus_small with a gme2b encoder (zero-shot or LoRA ckpt) into a .pt cache
that mmrag/eval_retrieval.py --cache_corpus picks up. Supports contiguous sharding across GPUs;
merge the shard files afterwards with --merge (order-preserving)."""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="gme2b")
ap.add_argument("--checkpoint", default="__profile__")
ap.add_argument("--cache", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()

D = os.environ["MMRAG_DATA"]

if a.merge:
    parts = [torch.load(f"{a.cache}.{i}of{a.nshards}") for i in range(a.nshards)]
    emb = torch.cat(parts)
    torch.save(emb, a.cache)
    print("MERGED", tuple(emb.shape), a.cache, flush=True)
    sys.exit(0)

from mmrag.encoder import load_encoder  # noqa: E402
from mmrag.eval_retrieval import encode_corpus, load_jsonl  # noqa: E402

corpus = load_jsonl(os.path.join(D, "built/corpus_small.jsonl"))
n = len(corpus)
lo = n * a.shard // a.nshards
hi = n * (a.shard + 1) // a.nshards
part = corpus[lo:hi]
out = f"{a.cache}.{a.shard}of{a.nshards}" if a.nshards > 1 else a.cache
print(f"corpus: {n} total, shard {a.shard}/{a.nshards} -> [{lo}:{hi}) = {len(part)}", flush=True)
enc = load_encoder(a.profile, checkpoint_path=a.checkpoint, device=a.device, max_len=512)
_t0 = time.time()
emb = encode_corpus(enc, part, a.bs, out)
_dt = time.time() - _t0
print(f"DONE {tuple(emb.shape)} {out} encode={_dt:.0f}s ({len(part)/max(_dt,1e-9):.0f} passages/s)", flush=True)
