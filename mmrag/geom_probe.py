"""Geometry probe (text-side exchange): mean pairwise cosine of a fixed seeded 2,000-doc
sample of corpus_small, encoded by base / v3pure / emaidx / emaenc. Answers whether the
EMA arms show the text side's signature (index dynamics buy spread without recall)."""
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from mmrag.encoder import load_encoder  # noqa: E402
from mmrag.eval_retrieval import encode_corpus, load_jsonl  # noqa: E402

D = os.environ["MMRAG_DATA"]
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"

corpus = load_jsonl(os.path.join(D, "built/corpus_small.jsonl"))
sample = random.Random(0).sample(corpus, 2000)

arms = {
    "base": "__profile__",
    "v3pure": f"{D}/runs/div45kv3-v3pure",
    "emaidx": f"{D}/runs/div45kv3-emaidx",
    "emaenc": f"{D}/runs/div45kv3-emaenc",
}
out = {}
for name, ck in arms.items():
    enc = load_encoder("gme2b", checkpoint_path=ck, device=DEV, max_len=512)
    emb = encode_corpus(enc, sample, 128, f"{D}/cache/geom_{name}.pt")
    e = torch.nn.functional.normalize(emb.float(), dim=-1)
    g = e @ e.T
    n = g.shape[0]
    mean_cos = (g.sum() - n) / (n * (n - 1))
    out[name] = round(mean_cos.item(), 4)
    print(f"GEOM {name} mean_pairwise_cos={out[name]}", flush=True)
    del enc, emb, e, g
    torch.cuda.empty_cache()

with open(f"{D}/results/geom_probe_div45kv3.json", "w") as f:
    json.dump(out, f, indent=1)
print("GEOM_DONE", out, flush=True)
