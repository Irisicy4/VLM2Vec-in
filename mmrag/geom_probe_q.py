"""Query-side geometry probe (text-side exchange, round 2): mean pairwise cosine of a
fixed seeded 1,000-query sample (image+question) of queries_test, encoded by base /
v3pure / emaidx / emaenc. Doc-side twin: geom_probe.py."""
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from mmrag.encoder import load_encoder            # noqa: E402
from mmrag.eval_retrieval import load_jsonl       # noqa: E402
from mmrag.image_store import open_stores         # noqa: E402

D = os.environ["MMRAG_DATA"]
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"

store = open_stores(D, "infoseek")
queries = [q for q in load_jsonl(os.path.join(D, "built/queries_test.jsonl"))
           if q["image_id"] in store]
sample = random.Random(0).sample(queries, 1000)

arms = {
    "base": "__profile__",
    "v3pure": f"{D}/runs/div45kv3-v3pure",
    "emaidx": f"{D}/runs/div45kv3-emaidx",
    "emaenc": f"{D}/runs/div45kv3-emaenc",
}
out = {}
BS = 32
for name, ck in arms.items():
    enc = load_encoder("gme2b", checkpoint_path=ck, device=DEV, max_len=512)
    embs = []
    for i in range(0, len(sample), BS):
        chunk = sample[i:i + BS]
        imgs = [store.get(q["image_id"]) for q in chunk]
        with torch.no_grad():
            embs.append(enc.encode_queries([q["question"] for q in chunk], imgs,
                                           batch_size=BS).float().cpu())
    e = torch.nn.functional.normalize(torch.cat(embs), dim=-1)
    g = e @ e.T
    n = g.shape[0]
    out[name] = round(((g.sum() - n) / (n * (n - 1))).item(), 4)
    print(f"GEOMQ {name} mean_pairwise_cos={out[name]}", flush=True)
    del enc, embs, e, g
    torch.cuda.empty_cache()

with open(f"{D}/results/geom_probe_q_div45kv3.json", "w") as f:
    json.dump(out, f, indent=1)
print("GEOMQ_DONE", out, flush=True)
