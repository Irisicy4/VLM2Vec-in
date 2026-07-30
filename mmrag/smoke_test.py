"""M0 gate smoke test (GPU): exercises every component end-to-end on tiny inputs.

  1. TarImageStore: index the val tar, load a few query images
  2. Encoder zero-shot: encode 300 passages + 24 image-queries, top-k sanity (gold-entity hit rate
     should be far above chance)
  3. Reader: score_logit + score_judge on gold vs random context (gold should score higher)

Run inside a SLURM job (compute nodes are offline: HF_HUB_OFFLINE=1).
    python3 mmrag/smoke_test.py
"""

import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.encoder import MMRagEncoder  # noqa: E402
from mmrag.image_store import TarImageStore  # noqa: E402
from mmrag.reader_vlm import LiveVLMReader  # noqa: E402

D = os.environ.get("MMRAG_DATA", "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data")
DEV = "cuda:0"
rng = random.Random(0)

print("=== 1. image store", flush=True)
store = TarImageStore([os.path.join(D, "images/Infoseek/infoseek_val_images.tar")])
print(f"indexed {len(store)} images", flush=True)

queries = [json.loads(l) for l in open(os.path.join(D, "built/queries_test.jsonl"))]
queries = [q for q in queries if q["image_id"] in store]
print(f"test queries with image: {len(queries)}", flush=True)
qs = rng.sample(queries, 24)
imgs = [store.get(q["image_id"]) for q in qs]
print("sample image sizes:", [im.size for im in imgs[:4]], flush=True)

print("=== 2. encoder zero-shot sanity", flush=True)
# small corpus: all passages of the 24 gold entities + 200 distractor passages
gold_urls = {q["entity_url"] for q in qs}
corpus = []
with open(os.path.join(D, "built/corpus_small.jsonl")) as f:
    for line in f:
        p = json.loads(line)
        if p["url"] in gold_urls:
            corpus.append(p)
distract = []
with open(os.path.join(D, "built/corpus_small.jsonl")) as f:
    for i, line in enumerate(f):
        if i % 997 == 0:
            p = json.loads(line)
            if p["url"] not in gold_urls:
                distract.append(p)
corpus += distract[:300]
print(f"mini corpus: {len(corpus)} passages", flush=True)

enc = MMRagEncoder("Qwen/Qwen2-VL-2B-Instruct", checkpoint_path="TIGER-Lab/VLM2Vec-Qwen2VL-2B",
                   device=DEV, max_len=512)
p_emb = enc.encode_docs([p["text"] for p in corpus], batch_size=32)
q_emb = enc.encode_queries([q["question"] for q in qs], imgs, batch_size=8)
print("emb shapes:", tuple(q_emb.shape), tuple(p_emb.shape), "norms:",
      q_emb.norm(dim=-1).mean().item(), p_emb.norm(dim=-1).mean().item(), flush=True)
sims = q_emb.float() @ p_emb.float().T
top5 = sims.topk(5, dim=1).indices.tolist()
hit = sum(any(corpus[i]["url"] == q["entity_url"] for i in t) for q, t in zip(qs, top5))
print(f"gold-entity in top-5: {hit}/24 (chance would be ~{24*5*len([c for c in corpus if c['url'] in gold_urls])//len(corpus)/24:.1f})", flush=True)
for j in range(2):
    print(" Q:", qs[j]["question"][:80], "| top1:", corpus[top5[j][0]]["title"][:60],
          "| gold:", qs[j]["entity_title"][:60], flush=True)
assert hit >= 8, f"zero-shot top-5 entity hit rate too low: {hit}/24"

del enc, p_emb, q_emb
torch.cuda.empty_cache()

print("=== 3. reader sanity", flush=True)
reader = LiveVLMReader("Qwen/Qwen2.5-VL-3B-Instruct", device=DEV, batch_size=4)
pid2text = {p["pid"]: p["text"] for p in corpus}
triples_gold, triples_rand = [], []
for q in qs[:8]:
    gold_pid = (q["ans_pids"] or q["gold_pids"])[0]
    gold_txt = pid2text.get(gold_pid)
    if gold_txt is None:
        continue
    im = store.get(q["image_id"])
    triples_gold.append({"image": im, "question": q["question"], "context": gold_txt, "answer": q["answer"]})
    triples_rand.append({"image": im, "question": q["question"],
                         "context": rng.choice(distract)["text"], "answer": q["answer"]})
lg = reader.score_logit(triples_gold)
lr = reader.score_logit(triples_rand)
print("logit gold:", [f"{x:.2f}" for x in lg], flush=True)
print("logit rand:", [f"{x:.2f}" for x in lr], flush=True)
better = sum(g > r for g, r in zip(lg, lr))
print(f"gold>rand: {better}/{len(lg)}", flush=True)
jg, preds = reader.score_judge(triples_gold, n_rollouts=1, temperature=0.0, return_preds=True)
print("judge gold acc:", sum(jg) / len(jg), "preds:", [p[:30] for p in preds[:4]],
      "answers:", [t["answer"][:30] for t in triples_gold[:4]], flush=True)

print("SMOKE TEST PASSED", flush=True)
