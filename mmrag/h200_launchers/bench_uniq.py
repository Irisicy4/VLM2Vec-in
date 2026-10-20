"""Discriminate two accounts of the measured 37.5 s/step, against real data.

(a) len(uniq) is lower than simulated      (b) 7.2-9.1 s/call was inflated by 1440-char filler

Scored against the RECONSTRUCTION, not against a band: every term is measured here, so
  ceil(uniq/8) * per_call + EMA_write + doc_encode + query_encode  vs  the measured 37.5 s.
Also settles the subsample-bias question directly (encode once, subset the embeddings) instead
of arguing about which of two competing effects dominates.
"""
import json, os, random, sys, time
import torch
sys.path.insert(0, "/raid/icy/VLM2Vec-in")
os.environ.setdefault("MMRAG_DATA", "/raid/icy/mmrag_data")
D = os.environ["MMRAG_DATA"]
from mmrag.encoder import load_encoder
from mmrag.image_store import open_stores
from mmrag.rl_core import pl_sample_lists
from mmrag.reader_vlm import LiveVLMReader

NBIG = int(os.environ.get("NBIG", "120000"))
NQ, TEMP, SUPPORT, GROUP, K = 16, 0.02, 24, 4, 4
MEASURED_STEP = 37.5

corpus = []
for _, l in zip(range(NBIG), open(f"{D}/built/corpus_small.jsonl")):
    corpus.append(json.loads(l))
texts = [c["text"] for c in corpus]
print(f"corpus: {len(texts)} passages, mean {sum(len(t) for t in texts)/len(texts):.0f} chars", flush=True)

rows = [json.loads(l) for _, l in zip(range(NQ), open(f"{D}/built/pool_train_1M.jsonl"))]
store = open_stores(D, "infoseek")
imgs = [store.get(r["image_id"]) for r in rows]
CKPT = os.environ.get("CKPT", "__profile__")
print(f"encoder checkpoint: {CKPT}", flush=True)
enc = load_encoder("gme2b", checkpoint_path=CKPT, device="cuda:0", max_len=512)

t0 = time.time()
with torch.no_grad():
    d_emb = enc.encode_docs(texts, batch_size=64).float()
enc_dt = time.time() - t0
print(f"ENCODE {len(texts)} docs in {enc_dt:.0f}s = {len(texts)/enc_dt:.1f} passages/s", flush=True)
with torch.no_grad():
    q_emb = enc.encode_queries([r["question"] for r in rows], imgs, batch_size=4).float()

def uniq_at(n):
    sims = q_emb @ d_emb[:n].T
    top = sims.topk(SUPPORT, dim=1)
    scaled = (top.values / TEMP).to("cuda:0")
    spread = (top.values[:, 0] - top.values[:, -1]).mean().item()
    cs = []
    for _ in range(400):
        idx = pl_sample_lists(scaled, k=K, n_lists=GROUP)
        cs.append(sum(len(set(idx[b].flatten().tolist())) for b in range(idx.shape[0])) / idx.shape[0])
    return spread, sum(cs) / len(cs)

print("\n--- subsample-bias check: same embeddings, growing index ---", flush=True)
res = {}
for n in [40000, 80000, NBIG]:
    if n > len(texts): continue
    sp, mu = uniq_at(n)
    res[n] = mu
    print(f"  index {n:>7}: top-24 cosine spread {sp:.4f}  unique/query {mu:5.2f}  "
          f"-> {4*mu:5.1f} triples/step, {int(-(-4*mu//8))} generate calls", flush=True)
if len(res) > 1:
    ks = sorted(res)
    print(f"  BIAS DIRECTION as index grows {ks[0]}->{ks[-1]}: unique/query "
          f"{res[ks[0]]:.2f} -> {res[ks[-1]]:.2f} "
          f"({'UP' if res[ks[-1]]>res[ks[0]] else 'DOWN' if res[ks[-1]]<res[ks[0]] else 'FLAT'})", flush=True)

mu = res[max(res)]
ntri = max(8, int(round(4 * mu)))
ncalls = int(-(-ntri // 8))

print("\n--- per-step encoder costs, measured ---", flush=True)
with torch.no_grad():
    for _ in range(2):
        torch.cuda.synchronize(); t = time.time()
        enc.encode_docs(texts[:96], batch_size=64)
        torch.cuda.synchronize(); doc96 = time.time() - t
    print(f"  doc encode 96 (pl_support 24 x batch 4): {doc96:.2f}s", flush=True)
    for _ in range(2):
        torch.cuda.synchronize(); t = time.time()
        enc.encode_queries([r["question"] for r in rows[:4]], imgs[:4], batch_size=4)
        torch.cuda.synchronize(); q4 = time.time() - t
    print(f"  query encode 4 (with images): {q4:.2f}s", flush=True)
    for _ in range(2):
        torch.cuda.synchronize(); t = time.time()
        enc.encode_docs(texts[:256], batch_size=64)
        torch.cuda.synchronize(); ema256 = time.time() - t
    print(f"  EMA write-back encode 256: {ema256:.2f}s", flush=True)
del d_emb; torch.cuda.empty_cache()

print("\n--- reader at TRUE shape with REAL passages ---", flush=True)
picks = [texts[i] for i in random.Random(0).sample(range(len(texts)), ntri)]
triples = [{"image": imgs[i % NQ], "image_id": rows[i % NQ]["image_id"],
            "question": rows[i % NQ]["question"], "context": picks[i],
            "answer": rows[i % NQ].get("answer_aliases") or rows[i % NQ]["answer"]}
           for i in range(ntri)]
print(f"  {ntri} triples, mean ctx {sum(len(t['context']) for t in triples)/ntri:.0f} chars "
      f"(earlier filler ~1440), {ncalls} generate calls", flush=True)
rd = LiveVLMReader("Qwen/Qwen2.5-VL-3B-Instruct", device="cuda:0", batch_size=16)
best = []
for t_ in range(3):
    torch.cuda.synchronize(); t = time.time()
    rd.score_judge(triples, n_rollouts=4, temperature=0.7)
    torch.cuda.synchronize(); el = time.time() - t
    best.append(el)
    print(f"  trial {t_}: {el:.2f}s total = {el/ncalls:.2f}s per generate call", flush=True)
rdr = sum(best) / len(best)

print("\n=== RECONSTRUCTION vs the measured 37.5 s/step ===", flush=True)
total = rdr + ema256 + doc96 + q4
for k, v in [("reader", rdr), ("EMA write 256", ema256), ("doc encode 96", doc96), ("query encode 4", q4)]:
    print(f"  {k:<16} {v:6.2f}s  ({100*v/total:4.1f}%)", flush=True)
print(f"  {'TOTAL':<16} {total:6.2f}s  vs measured {MEASURED_STEP}s "
      f"-> {100*(total-MEASURED_STEP)/MEASURED_STEP:+.1f}%", flush=True)
print(f"  VERDICT: {'reconstructs within 20%' if abs(total-MEASURED_STEP)/MEASURED_STEP<=0.20 else 'DOES NOT reconstruct'}", flush=True)
