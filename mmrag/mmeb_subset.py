"""Sample a reproducible random subset of MMEB(-V2) image tasks and evaluate a gme2b encoder
(zero-shot or an mmrag LoRA adapter) on it with MMEB's Precision@1.

Design:
  * QUERIES are subsampled (--per_task per dataset, seed-fixed); the CANDIDATE POOL is the full
    deduped pool of that dataset (optionally capped, positives always kept) so task difficulty
    is preserved -- only the number of queries shrinks.
  * The sampled subset is written to disk (subset.json manifest + per-task query indices) so the
    exact same queries can be re-evaluated for any other checkpoint.

    python3 mmeb_subset.py sample --out <dir> --per_task 100
    python3 mmeb_subset.py eval   --subset <dir> --checkpoint __profile__ --name gme2b-zs
"""
import argparse
import itertools
import json
import os
import random
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MMEB_DIR = os.environ.get("MMEB_DIR",
    "/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmeb_eval")  # HF TIGER-Lab/MMEB-eval snapshot
# Images live inside images.zip as "<Task>/<file>.jpg". Extracting all 285K files onto this NFS
# runs at ~1 GB/10 min, so we read them straight out of the zip's central directory instead.
IMG_ZIP = os.path.join(MMEB_DIR, "images.zip")
IMG_TOKEN = "<|image_1|>"


def task_dirs():
    return sorted(d for d in os.listdir(MMEB_DIR)
                  if os.path.isdir(os.path.join(MMEB_DIR, d)) and d != "images"
                  and any(f.endswith(".parquet") for f in os.listdir(os.path.join(MMEB_DIR, d))))


def load_task(task):
    d = os.path.join(MMEB_DIR, task)
    parts = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    return pd.concat([pd.read_parquet(os.path.join(d, p)) for p in parts], ignore_index=True)


def clean(t):
    """MMEB text -> plain text (image presence is carried separately)."""
    return (t or "").replace(IMG_TOKEN, "").strip()


# ----------------------------------------------------------------------------- sample
def do_sample(a):
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    manifest = {"seed": a.seed, "per_task": a.per_task, "max_pool": a.max_pool, "tasks": {}}
    for task in task_dirs():
        df = load_task(task)
        n = len(df)
        idx = sorted(rng.sample(range(n), min(a.per_task, n)))

        # full deduped candidate pool of the WHOLE task (difficulty preserved).
        # chain over the raw object arrays -- df.iterrows() is far too slow on OVEN-sized tasks.
        pool, seen = [], {}
        for t, p in zip(itertools.chain.from_iterable(df["tgt_text"].values),
                        itertools.chain.from_iterable(df["tgt_img_path"].values)):
            k = (clean(t), p or "")
            if k not in seen:
                seen[k] = len(pool)
                pool.append({"text": k[0], "img": p or None})
        # positives of the sampled queries (always retained if the pool gets capped)
        queries = []
        for i in idx:
            r = df.iloc[i]
            gold = seen[(clean(r["tgt_text"][0]), r["tgt_img_path"][0] or "")]
            queries.append({"row": int(i), "text": clean(r["qry_text"]),
                            "img": r["qry_img_path"] or None, "gold": gold})
        if a.max_pool and len(pool) > a.max_pool:
            keep = {q["gold"] for q in queries}
            rest = [j for j in range(len(pool)) if j not in keep]
            rng.shuffle(rest)
            keep_list = sorted(keep | set(rest[: max(0, a.max_pool - len(keep))]))
            remap = {old: new for new, old in enumerate(keep_list)}
            pool = [pool[j] for j in keep_list]
            for q in queries:
                q["gold"] = remap[q["gold"]]
            capped = True
        else:
            capped = False
        with open(os.path.join(a.out, f"{task}.json"), "w") as f:
            json.dump({"task": task, "n_total": n, "queries": queries, "pool": pool}, f)
        manifest["tasks"][task] = {"n_total": n, "n_queries": len(queries),
                                   "pool_size": len(pool), "pool_capped": capped}
        print(f"{task:24s} {len(queries):4d} queries / {n:6d}   pool {len(pool):6d}"
              f"{' (capped)' if capped else ''}", flush=True)
    with open(os.path.join(a.out, "subset.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    tot = sum(t["n_queries"] for t in manifest["tasks"].values())
    print(f"\nSAMPLED {tot} queries over {len(manifest['tasks'])} tasks -> {a.out}")


# ------------------------------------------------------------------------------- eval
def do_eval(a):
    from PIL import Image

    from mmrag.encoder import (GME_DEFAULT_INSTRUCTION, GME_IMAGE_TOKENS, GME_TEMPLATE,
                               load_encoder)

    manifest = json.load(open(os.path.join(a.subset, "subset.json")))
    tasks = sorted(manifest["tasks"]) if not a.tasks else a.tasks
    enc = load_encoder("gme2b", checkpoint_path=a.checkpoint, device=a.device, max_len=a.max_len)

    def fmt(text, has_img):
        return GME_TEMPLATE.format(instr=GME_DEFAULT_INSTRUCTION,
                                   content=(GME_IMAGE_TOKENS if has_img else "") + text)

    import io
    import zipfile

    zf = zipfile.ZipFile(IMG_ZIP)
    have = set(zf.namelist())

    def get_img(p):
        if not p or p not in have:
            return None
        im = Image.open(io.BytesIO(zf.read(p))).convert("RGB")
        w, h = im.size
        if min(w, h) < 28:
            s = 28 / min(w, h)
            im = im.resize((max(28, int(w * s)), max(28, int(h * s))))
        elif max(w, h) > 1024:
            s = 1024 / max(w, h)
            im = im.resize((max(28, int(w * s)), max(28, int(h * s))))
        return im

    def encode(items, bs):
        out = []
        for i in range(0, len(items), bs):
            ch = items[i: i + bs]
            texts = [fmt(c["text"], bool(c["img"])) for c in ch]
            imgs = [get_img(c["img"]) for c in ch]
            out.append(enc.encode(texts, imgs, batch_size=len(ch), to_cpu=True,
                                  max_length=a.max_len))
            if (i // bs) % 20 == 0:
                print(f"    {i + len(ch)}/{len(items)}", flush=True)
        return torch.cat(out)

    results = {}
    for task in tasks:
        d = json.load(open(os.path.join(a.subset, f"{task}.json")))
        qs, pool = d["queries"], d["pool"]
        print(f"[{task}] {len(qs)} queries, pool {len(pool)}", flush=True)
        try:
            p_emb = encode(pool, a.bs).to(a.device, torch.float16)
            q_emb = encode(qs, a.bs).to(a.device, torch.float16)
        except Exception as ex:                      # never lose the whole sweep to one task
            print(f"[{task}] SKIPPED: {type(ex).__name__}: {ex}", flush=True)
            results[task] = {"error": f"{type(ex).__name__}: {ex}"[:200]}
            torch.cuda.empty_cache()
            continue
        pred = (q_emb @ p_emb.T).argmax(dim=1).cpu().tolist()
        hit = sum(int(p == q["gold"]) for p, q in zip(pred, qs)) / len(qs)
        results[task] = {"prec@1": round(hit, 4), "n": len(qs), "pool": len(pool)}
        print(f"[{task}] Precision@1 = {hit:.4f}", flush=True)
        del p_emb, q_emb
        torch.cuda.empty_cache()

    ok = {k: v for k, v in results.items() if "prec@1" in v}
    macro = sum(v["prec@1"] for v in ok.values()) / max(len(ok), 1)
    out = {"name": a.name, "checkpoint": a.checkpoint, "subset": a.subset,
           "macro_prec@1": round(macro, 4), "per_task": results}
    fp = os.path.join(a.subset, f"results.{a.name}.json")
    json.dump(out, open(fp, "w"), indent=1)
    print(f"\n[{a.name}] MMEB subset macro Precision@1 = {macro:.4f} over {len(ok)}/{len(results)} tasks")
    print(f"wrote {fp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--per_task", type=int, default=100)
    s.add_argument("--max_pool", type=int, default=20000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=do_sample)
    e = sub.add_parser("eval")
    e.add_argument("--subset", required=True)
    e.add_argument("--checkpoint", default="__profile__")
    e.add_argument("--name", required=True)
    e.add_argument("--tasks", nargs="*", default=None)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--bs", type=int, default=32)
    e.add_argument("--max_len", type=int, default=2048)
    e.set_defaults(fn=do_eval)
    a = ap.parse_args()
    a.fn(a)
