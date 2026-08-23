"""Curate pool_div_1M — the canonical DIVERSE training pool (mmrag-ema campaign, step 1).

Composition (~1.0M rows, fixed, pre-shuffled; every run subsamples head-N so composition
is constant across the whole scaling axis):
  - InfoSeek        ~564k  (subsampled from pool_train_1M.jsonl)
  - E-VQA landmarks ~250k  (subsampled from the expanded rows inside pool_mix_1M1.jsonl)
  - OVEN train      ~177k  (M2KR; entity name is the answer; dirs 01-04 shards -> grows to
                            339k if shard00 is added later; rows filtered by on-disk images)
  - OKVQA train        9k  (M2KR; 10-annotator answer lists as aliases; COCO train2014 zip)

Corpus: corpus_div.jsonl = corpus_mix.jsonl (InfoSeek + E-VQA gold articles)
  + OVEN train passages  (pid offset 40M)
  + OKVQA train passages (pid offset 50M)
PID ranges stay disjoint with the InfoSeek (0..) / E-VQA-test (10M..) / E-VQA-train (20M..)
conventions, so eval corpora and pools remain mutually consistent.

Rows follow the standard pool schema; `pos`/`pos_pid` are populated where the source maps a
gold passage (OVEN pos_item_ids, OKVQA absent -> best alias-bearing passage else empty).
v3-pure (--no_force_gold) never consumes `pos`; it exists for diagnostics and SFT parity.

    python3 mmrag/build_div_mix.py --data_dir $MMRAG_DATA \
        --infoseek 564000 --evqa 250000 --oven 400000 --okvqa 999999 --seed 0
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.build_infoseek_data import norm  # noqa: E402

OVEN_PID0 = 40_000_000
OKVQA_PID0 = 50_000_000


def hf(repo, f):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, f, repo_type="dataset")


M2KR = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"


def oven_rows(D, cap, rng):
    import pyarrow.parquet as pq
    # image availability = union of the shard tar indexes present on disk
    from mmrag.image_store import build_tar_index
    avail = set()
    for i in range(6):
        t = os.path.join(D, "images/OVEN", f"shard{i:02d}.tar")
        if os.path.exists(t):
            avail.update(k.lstrip("./") for k in build_tar_index(t))
    print(f"[oven] {len(avail)} images on disk", flush=True)
    # passages -> corpus (only those referenced by kept rows, resolved after row pick)
    dp = hf(M2KR, "OVEN_data/train-00000-of-00001.parquet")
    rows = pq.read_table(dp).to_pylist()
    rng.shuffle(rows)
    kept, need_pids = [], set()
    for r in rows:
        if len(kept) >= cap:
            break
        if r["img_path"] not in avail:
            continue
        ent = (r["wiki_entity"] or "").strip()
        if not ent:
            continue
        pids = list(r.get("pos_item_ids") or [])
        kept.append((r, ent, pids))
        need_pids.update(pids)
    pp = hf(M2KR, "OVEN_passages/train_passages-00000-of-00001.parquet")
    ptab = pq.read_table(pp).to_pylist()
    id2new, passages = {}, []
    for j, p in enumerate(ptab):
        pid_str = str(p.get("passage_id") or p.get("id") or j)
        if pid_str in need_pids or not need_pids:
            id2new[pid_str] = OVEN_PID0 + j
            passages.append({"pid": OVEN_PID0 + j, "url": f"oven://{pid_str}",
                             "title": (p.get("page_title") or "")[:200], "section": "",
                             "text": p.get("passage_content") or p.get("passage") or ""})
    out = []
    for i, (r, ent, pids) in enumerate(kept):
        mapped = [id2new[p] for p in pids if p in id2new]
        pos_txt = next((x["text"] for x in passages if mapped and x["pid"] == mapped[0]), "")
        out.append({"qid": f"oven_{i:06d}", "image_id": r["img_path"],
                    "question": r["question"] or f"what is shown in this image?",
                    "answer": ent, "answer_aliases": [ent],
                    "entity_url": f"oven://{r.get('wiki_entity_id','')}",
                    "pos": [pos_txt] if pos_txt else [], "pos_pid": mapped[0] if mapped else -1})
    print(f"[oven] rows kept {len(out)}, passages {len(passages)}", flush=True)
    return out, passages


def okvqa_rows(D, cap, rng):
    import pyarrow.parquet as pq
    dp = hf(M2KR, "OKVQA_data/train-00000-of-00001.parquet")
    rows = pq.read_table(dp).to_pylist()
    pp = hf(M2KR, "OKVQA_passages/train_passages-00000-of-00001.parquet")
    ptab = pq.read_table(pp).to_pylist()
    passages = []
    for j, p in enumerate(ptab):
        passages.append({"pid": OKVQA_PID0 + j, "url": f"okvqa://{j}",
                         "title": "", "section": "",
                         "text": p.get("passage_content") or p.get("passage") or ""})
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows[:cap]):
        answers = [a for a in dict.fromkeys(r.get("answers") or []) if a]
        gold = r.get("gold_answer") or (answers[0] if answers else "")
        if not gold:
            continue
        # no pos_item_ids in this source: best-effort alias-bearing passage for diagnostics
        an = [norm(a) for a in answers[:3] if norm(a)]
        pos = next((p for p in passages[:5000] if any(a in norm(p["text"]) for a in an)), None)
        out.append({"qid": f"okvqa_{i:05d}", "image_id": r["img_file_name"],
                    "question": r["question"], "answer": gold,
                    "answer_aliases": answers or [gold], "entity_url": "okvqa://",
                    "pos": [pos["text"]] if pos else [], "pos_pid": pos["pid"] if pos else -1})
    print(f"[okvqa] rows kept {len(out)}, passages {len(passages)}", flush=True)
    return out, passages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA"))
    ap.add_argument("--infoseek", type=int, default=564000)
    ap.add_argument("--evqa", type=int, default=250000)
    ap.add_argument("--oven", type=int, default=400000)
    ap.add_argument("--okvqa", type=int, default=999999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    D = args.data_dir
    rng = random.Random(args.seed)

    inf = [json.loads(l) for l in open(f"{D}/built/pool_train_1M.jsonl")]
    rng.shuffle(inf)
    inf = inf[: args.infoseek]
    ev = [r for r in (json.loads(l) for l in open(f"{D}/built/pool_mix_1M1.jsonl"))
          if str(r.get("qid", "")).startswith("evqa_train_")]
    rng.shuffle(ev)
    ev = ev[: args.evqa]
    ov, ov_pass = oven_rows(D, args.oven, rng)
    ok, ok_pass = okvqa_rows(D, args.okvqa, rng)

    allrows = inf + ev + ov + ok
    rng.shuffle(allrows)
    with open(f"{D}/built/pool_div_1M.jsonl", "w") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{D}/built/corpus_div.jsonl", "w") as f:
        for line in open(f"{D}/built/corpus_mix.jsonl"):
            f.write(line)
        for p in ov_pass + ok_pass:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    comp = {"infoseek": len(inf), "evqa_landmarks": len(ev), "oven": len(ov),
            "okvqa": len(ok), "total": len(allrows), "seed": args.seed}
    json.dump(comp, open(f"{D}/built/pool_div_1M.composition.json", "w"), indent=1)
    print("DIV_MIX_DONE", comp, flush=True)


if __name__ == "__main__":
    main()
