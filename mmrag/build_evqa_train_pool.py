"""Data-mix experiment build: E-VQA TRAIN pool (+ its gold-article corpus) for mixed-dataset RL/SFT.

Mirrors build_evqa_data.py but over EchoSight's cleaned E-VQA train CSV, emitting TRAINING rows
(pool format: qid,image_id,question,answer,answer_aliases,entity_url,pos) with evidence-section
positives. pids offset by 20M (no collision with InfoSeek 0.. / E-VQA-test 10M..).

Outputs: built/corpus_evqa_train.jsonl, built/pool_evqa_train.jsonl
Then:    cat corpus_small.jsonl corpus_evqa_train.jsonl > corpus_mix.jsonl
         interleave pool_train.jsonl pool_evqa_train.jsonl > pool_mix.jsonl (done here)

    python3 mmrag/build_evqa_train_pool.py --data_dir $MMRAG_DATA --max_rows 25000 --per_entity 3
"""

import argparse
import csv
import json
import os
import random
import sys
import zipfile
from collections import defaultdict

import ijson

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.build_infoseek_data import chunk_article, norm  # noqa: E402

csv.field_size_limit(sys.maxsize)
PID0 = 20_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_rows", type=int, default=25000)
    ap.add_argument("--per_entity", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    D = args.data_dir
    out = os.path.join(D, "built")

    rows = list(csv.DictReader(open(os.path.join(D, "raw_evqa", "evqa_train.csv"))))
    rows = [r for r in rows if r.get("question_type") in ("templated", "automatic", "multi_answer", "infoseek")]
    by_ent = defaultdict(list)
    for r in rows:
        by_ent[r["wikipedia_url"]].append(r)
    picked = []
    for url, rs in by_ent.items():
        rng.shuffle(rs)
        picked.extend(rs[: args.per_entity])
    rng.shuffle(picked)
    picked = picked[: args.max_rows]
    gold_urls = {r["wikipedia_url"] for r in picked}
    print(f"train rows kept: {len(picked)} over {len(gold_urls)} entities", flush=True)

    id2name = {}
    for f in ("train_id2name.json", "val_id2name.json"):
        p = os.path.join(D, "raw_evqa", f)
        if os.path.exists(p):
            id2name.update(json.load(open(p)))

    pid = PID0
    url_arts, url_pids = {}, {}
    with open(os.path.join(out, "corpus_evqa_train.jsonl"), "w") as fout, \
         open(os.path.join(D, "raw_evqa", "encyclopedic_kb_wiki.zip"), "rb") as fz:
        kb_file = zipfile.ZipFile(fz).open("encyclopedic_kb_wiki.json")
        n = 0
        for url, art in ijson.kvitems(kb_file, ""):
            n += 1
            if url in gold_urls:
                ps = chunk_article(url, art, pid)
                if ps:
                    url_arts[url] = art
                    url_pids[url] = [p["pid"] for p in ps]
                    for p in ps:
                        fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                    pid += len(ps)
            if n % 400000 == 0:
                print(f"  scanned {n}; gold found {len(url_pids)}", flush=True)
    print(f"corpus_evqa_train: {pid - PID0} passages / {len(url_pids)} articles", flush=True)

    n_out = 0
    with open(os.path.join(out, "pool_evqa_train.jsonl"), "w") as f:
        for i, r in enumerate(picked):
            url = r["wikipedia_url"]
            if url not in url_pids:
                continue
            iid = (r["dataset_image_ids"].split("|")[0] or "").strip()
            image_key = id2name.get(iid) or (iid if r["dataset_name"] == "landmarks" else None)
            if not image_key:
                continue
            answers = [a for a in r["answer"].split("|") if a] or [r["answer"]]
            ps = chunk_article(url, url_arts[url], url_pids[url][0])
            sec_titles = {s.strip() for s in (r.get("evidence_section_title") or "").split("|") if s.strip()}
            gold_ps = [p for p in ps if p["section"] in sec_titles]
            if not gold_ps:
                ans_n = [norm(a) for a in answers[:3] if norm(a)]
                gold_ps = [p for p in ps if any(a in norm(p["text"]) for a in ans_n)] or ps[:1]
            f.write(json.dumps({
                "qid": f"evqa_train_{i:06d}", "image_id": image_key, "question": r["question"],
                "answer": answers[0], "answer_aliases": answers, "entity_url": url,
                "pos": [gold_ps[0]["text"]], "pos_pid": gold_ps[0]["pid"],
            }, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"pool_evqa_train: {n_out} rows", flush=True)

    # mixed corpus + interleaved mixed pool
    with open(os.path.join(out, "corpus_mix.jsonl"), "w") as f:
        for src in ("corpus_small.jsonl", "corpus_evqa_train.jsonl"):
            for line in open(os.path.join(out, src)):
                f.write(line)
    a = open(os.path.join(out, "pool_train.jsonl")).readlines()
    b = open(os.path.join(out, "pool_evqa_train.jsonl")).readlines()
    rng.shuffle(a); rng.shuffle(b)
    k = min(len(a), len(b))
    with open(os.path.join(out, "pool_mix.jsonl"), "w") as f:
        for x, y in zip(a[:k], b[:k]):
            f.write(x); f.write(y)
    print(f"corpus_mix + pool_mix ({2*k} rows, 50/50) written", flush=True)


if __name__ == "__main__":
    main()
