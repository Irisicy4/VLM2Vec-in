"""OOD eval data build: Encyclopedic-VQA test set + a corpus carved from the 2M-article wiki KB.

E-VQA (unlike our InfoSeek CSVs) ships true evidence sections, so gold passages here are the
passages chunked from the row's evidence section (fallback: whole gold article = entity-level).

Corpus scope: all passages of the test rows' gold entities + --distractor_articles random articles
(the full 2M KB is out of scope for the OOD check; we report the corpus size with the results).
The KB json (~19GB raw) is stream-parsed with ijson to dodge the login-node OOM killer.

Outputs (packed): built/corpus_evqa.jsonl, built/queries_evqa.jsonl

    python3 mmrag/build_evqa_data.py --data_dir $SCRATCH/mmrag_data --distractor_articles 25000
"""

import argparse
import csv
import json
import os
import random
import sys

import ijson

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.build_infoseek_data import chunk_article, norm  # noqa: E402

csv.field_size_limit(sys.maxsize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--distractor_articles", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    D = args.data_dir
    out_dir = os.path.join(D, "built")
    os.makedirs(out_dir, exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(D, "raw_evqa", "evqa_test.csv"))))
    # multi-answer rows use '|' separators; two-hop questions have '|'-joined sub-questions — keep
    # the standard single-hop subset (mirrors EchoSight's main table)
    rows = [r for r in rows if r["question_type"] in ("templated", "automatic", "multi_answer", "infoseek")]
    gold_urls = {r["wikipedia_url"] for r in rows}
    print(f"E-VQA test rows kept: {len(rows)}; gold entities: {len(gold_urls)}", flush=True)

    # image id -> archive key (iNat numeric ids via id2name; landmarks use the id as filename)
    id2name = {}
    for f in ("train_id2name.json", "val_id2name.json"):
        p = os.path.join(D, "raw_evqa", f)
        if os.path.exists(p):
            id2name.update(json.load(open(p)))
    print(f"inat id2name entries: {len(id2name)}", flush=True)

    # ---- stream the 2M KB; write gold articles' passages now, reservoir-sample distractors ----
    corpus_path = os.path.join(out_dir, "corpus_evqa.jsonl")
    url_pids, url_arts = {}, {}
    reservoir = []  # (url, art) distractor candidates
    n_seen = 0
    pid = 10_000_000  # offset so pids never collide with the InfoSeek corpora
    with open(corpus_path, "w") as fout, open(os.path.join(D, "raw_evqa", "encyclopedic_kb_wiki.zip"), "rb") as fz:
        import zipfile

        kb_file = zipfile.ZipFile(fz).open("encyclopedic_kb_wiki.json")
        for url, art in ijson.kvitems(kb_file, ""):
            n_seen += 1
            if url in gold_urls:
                passages = chunk_article(url, art, pid)
                if passages:
                    url_pids[url] = [p["pid"] for p in passages]
                    url_arts[url] = art
                    for p in passages:
                        fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                    pid += len(passages)
            else:
                if len(reservoir) < args.distractor_articles:
                    reservoir.append((url, art))
                else:
                    j = rng.randrange(n_seen)
                    if j < args.distractor_articles:
                        reservoir[j] = (url, art)
            if n_seen % 200000 == 0:
                print(f"  scanned {n_seen} articles; gold found {len(url_pids)}", flush=True)
        for url, art in reservoir:
            passages = chunk_article(url, art, pid)
            for p in passages:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
            pid += len(passages)
    print(f"corpus_evqa: gold articles {len(url_pids)}/{len(gold_urls)}, + {len(reservoir)} distractors, "
          f"{pid - 10_000_000} passages total", flush=True)

    # ---- queries: gold passages from the evidence section (fallback entity-level) ----
    n_img_miss = 0
    with open(os.path.join(out_dir, "queries_evqa.jsonl"), "w") as f:
        n_out = 0
        for i, r in enumerate(rows):
            url = r["wikipedia_url"]
            if url not in url_pids:
                continue
            img_ids = [x.strip() for x in r["dataset_image_ids"].split("|") if x.strip()]
            if not img_ids:
                continue
            # archive key: iNat numeric id -> member path; landmark hex id -> <id>.jpg basename
            iid = img_ids[0]
            if iid in id2name:
                image_key = id2name[iid]
            elif r["dataset_name"] == "landmarks":
                image_key = iid  # matched via key_fn=basename-without-ext on the landmarks tar
            else:
                n_img_miss += 1
                continue
            answers = [a for a in r["answer"].split("|") if a]
            gold_pids = url_pids[url]
            ans_pids = []
            ev = norm(r.get("evidence") or "")
            art = url_arts.get(url)
            if ev and art:
                ps = chunk_article(url, art, gold_pids[0])
                # passages overlapping the evidence text (or containing an answer)
                ans_pids = [p["pid"] for p in ps
                            if ev[:120] and ev[:120] in norm(p["text"])
                            or any(norm(a) and norm(a) in norm(p["text"]) for a in answers[:3])]
            f.write(json.dumps({
                "qid": f"evqa_test_{i:06d}", "image_id": image_key, "question": r["question"],
                "answer": answers[0], "answer_aliases": answers,
                "entity_url": url, "entity_title": r["wikipedia_title"],
                "gold_pids": gold_pids, "ans_pids": ans_pids,
                "dataset_name": r["dataset_name"],
            }, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"queries_evqa: {n_out} written; image-id misses: {n_img_miss}", flush=True)


if __name__ == "__main__":
    main()
