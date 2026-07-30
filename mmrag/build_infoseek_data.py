"""M1 data build: EchoSight-filtered InfoSeek CSVs + 100K wiki KB -> our training/eval files.

Inputs (downloaded on the login node, see mmrag/README.md):
  raw/infoseek_{train,test}_filtered.csv   EchoSight's cleaned InfoSeek splits (train / their test = InfoSeek val)
  raw/wiki_100_dict_v4.json                100K-article Wikipedia KB {url: {section_texts, section_titles, title, url}}

Outputs (all packed jsonl — inode quota!):
  corpus_full.jsonl    one line per ~PASSAGE_WORDS-word passage of every KB article {pid, url, title, section, text}
  corpus_small.jsonl   subset: all passages of train∪test gold entities + DISTRACTOR_ARTICLES random articles
  queries_train.jsonl  {qid, image_id, question, answer, entity_url, entity_title, gold_pids, ans_pids}
  queries_test.jsonl   same for test (EchoSight test = InfoSeek val)
  pool_train.jsonl     relevance pool for the SFT baseline {qid, image_id, question, answer, pos: [text], pos_pid}

gold_pids = all passages of the gold entity article (entity-level relevance — what InfoSeek/EchoSight R@k uses).
ans_pids  = the subset of gold_pids whose text contains the answer string (pseudo evidence; ~28% of train rows
have a verbatim-answer section, numbers/paraphrases don't match — so entity-level is the primary gold).

    python3 mmrag/build_infoseek_data.py --data_dir $SCRATCH/mmrag_data \
        --train_per_entity 10 --max_train 60000 --distractor_articles 20000
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

PASSAGE_WORDS = 100


def chunk_article(url, art, pid_start):
    """EchoSight-style chunking: concat title+section text, split into ~100-word passages."""
    out = []
    title = art.get("title") or url.rsplit("/", 1)[-1].replace("_", " ")
    for sec_i, (sec_title, sec_text) in enumerate(zip(art["section_titles"], art["section_texts"])):
        words = sec_text.split()
        if not words:
            continue
        for i in range(0, len(words), PASSAGE_WORDS):
            chunk = " ".join(words[i : i + PASSAGE_WORDS])
            if len(chunk.split()) < 10 and out:  # glue tiny tails onto the previous passage
                out[-1]["text"] += " " + chunk
                continue
            out.append({
                "pid": pid_start + len(out),
                "url": url,
                "title": title,
                "section": sec_title,
                "text": f"{title} ({sec_title}): {chunk}" if sec_title and sec_title != title else f"{title}: {chunk}",
            })
    return out


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_per_entity", type=int, default=10, help="cap on train queries per gold entity")
    ap.add_argument("--max_train", type=int, default=60000)
    ap.add_argument("--max_test", type=int, default=0, help="0 = keep all test rows")
    ap.add_argument("--distractor_articles", type=int, default=20000,
                    help="random non-gold KB articles added to corpus_small")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    D = args.data_dir
    out_dir = os.path.join(D, "built")
    os.makedirs(out_dir, exist_ok=True)

    print("loading KB ...", flush=True)
    kb = json.load(open(os.path.join(D, "raw", "wiki_100_dict_v4.json")))

    rows_by_split = {}
    for split in ("train", "test"):
        with open(os.path.join(D, "raw", f"infoseek_{split}_filtered.csv")) as f:
            rows_by_split[split] = list(csv.DictReader(f))
        print(f"{split}: {len(rows_by_split[split])} raw rows", flush=True)

    gold_urls = {r["wikipedia_url"] for rows in rows_by_split.values() for r in rows}
    missing = [u for u in gold_urls if u not in kb]
    print(f"gold entities: {len(gold_urls)} ({len(missing)} missing from KB)", flush=True)

    # ---- corpus_full: chunk every KB article; remember pid ranges per url ----
    pid = 0
    url_pids = {}
    small_urls = set(u for u in gold_urls if u in kb)
    distractors = [u for u in kb if u not in gold_urls]
    rng.shuffle(distractors)
    small_urls |= set(distractors[: args.distractor_articles])

    n_small = 0
    with open(os.path.join(out_dir, "corpus_full.jsonl"), "w") as ff, \
         open(os.path.join(out_dir, "corpus_small.jsonl"), "w") as fs:
        for url, art in kb.items():
            passages = chunk_article(url, art, pid)
            if not passages:
                continue
            url_pids[url] = [p["pid"] for p in passages]
            for p in passages:
                line = json.dumps(p, ensure_ascii=False) + "\n"
                ff.write(line)
                if url in small_urls:
                    fs.write(line)
                    n_small += 1
            pid += len(passages)
    print(f"corpus_full: {pid} passages over {len(url_pids)} articles; corpus_small: {n_small} passages "
          f"over {len(small_urls)} articles", flush=True)

    # answer-bearing pids need passage text again; keep a small url->passages cache pass
    def build_queries(split, per_entity_cap, max_rows):
        rows = rows_by_split[split]
        by_entity = defaultdict(list)
        for r in rows:
            by_entity[r["wikipedia_url"]].append(r)
        picked = []
        for url, rs in by_entity.items():
            rng.shuffle(rs)
            picked.extend(rs[:per_entity_cap] if per_entity_cap else rs)
        rng.shuffle(picked)
        if max_rows:
            picked = picked[:max_rows]
        out = []
        for r in picked:
            url = r["wikipedia_url"]
            if url not in url_pids:
                continue
            img_ids = [i.strip() for i in r["dataset_image_ids"].split("|") if i.strip()]
            if not img_ids:
                continue
            art = kb[url]
            gold_pids = url_pids[url]
            # answer-bearing passages (pseudo evidence)
            ans = norm(r["answer"])
            ans_pids = []
            if ans:
                ps = chunk_article(url, art, gold_pids[0])
                ans_pids = [p["pid"] for p in ps if ans in norm(p["text"])]
            out.append({
                "qid": r["data_id"],
                "image_id": img_ids[0],
                "question": r["question"],
                "answer": r["answer"],
                "entity_url": url,
                "entity_title": r["wikipedia_title"],
                "gold_pids": gold_pids,
                "ans_pids": ans_pids,
            })
        return out

    q_train = build_queries("train", args.train_per_entity, args.max_train)
    q_test = build_queries("test", 0, args.max_test)
    for name, qs in (("train", q_train), ("test", q_test)):
        with open(os.path.join(out_dir, f"queries_{name}.jsonl"), "w") as f:
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        n_ans = sum(1 for q in qs if q["ans_pids"])
        print(f"queries_{name}: {len(qs)} (with answer-bearing passage: {n_ans})", flush=True)

    # ---- SFT relevance pool: positive = answer-bearing gold passage if any, else lead passage ----
    pid2text = {}
    need = {p for q in q_train for p in ([q["ans_pids"][0]] if q["ans_pids"] else [q["gold_pids"][0]])}
    with open(os.path.join(out_dir, "corpus_full.jsonl")) as f:
        for line in f:
            p = json.loads(line)
            if p["pid"] in need:
                pid2text[p["pid"]] = p["text"]
    with open(os.path.join(out_dir, "pool_train.jsonl"), "w") as f:
        for q in q_train:
            pos_pid = q["ans_pids"][0] if q["ans_pids"] else q["gold_pids"][0]
            f.write(json.dumps({
                "qid": q["qid"], "image_id": q["image_id"], "question": q["question"],
                "answer": q["answer"], "entity_url": q["entity_url"],
                "pos": [pid2text[pos_pid]], "pos_pid": pos_pid, "gold_pids": q["gold_pids"],
            }, ensure_ascii=False) + "\n")
    print("pool_train.jsonl written", flush=True)


if __name__ == "__main__":
    main()
