"""div-mix v2: same composition as pool_div_1M, but the OVEN slice rebuilt from M-BEIR
train qrels — curated multi-positive gold + the full task-6 text candidate pool (676k
passages) merged into the corpus at the 40M pid range. InfoSeek / E-VQA / OKVQA slices
are carried over from v1 byte-identically (comparability); OKVQA gold stays flagged
heuristic. Answer string for an OVEN row = wikipedia_title of its first text gold.

    python3 mmrag/build_div_mix_v2.py --data_dir $MMRAG_DATA --oven_cap 400000
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OVEN_PID0 = 40_000_000
MB = "TIGER-Lab/M-BEIR"


def hf(f):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MB, f, repo_type="dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA"))
    ap.add_argument("--oven_cap", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    D = args.data_dir
    rng = random.Random(args.seed)

    from mmrag.image_store import build_tar_index
    avail = set()
    for i in range(6):
        t = os.path.join(D, "images/OVEN", f"shard{i:02d}.tar")
        if os.path.exists(t):
            avail.update(build_tar_index(t).keys())
    print(f"[v2] {len(avail)} oven images on disk", flush=True)

    # task-6 text candidate pool -> corpus passages; did "5:x" -> pid 40M + x
    cands = {}
    with open(hf("cand_pool/local/mbeir_oven_task6_cand_pool.jsonl")) as f:
        for line in f:
            c = json.loads(line)
            did = c["did"]
            if not did.startswith("5:"):
                continue
            sc = json.loads(c.get("src_content") or "{}")
            cands[did] = {"pid": OVEN_PID0 + int(did.split(":")[1]),
                          "url": f"oven://{sc.get('wikidata_id','')}",
                          "title": sc.get("wikipedia_title") or "", "section": "",
                          "text": c.get("txt") or ""}
    print(f"[v2] task6 text candidates: {len(cands)}", flush=True)

    gold = {}
    with open(hf("qrels/train/mbeir_oven_train_qrels.txt")) as f:
        for line in f:
            qid, _, did, rel = line.split()[:4]
            if rel != "0" and did.startswith("5:"):
                gold.setdefault(qid, []).append(did)
    print(f"[v2] queries with text gold: {len(gold)}", flush=True)

    kept = 0
    oven_rows = []
    with open(hf("query/train/mbeir_oven_train.jsonl")) as f:
        for line in f:
            if kept >= args.oven_cap:
                break
            q = json.loads(line)
            sc = json.loads(q["query_src_content"])
            img = sc["image_id"]
            if img not in avail:
                continue
            gs = [cands[d] for d in gold.get(q["qid"], []) if d in cands]
            if not gs:
                continue
            title = gs[0]["title"]
            if not title:
                continue
            oven_rows.append({"qid": f"oven_{kept:06d}", "image_id": img,
                              "question": q["query_txt"] or "what entity is shown?",
                              "answer": title, "answer_aliases": [title],
                              "entity_url": gs[0]["url"],
                              "pos": [g["text"] for g in gs[:3]],
                              "pos_pid": gs[0]["pid"]})
            kept += 1
    print(f"[v2] oven rows kept {len(oven_rows)}", flush=True)

    carry = [json.loads(l) for l in open(f"{D}/built/pool_div_1M.jsonl")
             if not json.loads(l)["qid"].startswith("oven_")]
    allrows = carry + oven_rows
    rng.shuffle(allrows)
    with open(f"{D}/built/pool_div_1M_v2.jsonl", "w") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{D}/built/corpus_div_v2.jsonl", "w") as f:
        for line in open(f"{D}/built/corpus_div.jsonl"):
            if json.loads(line)["pid"] < OVEN_PID0 or json.loads(line)["pid"] >= 50_000_000:
                f.write(line)          # keep v1 corpus minus the old 3.6k oven summaries
        for c in cands.values():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    comp = {"infoseek": sum(r["qid"].startswith(("infoseek", "q")) or not r["qid"].startswith(("oven", "okvqa", "evqa")) for r in carry),
            "evqa_landmarks": sum(r["qid"].startswith("evqa") for r in carry),
            "okvqa": sum(r["qid"].startswith("okvqa") for r in carry),
            "oven_mbeir": len(oven_rows), "total": len(allrows),
            "oven_gold": "M-BEIR train qrels, multi-positive, task6 text pool (676k passages in corpus)",
            "seed": args.seed}
    json.dump(comp, open(f"{D}/built/pool_div_1M_v2.composition.json", "w"), indent=1)
    print("DIV_MIX_V2_DONE", comp, flush=True)


if __name__ == "__main__":
    main()
