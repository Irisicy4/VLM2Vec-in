"""Build a 2M-passage mixed-domain retrieval corpus for mm-RAG.

Motivation: the RL index the cells retrieve against has been homogeneous encyclopedic
Wikipedia (corpus_small = 422k InfoSeek passages, of which only ~174k enter a run). Every
distractor looks like every gold, and nothing tests whether the retriever can reject an
off-domain passage. The text-only side of this project runs a 1M corpus spanning 13
domains; this is the multimodal analogue at 2M.

Composition = corpus_div_v2 (unchanged, carries EVERY gold passage for the div pools, pids
preserved) + BEIR passages as cross-domain distractors at pid 60M+, one pid range per domain
so provenance is recoverable from the pid alone.

    python3 mmrag/build_mix2m_corpus.py --data_dir $MMRAG_DATA --target 2000000

Note on sampling: BEIR domains are streamed and the head-N taken. That is deterministic and
needs no full download (msmarco is 8.8M passages), but it is a head slice, not a random
sample -- recorded in the composition file so nobody mistakes it for one.
"""

import argparse
import json
import os

# One pid range per source so a passage's origin is recoverable from its pid alone.
# 0..50M are taken by corpus_div_v2 (infoseek / evqa / oven / okvqa); BEIR starts at 60M.
BEIR_PID0 = 60_000_000
RANGE = 1_000_000  # pids per BEIR domain

# share-of-BEIR taken from the text-only side's corpus composition, renormalised
BEIR_DOMAINS = [
    ("msmarco",       "BeIR/msmarco",       18.2),
    ("quora",         "BeIR/quora",          8.3),
    ("cqadupstack",   "BeIR/cqadupstack",    7.4),
    ("trec-covid",    "BeIR/trec-covid",     6.6),
    ("webis-touche2020", "BeIR/webis-touche2020", 5.8),
    ("fiqa",          "BeIR/fiqa",           4.1),
    ("scidocs",       "BeIR/scidocs",        2.5),
    ("arguana",       "BeIR/arguana",        0.8),
    ("scifact",       "BeIR/scifact",        0.5),
    ("nfcorpus",      "BeIR/nfcorpus",       0.4),
]

# Fill domains, used by --append to reach the target when a primary domain is unavailable
# or smaller than its share. BeIR/cqadupstack has no corpus repo on HF (only -qrels and
# -generated-queries), so its 7.4% share is redistributed here rather than dumped into
# msmarco: more domains beats more of one domain.
FILL_DOMAINS = [
    ("hotpotqa",      "BeIR/hotpotqa",       1.0),
    ("climate-fever", "BeIR/climate-fever",  1.0),
    ("dbpedia-entity", "BeIR/dbpedia-entity", 1.0),
]
FILL_PID0 = 70_000_000


def stream_domain(repo, want):
    """Yield up to `want` passages from a BEIR corpus without downloading it whole."""
    from datasets import load_dataset
    try:
        ds = load_dataset(repo, "corpus", split="corpus", streaming=True)
    except Exception as e:                      # some BEIR repos have no config split
        print(f"    [{repo}] config 'corpus' failed ({str(e)[:70]}); trying default", flush=True)
        ds = load_dataset(repo, split="corpus", streaming=True)
    n = 0
    for r in ds:
        txt = (r.get("text") or "").strip()
        if len(txt) < 40:                        # drop stubs: they are free negatives, not passages
            continue
        yield r.get("_id"), (r.get("title") or "").strip(), txt
        n += 1
        if n >= want:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA"))
    ap.add_argument("--base", default="built/corpus_div_v2.jsonl",
                    help="carries every gold for the div pools; copied through unchanged")
    ap.add_argument("--out", default="built/corpus_mix2M.jsonl")
    ap.add_argument("--target", type=int, default=2_000_000)
    ap.add_argument("--append", action="store_true",
                    help="top up an existing --out to --target using FILL_DOMAINS")
    args = ap.parse_args()
    D = args.data_dir
    out_path = os.path.join(D, args.out)
    comp_path = out_path.replace(".jsonl", ".composition.json")

    if args.append:
        comp = json.load(open(comp_path))
        have = comp["_total"]
        need = max(args.target - have, 0)
        print(f"[mix2M] appending: have {have}, need {need} to reach {args.target}", flush=True)
        if need == 0:
            return
        added = 0
        with open(out_path, "a") as out:
            for i, (name, repo, _) in enumerate(FILL_DOMAINS):
                if added >= need:
                    break
                want = need - added                       # each domain fills what remains
                pid0 = FILL_PID0 + i * RANGE
                got = 0
                try:
                    for j, (did, title, text) in enumerate(stream_domain(repo, want)):
                        out.write(json.dumps({"pid": pid0 + j, "title": title, "section": "",
                                              "text": text, "url": f"beir://{name}/{did}"}) + "\n")
                        got += 1
                except Exception as e:
                    print(f"    [{name}] FAILED: {str(e)[:120]}", flush=True)
                comp[name] = comp.get(name, 0) + got
                added += got
                print(f"    [{name}] +{got}", flush=True)
        comp["_total"] = have + added
        comp["_note"] += (f"; topped up with FILL_DOMAINS at pid {FILL_PID0}+ because "
                          "BeIR/cqadupstack has no corpus repo on HF")
        json.dump(comp, open(comp_path, "w"), indent=1)
        print(f"[mix2M] total now {comp['_total']}", flush=True)
        return

    comp = {}
    n_base = 0
    tmp = out_path + ".tmp"
    with open(tmp, "w") as out:
        with open(os.path.join(D, args.base)) as f:
            for line in f:                        # byte-for-byte passthrough: golds must not move
                out.write(line)
                n_base += 1
        print(f"[mix2M] base {args.base}: {n_base} passages carried through", flush=True)
        comp["_base_" + os.path.basename(args.base)] = n_base

        budget = max(args.target - n_base, 0)
        total_share = sum(s for _, _, s in BEIR_DOMAINS)
        print(f"[mix2M] BEIR budget: {budget} passages across {len(BEIR_DOMAINS)} domains", flush=True)

        written = 0
        shortfall = 0
        for i, (name, repo, share) in enumerate(BEIR_DOMAINS):
            want = int(round(budget * share / total_share))
            if name == "msmarco":                 # largest corpus absorbs other domains' shortfall
                want += shortfall
            got = 0
            pid0 = BEIR_PID0 + i * RANGE
            try:
                for j, (did, title, text) in enumerate(stream_domain(repo, want)):
                    out.write(json.dumps({"pid": pid0 + j, "title": title, "section": "",
                                          "text": text, "url": f"beir://{name}/{did}"}) + "\n")
                    got += 1
            except Exception as e:
                print(f"    [{name}] FAILED: {str(e)[:120]}", flush=True)
            comp[name] = got
            written += got
            if got < want:
                shortfall += want - got           # smaller-than-target corpora hand budget onward
                print(f"    [{name}] {got}/{want} (corpus exhausted; +{want-got} to msmarco)", flush=True)
            else:
                print(f"    [{name}] {got}", flush=True)

    os.replace(tmp, out_path)
    total = n_base + written
    comp["_total"] = total
    comp["_note"] = ("base carried byte-identically so every div-pool gold keeps its pid; "
                     "BEIR domains are head-N of a streamed corpus (deterministic, NOT a random "
                     "sample); passages under 40 chars dropped; one pid range per domain from "
                     f"{BEIR_PID0} in steps of {RANGE}")
    json.dump(comp, open(out_path.replace(".jsonl", ".composition.json"), "w"), indent=1)
    print(f"[mix2M] wrote {out_path}: {total} passages", flush=True)
    for k, v in comp.items():
        if not k.startswith("_"):
            print(f"    {k:<20} {v:>8}  ({100*v/total:.1f}%)")


if __name__ == "__main__":
    main()
