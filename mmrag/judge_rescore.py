"""Re-grade stored reader answers with an external LLM judge, and measure whether that unlocks
usable REWARD GRADIENT rather than merely raising a score.

THE QUESTION. The training reward is `reader.score_judge`, which despite the name is the frozen
reader answering and **cover-EM** grading. cover-EM is strict: it demands a gold alias appear as a
token subsequence. So a correct answer phrased differently scores 0, and a query whose candidates all
score 0 produces a degenerate group whose z-scored advantage is exactly zero — dead gradient. If a
stronger judge recovers those, the reward signal widens.

WHY THIS IS NOT JUST "A BETTER METRIC". Whether recovered reward becomes *gradient* depends on
VARIANCE inside the group, not on the mean. The RAG taxonomy is built from two passes over the same
queries — the reader with the GOLD passage, and the reader with NO context — so the transition of each
query between taxonomy classes says exactly what happens to its gradient:

    hopeless -> benefits   gold now right, no-context still wrong  ->  NEW live gradient
    hopeless -> unneeded   both now right                          ->  still dead (saturated)
    benefits -> unneeded   no-context now right too                ->  gradient LOST
    unchanged                                                      ->  no effect

So this script reports a gradient ledger, not an accuracy delta.

A SEPARATE EFFECT NOT MEASURED HERE, stated so it is not conflated. A stronger *reader* makes more
queries answerable without retrieval, which moves rows into `unneeded` and *reduces* gradient — that is
the registered prediction in paper.md:956-960, and the text-side project measured it (their usable share
fell 36.9 -> 30.6 when they swapped in a 30B reader). A stronger *judge* is the opposite intervention:
the reader is unchanged, only the grading widens. Reader and judge push in opposite directions and must
not be argued for or against as one thing.

    python3 mmrag/judge_rescore.py --arm gme2b-zeroshot-3k-repro --workers 12
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data")
R = os.path.join(D, "results")
URL = os.environ.get("MMRAG_JUDGE_URL", "http://117.148.167.172:18901/v1/chat/completions")
KEY = os.environ.get("MMRAG_JUDGE_KEY", "judge-vM0ZWhvpPte7kt8LxcDzzvgYCo5d5OaMoGD3FuFj")
MODEL = os.environ.get("MMRAG_JUDGE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")


def render_gold(gold):
    """Dict golds carry a tolerance; give the judge the range explicitly or it cannot know it."""
    if isinstance(gold, list) and gold and isinstance(gold[0], dict):
        parts = []
        for g in gold:
            v = g.get("wikidata")
            rng = g.get("range")
            parts.append(f"{v} (acceptable range {min(rng)} to {max(rng)})" if rng else str(v))
        return "; ".join(parts)
    if isinstance(gold, (list, tuple)):
        return " | ".join(str(g) for g in gold)
    return str(gold)


def judge(question, gold, pred, retries=4):
    prompt = (f"Question: {question}\nGround truth: {render_gold(gold)}\nModel answer: {pred}\n\n"
              "Reply 1 if correct, 0 if incorrect. Output only the digit.\nJudgement:")
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 16, "temperature": 0.3, "stop": ["\n"]}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            m = re.search(r"[01]", txt)
            if m:
                return int(m.group())
        except Exception:
            pass
        time.sleep(1.5 * (a + 1))
    return None                                    # unresolved; counted separately, never guessed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gme2b-zeroshot-3k-repro")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="0 = all queries")
    args = ap.parse_args()

    meta = {}
    with open(os.path.join(D, "built", "queries_test.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            meta[r["qid"]] = r["question"]

    runs = {}
    for tag in ("gold", "noctx"):
        p = os.path.join(R, f"{args.arm}.vqa_{tag}.json")
        runs[tag] = {d["qid"]: d for d in json.load(open(p))["predictions"]}
    qids = sorted(set(runs["gold"]) & set(runs["noctx"]))
    if args.limit:
        qids = qids[: args.limit]

    cache_path = os.path.join(R, f"{args.arm}.judge.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [(q, t) for q in qids for t in ("gold", "noctx") if f"{q}|{t}" not in cache]
    print(f"  {len(qids)} queries x 2 passes; {len(todo)} judge calls to make "
          f"({len(cache)} cached)", flush=True)

    lock = threading.Lock()
    done = [0]

    def work(items):
        for q, t in items:
            d = runs[t][q]
            v = judge(meta.get(q, ""), d["gold"], d["pred"])
            with lock:
                cache[f"{q}|{t}"] = v
                done[0] += 1
                if done[0] % 200 == 0:
                    print(f"    {done[0]}/{len(todo)}", flush=True)
                    json.dump(cache, open(cache_path, "w"))

    chunks = [todo[i::args.workers] for i in range(args.workers)]
    ths = [threading.Thread(target=work, args=(c,), daemon=True) for c in chunks if c]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    json.dump(cache, open(cache_path, "w"))

    def cls(g, n):
        if g and not n: return "benefits"
        if g and n:     return "unneeded"
        if not g and not n: return "hopeless"
        return "harms"

    em, jd, unresolved = Counter(), Counter(), 0
    trans = Counter()
    for q in qids:
        ge, ne = runs["gold"][q]["acc"] > 0.5, runs["noctx"][q]["acc"] > 0.5
        gj, nj = cache.get(f"{q}|gold"), cache.get(f"{q}|noctx")
        em[cls(ge, ne)] += 1
        if gj is None or nj is None:
            unresolved += 1
            continue
        jd[cls(bool(gj), bool(nj))] += 1
        trans[(cls(ge, ne), cls(bool(gj), bool(nj)))] += 1

    n = len(qids)
    print(f"\n  TAXONOMY under each grader (n={n}, unresolved judge calls on {unresolved} queries)")
    print(f"  {'class':10s} {'cover-EM':>10s} {'JUDGE':>10s} {'delta':>8s}")
    for k in ("benefits", "unneeded", "hopeless", "harms"):
        print(f"  {k:10s} {100*em[k]/n:9.1f}% {100*jd[k]/max(n-unresolved,1):9.1f}% "
              f"{100*jd[k]/max(n-unresolved,1)-100*em[k]/n:+7.1f}")
    gained = trans[("hopeless", "benefits")] + trans[("harms", "benefits")]
    lost = trans[("benefits", "unneeded")] + trans[("benefits", "hopeless")]
    sat = trans[("hopeless", "unneeded")]
    print(f"\n  GRADIENT LEDGER (what the judge does to usable reward variance)")
    print(f"    hopeless/harms -> benefits   {gained:5d}  ({100*gained/n:+5.1f}%)  NEW live gradient")
    print(f"    benefits -> unneeded/hopeless{lost:5d}  ({-100*lost/n:+5.1f}%)  gradient LOST")
    print(f"    hopeless -> unneeded         {sat:5d}  ({100*sat/n:5.1f}%)  recovered but SATURATED")
    print(f"    NET usable-reward change     {100*(gained-lost)/n:+5.1f} percentage points")
    print(f"\n  top transitions:")
    for (a, b), c in trans.most_common(8):
        if a != b:
            print(f"    {a:9s} -> {b:9s} {c:5d}")


if __name__ == "__main__":
    main()
