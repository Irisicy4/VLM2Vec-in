"""Does the vLLM reward server return the same rewards as the in-process HF reader?

Switching the reward backend mid-campaign is only safe if the reward DISTRIBUTION is unchanged: every
arm already in the paper was trained against LiveVLMReader, so a backend that grades even slightly
differently makes new runs incomparable to old ones. Sampling is stochastic (reader_rollouts=4,
temperature=0.7), so bit-identity is not the bar; the bar is that the disagreement is no larger than
HF's disagreement with ITSELF across two independent samplings. This measures both, on the same triples.

Reports, over n queries scored by each side:
    mean reward       HF vs vLLM, and HF vs HF re-run   (levels must not shift)
    mean |delta|      per-query                          (vLLM-vs-HF must not exceed HF-vs-HF)
    exact agreement   fraction of queries with identical reward
    greedy agreement  the same with rollouts=1, temperature=0 -- this one SHOULD be near-identical,
                      since greedy decoding removes the sampling noise and isolates prompt/image drift

    python3 mmrag/verify_reward_parity.py --url http://127.0.0.1:8377 --n 120
"""

import argparse
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data")


def post(url, items, rollouts, temperature, timeout=1800):
    body = json.dumps({"kind": "judge", "items": items, "rollouts": rollouts,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/reward", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    if "rewards" not in out:
        raise RuntimeError(out.get("error", "no rewards"))
    return out["rewards"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8377")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--pool", default="built/pool_train_1M.jsonl")
    ap.add_argument("--corpus", default="built/corpus_small.jsonl")
    ap.add_argument("--reader", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    rng = random.Random(0)
    rows = []
    with open(os.path.join(D, args.pool)) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= 20000:
                break
    rows = rng.sample(rows, args.n)

    # pos holds the gold passage TEXT as a list (pos_pid holds the id) -- no corpus lookup needed.
    # An earlier version of this script treated pos as a pid, which silently produced EMPTY contexts and
    # ran the whole comparison on the no-context path at ~3% mean reward, where agreement is dominated
    # by both backends saying 0. The with-context path is the one training actually operates on.
    items = []
    for r in rows:
        ctx = (r.get("pos") or [""])[0]
        items.append({"image_id": r["image_id"], "question": r["question"],
                      "context": ctx, "answers": r["answer"]})
    filled = sum(1 for it in items if it["context"])
    print(f"  contexts filled: {filled}/{len(items)}")
    if filled < 0.9 * len(items):
        print("  REFUSING: too many empty contexts; this would not test the operating point")
        sys.exit(2)

    from mmrag.image_store import open_stores
    from mmrag.reader_vlm import LiveVLMReader
    store = open_stores(D, "infoseek")
    triples = [{"image": store.get(it["image_id"]), "question": it["question"],
                "context": it["context"], "answer": it["answers"]} for it in items]
    hf = LiveVLMReader(args.reader, device=args.device, batch_size=16)

    def report(tag, a, b):
        n = len(a)
        mad = sum(abs(x - y) for x, y in zip(a, b)) / n
        agree = sum(1 for x, y in zip(a, b) if abs(x - y) < 1e-9) / n
        print(f"  {tag:22s} mean {sum(a)/n:.4f} vs {sum(b)/n:.4f}   "
              f"mean|delta| {mad:.4f}   exact agreement {100*agree:5.1f}%")

    print(f"\n  n={args.n}  rollouts={args.rollouts}  temperature={args.temperature}")
    hf1 = hf.score_judge(triples, n_rollouts=args.rollouts, temperature=args.temperature)
    hf2 = hf.score_judge(triples, n_rollouts=args.rollouts, temperature=args.temperature)
    vl = post(args.url, items, args.rollouts, args.temperature)
    report("HF vs HF (noise floor)", hf1, hf2)
    report("HF vs vLLM", hf1, vl)

    print("\n  greedy (rollouts=1, temperature=0) — sampling noise removed")
    hg = hf.score_judge(triples, n_rollouts=1, temperature=0.0)
    vg = post(args.url, items, 1, 0.0)
    report("HF vs vLLM, greedy", hg, vg)
    print("\n  VERDICT: the switch is safe iff HF-vs-vLLM is no worse than HF-vs-HF above,")
    print("           and the greedy row agrees near-exactly.\n")


if __name__ == "__main__":
    main()
