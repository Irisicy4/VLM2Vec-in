import os
"""Audit the reward metric. cover-EM (substring match against an alias list) is what scores every
reward in this campaign. If it under-credits, the "reward starvation" in Q2 is partly an artifact of
the GRADER rather than a limit of the reader. Re-grade stored predictions with an LLM judge and
measure the disagreement in both directions."""
import json, random, sys, re, ast, urllib.request, concurrent.futures as cf

URL="http://183.213.219.114:18901/v1/chat/completions"
KEY = os.environ["JUDGE_API_KEY"]  # redacted for harvest; set in env
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

def judge(q, gold, pred):
    golds = gold if isinstance(gold, list) else ast.literal_eval(gold)
    prompt = (f"Question: {q}\nGround truth (any of): {'; '.join(map(str,golds[:12]))}\n"
              f"Model answer: {pred}\n\nReply 1 if the model answer is correct (matches any ground "
              f"truth, allowing paraphrase and formatting differences), 0 if incorrect. "
              f"Output only the digit.\nJudgement:")
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt}],
                     "max_tokens":8,"temperature":0.0,"stop":["\n"]}).encode()
    req=urllib.request.Request(URL, data=body, headers={
        "Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                t=json.load(r)["choices"][0]["message"]["content"]
            m=re.search(r"[01]", t)
            if m: return int(m.group())
        except Exception as e:
            err=e
    return None

name=sys.argv[1] if len(sys.argv)>1 else "emaidx-1M-s3000"
d=json.load(open(f"/raid/icy/mmrag_data/results/{name}.vqa_top5.json"))
qs={}
for l in open("/raid/icy/mmrag_data/built/queries_test.jsonl"):
    r=json.loads(l); qs[r["qid"]]=r["question"]
preds=[p for p in d["predictions"] if p["qid"] in qs]
rng=random.Random(0)
zero=[p for p in preds if float(p["acc"])==0.0]; one=[p for p in preds if float(p["acc"])==1.0]
S0=rng.sample(zero, min(250,len(zero))); S1=rng.sample(one, min(150,len(one)))
print(f"{name}: cover-EM acc {d['acc']*100:.2f} (n={d['n']}); auditing {len(S0)} scored-0 and {len(S1)} scored-1", flush=True)
def run(sample, label):
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        res=list(ex.map(lambda p: judge(qs[p["qid"]], p["gold"], p["pred"]), sample))
    ok=[(p,j) for p,j in zip(sample,res) if j is not None]
    flip=sum(1 for p,j in ok if j != int(float(p["acc"])))
    print(f"  {label}: {len(ok)}/{len(sample)} judged, judge DISAGREES on {flip} ({100*flip/max(len(ok),1):.1f}%)", flush=True)
    return ok, flip
ok0,f0=run(S0,"cover-EM said WRONG")
ok1,f1=run(S1,"cover-EM said RIGHT")
print(f"\n  -> cover-EM false-NEGATIVE rate (judge says right, EM said wrong): {100*f0/max(len(ok0),1):.1f}%")
print(f"  -> cover-EM false-POSITIVE rate (judge says wrong, EM said right): {100*f1/max(len(ok1),1):.1f}%")
p0=f0/max(len(ok0),1); p1=f1/max(len(ok1),1); a=d["acc"]
print(f"  -> implied judge-graded accuracy: {100*(a*(1-p1) + (1-a)*p0):.2f}  vs cover-EM {100*a:.2f}")
print("\n  examples the judge overturned (EM=0 -> judge=1):")
for p,j in ok0:
    if j==1: print(f"     Q: {qs[p['qid']][:60]}\n        pred={p['pred'][:40]!r}  gold={str(p['gold'])[:70]}")
    if sum(1 for x,y in ok0 if y==1 and ok0.index((x,y))<=ok0.index((p,j)))>=5: break
