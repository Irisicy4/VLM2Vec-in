"""Concurrent client for an external LLM judge, used as a training reward grader.

WHY. The reward path is `reader.score_judge`, which is the frozen reader answering and **cover-EM**
grading. cover-EM is strict, so a correct answer phrased differently scores 0 and a query whose
candidates all score 0 yields a degenerate group with exactly zero advantage — dead gradient. Measured
on 1,500 held-out queries (2026-09-14): re-grading with Qwen3-30B moves the live-gradient population
from 32.3% to 39.6%, net +7.7 points.

THE GROUP MUST STAY COMPARABLE. GRPO z-scores advantages *within* a group, so a group graded by two
different graders would have its advantage computed across incommensurable scales. The switch is
therefore **per step**: every reward in a step comes from one grader. That is stricter than necessary
(a group is one query, so per-query switching would also be safe) and leaves the step-level metrics
interpretable.

Failures return None and the caller falls back to cover-EM for that item, so a flaky endpoint degrades
to the current behaviour rather than injecting zeros — injecting a 0 would look exactly like a wrong
answer and would poison the advantage silently.
"""

import json
import os
import re
import threading
import time
import urllib.request

URL = os.environ.get("MMRAG_JUDGE_URL", "http://117.148.167.172:18901/v1/chat/completions")
KEY = os.environ.get("MMRAG_JUDGE_KEY", "judge-vM0ZWhvpPte7kt8LxcDzzvgYCo5d5OaMoGD3FuFj")
MODEL = os.environ.get("MMRAG_JUDGE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")


def render_gold(gold):
    """Dict golds carry a tolerance range; the judge cannot apply it unless it is spelled out."""
    if isinstance(gold, list) and gold and isinstance(gold[0], dict):
        out = []
        for g in gold:
            rng = g.get("range")
            out.append(f"{g.get('wikidata')} (acceptable range {min(rng)} to {max(rng)})"
                       if rng else str(g.get("wikidata")))
        return "; ".join(out)
    if isinstance(gold, (list, tuple)):
        return " | ".join(str(g) for g in gold)
    return str(gold)


def _one(question, gold, pred, url, model, key, retries=3, timeout=45):
    prompt = (f"Question: {question}\nGround truth: {render_gold(gold)}\nModel answer: {pred}\n\n"
              "Reply 1 if correct, 0 if incorrect. Output only the digit.\nJudgement:")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 16, "temperature": 0.3, "stop": ["\n"]}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            m = re.search(r"[01]", txt)
            if m:
                return float(m.group())
        except Exception:
            pass
        time.sleep(0.8 * (a + 1))
    return None


class JudgeClient:
    def __init__(self, url=URL, model=MODEL, key=KEY, workers=16):
        self.url, self.model, self.key, self.workers = url, model, key, workers
        self.n_called = 0
        self.n_failed = 0

    def score(self, items, fallback):
        """items: list of dicts with 'question' and 'answer'. fallback: list of cover-EM scores,
        used verbatim wherever the judge does not resolve. Returns (scores, n_unresolved)."""
        out = [None] * len(items)
        lock = threading.Lock()

        def work(idxs):
            for i in idxs:
                it = items[i]
                v = _one(it.get("question", ""), it.get("answer"), it.get("pred", ""),
                         self.url, self.model, self.key)
                with lock:
                    out[i] = v

        chunks = [list(range(len(items)))[k::self.workers] for k in range(self.workers)]
        ths = [threading.Thread(target=work, args=(c,), daemon=True) for c in chunks if c]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        n_bad = sum(1 for v in out if v is None)
        self.n_called += len(items)
        self.n_failed += n_bad
        return [fallback[i] if out[i] is None else out[i] for i in range(len(items))], n_bad
