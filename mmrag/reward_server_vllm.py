"""vLLM-backed reward server: the same /reward API as mmrag/reward_server.py, served by vLLM.

WHY. The RL reward path is the bottleneck, not the retriever. Per training step the reader answers
pl_support x batch triples with reader_rollouts samples each -- with the defaults that is up to
4 x 24 x 4 = 384 generations of 24 tokens per step -- and LiveVLMReader does that with HF
`model.generate` on a padded batch, which idles on the longest sequence and re-runs prefill every call.
Measured on this box the whole step rate is 0.6-2.6 steps/min, i.e. 2.4-10.2 queries/min, which puts a
500k-query mixed-domain run at 34-145 days. vLLM's continuous batching and paged KV cache attack exactly
that shape of work.

WHAT IS HELD IDENTICAL, because the reward distribution must not move. Anything that changes the reward
silently invalidates comparability with every arm already in the paper:
  * the prompt: PROMPT / PROMPT_NOCTX and apply_chat_template are imported from reader_vlm, not retyped
  * context truncation: the same max_ctx_chars slice, applied before templating
  * image pipeline: the same min_pixels / max_pixels passed to the processor
  * grading: answer_correct imported from reader_vlm -- the identical cover-EM normaliser
  * decoding: max_tokens 24, temperature and n taken from the request as before
Only the generation ENGINE differs. Sampling will still not be bit-identical to HF, so
mmrag/verify_reward_parity.py exists to measure agreement before any run is switched over.

WHY A SEPARATE INTERPRETER. vLLM lives in /raid/icy/iris/.conda/envs/cu13rl (vllm 0.28, torch 2.13,
transformers 5.10). The training venv at $MMRAG_DATA/venv must NOT have vLLM installed into it -- that
would upgrade torch under the running trainers. This server is a standalone process; the trainer reaches
it over HTTP through the existing --reward_server_url / RemoteReader path, which needs no change.

    PYTHONPATH=/raid/icy/VLM2Vec-in CUDA_VISIBLE_DEVICES=7 \
      /raid/icy/iris/.conda/envs/cu13rl/bin/python3 mmrag/reward_server_vllm.py \
      --model Qwen/Qwen2.5-VL-3B-Instruct --port 8377 --gpu_frac 0.30
"""

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.image_store import open_stores                              # noqa: E402
from mmrag.reader_vlm import PROMPT, PROMPT_NOCTX, answer_correct      # noqa: E402

ENGINE = None
PROC = None
STORE = None
ARGS = None
LOCK = threading.Lock()


class VLLMJudge:
    """score_judge with LiveVLMReader's semantics, executed by vLLM."""

    def __init__(self, model_id, gpu_frac, max_ctx_chars, max_pixels, max_model_len):
        from transformers import AutoProcessor
        from vllm import LLM

        self.max_ctx_chars = max_ctx_chars
        self.proc = AutoProcessor.from_pretrained(
            model_id, min_pixels=28 * 28 * 4, max_pixels=max_pixels)
        self.llm = LLM(model=model_id, dtype="bfloat16",
                       gpu_memory_utilization=gpu_frac,
                       limit_mm_per_prompt={"image": 1},
                       max_model_len=max_model_len,
                       mm_processor_kwargs={"min_pixels": 28 * 28 * 4, "max_pixels": max_pixels},
                       enforce_eager=False, disable_log_stats=True)

    def _chat(self, t):
        """Byte-identical to LiveVLMReader._chat (no add_answer path; that is score_logit's)."""
        ctx = (t.get("context") or "")[: self.max_ctx_chars]
        text = PROMPT.format(ctx=ctx, q=t["question"]) if ctx else PROMPT_NOCTX.format(q=t["question"])
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def score_judge(self, triples, max_new_tokens=24, n_rollouts=1, temperature=0.7,
                    return_preds=False):
        from vllm import SamplingParams

        sample = n_rollouts > 1 and temperature > 0
        sp = SamplingParams(n=n_rollouts, max_tokens=max_new_tokens,
                            temperature=temperature if sample else 0.0,
                            top_p=0.95 if sample else 1.0)
        reqs = [{"prompt": self._chat(t), "multi_modal_data": {"image": t["image"]}} for t in triples]
        outs = self.llm.generate(reqs, sp, use_tqdm=False)
        scores, preds = [], []
        for t, o in zip(triples, outs):
            got = [c.text.strip() for c in o.outputs]
            scores.append(sum(answer_correct(a, t["answer"]) for a in got) / max(len(got), 1))
            preds.append(got[0] if got else "")
        return (scores, preds) if return_preds else scores


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": ARGS.model, "backend": "vllm"})
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/reward":
            self._send(404, {"error": "unknown path"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            triples = [{"image": STORE.get(it["image_id"]), "question": it["question"],
                        "context": it.get("context") or "",
                        "answer": it.get("answers") or it.get("answer")} for it in req["items"]]
            if req.get("kind", "judge") == "logit":
                # score_logit needs per-token logprobs of a forced answer; vLLM's prompt_logprobs path
                # is a different contract, so it is not served here rather than silently approximated.
                self._send(400, {"error": "kind=logit not supported by the vLLM backend; "
                                          "use mmrag/reward_server.py for logit rewards"})
                return
            with LOCK:
                if req.get("max_ctx_chars"):
                    ENGINE.max_ctx_chars = int(req["max_ctx_chars"])
                rewards = ENGINE.score_judge(triples,
                                             n_rollouts=int(req.get("rollouts", 4)),
                                             temperature=float(req.get("temperature", 0.7)))
            self._send(200, {"rewards": rewards})
        except Exception as e:  # noqa: BLE001 — report, never kill the server
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    global ENGINE, STORE, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--port", type=int, default=8377)
    ap.add_argument("--gpu_frac", type=float, default=0.30)
    ap.add_argument("--max_ctx_chars", type=int, default=1600)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--image_dataset", default="infoseek")
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data"))
    ARGS = ap.parse_args()
    STORE = open_stores(ARGS.data_dir, ARGS.image_dataset)
    ENGINE = VLLMJudge(ARGS.model, ARGS.gpu_frac, ARGS.max_ctx_chars,
                       28 * 28 * 576, ARGS.max_model_len)
    print(f"reward_server_vllm ready: {ARGS.model} on port {ARGS.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
