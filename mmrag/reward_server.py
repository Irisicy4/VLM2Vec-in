"""HTTP reward server: downstream RAG QA accuracy as a service (multimodal port of the text
project's reward_server.py). Holds ONE reader VLM on a dedicated GPU; any number of trainer
processes POST (image_id, question, context, answers) batches and get rewards back — so the
reward reader can be the eval-grade 7B model shared across trainers instead of a per-trainer 3B.

    python3 mmrag/reward_server.py --model Qwen/Qwen2.5-VL-7B-Instruct --device cuda:0 --port 8377

API:
  GET  /health            -> {"status": "ok", "model": ...}
  POST /reward            JSON {"kind": "judge"|"logit", "rollouts": int, "temperature": float,
                                "max_ctx_chars": int|null,
                                "items": [{"image_id", "question", "context", "answers"}...]}
                          -> {"rewards": [float,...]}
Images are resolved server-side from the InfoSeek tars via image_store (payloads stay tiny).
"""

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.image_store import open_stores  # noqa: E402
from mmrag.reader_vlm import LiveVLMReader  # noqa: E402

READER = None
STORE = None
LOCK = threading.Lock()   # one GPU — serialize scoring; ThreadingHTTPServer still overlaps IO
ARGS = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # quiet
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
            self._send(200, {"status": "ok", "model": ARGS.model})
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/reward":
            self._send(404, {"error": "unknown path"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            items = req["items"]
            triples = []
            for it in items:
                triples.append({"image": STORE.get(it["image_id"]),
                                "question": it["question"],
                                "context": it.get("context") or "",
                                "answer": it.get("answers") or it.get("answer")})
            kind = req.get("kind", "judge")
            with LOCK:
                if req.get("max_ctx_chars"):
                    READER.max_ctx_chars = int(req["max_ctx_chars"])
                if kind == "logit":
                    rewards = READER.score_logit(triples)
                else:
                    rewards = READER.score_judge(triples,
                                                 n_rollouts=int(req.get("rollouts", 4)),
                                                 temperature=float(req.get("temperature", 0.7)))
            self._send(200, {"rewards": rewards})
        except Exception as e:  # noqa: BLE001 — report, don't kill the server
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


class RemoteReader:
    """Client with the LiveVLMReader interface; triples must carry image_id (image is dropped)."""

    def __init__(self, url, timeout=600, max_ctx_chars=None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.max_ctx_chars = max_ctx_chars

    def _post(self, kind, triples, rollouts=4, temperature=0.7):
        import time
        import urllib.request

        items = [{"image_id": t["image_id"], "question": t["question"],
                  "context": t.get("context") or "", "answers": t["answer"]} for t in triples]
        payload = json.dumps({"kind": kind, "items": items, "rollouts": rollouts,
                              "temperature": temperature,
                              "max_ctx_chars": self.max_ctx_chars}).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url + "/reward", data=payload,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                if "rewards" not in out:
                    raise RuntimeError(out.get("error", "no rewards in response"))
                return out["rewards"]
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"[RemoteReader] retry {attempt + 1}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))

    def score_judge(self, triples, n_rollouts=4, temperature=0.7, **_):
        return self._post("judge", triples, n_rollouts, temperature)

    def score_logit(self, triples):
        return self._post("logit", triples)


def main():
    global READER, STORE, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8377)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_ctx_chars", type=int, default=4500)
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA", "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data"))
    ARGS = ap.parse_args()
    STORE = open_stores(ARGS.data_dir, "infoseek")
    READER = LiveVLMReader(ARGS.model, device=ARGS.device, batch_size=ARGS.batch_size,
                           max_ctx_chars=ARGS.max_ctx_chars)
    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler)
    print(f"reward server up on :{ARGS.port} model={ARGS.model}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
