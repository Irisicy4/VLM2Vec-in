"""In-process VLM reader for live RL rewards and L2 eval (multimodal port of ../rag/live_reader.py).

Scores (image, question, context, answer) two ways:
  - logit: length-normalized logP(answer | image, question, context), teacher-forced — dense reward.
  - judge: greedy/sampled generation -> binary cover-EM accuracy vs the gold answer (+aliases).

Reader = Qwen2.5-VL-Instruct (3B for reward, 7B for eval), loaded with flash-attn in bf16.
"""

import re
import string
from collections import Counter

import torch

PROMPT = ("Context: {ctx}\n\nAnswer the question about the image using the context. "
          "Reply with a short answer only.\nQuestion: {q}\nAnswer:")
PROMPT_NOCTX = ("Answer the question about the image. Reply with a short answer only.\n"
                "Question: {q}\nAnswer:")


def _normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_correct(pred, golds):
    """Binary cover-EM vs any gold alias: exact normalized match OR the gold's TOKEN SEQUENCE
    contained in the prediction's tokens. Token-boundary containment, not substring — raw
    substring over-credits ('us' in 'museum', numbers inside years)."""
    p = _normalize(pred)
    pt = p.split()
    for g in golds if isinstance(golds, (list, tuple)) else [golds]:
        g = _normalize(str(g))
        if not g:
            continue
        if p == g:
            return 1.0
        gt = g.split()
        n = len(gt)
        if n and any(pt[i:i + n] == gt for i in range(len(pt) - n + 1)):
            return 1.0
    return 0.0


def answer_strict_em(pred, golds):
    """Strict EM: normalized prediction exactly equals some gold alias."""
    p = _normalize(pred)
    return 1.0 if any(p == _normalize(str(g)) and p
                      for g in (golds if isinstance(golds, (list, tuple)) else [golds])) else 0.0


def squad_f1(pred, golds):
    p = _normalize(pred).split()
    best = 0.0
    for g in golds if isinstance(golds, (list, tuple)) else [golds]:
        gt = _normalize(str(g)).split()
        if not p or not gt:
            best = max(best, float(p == gt))
            continue
        common = sum((Counter(p) & Counter(gt)).values())
        if common == 0:
            continue
        prec, rec = common / len(p), common / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


class LiveVLMReader:
    """triples: {"image": PIL.Image, "question": str, "context": str|None, "answer": str|list}"""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda:0", batch_size=8,
                 max_ctx_chars=1600, max_pixels=28 * 28 * 576):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=28 * 28 * 4, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to(device).eval()
        self.tok = self.processor.tokenizer
        self.device, self.batch_size, self.max_ctx_chars = device, batch_size, max_ctx_chars

    def _chat(self, t, add_answer=None):
        ctx = (t.get("context") or "")[: self.max_ctx_chars]
        text = PROMPT.format(ctx=ctx, q=t["question"]) if ctx else PROMPT_NOCTX.format(q=t["question"])
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]}]
        s = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if add_answer is not None:
            s = s + add_answer
        return s

    def _encode(self, texts, images):
        enc = self.processor(text=texts, images=[[im] for im in images],
                             padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def score_logit(self, triples):
        """Mean logP(answer tokens | image, question, context) per triple."""
        out = []
        for i in range(0, len(triples), self.batch_size):
            chunk = triples[i : i + self.batch_size]
            answers = [(t["answer"][0] if isinstance(t["answer"], list) else t["answer"]) for t in chunk]
            texts = [self._chat(t, add_answer=a) for t, a in zip(chunk, answers)]
            enc = self._encode(texts, [t["image"] for t in chunk])
            logits = self.model(**enc).logits[:, :-1]           # (B, L-1, V) bf16
            tgt = enc["input_ids"][:, 1:]
            tok_logit = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
            # reduce logsumexp in bf16 — a .float() copy of the (B, L, V) tensor is ~10GB at bs16
            tok_logp = tok_logit - torch.logsumexp(logits, dim=-1).float()   # (B, L-1)
            for j, a in enumerate(answers):
                n_ans = len(self.tok(a, add_special_tokens=False)["input_ids"])
                mask = enc["attention_mask"][j, 1:].bool()
                real = tok_logp[j][mask]
                ans_lp = real[-n_ans:] if 0 < n_ans <= real.numel() else real
                out.append(ans_lp.mean().item())
        return out

    @torch.no_grad()
    def score_judge(self, triples, max_new_tokens=24, n_rollouts=1, temperature=0.7, return_preds=False):
        """Reward = fraction of sampled reader answers that are correct (cover-EM); n=1 → greedy 0/1."""
        sample = n_rollouts > 1 and temperature > 0
        bs = max(1, self.batch_size // (2 if sample else 1))
        out, preds = [], []
        for i in range(0, len(triples), bs):
            chunk = triples[i : i + bs]
            texts = [self._chat(t) for t in chunk]
            enc = self._encode(texts, [t["image"] for t in chunk])
            gen = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=sample,
                temperature=temperature if sample else None, top_p=0.95 if sample else None,
                num_return_sequences=n_rollouts, pad_token_id=self.tok.eos_token_id)
            plen = enc["input_ids"].shape[1]
            for j, t in enumerate(chunk):
                correct, first = 0.0, None
                for k in range(n_rollouts):
                    ans = self.tok.decode(gen[j * n_rollouts + k, plen:], skip_special_tokens=True).strip()
                    first = ans if first is None else first
                    correct += answer_correct(ans, t["answer"])
                out.append(correct / n_rollouts)
                preds.append(first)
        return (out, preds) if return_preds else out
