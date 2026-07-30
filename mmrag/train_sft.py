"""Relevance-SFT baseline (the "direct supervision" arm): contrastive InfoNCE fine-tuning of the
multimodal encoder on InfoSeek gold-evidence pairs — multimodal port of ../rag/train_mnrl.py.

query = image + question, positive = gold-entity passage (answer-bearing if available), negatives =
in-batch (+ optional mined hard negatives from a `neg` key produced by mine_hard_negs.py).
Trains fresh LoRA adapters on the merged VLM2Vec encoder — same trainable budget as the RL arm.

    python3 mmrag/train_sft.py --pool built/pool_train.jsonl --output_dir runs/sft-2b \
        --batch_size 32 --learning_rate 1e-4 --max_steps 600
"""

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.encoder import ENCODER_PROFILES, MMRagEncoder  # noqa: E402
from mmrag.image_store import TarImageStore, open_stores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="gme2b", choices=list(ENCODER_PROFILES))
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA", "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data"))
    ap.add_argument("--pool", default="built/pool_train.jsonl")
    ap.add_argument("--image_tars", nargs="+", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_hard_negs", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.03)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--max_len_doc", type=int, default=512)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--query_instruction", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    D = args.data_dir

    rows = [json.loads(l) for l in open(os.path.join(D, args.pool))]
    store = (TarImageStore([t for t in args.image_tars if os.path.exists(t)])
             if args.image_tars else open_stores(D, "infoseek"))
    rows = [r for r in rows if r["image_id"] in store and r.get("pos")]
    print(f"train rows with images: {len(rows)}", flush=True)

    enc = MMRagEncoder.from_profile(args.profile, device=args.device,
                                    max_len=args.max_len_doc, new_lora_r=args.lora_r,
                                    new_lora_alpha=args.lora_alpha,
                                    query_instruction=args.query_instruction)
    enc.gradient_checkpointing_enable()
    enc.train()

    params = enc.trainable_parameters()
    n_train = sum(p.numel() for p in params)
    print(f"trainable params: {n_train/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.learning_rate)
    rng = random.Random(args.seed)
    B = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    for step in range(args.max_steps):
        batch = rng.sample(rows, B)
        imgs = [store.get(r["image_id"]) for r in batch]
        positives = [r["pos"][0] for r in batch]
        hard = [n for r in batch for n in (r.get("neg") or [])[: args.num_hard_negs]]

        Q = enc.encode_queries([r["question"] for r in batch], imgs, batch_size=B, grad=True).float()
        C = enc.encode_docs(positives + hard, batch_size=B, grad=True).float()
        scores = (Q @ C.T) / args.temperature      # (B, B+H) in-batch positives + hard negs
        labels = torch.arange(B, device=Q.device)
        loss = F.cross_entropy(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        if step % 10 == 0:
            with torch.no_grad():
                acc = (scores.argmax(1) == labels).float().mean().item()
            print(f"step {step:>4} loss={loss.item():.4f} inbatch_acc={acc:.3f}", flush=True)
        if args.save_steps and step and step % args.save_steps == 0:
            enc.save_adapters(os.path.join(args.output_dir, f"checkpoint-{step}"))

    enc.save_adapters(args.output_dir)
    print(f"saved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
