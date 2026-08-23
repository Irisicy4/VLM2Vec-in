"""Indirect-RL training mode ("Ours"): online PPO/GRPO on the multimodal retriever's LoRA adapters,
rewarded by a frozen VLM reader's answer success — multimodal port of ../rag/train_online_v2.py +
trl.experimental.embedding_ppo (single-hop).

Each step:
  1. sample B (image, question, answer, gold-passage) rows
  2. encode queries with the current policy; retrieve top-(N-1) passages from the RL corpus
     (+ force-include the gold passage at slot 0 as the InfoNCE positive, unless --no_force_gold)
  3. frozen VLM reader scores each candidate: logP(answer | image, question, passage) [logit] or
     sampled answer accuracy [judge]; per-pool z-score
  4. PPO-clip policy gradient over softmax(sim/temperature) pools (+ critic V(q), or GRPO with
     group-relative advantages and no critic) + InfoNCE anchor against collapse
Corpus embeddings are re-encoded with the current policy every --refresh_steps.

    python3 mmrag/train_rl.py --output_dir runs/rl-2b --algo grpo --reward judge \
        --batch_size 4 --num_candidates 8 --max_steps 500 --reader_device cuda:1
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmrag.encoder import ENCODER_PROFILES, load_encoder  # noqa: E402
from mmrag.image_store import TarImageStore, open_stores  # noqa: E402
from mmrag.reader_vlm import LiveVLMReader  # noqa: E402
from mmrag.rl_core import (  # noqa: E402
    ValueHead, compute_advantages_and_targets, embedding_ppo_loss, infonce_loss, pl_list_logps,
    pl_sample_lists, pool_logps, rloo_advantages,
)


def build_rl_corpus(data_dir, corpus_file, rows, n_distractor_articles, seed=0):
    """RL corpus = all passages of the train queries' gold entities + N distractor articles,
    filtered out of the prebuilt corpus (keeps pids consistent with eval corpora)."""
    gold_urls = {r["entity_url"] for r in rows}
    by_url = {}
    with open(os.path.join(data_dir, corpus_file)) as f:
        for line in f:
            p = json.loads(line)
            by_url.setdefault(p["url"], []).append(p)
    distract = sorted(u for u in by_url if u not in gold_urls)
    random.Random(seed).shuffle(distract)
    keep = gold_urls | set(distract[:n_distractor_articles])
    corpus = [p for u in keep if u in by_url for p in by_url[u]]
    return corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="gme2b", choices=list(ENCODER_PROFILES))
    ap.add_argument("--data_dir", default=os.environ.get("MMRAG_DATA", "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data"))
    ap.add_argument("--pool", default="built/pool_train.jsonl", help="rows: qid,image_id,question,answer,pos")
    ap.add_argument("--corpus", default="built/corpus_small.jsonl")
    ap.add_argument("--image_tars", nargs="+", default=None)
    ap.add_argument("--image_dataset", default="infoseek", choices=["infoseek", "evqa", "mix", "oven", "okvqa", "div"])
    ap.add_argument("--output_dir", required=True)
    # policy / optimization
    ap.add_argument("--algo", choices=["ppo", "grpo", "plgrpo"], default="grpo",
                    help="ppo/grpo: v1 listwise update over deterministic top-N pools (all pool "
                         "members weighted 1 — see tag v1-listwise-top8). plgrpo: proper "
                         "conditional-probability policy gradient — sample G ordered k-lists from "
                         "the Plackett-Luce policy over a top-M support, exact factorized list "
                         "log-probs, GRPO group = the G lists (advantage z-scored across them)")
    ap.add_argument("--pl_group", type=int, default=4, help="plgrpo: lists sampled per query (G)")
    ap.add_argument("--pl_k", type=int, default=4, help="plgrpo: list length (k)")
    ap.add_argument("--pl_support", type=int, default=24,
                    help="plgrpo: candidate support M = top-M by current policy; the PL denominator "
                         "runs over this set")
    ap.add_argument("--kl_beta", type=float, default=0.0,
                    help="plgrpo: k3 KL-to-init on the sampled lists (reference = adapters "
                         "disabled == the init policy). RL-native anchor replacing InfoNCE.")
    ap.add_argument("--entropy_coef", type=float, default=0.0,
                    help="plgrpo: first-position PL entropy bonus over the support (anti-collapse)")
    ap.add_argument("--baseline", choices=["group_z", "rloo"], default="group_z",
                    help="plgrpo advantage baseline: per-group z-score (v2 default) or RLOO "
                         "leave-one-out mean (unbiased, no std division)")
    ap.add_argument("--pl_behavior_temperature", type=float, default=None,
                    help="plgrpo: SAMPLE lists at this temperature while scoring log-probs at "
                         "--temperature; old_logps become the behavior log-probs, so the PPO "
                         "ratio is the true importance weight (off-policy clipped PG)")
    ap.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant",
                    help="cosine: linear warmup then cosine decay to 10% of peak over max_steps. "
                         "Every recorded run used constant LR; the rise-then-decay consumption "
                         "shape is the classic constant-LR overtraining signature.")
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--inner_epochs", type=int, default=1,
                    help=">1 re-runs the grad forward + optimizer step on the SAME rollout, making "
                         "the PPO ratio/clip meaningful (with 1 epoch ratio==1 identically and the "
                         "clip never engages)")
    ap.add_argument("--temperature", type=float, default=0.02, help="policy softmax temperature")
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_candidates", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.1, help="0 with --algo grpo")
    ap.add_argument("--value_head_lr", type=float, default=1e-2)
    ap.add_argument("--value_init", type=float, default=0.3)
    ap.add_argument("--critic_warmup_steps", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.0, help="KL-to-init coefficient (0 = off)")
    ap.add_argument("--contrastive_coef", type=float, default=0.3)
    ap.add_argument("--contrastive_temperature", type=float, default=0.03)
    ap.add_argument("--no_force_gold", action="store_true",
                    help="annotation-free pools: pure top-N (gold only enters if retrieved). "
                         "Disables the InfoNCE anchor's labeled positive -> uses top-1 as anchor.")
    ap.add_argument("--pool_sampling", action="store_true",
                    help="HARR-style stochastic pools: sample the candidates from "
                         "softmax(sims/temperature) over the top-4N without replacement, instead "
                         "of deterministic top-N (text finding: unstable without a gold anchor)")
    ap.add_argument("--pool_sample_temperature", type=float, default=None,
                    help="temperature for --pool_sampling (default: the policy temperature; "
                         "at tau=0.02 sampling is near-argmax — raise to test true stochasticity)")
    ap.add_argument("--reward_gate_std", type=float, default=0.0,
                    help="zero the advantage of pools whose raw reward std < this (no signal)")
    ap.add_argument("--reward_gate_noctx", action="store_true",
                    help="NO-CONTEXT UTILITY GATE (text-side port): keep a query only if its BEST "
                         "candidate beats the reader's no-context accuracy, i.e. "
                         "max_c judge(c) > judge(no context). Drops queries retrieval cannot help "
                         "on — either already answerable from the image, or not answerable at all. "
                         "Unlike --reward gain (which SHIFTS every reward by the no-context "
                         "baseline) this is a per-query DATA FILTER: kept queries train on their "
                         "unmodified rewards, dropped queries contribute no gradient.")
    # reward
    ap.add_argument("--reward", choices=["logit", "judge", "ragacc", "gain", "mix",
                                         "goldrel", "ansmatch"], default="logit",
                    help="logit/judge score each passage ALONE; ragacc = downstream RAG accuracy: "
                         "sample a k-slate from the policy, reader answers over the JOINT top-k "
                         "context, one scalar reward per query, REINFORCE on the slate members; "
                         "gain = judge(passage) - judge(no context) per query (answer-GAIN: kills "
                         "the spurious reward on queries the reader answers without retrieval); "
                         "mix = mean of per-pool z-scored judge and logit rewards; "
                         "CONTROLS (no reader, benchmark the indirect signal through the SAME RL "
                         "loop): goldrel = 1 iff passage belongs to the gold entity (direct "
                         "relevance-label reward); ansmatch = 1 iff an answer alias appears "
                         "token-bounded in the passage (lexical oracle, answers-only supervision)")
    ap.add_argument("--slate_k", type=int, default=5, help="slate size for --reward ragacc")
    ap.add_argument("--reward_server_url", default=None,
                    help="http://host:port of mmrag/reward_server.py; if set, rewards are fetched "
                         "remotely (shared, possibly eval-grade reader) instead of in-process")
    ap.add_argument("--reader_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--reader_device", default=None, help="default: same as --device")
    ap.add_argument("--reader_rollouts", type=int, default=4)
    ap.add_argument("--reader_temperature", type=float, default=0.7)
    ap.add_argument("--reader_batch_size", type=int, default=16)
    # corpus
    ap.add_argument("--n_distractor_articles", type=int, default=3000)
    ap.add_argument("--refresh_steps", type=int, default=100)
    ap.add_argument("--index_ema", type=float, default=0.0,
                    help="EMA index (mmrag-ema scheme 1+2): 0 = legacy full re-encode every "
                         "refresh_steps. m>0 = NO periodic full refresh; instead every step the "
                         "docs already encoded fresh for the pools (the policy's own top "
                         "candidates + the near-boundary docs just past the support) plus "
                         "--index_refresh_extra of the STALEST corpus docs are written back as "
                         "index[p] <- normalize(m*old + (1-m)*new). Index maintenance becomes "
                         "O(pool+extra) per step instead of O(corpus) per refresh.")
    ap.add_argument("--index_refresh_extra", type=int, default=256,
                    help="EMA index: stalest docs re-encoded and written back per step")
    ap.add_argument("--encode_bs", type=int, default=64)
    ap.add_argument("--max_len_doc", type=int, default=512)
    ap.add_argument("--max_train_rows", type=int, default=0)
    # misc
    ap.add_argument("--query_instruction", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem_frac", type=float, default=0.0,
                    help="hard per-process VRAM cap for GPU co-location (0 = uncapped)")
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # --beta was parsed but never read (the live KL flag is --kl_beta) — the same dead-flag
    # defect class the text side found on their PL path. Alias it so a KL request can never
    # silently no-op; conflicting nonzero values are a hard error.
    if args.beta and not args.kl_beta:
        args.kl_beta = args.beta
    elif args.beta and args.kl_beta and args.beta != args.kl_beta:
        ap.error(f"--beta {args.beta} conflicts with --kl_beta {args.kl_beta}; set only one")
    if args.reward == "ragacc":
        args.algo = "grpo"  # slate reward is per-query scalar; critic/value path not wired for it
    if args.algo == "grpo":
        args.vf_coef = 0.0
    D = args.data_dir
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(l) for l in open(os.path.join(D, args.pool))]
    store = (TarImageStore([t for t in args.image_tars if os.path.exists(t)])
             if args.image_tars else open_stores(D, args.image_dataset))
    rows = [r for r in rows if r["image_id"] in store and r.get("pos")]
    if args.max_train_rows:
        rows = rng.sample(rows, min(args.max_train_rows, len(rows)))
    print(f"train rows: {len(rows)}", flush=True)
    if len(rows) < args.batch_size:
        sys.exit(f"SKIP: only {len(rows)} usable rows (image archives missing?) — nothing to train on")

    corpus = build_rl_corpus(D, args.corpus, rows, args.n_distractor_articles, args.seed)
    corpus_texts = [p["text"] for p in corpus]
    text2url = {p["text"]: p["url"] for p in corpus}   # for the goldrel benchmark reward
    print(f"RL corpus: {len(corpus)} passages", flush=True)

    # Stacking convention (shared with the text session): a hard per-process VRAM cap so a
    # co-located lane takes an allocator error instead of OOMing its neighbour. 0 = uncapped.
    if args.mem_frac > 0:
        dev = int(args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, dev)
    enc = load_encoder(args.profile, device=args.device,
                                    max_len=args.max_len_doc, new_lora_r=args.lora_r,
                                    new_lora_alpha=args.lora_alpha,
                                    query_instruction=args.query_instruction)
    enc.gradient_checkpointing_enable()
    enc.train()
    # deterministic policy: kill dropout so rollout logps == loss-forward logps (PPO ratio noise)
    for m in enc.model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    value_head = ValueHead(enc.hidden_size, init_bias=args.value_init).to(args.device)
    params = enc.trainable_parameters()
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(
        [{"params": params, "lr": args.learning_rate},
         {"params": value_head.parameters(), "lr": args.value_head_lr, "weight_decay": 0.0}])
    sched = None
    if args.lr_schedule == "cosine":
        import math

        def _lr_lambda(step):
            if step < args.warmup_steps:
                return (step + 1) / max(args.warmup_steps, 1)
            t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    if args.reward in ("goldrel", "ansmatch"):
        reader = None  # gold-derived benchmark rewards need no reader
    elif args.reward_server_url:
        from mmrag.reward_server import RemoteReader

        reader = RemoteReader(args.reward_server_url,
                              max_ctx_chars=(args.slate_k * 900 if args.reward == "ragacc" else 1600))
    else:
        reader = LiveVLMReader(args.reader_model, device=args.reader_device or args.device,
                               batch_size=args.reader_batch_size,
                               max_ctx_chars=(args.slate_k * 900 if args.reward == "ragacc" else 1600))

    corpus_emb = None

    @torch.no_grad()
    def refresh_corpus():
        nonlocal corpus_emb
        t0 = time.time()
        embs = []
        for i in range(0, len(corpus_texts), args.encode_bs):
            e = enc.encode_docs(corpus_texts[i : i + args.encode_bs], batch_size=args.encode_bs)
            embs.append(e.float())
        corpus_emb = torch.cat(embs)  # (C, d) on device fp32
        print(f"  corpus refreshed in {time.time()-t0:.0f}s", flush=True)

    # EMA index state: per-doc last-refresh step (staleness drives the extra-refresh sample)
    doc_last_upd = None

    @torch.no_grad()
    def ema_write(idxs, new_emb, step, m):
        """index[p] <- l2norm(m*old + (1-m)*new) for corpus rows idxs; marks them fresh."""
        nonlocal corpus_emb
        idx_t = torch.as_tensor(idxs, device=corpus_emb.device, dtype=torch.long)
        mixed = m * corpus_emb[idx_t] + (1.0 - m) * new_emb.to(corpus_emb.device).float()
        corpus_emb[idx_t] = torch.nn.functional.normalize(mixed, dim=-1)
        doc_last_upd[idx_t] = step

    @torch.no_grad()
    def ema_refresh_stale(step):
        """Re-encode the stalest --index_refresh_extra docs with the CURRENT policy and
        EMA-write them back — the deliberate less-frequent-sample refresh (scheme 2)."""
        k = min(args.index_refresh_extra, len(corpus_texts))
        if k <= 0:
            return
        idxs = torch.topk(-doc_last_upd, k).indices.tolist()
        e = enc.encode_docs([corpus_texts[i] for i in idxs], batch_size=args.encode_bs).float()
        ema_write(idxs, e, step, args.index_ema)

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=2)
    N = args.pl_support if args.algo == "plgrpo" else args.num_candidates
    if args.algo == "plgrpo":
        assert args.no_force_gold, "plgrpo is defined on pure policy pools (--no_force_gold)"
    metrics_log = open(os.path.join(args.output_dir, "metrics.jsonl"), "a")

    for step in range(args.max_steps):
        extra_metrics = {}  # per-step; only the plgrpo branch populates it
        if corpus_emb is None:
            refresh_corpus()  # one full encode at init, both modes
            if args.index_ema > 0:
                assert args.no_force_gold, "EMA index write-back needs pure policy pools"
                doc_last_upd = torch.zeros(len(corpus_texts), dtype=torch.long,
                                           device=corpus_emb.device)
        elif args.index_ema == 0 and args.refresh_steps and step % args.refresh_steps == 0:
            refresh_corpus()

        batch = rng.sample(rows, args.batch_size)
        imgs = [store.get(r["image_id"]) for r in batch]
        questions = [r["question"] for r in batch]

        # ---- rollout (no grad): retrieve pools, reward, old logps/values ----
        with torch.no_grad():
            q_emb = enc.encode_queries(questions, imgs, batch_size=args.batch_size).float()
            sims = q_emb @ corpus_emb.T                              # (B, C)
            K = (4 * N + 4) if args.pool_sampling else (N + 4)
            topk = sims.topk(min(K, sims.shape[1]), dim=1).indices.tolist()
            pools, triples, gold_hit = [], [], 0
            gold_in_pool = []          # per-query, for the gate x pool-content contingency
            pool_ids, boundary_ids = [], []   # corpus indices for EMA index write-back
            for b, r in enumerate(batch):
                gold = (r.get("pos") or [""])[0]   # div-pool rows may carry no gold passage
                retrieved = [corpus_texts[i] for i in topk[b]]
                _hit = float(gold in retrieved[:N])
                gold_in_pool.append(_hit)
                gold_hit += _hit
                if args.pool_sampling:
                    # stochastic pools: sample w/o replacement from softmax over the wider top-K
                    pool_idx = [i for i in topk[b] if args.no_force_gold or corpus_texts[i] != gold]
                    want = N if args.no_force_gold else N - 1
                    samp_t = args.pool_sample_temperature or args.temperature
                    logits = sims[b, pool_idx] / max(samp_t, 1e-6)
                    pick = torch.multinomial(torch.softmax(logits, dim=-1),
                                             min(want, len(pool_idx)), replacement=False)
                    chosen = [corpus_texts[pool_idx[j]] for j in pick.tolist()]
                    cands = chosen if args.no_force_gold else [gold] + chosen
                elif args.no_force_gold:
                    cands = retrieved[:N]
                    pool_ids.extend(topk[b][:N])          # exact ids of the encoded pool docs
                    boundary_ids.extend(topk[b][N:])      # near-boundary = free hard negatives
                else:
                    cands = [gold] + [t for t in retrieved if t != gold][: N - 1]
                pools.append(cands)
                ans = r.get("answer_aliases") or r["answer"]
                triples.extend({"image": imgs[b], "image_id": r["image_id"],
                                "question": r["question"], "context": c, "answer": ans}
                               for c in cands)

            flat = [c for pool in pools for c in pool]
            c_emb = enc.encode_docs(flat, batch_size=args.encode_bs).float().view(args.batch_size, N, -1)
            if args.index_ema > 0 and len(pool_ids) == args.batch_size * N:
                # scheme 1+2: pool docs were just encoded fresh — EMA them into the index,
                # re-encode the near-boundary docs (hard negatives) and the stalest tail.
                ema_write(pool_ids, c_emb.view(-1, c_emb.shape[-1]), step, args.index_ema)
                if boundary_ids:
                    be = enc.encode_docs([corpus_texts[i] for i in boundary_ids],
                                         batch_size=args.encode_bs).float()
                    ema_write(boundary_ids, be, step, args.index_ema)
                ema_refresh_stale(step)
                extra_metrics["index/stale_mean"] = float(
                    (step - doc_last_upd).float().mean())
            old_logps, _ = pool_logps(q_emb, c_emb, args.temperature)
            old_values = value_head(q_emb)
            gate = None
            noctx_keep_rate = None
            noctx_gate_xtab = None

            pl_idx = old_list_lp = ref_list_lp = None
            if args.algo == "plgrpo":
                # ---- conditional-probability GRPO: sample G ordered k-lists from the PL policy
                # over the top-M support; group = the G lists of one query ----
                assert args.reward in ("judge", "logit"), "plgrpo: reward must be judge or logit"
                _, old_scaled = pool_logps(q_emb, c_emb, args.temperature)      # (B, M) sims/tau
                if args.pl_behavior_temperature:
                    # explore at T_b, score at the target temperature; old_logps = BEHAVIOR
                    # log-probs so the clipped ratio is the true importance weight
                    _, beh_scaled = pool_logps(q_emb, c_emb, args.pl_behavior_temperature)
                    pl_idx = pl_sample_lists(beh_scaled, args.pl_k, args.pl_group)
                    old_list_lp = pl_list_logps(beh_scaled, pl_idx)
                else:
                    pl_idx = pl_sample_lists(old_scaled, args.pl_k, args.pl_group)  # (B, G, k)
                    old_list_lp = pl_list_logps(old_scaled, pl_idx)                 # (B, G)
                ref_list_lp = None
                if args.kl_beta > 0:
                    with enc.ref_ctx():
                        ref_q = enc.encode_queries(questions, imgs, batch_size=args.batch_size).float()
                        ref_c = enc.encode_docs(flat, batch_size=args.encode_bs).float().view(
                            args.batch_size, N, -1)
                    _, ref_scaled = pool_logps(ref_q, ref_c, args.temperature)
                    ref_list_lp = pl_list_logps(ref_scaled, pl_idx)             # (B, G) fixed
                # reader scores each UNIQUE sampled doc once; list reward = mean over positions
                # (per-document credit — the joint-context scalar variant collapsed, see ragacc)
                uniq, owner = [], []
                for b in range(args.batch_size):
                    docs = sorted(set(pl_idx[b].flatten().tolist()))
                    owner.append({j: len(uniq) + i for i, j in enumerate(docs)})
                    r = batch[b]
                    ans = r.get("answer_aliases") or r["answer"]
                    uniq.extend({"image": imgs[b], "image_id": r["image_id"],
                                 "question": r["question"], "context": pools[b][j],
                                 "answer": ans} for j in docs)
                if args.reward == "judge":
                    r_doc = reader.score_judge(uniq, n_rollouts=args.reader_rollouts,
                                               temperature=args.reader_temperature)
                else:
                    r_doc = reader.score_logit(uniq)
                r_doc = torch.tensor(r_doc, dtype=torch.float32, device=args.device)
                R = torch.zeros(args.batch_size, args.pl_group, device=args.device)
                for b in range(args.batch_size):
                    for g in range(args.pl_group):
                        ids = [owner[b][j] for j in pl_idx[b, g].tolist()]
                        R[b, g] = r_doc[ids].mean()
                raw_mean, gold_mean = R.mean().item(), float("nan")
                # Within-group reward discriminability: std=0 for a group forces its z-scored
                # advantage (and hence its gradient contribution) to exactly zero.
                grp_std = R.std(1)
                extra_metrics = {"reward/group_std_mean": grp_std.mean().item(),
                                 "reward/group_degenerate_frac": (grp_std == 0).float().mean().item()}
                if args.baseline == "rloo":
                    advantages = rloo_advantages(R)
                else:
                    advantages = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)
                value_target = R.new_zeros(args.batch_size)
                rewards = R
            elif args.reward == "ragacc":
                # downstream RAG accuracy as the reward: sample a k-slate per query from the
                # current policy, reader answers over the JOINT slate context; the scalar reward
                # is batch-z-scored and applied to the sampled members (REINFORCE on the slate).
                k = min(args.slate_k, N)
                slates = torch.multinomial(old_logps.exp(), k, replacement=False)  # (B, k)
                slate_triples = []
                for b, r in enumerate(batch):
                    ctx = "\n\n".join(pools[b][j] for j in slates[b].tolist())
                    slate_triples.append({"image": imgs[b], "image_id": r["image_id"],
                                          "question": r["question"], "context": ctx,
                                          "answer": r.get("answer_aliases") or r["answer"]})
                slate_r = reader.score_judge(slate_triples, n_rollouts=args.reader_rollouts,
                                             temperature=args.reader_temperature)
                slate_r = torch.tensor(slate_r, dtype=torch.float32, device=args.device)  # (B,)
                raw_mean, gold_mean = slate_r.mean().item(), float("nan")
                adv_q = (slate_r - slate_r.mean()) / (slate_r.std() + 1e-6)              # (B,)
                advantages = torch.zeros_like(old_logps)
                advantages.scatter_(1, slates, adv_q.unsqueeze(1).expand(-1, k))
                value_target = advantages.new_zeros(args.batch_size)
                rewards = advantages  # for logging shape-compat
            elif args.reward in ("goldrel", "ansmatch"):
                # gold-derived benchmark rewards: same RL loop, no reader involved
                from mmrag.reader_vlm import answer_correct as _am

                vals = []
                for b, r in enumerate(batch):
                    for c in pools[b]:
                        if args.reward == "goldrel":
                            vals.append(1.0 if text2url.get(c) == r["entity_url"] else 0.0)
                        else:
                            vals.append(_am(c, r.get("answer_aliases") or [r["answer"]]))
                rewards = torch.tensor(vals, dtype=torch.float32,
                                       device=args.device).view(args.batch_size, N)
                raw_mean, gold_mean = rewards.mean().item(), rewards[:, 0].mean().item()
                rewards = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)
                if args.algo == "grpo":
                    advantages, value_target = rewards, rewards.new_zeros(args.batch_size)
                else:
                    advantages, value_target = compute_advantages_and_targets(rewards, old_values, old_logps)
            else:
                if args.reward in ("judge", "gain"):
                    rewards = reader.score_judge(triples, n_rollouts=args.reader_rollouts,
                                                 temperature=args.reader_temperature)
                elif args.reward == "mix":
                    rj = torch.tensor(reader.score_judge(triples, n_rollouts=args.reader_rollouts,
                                                         temperature=args.reader_temperature),
                                      dtype=torch.float32, device=args.device).view(args.batch_size, N)
                    rl_ = torch.tensor(reader.score_logit(triples), dtype=torch.float32,
                                       device=args.device).view(args.batch_size, N)
                    z = lambda x: (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
                    rewards = (0.5 * z(rj) + 0.5 * z(rl_)).flatten().tolist()
                else:
                    rewards = reader.score_logit(triples)
                rewards = torch.tensor(rewards, dtype=torch.float32,
                                       device=args.device).view(args.batch_size, N)
                r_no = None
                if args.reward == "gain" or args.reward_gate_noctx:
                    noctx = [{"image": imgs[b], "image_id": r["image_id"], "question": r["question"],
                              "context": "", "answer": r.get("answer_aliases") or r["answer"]}
                             for b, r in enumerate(batch)]
                    r_no = torch.tensor(reader.score_judge(noctx, n_rollouts=args.reader_rollouts,
                                                           temperature=args.reader_temperature),
                                        dtype=torch.float32, device=args.device)
                if args.reward == "gain":
                    # subtract the per-query NO-CONTEXT accuracy: only passages that CHANGE the
                    # reader's answer earn reward (queries answerable without retrieval give 0)
                    rewards = rewards - r_no.unsqueeze(1)
                raw_mean, gold_mean = rewards.mean().item(), rewards[:, 0].mean().item()
                if args.reward_gate_noctx:
                    # DATA FILTER, not a reward shift: keep the query only if retrieval can beat
                    # answering with no context at all. Kept queries keep their raw rewards.
                    keep = (rewards.max(dim=1).values > r_no).float().unsqueeze(1)
                    gate = keep if gate is None else gate * keep
                    noctx_keep_rate = keep.mean().item()
                    # Can reader-utility gating tell a USELESS pool from a useful one? Cross the
                    # keep decision with whether the pool actually contains the gold passage.
                    # A gate that keeps gold-free pools at a similar rate is not a pool-quality
                    # detector, however well it separates rewards.
                    _g = torch.tensor(gold_in_pool, dtype=torch.float32, device=rewards.device)
                    _k = keep.squeeze(1)
                    noctx_gate_xtab = {
                        "n_gold_in_pool": int(_g.sum().item()),
                        "n_no_gold_in_pool": int((1 - _g).sum().item()),
                        "keep_given_gold": (_k * _g).sum().item() / max(_g.sum().item(), 1e-9),
                        "keep_given_no_gold": (_k * (1 - _g)).sum().item() / max((1 - _g).sum().item(), 1e-9),
                    }
                if args.reward_gate_std > 0:
                    gate = (rewards.std(1, keepdim=True) >= args.reward_gate_std).float()
                rewards = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)
                if args.algo == "grpo":
                    advantages, value_target = rewards, rewards.new_zeros(args.batch_size)
                else:
                    advantages, value_target = compute_advantages_and_targets(rewards, old_values, old_logps)
                if gate is not None:
                    advantages = advantages * gate
                    if args.algo == "ppo":
                        value_target = value_target * gate.squeeze(1) + old_values * (1 - gate.squeeze(1))

        # ---- loss forward (grad): re-encode the same pools with the trainable policy.
        # inner_epochs > 1 repeats this on the SAME rollout, so the PPO ratio moves off 1
        # and the clip actually constrains the update. ----
        skip_step = False
        for _ep in range(max(1, args.inner_epochs)):
            q_emb_g = enc.encode_queries(questions, imgs, batch_size=args.batch_size, grad=True).float()
            c_emb_g = enc.encode_docs(flat, batch_size=args.encode_bs, grad=True).float().view(args.batch_size, N, -1)
            if args.algo == "plgrpo":
                _, scaled_g = pool_logps(q_emb_g, c_emb_g, args.temperature)
                logps = pl_list_logps(scaled_g, pl_idx)                       # (B, G) list logps
                eff_old = old_list_lp
            else:
                logps, _ = pool_logps(q_emb_g, c_emb_g, args.temperature)
                eff_old = old_logps
            values = value_head(q_emb_g)
            loss, m = embedding_ppo_loss(
                logps=logps, old_logps=eff_old, advantages=advantages,
                values=values, old_values=old_values, value_target=value_target,
                epsilon_low=args.epsilon, epsilon_high=args.epsilon, vf_coef=args.vf_coef,
                disable_policy=(args.algo == "ppo" and step < args.critic_warmup_steps))
            if args.algo == "plgrpo" and args.kl_beta > 0 and ref_list_lp is not None:
                # k3 estimator on the sampled lists: exp(ref-pi) - (ref-pi) - 1  >= 0
                d_lp = ref_list_lp - logps
                kl3 = (d_lp.exp() - d_lp - 1).mean()
                loss = loss + args.kl_beta * kl3
                m["loss/kl_init"] = kl3.item()
            if args.algo == "plgrpo" and args.entropy_coef > 0:
                p1 = torch.softmax(scaled_g, dim=-1)
                ent = -(p1 * torch.log(p1 + 1e-9)).sum(-1).mean()
                loss = loss - args.entropy_coef * ent
                m["policy/entropy1"] = ent.item()
            if args.contrastive_coef > 0:
                nce = infonce_loss(q_emb_g, c_emb_g, args.contrastive_temperature)
                loss = loss + args.contrastive_coef * nce
                m["loss/infonce"] = nce.item()

            if not torch.isfinite(loss):
                opt.zero_grad()
                nan_streak = getattr(main, "_nan_streak", 0) + 1
                main._nan_streak = nan_streak
                print(f"step {step:>4} NON-FINITE loss — skipping optimizer step ({nan_streak} in a row)", flush=True)
                if nan_streak >= 30:
                    sys.exit("ABORT: 30 consecutive non-finite losses — model has diverged")
                skip_step = True
                break
            main._nan_streak = 0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
        if sched is not None:
            sched.step()
            m["lr"] = sched.get_last_lr()[0]
        if skip_step:
            continue

        m.update({"step": step, "loss": loss.item(), "reward/raw_mean": raw_mean,
                  "reward/gold_mean": gold_mean, "retrieval/gold_in_topN": gold_hit / args.batch_size})
        m.update(extra_metrics)
        if noctx_keep_rate is not None:
            m["gate/noctx_keep_rate"] = noctx_keep_rate
        if noctx_gate_xtab is not None:
            m.update({f"gate/{k}": v for k, v in noctx_gate_xtab.items()})
        metrics_log.write(json.dumps(m) + "\n")
        metrics_log.flush()
        if step % args.log_every == 0:
            print(f"step {step:>4} loss={loss.item():.4f} R_raw={raw_mean:.3f} "
                  f"R_gold={gold_mean:.3f} gold@N={gold_hit/args.batch_size:.2f} "
                  f"pg={m['loss/policy']:.4f} nce={m.get('loss/infonce', 0):.4f}", flush=True)
        if args.save_steps and step and step % args.save_steps == 0:
            enc.save_adapters(os.path.join(args.output_dir, f"checkpoint-{step}"))
            torch.save(value_head.state_dict(), os.path.join(args.output_dir, f"checkpoint-{step}", "value_head.pt"))

    enc.save_adapters(args.output_dir)
    torch.save(value_head.state_dict(), os.path.join(args.output_dir, "value_head.pt"))
    print(f"saved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
