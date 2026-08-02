# mm-RAG experiment ledger — tested vs. works

All VQA numbers: v2 token-boundary cover-EM, top-5, Qwen2.5-VL-7B reader, n=1500 fixed InfoSeek-val
sample. Reference points: zero-shot GME-2B **0.303**, best-tuned SFT **0.321** (lr2e-4 h0),
gold-oracle context 0.503. PROVISIONAL pending review. Evidence = run name in
`results_summary.json` (→ job id, config, log).

| Axis | Cells tested | Verdict | Evidence |
|---|---|---|---|
| **[no-gold] annotation-free RL (HEADLINE)** | judge det lr2e-5, seeds 0/1/2 (+3/4 running) | **WORKS: 0.341/0.339, > SFT p=0.024/0.027 per seed; no collapse** — text side could not do this | rl-j2e5-nogold-det{,-s1,-s2} |
| Grounding (gold seed in pool) | gold vs no-gold, det+samp | **Not needed in-domain** (0.343 vs 0.341, p=0.93); **needed for OOD transfer** (E-VQA: gold 0.462 > zs 0.426 > no-gold 0.397) | m6b, m6q, m6x |
| Pool construction | det vs sampled (τ=0.02) vs sampled τ=0.1 | det best (0.341 > 0.333 > 0.326); **sampling never collapses** (text: collapsed unconditionally) | rl-j2e5-nogold-{det,samp}, -samp-t01 |
| Pool size N | 8 / 16 / 32 | N=8 best (0.341 vs 0.319–0.333); N16 rose during training but evals worse | m6l/m6o/m6p N16, N32 |
| Reward: judge (binary acc) | 2B+7B, gold+nogold | **best indirect reward** (0.341 nogold, 0.359 7B-gold) | rl-grpo-judge-*, rl7b-j-* |
| Reward: logit (dense logP) | gold, nogold | OK gold (0.329–0.341); **FAILS no-gold (0.315)** — inverts text preference | rl-logit2e5-nogold-det |
| Reward: goldrel (relevance oracle) | gold, nogold | ties at gold (0.341); **loses no-gold (0.317)** — indirect > direct-as-reward on policy pools | m6v |
| Reward: ansmatch (lexical, no VLM) | 2B gold/nogold, 7B nogold | competitive (0.346/0.334; 7B 0.352 ≈ judge) — reader adds little over lexical at 7B, small edge at 2B (n.s. p=0.32) | m6v, m6w |
| Reward: gain (judge − noctx) | gold, nogold | no gain (0.337 both) | m6t |
| Reward: mix (judge+logit) | gold | ~parity (see results file) | m6t |
| Reward: ragacc (slate top-k acc) | lr2e-5/1e-5, 500/1500, N16, 7B-server | **FAILS/collapses** (0.14–0.21; OOD 0.097) — per-passage credit needed | m6k/m6p ragacc cells |
| Reward reader size | 3B in-process vs 7B server | 3B suffices (0.341 vs 0.337) | m6n |
| Reward rollouts | 1 / 4 / 8 | 4 fine; 1 and 8 no better (0.3?–0.339) | m6t roll1, m6g roll8 |
| InfoNCE anchor coef | 0 / 0.1 / 0.3 (gold; no-gold cells running) | helps ~+1pt (0.333/0.341/0.343); **RL works without it** (cc0 still > SFT) | m6g cc0/cc01 |
| Index liveness | live (refresh 100) vs frozen | ~tie no-gold (0.336–0.341); frozen not needed (text: frozen was the only no-gold rescue) | rl-judge-nogold-frozen |
| Algorithm | GRPO vs PPO-critic | tie (0.343 vs 0.337–0.345) | m6a/m6k ppo-judge |
| LR (2B) | 1e-5 / 2e-5 / 3e-5 / 5e-5 | 2e-5–3e-5 best; 5e-5 degrades | m4/m6a/m6j sweeps |
| LR (7B) | 5e-6 / 1e-5 | 1e-5 best (0.359) — lower than 2B optimum ✓ capacity rule | m6e |
| Capacity ladder | SigLIP2 / CLIP-L / 2B / 7B | **RL−SFT gap grows with capacity**: −19pts (SigLIP2 R@5) / −4 (CLIP) / +2.2 (2B) / +3.3 (7B, p=4e-4) | m5, m6e, m6w |
| Data scale (fixed 500 steps) | 2k / 8k / 20k / 41k, all arms | RL rises monotonically (0.329→0.343 gold; 0.333→0.341 nogold); SFT non-monotone/flat | m6d/m6r, fig_mm_scaling.pdf |
| Budget × data | 1500 steps × 2k/8k/41k | longer training does NOT help (0.300–0.344); 2k+1500 overfits (0.300); no-gold 1500 drifts (0.319) | m6g/m6o/m6u |
| Data mix | InfoSeek+EVQA 50/50, EVQA-only | **mix preserves both domains** (0.344 IS + 0.477 EV vs 0.313/0.479) — diversity pays in-structure ✓ | m6s |
| E-VQA in-domain | zs/SFT/RL | RL 0.479 > SFT 0.450 > zs 0.426 — replicates on 2nd dataset | m6s |
| OOD transfer (IS→EVQA) | zs/SFT/gold-RL/no-gold-RL | gold-RL 0.462 > zs 0.426 > no-gold 0.397 ≈ frozen 0.398 > SFT 0.395; **no-gold-mix cells running** (m6ac) | m6q/m6x/m6f |
| Corpus scale | 422K vs 1.42M passages | ranking preserved (RL-judge 0.328 > logit 0.317 > SFT 0.305 > zs 0.289) | m6h |
| Statistical significance | paired McNemar + bootstrap, per-question | nogold>SFT SIG per seed; 7B all SIG; gold≈nogold n.s.; 2B-vs-zs alignment eval running (m6ab) | mmrag/significance.py output |

Running at ledger time: m6y (nogold s3/s4 + nogold cc0/cc01), m6z (SFT-lr2e4 seeds), m6ab
(aligned zero-shot), m6ac (no-gold data-mix transfer, 2B×2 + 7B + evqa-only).
