# RL redesign (branch `rl-redesign`, sibling repo VLM2Vec-rl)

## Thesis

The v1 recipe is RL-shaped supervision with supervised-learning scaffolding. Strip the
scaffolding and replace it with RL-native machinery:

| v1 scaffold | why it's a crutch | RL-native replacement |
|---|---|---|
| InfoNCE anchor (cc=0.3) | supervised contrastive loss inside a "policy gradient"; its positive is a pseudo-label (top-1 / gold) | KL-to-init trust region + entropy bonus |
| deterministic top-8 pools | no exploration; policy only re-ranks its own argmax set | Plackett–Luce sampled lists (in since a3b734e) |
| uniform all-actions weighting | not a policy-gradient estimator | exact PL conditional log-probs (a3b734e) |
| per-pool z-score baseline | std-division couples advantage scale to pool degeneracy | RLOO leave-one-out baseline |
| sampling temp == scoring temp (0.02) | near-argmax sampling → no exploration even when "sampling" | decoupled behavior temperature + clipped IS correction (off-policy PPO) |

Evidence this is plausible: recorded cc=0 no-gold = 0.337 (vs 0.341 anchored) — RL already
works without the anchor, losing only ~0.4pt that principled regularization may recover.
Discovered while building this: `--beta` (KL-to-init) was DEAD CODE in v1 — `ref_logps` was
never computed — so the anchor was the only anti-collapse device ever actually active.

## New knobs (all on the plgrpo path)

- `--kl_beta B` — k3 KL-to-init on sampled lists. Reference policy = the base encoder
  (LoRA-B init is zero, so the initial policy IS the base; ref log-probs are computed with
  adapters disabled — no second model in memory).
- `--entropy_coef C` — first-position PL entropy bonus over the support (direct anti-collapse:
  keeps the sampling distribution from spiking, which is what the anchor was papering over).
- `--baseline {group_z,rloo}` — RLOO: A_g = R_g − mean of the OTHER G−1 list rewards.
  Unbiased, no std division (z-score's std blows up advantages precisely when the pool is
  degenerate, e.g. all-zero-reward pools).
- `--pl_behavior_temperature T_b` — sample lists at T_b (explore), score at `--temperature`
  (target). old_logps become the BEHAVIOR log-probs, so the PPO ratio is the true importance
  weight and the clip bounds the off-policyness. This is off-policy clipped PG, the standard
  fix for "the deployed policy is near-argmax but training needs exploration".

## First wave (all cc=0 — the point is no InfoNCE anywhere)

| cell | config | isolates |
|---|---|---|
| v3-pure   | plgrpo, cc0, nothing else | does pure RL collapse without the anchor? |
| v3-kl     | + kl_beta 0.05 | KL trust region as the anchor replacement |
| v3-ent    | + entropy_coef 0.01 | entropy as the anchor replacement |
| v3-rloo   | + baseline rloo | baseline quality |
| v3-full   | kl 0.05 + ent 0.01 + rloo + T_b 0.05 | the whole redesign |

References on identical protocol: v1 anchored 0.3413 (recorded) / 0.3333 (fresh retrain);
v1 cc0 0.3373 (recorded); plgrpo anchored 0.3427/0.3293 (s0/s1). Fresh-retrain noise is
~±0.005 — judge against fresh-retrain controls, not recorded numbers.

Success criterion: any cc0 cell ≥ anchored v1 fresh-retrain (0.333) means the anchor is
replaceable; matching the anchored plgrpo mean (~0.336) means the redesign is strictly
cleaner at no cost; beating 0.342 means it's better.

## Cross-side notes (from the text session, 2026-08-12)

- **Binding vs inert anchor.** Our anchored runs' InfoNCE loss stays at 0.48–0.63 through the
  final quarter of training (nogate 0.80→0.48, plgrpo 1.02→0.61, plgrpo-s3 1.04→0.63) — the
  anchor gradient never dies. The text side's equivalent decayed to 0.0000 by step 240 (its
  positive = own top-1 was trivially satisfied). Registered joint prediction: anchor removal
  helps where the anchor is binding (us: +0.009–0.011, observed) and does nothing where it is
  inert (them: cells rag-plgcc0 in flight). If both hold, the rule is "check your anchor's
  loss curve before concluding anything about anchors."
- **Scope warning (theirs, adopted):** dropping the anchor is safe for POOL-RESTRICTED PL
  (top-M support, our v3) but collapsed outright on text when the partition function was
  normalized over the full corpus (R@5 1–11 vs base 55) until a frozen-base trust region was
  added. If v4 ever moves to corpus-wide normalization, expect to need KL-to-frozen-base —
  and note their failure was a drift-blind KL reference, not KL itself.
- **Ceiling story confirmed on both sides.** Their step sweep (SQuAD 10k pool, 0.6B):
  47.2 → 49.0 (peak at ~6.4k visited) → 47.8 → 48.6 with recall monotonically falling; our 2B:
  plateau ~0.345 at 2–6k visited, degradation at 12k. Joint conclusion: these recipes are a
  few-thousand-example correction, not a scalable training paradigm; capacity is the only
  lever that moved either ceiling.

## Anchor taxonomy, amended (2026-08-12, after cross-side falsification)

The text side proposed: external positive -> binding; self-argmax positive -> inert BY
CONSTRUCTION; stochastic positive -> destructive. Our data falsifies the middle clause as
stated: every binding anchor quoted above (nce 0.48-0.63 at end of training) is a SELF-ARGMAX
positive (no_force_gold: candidate 0 = own top-1), not an external one.

The repair is the negative SCOPE, not the positive type alone. Our infonce_loss ranks the
positive against ALL B x P docs in the batch — cross-query negatives from other entities'
pools — while the argmax was taken only over the query's own pool. The CE task therefore
contains comparisons the argmax does not automatically win, and at contrastive temperature
0.03 those cross-entity margins keep the loss alive.

  Amended rule: an anchor is inert iff its negative set is contained in the comparison set
  over which its positive was selected. Self-argmax + own-pool negatives = born satisfied.
  Self-argmax + cross-query negatives = binding (ours). External positive = binding
  regardless. Stochastic positive = harmful regardless.

Consequence for the joint writeup: our +0.009-0.011 from anchor removal is the removal of a
binding SELF-ARGMAX anchor — i.e. the proxy being traded away was "top-1 stability under
cross-entity contrast," not gold relevance. That makes the objective/proxy trade purer: no
label enters even through the anchor, and pure RL still wins.

## Anchor taxonomy, final (margin-at-temperature) — 2026-08-12

Both single-factor rules died on the other side's data: "self-argmax -> inert by construction"
(theirs) is falsified by our binding self-argmax runs; "inert iff negatives within the
positive's comparison set" (ours) is falsified by their inert self-argmax + cross-query cell.

Surviving rule (theirs, adopted): **an anchor is inert iff no in-scope negative sits within
~O(few*tau) of the positive.** Positive type and negative scope matter only through the
margins they induce.

Direct measurement on our side (base policy, B=4 batches, P=24 pools, 48 queries, ctemp=0.03):
gap between own-top-1 and the best cross-query doc, in units of tau: min 0.00, p10 2.21,
median 6.38, p90 9.07; 17% of queries competitive (gap < 3*tau); 0% inverted (gap < 0).
Exactly the predicted signature of our runs: a minority of live margins keeps InfoNCE at
~0.5 (binding) while the argmax positive always wins its own row (never destructive).
Their inert cell: same construction, but B=4 Trivia batches put cross-query docs topically
far away — gaps >> tau, loss born at 0.44 and gone by step 240.

Reading of the v3 result under the final rule: what anchor removal deleted was hard-margin
uniformity pressure on the ~17% of queries with near-entity competitors — which is
recall-shaped by definition. Pure RL reallocates those queries' gradient from "keep the top-1
separated" to "rank what the reader can answer from," hence R@5 down, accuracy up.

Discriminating cell (pre-registered, in flight on the text side): plgcc0-ng removes their
inert anchor -> margin rule predicts no-op; our withdrawn scope rule predicted small positive.
plgcc0-gold removes a binding external anchor -> tests proxy-for-objective transfer.

## Cross-modal split on anchor removal (2026-08-13) — scope condition for v3

Text-side plgcc0 cells landed: their no-anchor no-gold arms COLLAPSE (Trivia R@5 43.1 vs base
58.1; WQ R@5 6.3 vs base 55.5) while our v3-pure improved both in-domain (+0.9 over anchored)
and OOD (+3.3). So "drop the anchor" is NOT modality-portable as stated. The margin rule
correctly predicted the anchor's RANKING pressure was inert on text — what it missed is the
anchor's second job: absolute cross-query collapse prevention, which becomes load-bearing when
the reward is sparse (their live-pool rates 21-75% pre-fix; ours healthy at mean ~0.26).

MM datapoints against a pure-density story: v1-cc0 (all-actions, no anchor) never collapsed
either (recorded 0.3373), so on MM BOTH estimators survive anchor removal.

PRE-REGISTERED DISCRIMINATOR (in flight, job a1eb5830): clip-v3pure = no anchor + weak
retriever (CLIP-L, zs R@5 0.054) + mostly-junk pools = the text collapse regime reproduced on
MM. If reward density/liveness is the deciding variable, clip-v3pure should collapse like
their WQ cell; if the split is modality- or action-structure-driven (their pool construction
vs our sampled lists), it should merely underperform its anchored twin (clip-nogold-base,
R@5 0.291). Logged here before the cell reports.

Consequence for the paper: the v3 no-anchor claim carries a scope sentence — demonstrated in
a reward-dense regime; the text twin collapses in reward-sparse ones; the anchor's
collapse-prevention role is real and regime-dependent.

## Discriminator verdict (2026-08-12, hours after pre-registration): DENSITY BRANCH CONFIRMED

clip-v3pure landed: entity R@5 = 0.0043 — total retrieval collapse, 12x BELOW its own
zero-shot base (0.054), let alone its anchored twin (clip-nogold-base, 0.291). Accuracy 0.190
= the reader answering from the image alone. This is the text side's WQ collapse (6.3 vs base
55.5) reproduced within our modality, under the identical sampled-list PL estimator (their
plgcc0 confirmed G=8 k=5 Gumbel-top-k exact-factorized — the estimator x anchor confound is
closed on both sides).

Both pre-registrations fired the same way (ours: density branch; theirs: collapse expected).
The joint rule, now measured on a 2x2 across modalities:

                      reward-dense pools        reward-sparse pools
  anchor removed      HELPS (+0.9 ID, +3.3 OOD  COLLAPSES (MM clip: R@5 0.004;
                      — mm v3-pure; both MM      text WQ: 6.3; text Trivia: partial)
                      estimators survive)
  anchor kept         costs accuracy for recall  prevents collapse (its load-bearing job)

The InfoNCE anchor's two jobs are now separable and regime-tagged: ranking pressure (inert or
harmful depending on margins — the margin-at-temperature rule) and collapse prevention
(irrelevant when reward is dense, essential when sparse). "Drop the anchor" is a
reward-density-conditional recommendation, on both modalities, full stop.

Corollary for the ladder: qwen2b-zeroshot R@5 = 0.0003 (raw VLM embeddings = random retrieval).
By the density rule, qwen2b-v3pure should also collapse and qwen2b-nogold (anchored) is the
interesting cell — can the anchor keep a from-scratch policy alive long enough for the reward
to shape it? Registered expectation: anchored survives, unanchored collapses, locating the
boot-strap floor between clipb32 and gme2b.

## The anchor as a restoring force (text-side insight, 2026-08-12, PAPER_TABLES 8f @ 0cdc69c2)

Resolution of why the margin-rule pre-registration missed on collapse: 8d margins are measured
at the BASE model — a POINT statement. Their anchored-ng twins are anchor(self), inert at init
by that measurement, yet removal still collapsed training. The anchor is a RESTORING FORCE:
zero at equilibrium, re-engaging exactly as reward-driven drift shrinks the margins that made
it inert. Static margins predict ranking pressure; they cannot price collapse prevention,
because inertness is about a point and collapse is about a trajectory.

Cross-project vocabulary (both papers): anchor(self) = positive is the policy's own top-1
(all 39 of our anchored runs, audited); anchor(ext) = external/gold positive (their gold arms);
anchor(stoch) = sampled positive (destructive, both sides' history). "Anchored" is NOT the
same column across projects without these tags.

CORRECTION (writer-caught, 8f cross-checked): my earlier claim that collapse prevention is
"demonstrated by both anchor types" was WRONG. Both text-side collapse cells are anchor(self)
— the same type as all 39 of ours — so the cross-modal collapse result is a SAME-TYPE
replication. Their one anchor(ext) removal (mus_gold cc0: Hot cEM 39.6->30.8) lost the gold
advantage WITHOUT retrieval collapse — a different failure mode.

Refinement on the refinement: "collapse prevention belongs to the self type" also overreaches,
because type and regime are CONFOUNDED in the ext cell — external positives exist only where
gold labels exist, and gold seeding keeps pools reward-dense, which is precisely the regime
where NO anchor is needed for stability. The ext cell therefore never faced collapse risk and
cannot test whether ext anchors prevent collapse. Accurate statement: collapse prevention is
demonstrated ONLY for anchor(self), on two modalities; anchor(ext)'s collapse-prevention
capacity is untested and structurally hard to test.

PRE-REGISTERED READINGS for the ladder cells in flight, per the restoring-force account:
- siglip2-nogold (anchor(self), zs R@5 0.082): rescue should repeat (CLIP-L pattern).
- qwen2b-nogold (anchor(self), zs R@5 0.0003 — the top-1 positive is a RANDOM doc):
  * if it still bootstraps -> the anchor needs only a FIXED POINT, not a meaningful one;
  * if it fails where CLIP-L (0.054) succeeded -> there is a liveness floor below which
    anchor(self) has nothing to restore to.
  Either outcome is one clean sentence; both are written here before the cells report.

## Two collapse phenotypes (2026-08-12, joint w/ text side, their 8f @ 1bb562ef)

The distinct-top-k metric run on text-side collapse cells fires the second live outcome:
their TOTAL collapse (wq_ng cc0, R@5 6.3 vs base 55.5) is 242/276 distinct sets (top set
2.2%) — varied-but-useless. Ours is 67-338/3000 with top sets up to 45% — contracted.

  MM collapse   = DEGENERACY: the embedding contracts; policy maps most queries to a few
                  fixed sets. Detectable statically (distinct-count), no labels needed.
  text collapse = MISALIGNMENT DRIFT: query/doc geometry decouples while staying spread out.
                  Invisible to the distinct-count (their anchored-vs-pure pairing: 276 vs 242,
                  1.14x — nothing); visible only against ground truth (recall).

Same anchor term, two restoring jobs by regime AND modality: against contraction here,
against drift there. Joint-table rule: a signature column, never the bare word "collapse".
Symmetric limitation now explicit: our static count would have missed their failure exactly
as their static margin measurement missed ours.

## Boundary probe verdict + phenotype revision (2026-08-12 evening)

vlm2vec2b-v3pure (zs R@5 0.157, the registered boundary probe): **FAILED — acc 0.1720,
R@5 0.007 (22x below its zero-shot). So v3-pure's working range is bracketed: fails at 0.157,
works at 0.685. Pure RL requires a GME-class base.**

**And it failed by the TEXT phenotype: distinct-top5 = 3000/3000 (max 1) — maximal diversity,
zero utility. Misalignment DRIFT, not contraction.** The "degeneracy=MM, drift=text" split
recorded earlier today is falsified as a modality rule: the MM ladder now contains BOTH
phenotypes (contraction at clip/siglip2/raw-qwen2b; drift at vlm2vec2b). Phenotype correlates
with something else — candidate: contrastive-pretraining structure of the base (VLM2Vec is
MMEB-contrastive-trained with strong uniformity, resisting contraction but free to drift;
CLIP/SigLIP dual towers and raw VLMs contract). One-seed evidence, hypothesis only.
Consequence: the diagnostic-PAIR recommendation (distinct-count + recall-vs-base) upgrades
from "covers both modalities" to "required within a single modality."

## Checklist-caught ledger gaps (2026-08-12 late — items sent in messages but never written here)

- HEADLINE, final: v3-pure [InfoSeek acc, top-5, n=1500] = 0.3471 ± 0.0019 (3 seeds:
  0.3447/0.3473/0.3493), SIG vs SFT 0.3100 and base 0.3080 per seed (McNemar p <= 4.8e-05).
  ck250 lower on all seeds (0.3373 mean) -> endpoint is the checkpoint max; headline unrevised.
- PHENOTYPE HYPOTHESIS, corrected name: **domain-uniformity** (not "contrastive-pretrained" —
  CLIP/SigLIP are contrastive yet contract). Bases whose pretraining spread the retrieval-domain
  (passage-side) embedding space fail by drift (VLM2Vec, Qwen3-Emb); bases without passage-side
  uniformity contract (CLIP/SigLIP cross-modal alignment, raw VLMs). 1 seed per cell.
- CAPACITY, corrected: the recorded "gap grows with capacity" was a baseline-pairing artifact
  (gold-RL vs mid-rung SFT). Like-for-like: 2B +2.08 / 7B +2.17 — FLAT. See main-repo LEDGER
  erratum. Fresh v3-7B: 0.3460 (1 seed) — no accuracy gain over 2B, +4.9 entR@5 only.
- v3 boundary bracket: pure RL fails at zs entR@5 0.157 (vlm2vec2b, by DRIFT) and works at
  0.685 (gme2b); working range (0.157, 0.685] pending finer rungs.

## Base-model wave 2 (registered 2026-08-13, pre-launch)
New cells: bgevl_b-v3pure (5310cbb94793e62c), bgevl_l-v3pure (b8803548161beb61), qwen25_3b-v3pure
(7ece3f2262c030cf); zs chains bgevl_b/bgevl_l/qwen25_3b local. BGE-VL = MegaPairs
retrieval-FINETUNED CLIP-arch — same architecture as clipb16/clip, different pretraining.
PREDICTION (registered before any number lands): if the working/failing boundary is about
retrieval pretraining (domain-uniformity account), bgevl_l-v3pure avoids the contraction
phenotype (distinct-top5 >> 338, R@5 within a few points of its own zs) and shows positive
delta-acc; if the boundary is about VLM-class architecture/capacity, bgevl_l collapses like
clip (R@5 ~0). Secondary: bgevl_b (0.15B) tests whether size alone blocks training at
matched pretraining. qwen25_3b slots the raw-VLM size ladder between qwen2b and qwen7b;
expectation from domain-uniformity: drift phenotype like qwen2b, not contraction.

### Wave-2 amendment (2026-08-13, zs landed, TRAINING CELLS STILL BLIND)
zs placements: bgevl_b entR@5 0.1473 / acc 0.1593; bgevl_l entR@5 0.1573 / acc 0.1587 —
both AT the known failure boundary (<=0.157), and both BELOW the no-context floor (0.1927):
their retrieved context actively hurts the reader. MegaPairs CIR finetuning does not
transfer to InfoSeek wiki-KB retrieval; BGE-VL sits in the raw-CLIP zs band here.
AMENDED READING (registered before any training number lands): the original clean
discrimination (same arch, retrieval pretraining, higher start) is unavailable — BGE-VL is
NOT higher-start. The cell now tests: retrieval-style objective at matched-bad zs level.
If bgevl_l-v3pure still trains where clip collapsed -> objective/geometry mechanism result
(stronger than the original claim). If it collapses -> consistent with a zs-competence
floor; pretraining-vs-architecture question stays OPEN (not refuted).

### LR-sensitivity control wave (2026-08-13, user-directed, registered pre-landing)
CORRECTION (writer, from args.json): rates were NOT uniform and 1e-4 is NOT the GME rate.
Actual rates: gme2b/qwen2b/vlm2vec2b 2e-5; clip/siglip2 (and wave-2 bgevl_*/qwen25_3b,
which copied the clip config) 1e-4 (best-for-CLIP in the recorded weak-rung sweep);
qwen7b/gme7b 1e-5. Each failure is a failure at that rung's own inherited rate, chosen on
a different base. Control cells submitted:
clip-v3pure-lr{1e5:MLX b530a228, 3e5:local}, bgevl_l-v3pure-lr{1e5:MLX b319c018, 3e5:local},
siglip2-v3pure-lr1e5 (local), qwen25_3b-v3pure-lr3e5 (local); coverage-gap cells added
after writer's design note: qwen25_3b-v3pure-lr2e5 (its family's rate), qwen2b-v3pure-lr5e6
and vlm2vec2b-v3pure-lr5e6 (low-end controls for the 2e-5 collapses, previously uncovered).
PREDICTION: if the CLIP-class contraction phenotype is an LR overshoot artifact, lower LR
rescues R@5 toward zs level and the boundary claim must be rewritten as "fails at
GME-tuned lr" (per-family tuning required, boundary table gains an lr column). If
contraction persists across the 10x ladder, the boundary claim survives LR as a confound.
Registered expectation from the margin account: contraction persists (the failure is
signal-starvation in the top-M support, not step size) — but genuinely uncertain.

### Wave-2 blind cells landed (2026-08-13 03:13) — SCORED
bgevl_b-v3pure acc 0.1780, entR@5 0.1473->0.0007, distinct-top5 133/3000.
bgevl_l-v3pure acc 0.1827, entR@5 0.1573->0.0070, distinct-top5 197/3000.
Both CONTRACT at lr 1e-4 — same phenotype as clip/siglip2. Amended registered reading
scores as: retrieval finetuning does NOT rescue CLIP-arch at matched-bad zs;
pretraining-vs-architecture stays OPEN (zs-competence-floor account consistent).
Weight shifts to the LR controls (bgevl_l 1e-5/3e-5, running).
ANTI-DIAGNOSTIC INSTANCE #2 (cleanest yet): acc ROSE +2 to +2.4 pts while retrieval
died 20-200x — with R@5 ~0 the context stream approaches uniform junk and the reader
drifts toward its no-ctx floor (0.1927) from below. Accuracy alone would have called
these cells "improved." The diagnostic PAIR is mandatory; log this instance in the paper's
diagnostics paragraph.
qwen25_3b-zeroshot: acc 0.1607, entR@5 0.0373 (distinct 2992/3000) — raw Qwen2.5-VL-3B
zs retrieval is near-random, far below the boundary; its v3 cells now test training from
a near-zero start.
sft-curve ck750 = 0.3300 — NEW SFT MAX, curve non-monotone (.2953/.3207/.3227/.3107/.3300);
provisional fair 2B gap 0.3471-0.3300 = +0.0171; McNemar at fair pair when curve completes.

### Text-side composed cell scored (2026-08-13, their wave) — my registration: HIT
plgTa_triv (PL + anchor(self) cc=1.0 + 0.727 reader, 0.6B) ck175: Trivia R@5 55.6 vs base
58.1 (within 2.5; registered ~2), cEM 68.0 vs 69.8 ~parity. 30%-clears-base branch did NOT
fire (score that miss-of-branch). Decomposition as registered: anchor removed the early
DROP (+5.5-7.4 R@5 vs both cc0 arms), PL removed the decay. Headline: NQ-transfer cEM 47.6
= +3.0 over base = matches gold-seeded RL exactly; swept SFT 49.2 still +1.6 ahead (their
SFT verdict survives). Estimator x capacity, text column: PL > enumeration at 0.6B AND 4B
(plgT4b_tj ck175 61.4/72.0 vs enum 57.5/70.8). MM column: our gme 2B/7B pair — with the
caveat that MM 7B RL-SFT gap compresses to n.s.; text 4B still shows PL>enum clearly
(different comparison: estimator-vs-estimator, not RL-vs-SFT).

### LR-confound propagation from text side (2026-08-13, their config audit)
Their anchored plg twins/klfix/ppofix ran 1e-5; T-era family (plgT*, plgTa, cc0 arms) ran
2e-5. CONSEQUENCE FOR THIS LEDGER: the "text no-anchor no-gold arms COLLAPSE" attribution
(entry above citing Trivia R@5 43.1) compared unanchored-2e-5 vs anchored-1e-5 — the
anchor-prevents-collapse reading carries an LR confound there. PARTIAL RESCUE already in
hand: the composed-cell pair plgTa vs plgT_cc0 is same-rate (both 2e-5) and shows the
anchor removing the early drop (+5.5-7.4 R@5) at matched LR — anchor attribution survives
at matched rate, magnitude of the old "collapse" rows awaits their wave-3 plgTlr1e5 cell.
Cross-modality claims should cite the T-era same-rate pair, not the old twins.

### clipb32-v3pure landed (2026-08-13 03:55): THIRD PHENOTYPE — INERT
acc 0.1513->0.1473, entR@5 0.0260->0.0257, distinct 2825/3000 (healthy). Training was a
no-op. New failure vocabulary: INERT (nothing moves) vs CONTRACTION vs DRIFT.
HYPOTHESIS (registered now, qwen25_3b-v3pure still blind): phenotype tracks starting
signal level — zs entR@5 ~0.03 -> inert (reward ~uniform, no gradient direction);
~0.15 -> contraction (bgevl_b 0.147, bgevl_l 0.157, clip, siglip2 — enough signal to
chase, not enough to rank within); ~0.16 raw-VLM -> drift (vlm2vec2b, qwen2b);
0.69 -> healthy (gme). AMENDED PREDICTION for qwen25_3b-v3pure (zs entR@5 0.0373,
supersedes my earlier drift registration, recorded pre-landing): INERT, not drift.
If it drifts instead, the ladder hypothesis loses the level-not-family part (raw-VLMs
may drift at any level); if inert, phenotype-by-signal-level gains its second point.

### Taxonomy correction (writer catch #8, accepted) + boundary bracket
My "inert is where distinct-count IS informative alone" INVERTED — distinct for inert
(2825) is indistinguishable from healthy (2899). Corrected taxonomy (theirs, adopted):
recall-vs-base separates healthy/inert/damaged but cannot split contraction from drift;
distinct-count detects ONLY contraction (healthy/drift/inert all 2800-3000). The pair is
minimal and sufficient — neither dominates, neither redundant.
BRACKET (sharper than my three-band sketch): clip-L already CONTRACTS at zs entR@5 0.054,
clipb32 INERT at 0.026 -> the inert/contraction boundary lies in (0.026, 0.054). My
ladder bands ("~0.03 inert, ~0.15 contraction") were sloppy — the contraction band
extends down to at least 0.054. qwen25_3b (zs 0.0373) sits MID-BRACKET: whichever
phenotype it lands, the bracket halves. Queued clipb16-zeroshot to place a further
intermediate point (expected zs between clipb32 and clip-L); its v3 cell follows if its
zs lands inside the surviving half-bracket.

### Bracket wave-audit (writer) + fresh clip-zeroshot launched
Writer caught the (0.026, 0.054) bracket mixing waves: clip-L's 0.054 upper bound is
RECORDED (no fresh clip-zeroshot existed on this cluster). Fully-fresh primary bracket:
(0.026, 0.082) — clipb32 inert / siglip2 contracting, both endpoints fresh-zs + fresh-v3.
Fresh clip-zeroshot chain launched (GPU 5) — makes the tight bracket within-wave and is
PREREQUISITE for interpreting clipb16-v3pure as a tight-bracket bisection. qwen25_3b
(0.0373) bisects either bracket. Print-once discipline: nothing writable until qwen25_3b
lands.

### LR ladder first landings + gme7b-s1 (2026-08-13 04:21) — MY LR PREDICTION: MISS
**bgevl_l-v3pure-lr1e5: TRAINS.** entR@5 0.1573 -> 0.2123 (+5.5, 1.35x), distinct
2966/3000, acc 0.1587 -> 0.1693. Contraction at 1e-4 was an LR ARTIFACT for BGE-VL-L.
My registered prediction ("contraction persists; failure is signal-starvation not step
size") SCORES AS MISS on the phenotype claim. User's tuning instinct was right.
**clip-v3pure-lr1e5: rescued phenotype, NO learning.** entR@5 0.0410 (vs recorded zs
0.054), distinct 2882 — no contraction, but no improvement either (slight sag).
=> THE WAVE-2 DISCRIMINATION RETURNS WITH AN ANSWER: at matched tuned LR (1e-5), the
retrieval-pretrained CLIP-arch (BGE-VL-L) LEARNS (+5.5 R@5) while raw CLIP-L does not
(-1.3). Retrieval pretraining is an enabling ingredient; step size was a masking artifact
on top. Margin account survives only its signal-starvation half (clip has nothing to
learn from); its phenotype half was wrong.
HONESTY CLAUSE: bgevl_l-lr1e5 acc 0.1693 remains BELOW the no-ctx floor (0.1927) — its
retrieval improved but is still net-harmful to the reader. "Trains" means retrieval
learns; it does NOT yet mean useful RAG. 500 steps; longer runs unregistered.
**gme7b-v3pure-s1 = 0.3573** (entR@5 0.8030, distinct 2895): vs gme7b-sft +0.0253
McNemar p=0.0014 SIG (87/49); vs 2B base +0.0493 p=5.9e-08. 7B RL 2-seed: 0.3460/0.3573
(mean 0.3517, spread 0.0113). The s0-only "n.s. convergence at 7B" verdict does NOT
survive s1: RL>SFT at 7B is SIG in the stronger seed, mean gap +2.0 vs 2B's +3.7.
Capacity story v4: margin persists at 7B, somewhat smaller than 2B; "capacity helps SFT
not RL" weakens (RL-7B mean 0.3517 vs RL-2B 0.3471, +0.5 within seed noise).
sft-curve ck900 = 0.3133 (max still ck750 0.3300; ck1050/1200 pending).

### 05:18 landings: bracket promotes; LR dose-response; b16 consumption point
clip-zeroshot FRESH entR@5 0.0537 (recorded 0.054, delta 0.0003 — cluster repro fidelity
holds). Tight bracket (0.026, 0.054) PROMOTES to fully-fresh. clipb16-zeroshot 0.0310 —
inside the bracket, below qwen25_3b 0.0373; bracket coverage now .026/.031/.0373/.054.
clipb16-v3pure stays queued (in-bracket condition met).
LR DOSE-RESPONSE (phenotype=f(zs,lr) now has gradients, not just endpoints):
clip-L: 1e-5 no contraction/no learning (2882, R@5 .041) | 3e-5 PARTIAL contraction
(1795, .0170) | 1e-4 full (214, .0043).
bgevl_l: 1e-5 LEARNS (+5.5, 2966) | 3e-5 mild damage (2364, R@5 .1477 < zs) | 1e-4 full
contraction (197). Learning window closes between 1e-5 and 3e-5. QUEUED bgevl_l-v3pure-lr3e6
to probe below 1e-5 — 1e-5 is a boundary cell, not established as the peak.
sft-curve ck1050 = 0.3133 (plateau; max still ck750 0.3300; ck1200 last).
v3-b16-s750 = 0.3487 acc, entR@5 0.7590, distinct 2891 — batch-16, 12k draws: consumption
curve gains a >=headline point (0.3487 vs 0.3471), consistent with consumption-rising.

### 05:59 landings — fairness audit COMPLETE; qwen25_3b prediction HIT; bracket halves
**2B fairness verdict (final): SFT curve complete** (.2953/.3207/.3227/.3107/.3300/.3133/
.3133/.3153, ck150-1200; max ck750 = 0.3300; released 600-step ckpt confirmed post-peak).
v3-pure fresh seeds vs SFT-max McNemar: s0 +0.0147 p=0.084 n.s.; s1 +0.0173 p=0.042 SIG;
s2 +0.0193 p=0.02 SIG. AUDITED HEADLINE: RL 0.3471 (3-seed mean) vs SFT-best-checkpoint
0.3300 = +1.7, SIG in 2/3 seeds. (Old +3.7-vs-endpoint number retired from headline use.)
Residual asymmetry, disclosed: SFT max is from ONE retrain (8 ckpts); RL endpoints
verified >= their ck250 on s0 only.
**qwen25_3b-v3pure: INERT, exactly** — entR@5 0.0373 -> 0.0373, distinct 2992, acc
identical. Amended prediction (drift->inert, registered pre-landing) SCORES HIT. Bracket
halves: at lr 1e-4, inert/contraction boundary now (0.0373, 0.054), fully fresh. Note the
bracket is LR-CONDITIONED (at 1e-5 nothing contracts anywhere). clipb16-v3pure (zs .031,
queued) now tests inert-region consistency rather than bisecting.
**siglip2-v3pure-lr1e5: no contraction (2917), no learning (R@5 .0823 -> .0660)** — same
class as clip-L at 1e-5. At 1e-5 the only base that LEARNS remains bgevl_l. Wording
nuance for the discrimination claim: BGE-VL is retrieval-FINETUNED (MegaPairs CIR);
clip/siglip are image-text contrastive — say "retrieval-finetuned", not "-pretrained".
**v3-rows200k-s3000 = 0.3407** (R@5 .764, distinct 2885): big-pool consumption curve now
2k:0.3400(3s) / 6k:0.3480(1s) / 12k:0.3407(1s) — plateau ~0.34-0.35, no clear rise
beyond 6k; 1-seed points, do not claim a sag.
ERRATUM (writer catch): I doubled these draw labels in the 05:59 entry and in messages
(4k/12k/24k); batch is 4, so draws = 4 x steps = 2k/6k/12k. Accuracies were mapped
correctly. The b16-s750 line above (16 x 750 = 12k) was already correct — note it now
COINCIDES in draws with rows200k-s3000 (12k): 0.3487 (b16, 3k pool) vs 0.3407 (b4,
200k pool) — a matched-consumption batch/pool comparison, confounded pairwise, listed
as observation only.

### Zero-gradient note (writer suggestion, adopted)
qwen25_3b-v3pure's retrieval metrics unchanged to FOUR DECIMALS is evidence the policy
gradient was exactly zero (uniform reward -> zero advantage -> no update), not merely
small. Verifiable from train logs (loss/adv should be ~0 throughout); stronger inertness
evidence than "flat metrics". Same check applies to clipb32 (near-flat, small nonzero).

### RETRACTED (writer control): the entry below claimed verification from an identity
loss/policy = -mean(A) with A group-z-scored is ZERO BY CONSTRUCTION for every plgrpo
run — the healthy v3-pure logs |loss/policy|max 1.49e-08 too. A zero scalar loss also
does not imply zero gradient (grad = -mean(A*gradlogpi) at rho=1, generally nonzero).
The within-group-degeneracy account is DEMOTED to hypothesis (still the best one; the
across-query reward variation stands). Verification requires per-group reward std,
which was not logged. FIX: train_rl.py now logs reward/group_std_mean and
reward/group_degenerate_frac; 50-step probe cells queued (qwen25_3b + gme2b control).
Prediction, registered: qwen25_3b degenerate_frac ~1.0, gme2b well below 1.

### Zero-gradient claim (RETRACTED ABOVE — kept for the record) — mechanism sharpened
loss/policy and ppo/advantage_mean = 0.000000 exactly, all 500 steps. BUT reward/raw_mean
= 0.135 (max 0.75) — rewards are NOT zero/uniform across queries. Mechanism: inertness =
WITHIN-GROUP reward degeneracy. With near-random retrieval, the G=4 sampled lists from a
junk top-24 are interchangeable to the reader — identical reward within every group —
so the group-z advantage is identically zero even though the reader sometimes answers
right (parametric knowledge). Inert cells are where the reward cannot DISCRIMINATE
between lists, not where it is absent. (This also predicts: any base whose top-24
contains at least occasional signal escapes inertness — consistent with contraction
starting by 0.054.)

### Conventions (adopted from writer exchange, 2026-08-13 morning)
Diagnostic runs use name suffixes -stdprobe/-probe/-smoke/-debug; the writer's collectors
exclude these BY NAME (EXCLUDE_RUNS), because config rules deliberately free max_steps on
the consumption axis and a 50-step diagnostic would otherwise be admitted as a legitimate
200-draw curve point. General shape, theirs, worth keeping: "the more permissive an axis
is by design, the more it admits things that are not results" — any future axis that
frees a config key inherits the same exposure. New diagnostic suffixes must be announced
(protocol runs both directions). Methods-narrative decision: process near-misses stay
OUT of the paper; rigour is shown by scored misses, withdrawn registrations, and the
audited headline, not by narrating the catch.

### INVALIDATION (2026-08-13 08:47): qwen25_3b-v3pure "inert" was an EVAL ARTIFACT
Chain that exposed it: stdprobe prediction MISSED (qwen25_3b degenerate_frac 0.71, NOT
~1.0; healthy gme2b control itself 0.55) -> partial degeneracy cannot give four-decimal
flatness -> checked adapter: lora_B abs-sum 44k (HUGE — bgevl_l's +5.5 R@5 came from 812)
-> loaded ckpt vs base embeddings: diff 0.0 EXACTLY -> src/model/model.py reload path
attached qwen2_5_vl adapters to base_model.model, keys matched NOTHING, merge_and_unload
was identity. FIXED (QWEN2_5_VL added to full-model-attach set; verified diff 0.328) and
pushed. SCOPE: only qwen2_5_vl CHECKPOINT RELOADS — training was real (fresh-LoRA attach
is a different path); qwen25_3b-zeroshot valid (no ckpt); qwen2_vl/gme/clip families
unaffected (their landed metrics visibly moved); reader unaffected.
CONSEQUENCES: (1) qwen25_3b-v3pure verdict UNKNOWN pending re-eval (running, GPU 0) —
adapter moved a lot, could be anything; (2) "amended prediction HIT" un-scored — pending;
(3) bracket REVERTS to (0.026, 0.054) with interior point unresolved; phenotype-ladder
%-comment must drop the qwen25_3b point; (4) lr2e5/lr3e5 cells' upcoming evals are
CORRECT (fix landed before their eval step); (5) stdprobe data reinterpreted: 29% live
groups is CONSISTENT with the large adapter movement — there never was a reward-side
inertness at this rung.

### Zero-gradient account DEAD on both rungs (writer refutation, accepted)
Writer's checks, both sharper than mine: (1) artifact signature is BIT-IDENTICAL outputs
— qwen25_3b identical on every metric; clipb32 moved (.0260->.0257, 2971->2825, .1513->
.1473) => different phenomenon, clipb32 verdict stands (test works without reading code).
(2) lora_B abs-sums: clipb32 3048 — FOUR TIMES bgevl_l-lr1e5's 812 (which gained +5.5
R@5). clipb32's gradient was emphatically nonzero; retrieval didn't follow. With gme2b
control 0.55 degenerate, uniform-reward -> zero-advantage -> no-update is dead on BOTH
rungs it was proposed for. INERTNESS IS NOW AN OBSERVATION, NOT A MECHANISM: substantial
parameter movement buying no measurable retrieval change, cause unresolved. Do not
restore the mechanism without new evidence.
CONVENTIONS: lora_B abs-sum added to the standard harvest (mmrag/adapter_stats.py, no
GPU) — separates "didn't train" from "trained and it didn't help"; would have flagged
44041 on landing. Re-eval of qwen25_3b treated as a FRESH CELL: first check outputs
differ from base, then interpret.

### clipb16-v3pure (09:14): CONTRACTS — bracket tightens to (0.026, 0.031) CLIP-family
zs entR@5 0.0310 -> 0.0003, distinct 96/3000 (deepest contraction recorded), acc ROSE
0.1680 -> 0.1947 = the no-ctx floor (0.1927) within noise. ANTI-DIAGNOSTIC INSTANCE #3,
the cleanest: retrieval fully dead, reader reverts exactly to context-free behavior —
acc-at-floor is now a predicted signature of full contraction (bgevl cells were en route:
0.178/0.183 rising toward floor from 0.159).
BRACKET (lr 1e-4, CLIP family, both endpoints fresh): inert/contraction boundary in
(0.0260, 0.0310) — clipb32 inert, clipb16 contracts. Print-once: this version can print;
the qwen25_3b point (zs 0.0373) now sits ABOVE the bracket and its re-eval tests whether
the boundary TRANSFERS ACROSS FAMILIES (raw-VLM), not bisection. FAMILY-SCOPE the claim.
lora_B: clipb16 3126 ~= clipb32 3048 — same optimiser movement, opposite phenotype;
movement magnitude does not determine outcome (second confirmation).

### qwen25_3b corrected verdicts (10:41, fresh-cell rule applied: outputs differ from base)
qwen25_3b-v3pure (lr 1e-4, RE-EVAL): **CONTRACTS, totally** — entR@5 0.0373 -> 0.0000,
distinct 32/3000 (deepest recorded), acc 0.1607 -> 0.1707 (climbing toward floor 0.1927;
acc-at-floor pathway consistent). My original amended prediction (INERT) now definitively
scores MISS — the artifact had hidden a contraction.
qwen25_3b-v3pure-lr2e5 (family rate): retrieval also dies — entR@5 0.0003, distinct
2033/3000 (partial diversity loss), acc 0.1853 (near floor). Raw Qwen2.5-VL-3B fails at
BOTH rates by retrieval-death; dose-response in diversity (32 @1e-4 vs 2033 @2e-5).
CONSEQUENCES: (1) BOUNDARY TRANSFERS ACROSS FAMILIES at the one interior point — zs
0.0373 > bracket upper 0.031 predicts contraction; observed. (2) The INERT phenotype has
exactly ONE member (clipb32, zs 0.026) — it is an edge case at near-zero signal, not a
band; the taxonomy's load-bearing split is contract-vs-drift-vs-learn, with inert as the
degenerate corner. (3) stdprobe data now coherent: 29% live groups = gradient existed =
contraction had fuel.

### bgevl_l-v3pure-lr3e6 (11:52): flat — 1e-5 is an INTERIOR max of the tested window
entR@5 0.1573 -> 0.1523 (~flat), distinct 2952 (healthy), acc 0.1640, lora_B 260 (1/3 of
the 1e-5 run's 812) — undertrained at 500 steps. BGE-VL-L window over the full ladder:
3e-6 flat | 1e-5 LEARNS +5.5 | 3e-5 damaged -1.0 | 1e-4 contracts. 1e-5 upgrades from
boundary cell to interior maximum; the +5.5 rescue number is no longer an underestimate
by construction (longer-run/finer-grid gains remain possible but unregistered).

### 5e-6 low-end controls (13:03) — drift is NOT an LR artifact; qwen2b zs corrected
**vlm2vec2b-v3pure-lr5e6: STILL DRIFTS** — entR@5 0.157(zs) -> 0.0017, distinct 2999/3000
(full diversity, retrieval destroyed), acc 0.1793. With the 2e-5 twin, vlm2vec2b's drift
is robust across a 4x LR range — the writer's coverage-gap cell answers its question:
unlike BGE-VL's contraction (an LR artifact), vlm2vec2b's drift is not rescued by gentler
LR at 500 steps. SHARPENS THE DISCRIMINATION at matched zs signal (~0.157): retrieval-
finetuned CLIP-arch (bgevl_l) LEARNS at 1e-5; VLM2Vec-trained Qwen2-VL drifts at every
tested rate.
**qwen2b ZS CORRECTION (my error in messages/summary): qwen2b-zeroshot entR@5 = 0.0003**,
near-zero — NOT ~0.15 (that was vlm2vec2b). qwen2b sits in the near-zero-signal corner
with clipb32; its "drift/contraction" labels are largely vacuous (nothing to lose:
0.0003 -> 0.0000 at 5e-6, distinct 2503, acc 0.1993 -> 0.2187 — both accs ABOVE the
floor, consistent with near-random context being ignorable). Any ladder/phenotype text
placing "raw-qwen2b" at mid-signal is wrong and must cite vlm2vec2b instead.

### Cross-modality headroom exchange (13:30-14:00) — my axis-1 premise CORRECTED
Text long-run (plgTa_long, 1e-5, 3x draws) DECLINED monotonically (67.4 -> 66.2 -> 63.8
in-domain cEM). Consumption paying is NOT modality-independent: MM rises 2k->6k and
plateaus; text declines by 3x. My "extend to 10x" advice was premised on rise — corrected
by their data. Divergence hypothesis (unregistered): draws pay only while the reward
stays informative per group; candidates for what saturates on text unresolved.
THEIR NEW RESULTS, noted for cross-reference: (1) movement (lora_B) linear in LR on text
too, and the BEST cell moved LEAST (70.0 cEM @ 1226) — movement-vs-outcome decoupling is
now cross-modality; (2) anchor STRENGTH adds +2.4 cEM at MATCHED movement (cc05/cc2 ~4030
moved, 70.4/70.6 vs baseline 68.2) — their frontier is (movement budget x anchor
strength); (3) movement SUBLINEAR in steps (3x steps = 1.75x movement) — steps and LR are
not interchangeable movement currencies; (4) above-base candidate (70.4-70.6 vs base
69.8) pre-registered with 3 seeds before any headline. Their discriminator cell
plgTaBudget (1.7e-6 x 1500) + anneal cell (cc 1->0 first quarter) submitted.

### Writer pass (14:3x): consumption series complete; sublinearity replicates on MM
(1) v3 consumption series COMPLETE: 2k 34.00+/-0.47 (3s) / 6k 34.80 (1s) / 12k 34.07 (1s)
— plateau in the 34-35 band; 34.80 is NOT quotable as a maximum (span < 3-seed spread).
Shared-cell warning fired correctly on their side (two paragraphs, one cell, different
numbers) — fixed; honest scoring adopted: replicates .3433/.3420 => "no pool-composition
effect at matched consumption (vs 25k cell: exactly 0.00)", NOT "deficit was a low draw"
(3-seed mean .3400 came in below the .347 the low-draw branch anticipated).
(2) Trajectory form (theirs, better): early-peak is conditioned on BUDGET, not modality —
at matched short budgets neither estimator has peaked; modalities separate only under
extended consumption (MM plateaus, text declines).
(3) SUBLINEARITY REPLICATES ON OUR ADAPTERS (their harvest, lr/pool fixed, only
max_steps differs): 500st 7194 / 1500st 11915 (x1.66) / 3000st 15584 (x2.17). With the
LR ladder (~linear in rate), displacement is ~LINEAR IN RATE, SUBLINEAR IN BUDGET.
RULES ADOPTED: collapse boundary stated PER LEARNING RATE, never per budget (a long safe-
rate run does not reach where a short damaging-rate run goes); budget sweeps are NOT
movement sweeps — applies to interpreting v3-cos/b16cos tails (cos reduces late LR;
movement even more sublinear there). Best line, theirs: across the plateau "the optimiser
keeps moving; it stops converting."
(4) Cross-side provenance rule tightened: FIVE of my relayed text-side figures failed
their reconciliation against trl PAPER_TABLES.md (numbers were from messages, not tables
— summarisation drift, not fabrication). New rule: cross-side numbers quoted in either
paper must come from the source repo's TABLES file; message numbers are coordination
only.
(5) v1-vs-v3 over-training contrast held soft (1 seed/arm at upper points) — the
scale200k-ck dense checkpoints are its test; flag to writer the moment they land.

### Anchor-slot gate 3 INVERTED (their config check, 14:5x)
Text composed cells all run online_force_gold: false — their anchor positive is the
policy's own top-1 = anchor(self), SAME TYPE as our 39 audited runs, not the annotated
passage. (Consistent with the writer's own early catch #2 — "text collapse cells are
anchor(self)" — the gate-3 premise had regressed against that.) Same-rate confirmed in
their Table 8g: cc05/cc2/cc1 all 2e-5, movement 4038/4035/4028 (matched). Slot sentence
upgrades and is PRE-REGISTERED, dies with cc2 seeds: "the SAME self-positive anchor term
is inert insurance under dense image-conditioned reward and an active performance knob
under sparse same-modality reward." Outstanding gate: cc2 3-seed mean only. Table-first
executed on their side (8g @ 2635af6b carries all relayed figures incl. lora_B column and
our adapter-ladder replication credited).

### Conventions round (writer, 15:1x) — three adopted
(1) NAME THE METRIC IN THE CLAIM: "best cell moved least" is true in-domain (R@5/cEM)
and INVERTS on NQ-transfer (most-displaced best: 47.6 @2e-5 vs 46.6 @5e-6). Displacement
predicts neither quality nor damage on its own; what it predicts depends on the metric
asked. Same shape as the ordering-holds-everywhere catch from the 12th: the omitted
column is where cross-side claims fail.
(2) ANCHOR TYPE RIDES WITH ARM NAME, always: text composed cells = anchor(self); text
GOLD arm = anchor(ext) (their table lines 449-450 carry both). Three true statements
coexisted and the arms crossed twice in correspondence; the label prevents the third.
(3) MM consumption wording: PLATEAU (34.00+/-0.47 / 34.80 / 34.07, upper two 1-seed,
span inside spread) — decline to read the rise. My earlier "consumption possibly rising"
phrasings are superseded.
Also recorded: gate 2 discharged (+2.2/+2.4 cEM is anchor strength at matched rate AND
movement); first relay scored accurate-but-unsourced in their provenance block.

### qwen25_3b-v3pure-lr3e5 (14:13) — ladder complete: dead at all rates, phenotype slides
entR@5 0.0373(zs) -> 0.0060, distinct 2901/3000 (full diversity), acc 0.1733. Full ladder:
2e-5 dead/partial-diversity(2033) | 3e-5 dead/full-diversity(2901, drift-like) | 1e-4
dead/contracted(32). Raw Qwen2.5-VL-3B fails at every tested rate; LR slides the failure
MODE (drift-like at gentle, contraction at hot) without changing the verdict. Non-
monotone diversity across the ladder (2033 < 2901 at gentler rate) noted, uninterpreted.

### vlm2vec size axis (landed during session-restart gap, ~15:3x)
vlm2vec7b-zeroshot: entR@5 0.1437, distinct 2963, acc 0.1733.
vlm2vec7b-v3pure (2e-5): entR@5 0.1437 -> 0.0290, distinct 2983/3000 (full diversity),
acc 0.1460 — **DRIFT, same phenotype as 2B** (0.157 -> 0.007/0.0017 at 2e-5/5e-6).
VLM2Vec-trained Qwen2-VL drift is now LR-ROBUST (4x range, 2B) AND CAPACITY-ROBUST
(2B->7B): the failure follows the training recipe, not scale. (In-domain entR@5 metric;
lora_B 16037 not comparable across model sizes.)
vlm2vec4b-zeroshot (Phi-3.5-V, VLM2Vec-Full): entR@5 0.1277, distinct 2925, acc 0.1553 —
mid-signal ladder point established. vlm2vec4b-v3pure (MLX f1a4695950630a51) pending:
tests whether drift follows the VLM2Vec RECIPE across backbones (Phi vs Qwen2-VL).

### 2026-08-18: text-side return — RETRACTIONS BINDING, board recovery, protocol adopted
AUTHORITATIVE text-side source: trl .../rag/HANDOFF_MM.md (supersedes all pre-Aug-17
messages). RETRACTED there and marked in sec_mm_rag.tex here: (1) beats-Rel-SFT (both
domains; 1-seed max-over-ck comparator; seeded SFT bar rose 46.4->47.76; ties at 0.97/
1.15 SE — surviving claim: annotation-free PARITY from cheaper supervision, 3x tighter
seed SD, no retrieval damage); (2) anchor-strength knob (cc2 3-seed 69.1+/-1.4 < base;
slot in tex discharged DEAD with the approved replacement sentence).
PROTOCOL ADOPTED (their sec 5): matched seed counts + same estimator before ANY margin is
quoted; small evals conceal failures. APPLIED TO US: our +1.71-vs-SFT margin uses a
1-seed SFT comparator — ESTIMATOR-RISK note added above the headline paragraph in tex;
SFT seeds s1/s2 queued (sft_seed.sh, GPUs 5/6, 6 cks each, per-seed max then mean).
+3.91-vs-base is comparator-independent and unaffected.
COS PRE-REGISTRATION (acb060a) SCORED: CONFIRMED both clauses from the dead partial's
checkpoints (matched seed/LR/pool): cos ck500/1500/2500 = 7227/10365/10779 lora_B vs
constant 500/1500/3000 = 7194/11915/15584 — equal at 500 (schedule barely decayed),
-13% at 1500, ratio ladder 1.00/1.43/1.49 vs 1.00/1.66/2.17. Integral-of-schedule
reading and per-LR boundary framing STAND. (Writer session gone; scoring recorded here.)
BOARD RECOVERY: quota flap killed all in-flight Aug-13 work. Twins all TRAINED (500
steps, adapters on disk) — only evals died; all five eval chains relaunched locally
(GPUs 3-6). Dead runs resubmitted to MLX: vlm2vec4b-v3pure 885a9aad, v3-cos-s3000
f35eb3d3, v3-b16cos-s3000 ad2f9608 (partials preserved as *.partial / *.old).
CHECKPOINT LOSS (my Aug-13 cleanup, honest record): scale200k + v3-rows200k dense
checkpoint dirs were deleted as "results-landed" — the v1-vs-v3 over-training test lost
its arms; hypothesis stays SOFT unless a checkpointed v1 re-run (~2500 steps) is ordered.
CODE: --beta was DEAD (parsed, never read; kl_beta is live) — same defect class as their
PL-path KL; aliased with conflict guard. No landed cell affected (all intended KL=0).
THEIR OTHER RESULTS FOR CROSS-REFERENCE: KL-to-base beats base 4/4 seeds and INVERTS the
consumption decline (anchor recipe fell at 3x, KL rose); 4B chain inverts (base > RL >
SFT — matches our raw-VLM damage findings); pool is not the constraint (oracle pool <=
K=8); data scaling flat (novel ~= recycled); my saturation instrument arbitrated their
plateau as CONVERSION-side (reward std RISES, degenerate frac 1.00->0.00).

### 2026-08-18 06:1x: NODE HANDOVER — profile campaign design (charter: taxonomy > point-wins)
Text-side taxonomy axes to test on MM: base headroom / domain saturation / regularizer
type / budget. MM state per axis + cells launched:
- HEADROOM: largely profiled (phenotype ladder; learners = retrieval-finetuned bases).
  NEW: bgevl_b-v3pure-lr1e5 [GPU1] — does the LR window transfer within family size?
- REGULARIZER: kl_beta now LIVE (was dead flag). v3kl001 (0.01) [GPU7], v3kl01 (0.1)
  [GPU2] vs v3-pure baseline; if either wins, KLxbudget long cell follows (text: KL
  INVERTED their consumption decline — MM analog tests whether KL lifts our plateau).
- DOMAIN/TRANSFER: found landed OOD row — E-VQA transfer (gme2b, InfoSeek-trained):
  zs 42.07/76.31, v3pure 45.87/81.34 (+3.8 acc/+5.0 R@5), v2 42.60/81.53 (retrieval
  transfers, acc doesn't), v1 41.60 (BELOW zs acc), SFT 39.33/71.05 (ANTI-transfers,
  -2.7/-5.3). RL-transfers-SFT-damages mirrors text's in-domain stability finding, OOD.
  1-seed; protocol fix launched [GPU0]: evqa-v3pure-s1/-s2 + evqa-sft-ck750 (tuned SFT,
  replacing released-config comparator).
- BUDGET: profiled (plateau; movement sublinear). Await KL cells before KLxbudget.
Estimator upgrades in force: SFT seeds s1/s2 [GPU5/6 chained]; decisive cells to use
vqa n=3000 going forward. GPUs 3/4 finishing twin evals; MLX: 4b + 2 cos runs.
Charter endpoint: 3+ working recipes OR the profile "gains require retrieval-finetuned
base + in-window LR + (regularizer TBD); gains transfer OOD; SFT's do not".

### Architecture page published (2026-08-18)
"The VQA Retriever Loop" — https://claude.ai/code/artifact/e07889f2-a50d-4911-9051-ed2e77bf2f94
MM twin of the text side's "The No-Gold Retriever Loop" (0bd9cda2). Same design system
(teal=trainable, ochre=frozen env); MM-specific content: the image-enters-twice edge,
win-by-subtraction recipe, diff tables vs standard GRPO AND vs the text twin, base-is-
part-of-the-recipe scope block, acc-at-floor warning. All numbers carry seeds/SDs; the
SFT margin carries the estimator-risk caveat inline; EVQA transfer marked 1-seed.

### 2026-08-18: 1M-draw scaling run launched (user authorization: "explore up to 1M data")
Consumption plot REMOVED from the architecture page (3 points too few — user call).
v3-b16-1M submitted (MLX fd14405895935cd3): v3-pure recipe, b16, lr 2e-5 CONSTANT
(per-LR rule; no schedule), big pool 200k rows (~5 epochs at 1M draws — novel~=recycled
is established on both sides), 62,500 steps = 1,000,000 reader-scored draws, checkpoint
every 3,125 steps (50k-draw resolution, 20 ckpts). Eval plan when ckpts appear:
log-spaced subset {50k, 100k, 200k, 400k, 700k, 1M} + lora_B per ckpt (movement law at
scale: does displacement stay sublinear to 80x?). Risks accepted: no resume support —
if the job dies at step N the curve to N survives via checkpoints; MLX walltime unknown
at this length. Est. 3-3.5 days. Curve replaces the 3-point series everywhere when it
lands; the b4 2k/6k/12k points stay as the separate low-batch lane (batch is a
confound — never merge the lanes in one series).

### Anchored twins landed (06:52) — ANCHOR RESCUES CONTRACTION, NOT DRIFT; 3rd recipe found
[entR@5, 1 seed each, anchored = cc 0.3 same-config twin]
- siglip2: zs 8.23 | bare-RL 0.40 (contracted) | ANCHORED **17.47** (distinct 2464), acc
  13.13 -> 19.73. The anchor doesn't just prevent contraction — SigLIP2 LEARNS: largest
  R@5 gain on record (+9.2 > bgevl's +5.5), largest acc gain (+6.6 > headline's +3.91).
  Same movement as the collapsed bare run (lora_B 15394 vs 14610) — same movement,
  opposite outcome, THIRD confirmation movement doesn't determine fate. lr 1e-4.
  ** BREAKS the "only retrieval-finetuned bases learn" claim — the correct form is:
  retrieval-finetuned bases learn bare; contraction-class bases can learn WITH the
  self-anchor; drift-class cannot learn at all. **
- vlm2vec2b: zs 15.67 | bare 0.70 | ANCHORED 1.30 — drift NOT rescued by the anchor
  (text side's anchor DID hold drift at bay: 0.431 vs 0.063 — CROSS-MODALITY INVERSION:
  anchor fixes drift on text, fixes contraction on MM, not vice versa).
- qwen2b: zs 0.03 | anchored 0.00 (distinct 535) — nothing to rescue, corner stays.
ANCHOR ROLE REVISED (3rd time, each time richer): not "inert insurance" — it is
BASE-CONDITIONED: enabler for contraction-prone bases, no-op on stable bases (gme2b),
useless against drift. CAMPAIGN STATUS: 3 working recipes = (1) v3-pure on retrieval-
finetuned bases (gme2b 3s, gme7b 2s), (2) bgevl_l @1e-5 in-window (1s), (3) siglip2 +
anchor (1s — REPLICATION CELLS NEEDED: 2 seeds + does it generalize to clip/clipb16?).
EVQA transfer s1: +1.3 acc/+2.6 R@5 (s0: +3.8/+5.0) — margin real but varies; s2 pending
before quoting a mean. Artifact updated: retrieval-first tables, 7B ordering table,
siglip2 anchored row, base-row numbers.

### 2026-08-18 21:xx: overnight harvest — two claims die, one big one survives seeding
[all entR@5/acc, seeds noted]
1. **SIGLIP2 ANCHOR RECIPE RETRACTED**: s1 1.10 (distinct 1118), s2 0.13 (77) vs s0's
   17.47 — one-seed lottery, 3-seed mean below zs. Generalization agrees: clipb16-anchor
   1.13, bgevl_b-anchor 3.77 (both still contract WITH anchor). "Anchor rescues
   contraction" is DEAD; anchor-rescue was never real. Recipe count back to 2. The seed
   protocol caught it pre-print, exactly as designed. Artifact corrected (row kept as a
   warning exhibit).
2. **BEATS-SFT SURVIVES SEEDING ON MM** (the claim text retracted): SFT per-seed maxes
   33.00/33.67/31.87 -> 32.85 +/- 0.90 (3 seeds, same estimator both legs). RL 34.71
   +/- 0.19 -> +1.86 = 3.5 SE. MM SFT bar did NOT rise with seeds (text's rose +1.36).
   Estimator-risk fence in tex can now close with these numbers.
3. **KL hurts on MM**: v3kl001 33.80/76.73 (acc -0.7, R@5 +1.5); v3kl01 32.73/74.50
   (-1.7 acc). Regularizer axis closed: bare > KL on MM; KL essential on text — the
   regularizer must match the failure mode the base actually has, and gme2b has none.
4. **LR window is per-MODEL not per-family**: bgevl_b at bgevl_l's window rate 1e-5:
   14.73 -> 8.90 (fails). BGE-VL-large's +5.5 does not extend to its smaller twin.
5. **gme7b-zeroshot 30.67/69.37/53.59**: 7B chain complete — base 30.67 < SFT 33.20 <
   RL 34.60/35.73; RL>SFT>base at both sizes. Artifact 7B row filled.
6. **EVQA transfer 3 seeds complete**: v3 +2.6+/-1.3 acc / +3.9+/-1.2 R@5. TUNED SFT
   ck750 transfers POSITIVELY (+1.4/+2.1, 1 seed) — "SFT anti-transfers" was an artifact
   of the weaker released-config comparator; corrected everywhere. RL transfers ~2x SFT.
7. **cos eval side**: v3-cos-s3000 34.33/77.13 ~= constant 34.07 at 13% less movement —
   schedule-integral story consistent both halves now (movement + eval).
### 30h scaling infrastructure (user charter: try harder on data scaling + art's curve)
Node contended: user's own Qwen3-4B math-eval sweep (evaluate_math.py, vLLM x8) took all
GPUs at 21:2x. Deployed scaling_fleet.sh: claims GPUs as they free, launches in priority
order — sft-big-1M (SFT art curve, 42k steps = 1M pairs, ck/50.4k), v3-b16-local (RL
consumption, 6250 steps = 100k draws, ck/10k, reader_batch 64), v3-rows1M + v3-rows1M-
s3000 (dataset-SIZE axis: 25k->200k->1M rows at 2k and 12k draws), v3-b16-local-s1
(low-end error bars), then eval daemons on remaining GPUs (auto-eval every checkpoint +
lora_B). pool_train_1M building (full InfoSeek train, target ~1M rows). MLX v3-b16-1M
kept as long-tail insurance (35s/step — reaches ~80-100k draws by window end).

### 2026-08-21 ~07:00 INTERIM — pool-size x consumption interaction materializing
At matched 3k draws (ck750, 1 seed each): 45k rows 33.20 vs 810k rows 35.13. Small pool
has fallen below its own 2k-draw score (34.71); big pool has risen above its own (34.73).
REGISTERED READING (falsifiable by ck1500-3000, in eval pipeline): pool size is null at
2k draws and decisive beyond — small pools EXHAUST under continued consumption, large
pools keep converting. If it holds to 12k draws, the "consumption plateau" was a small-
pool artifact and the honest claim becomes "RL scales with draws GIVEN sufficient data
diversity" — which would also reconcile the MM/text divergence (their pools were small).
SFT rows-line complete & flat: 25k 32.85+/-0.90 | 200k ck600 32.73 | 810k ck600 32.47.
1.1M mixed-source pool built (1,100,589 = InfoSeek 810k + EVQA-landmarks expanded 290k);
v3-rows1p1M cell chained. Mixed-source + larger-index caveats logged for the chart.

### 2026-08-22 01:2x: SCALING CHART COMPLETE — both finals + full mixed line
12k-draw finals [acc/entR@5, 1 seed]: 45k 33.13/76.17 | 810k 33.80/72.63.
REGISTERED INTERIM READING SCORED: "big pools keep converting" = MISS (falsified by
ck1500 on). Surviving shape, all pool sizes and batches: RISE-THEN-DECLINE in acc; more
data RAISES THE PEAK slightly and SHIFTS it right (~2k->3k) but no pool escapes turnover.
On RETRIEVAL the story inverts: 810k R@5 peaks at 2k draws (78.70, best on record) and
decays 6.1 points by 12k; 45k R@5 ~flat (76.1-77.2); acc and R@5 peak at DIFFERENT draws
(the anti-diagnostic decoupling inside the scaling curve). BEST RET RECIPE: biggest
pool, shortest training.
MIXED LINE @2k draws (composition-constant subsets): 69k 33.13/77.20 | 275k 34.67/76.53
| 1.06M 34.33/76.77 — flat; diversity neither helps nor hurts at 2k; the same-source
810k R@5 spike (78.70) does NOT reproduce under mixing (76.77) [1 seed both].
EVERY RL config > entire SFT band > base on both metrics at every point measured.
SFT art: flat 31.5-33.3 acc / 73.4-76.2 R@5 across 50k-655k pairs AND 25k-810k rows.
b16 lane consistent (35.07@10k -> 33.47@30k draws). CAVEATS: non-2k points 1 seed;
810k-s1 + remaining interim cks still in eval queue (GPUs 5/6).
GPU map: deepeyes exited; split with text side = they 0-3, we 4-7.

### 2026-08-22 23:5x: dense checkpoints REVISE the scaling story — name the metric
With ck2250s + b16-1M cks in, "rise-then-decline in acc" was an OVER-READ of sparse
1-seed points. ACCURACY is NOISY-FLAT at every pool size (810k oscillates 33.8-35.1
across 2k-12k draws — swings within the n=1500 seed/eval noise band; 45k similar;
b16-200k drifts 34.4->33.9 over 50k->150k draws, slow, no cliff). The 35.13 "peak" is
inside noise. RETRIEVAL is where the direction is real and consistent: 810k entR@5
declines 78.70 -> 72.63 (monotone-ish, -6.1) while 45k stays flat 76.1-77.2 and
b16-200k drifts down slowly. CORRECTED CLAIMS, metric-named: (1) acc: no data-scaling
gain, no dramatic over-training loss — flat within noise everywhere above SFT+base;
(2) retrieval: big-pool SHORT training reaches the best index quality on record (78.70
@ 2k draws, 1 seed, s1 validation job running); continued training CONVERTS INDEX
QUALITY DOWN while acc stays flat — the decoupling is the scaling story. b16-1M run
alive at 150k+ draws (34/72.3 — 12x beyond old max, still > SFT band).

### 2026-08-23: pool_div_1M CURATED (mmrag-ema step 1 data) — 998,918 rows, seed 0
Composition (fixed, pre-shuffled; ALL runs subsample head-N): InfoSeek 564,000 |
E-VQA-landmarks expanded 249,083 | OVEN train 176,826 (dirs 01-04; +162k more if shard00
is ever added) | OKVQA 9,009. Corpus_div = corpus_mix + OVEN gold passages (3,629, pid
40M+) + OKVQA google-search passages (114,809, pid 50M+). Images: OVEN shards 98GB +
OKVQA COCO 13.5GB downloaded under quota guard; all keyed by bare img_id via tar/zip
offset indexes (an img_path-vs-img_id keying bug cost one build round — caught by the
rows-kept-0 signal, fixed, recovered exactly the predicted 176,826).
Sources vetted and REJECTED: MMEB WebQA (no answer strings), ChartQA/DocVQA (answer from
image -> retrieval reward vacuous), KVQA (M2KR dropped answers), WIT (200GB, no QA),
OVEN shard00 (76GB split-zip, deferred), EVQA-iNat (images only in 224GB monolith).
EMA-index A/B (div45k-v3pure vs div45k-emaidx m=0.9 extra=256) auto-submits next.

### 2026-08-23: pool_div_1M_v2 — M-BEIR gold upgrade (hybrid), 1,073,002 rows
OVEN slice: 78,953 rows with M-BEIR train-qrels gold (multi-positive, up to 3 text golds,
answer = gold's wikipedia_title) + 171,900 rows keeping M2KR coarse gold (distinct images;
per-row gold_src tag). InfoSeek 564k / EVQA 249k / OKVQA 9k carried byte-identical.
Corpus_div_v2 = 1,259,544 passages: prior articles + OKVQA google (50M..) + full M-BEIR
task6 text pool 676k (40M..) + v1 oven summaries remapped to 39M.. (a 40M pid collision
between v1 j-indexing and M-BEIR did-indexing was caught by the dup/dangling audit:
0 duplicate pids, 0 dangling pos_pids after remap). M-BEIR non-QA tasks rejected for the
RL pool (no answer strings -> judge reward vacuous); WebQA-with-answers still open.

### 2026-08-23: query-diversity audit -> pool_div_1M_v3 (TIERED)
User flagged template monoculture; audit confirmed: unique-question rates InfoSeek 0.2%
(860 q / 564k rows), OVEN 0.4% (three generic templates = 83k rows), EVQA 22%, OKVQA 92%.
v3 = per-question caps (InfoSeek 1500, OVEN 600, OVEN-generic 1000 total) + all sources'
full diverse mass -> 702,271-row DIVERSE CORE; template surplus (371k) appended after as
tier 2. Subsample convention unchanged (head-N) and now diversity-optimal by construction.
WebQA (33k natural questions) blocked on answers: M-BEIR strips them, official release is
Drive-only — open item. EVQA expanded set fully consumed (249,083 = all corpus-groundable).

### 2026-08-23 11:20: 810k retrieval peak FAILS seed validation
v3-rows1M-s1 (810k, 2k draws): acc 34.40 / entR@5 74.37 vs seed 0's 34.73 / 78.70.
2-seed R@5 = 76.54 +/- 3.06 — indistinguishable from 45k's 76.09 (3s). RETRACT: "more
data helps retrieval (+2.6)" and "best ret recipe: biggest pool shortest training" as a
POOL-SIZE claim. Acc 34.57 +/- 0.23 (2s) — flat, consistent. What SURVIVES: the
within-run retrieval decline under continued training (s0 trajectory 78.7 -> ~73 across
2k->12k draws exceeds the wiggle; softened to "large-pool retrieval declines along
training in the one run instrumented"). Seed protocol's third single-seed spike kill
(cc2-anchor, siglip2-anchor, now this). Data-scaling verdict is now fully null on BOTH
metrics at matched budget: pool size buys neither accuracy nor retrieval.

### 2026-08-23: text-side EMA convergence + audits answered pre-landing
Text side independently implemented the same ladder (online_kl_ema = EMA-as-KL-reference
bounding rate-of-change; online_dynamic_index + index EMA + stale-frac sweep). Their two
audits, answered from code: (1) blend renormalization PRESENT (train_rl.py:318); (2) our
extra-refresh is a GLOBAL stalest-first sweep, not retrieved-set-biased — no rich-get-
richer. REGISTERED SUSPECT pre-landing: coverage — ~368 docs/step touched = ~53% of the
~350k div index per 500 steps; untouched half stays at init embeddings (max staleness 500
vs legacy <=100). If emaidx underperforms, coverage fraction first, --index_refresh_extra
the knob. ADOPTED: mean-pairwise-cosine from checkpoints joins the standard harvest
(their base bar 0.166 on their 1M corpus). PRE-REGISTERED for scheme 4: EMA-as-reference
degenerate solution (shadow tracks policy, KL->0, geometry unconstrained) — sweep the
EMA rate, measure cosine from checkpoints, never training metrics. GPU map: text takes
GPU 3 (capped); my three arms on 0-2 evaluate on their own GPUs.

### Coverage becomes a cross-side ablation (text-side discovery via our registered suspect)
Their computed coverage: 9.6% (64 sweep rows x 1500 steps / 1M corpus) — their dynamic-
index arm was "a frozen index with a moving corner". They submitted dmixDynCov (668
rows/step = exactly 1x coverage) making it an ablation: 9.6% / 53% (ours) / 100% bracket
the coverage question; if low loses and full wins, coverage is CONFIRMED as mediator.
ADOPTED their rule: sweep coverage now prints at startup (budgets must never be
implicit). Geometry watch baselines: their DMIX corpus base pairwise-cos 0.166 (diverse
corpus starts LOW; their old 0.41 was pool-shape-specific); 40-step smoke SPREAD 0.266->
0.254. Our cc=0+beta=0 500-step stability on gme2b = negative control: collapse risk is
BASE-dependent, not intrinsic to the anchorless objective.

### Coverage ablation v2 — ordered by STALENESS, not reach (their correction, adopted)
Mean staleness for undercovered sweeps = T*(1 - coverage/2). Joint ladder (steps stale):
their dmixDyn 1428/1500 | their dmixDynCov 748/1500 | OUR emaidx ~368/500 | their new
dmixDynFreq 187/1500 (4x coverage, ~0.5s/step — frequency is CHEAP). Reading rules
REVISED: arms ordered by staleness; "all tie => relax" branch weakened (their low arms
were never far apart in freshness); dmixDyn~dmixDynCov tie + dmixDynFreq win reads as
"frequency matters, reach was the wrong knob".
MECHANISTIC NOTE for reading our arms: pool+boundary docs are re-encoded fresh EVERY
step, so the scored candidates are never stale — staleness lives only in RETRIEVAL
RANKING (whether the right rows reach the pool at all). emaidx underperformance would
therefore indict ranking-staleness specifically. If it does: coverage>1x on our side =
--index_refresh_extra 700 (1x) / 2800 (4x), cost ~seconds/step.
(Their latent-defect find from this exchange: single-forward refresh OOM at large
budgets — ours already chunks via encode_docs batch_size.)

### Eval-parity audit (prompted by text-side's two eval defects) — one clean, one fixed
(1) HEAD-SLICE: our queries_test.jsonl is SHUFFLED — first-3000 split mix 75.0/25.0 vs
full-file 74.6/25.4, adjacent same-entity 116/71k. No bias; all campaign numbers stand.
(Their DMIX first-500 was 49% MuSiQue — R@5 understated 16 pts; their internal ladders
stay valid, absolute numbers relabeled.)
(2) BASE-INDEX: our eval re-encodes the corpus with the trained checkpoint — clean. BUT
the emaenc arm had the MIRROR defect: training scores q(live)·d(EMA), eval encodes docs
with live weights (EMA lags ~1/(1-m)=100 steps of movement). FIX: final save now writes
the EMA tower as runs/<name>/ema_tower/ (artifact, not cache). TONIGHT'S emaenc cell
predates the patch -> its eval is live-weights-only; result carries that caveat, and the
queued MLX duplicate (patched code at run time) will save both towers for the two-way
eval (live vs EMA doc encoding — itself an informative comparison).
Their scoring-vs-ranking answer: their _prepare_inputs also re-encodes scored candidates
-> both sides vary RANKING-staleness only; shared knob confirmed.

### Head-slice class-grep — harness IMMUNE; my own audit produced a FALSE ALARM (retracted)
Class-grep outcome: eval_retrieval.load_jsonl AND eval_vqa both take seeded
random.sample slices — never head-N — so grouped files (queries_evqa IS grouped: 2755/
4749 adjacent same-entity) cannot bias any eval. Verified by reproducing the exact
seed-0 samples: EVQA retrieval slice 52.7/47.3 iNat/landmarks vs full-file 52.6/47.4;
vqa sub-slice 51.9/48.1. ALL EVQA numbers stand unchanged.
PROCESS LESSON (mine, logged at my expense): I declared "confirmed bias" after measuring
the FILE HEAD — the wrong stage; the sampler's output is the only thing that counts.
Audit the quantity the code actually consumes, not the artifact it starts from. The
phantom was caught before any code change, re-eval, or relabeling happened.

### Conventions, final forms from the audit exchange
(1) BIAS CLAIMS: a file-level observation is a HYPOTHESIS regardless of which way it
points; no bias claim until measured through the code path (the consumed quantity). Their
correct call was right only by coincidence of implementation; my phantom was the same
error pointing the other way. One rule covers both.
(2) RECORDED NON-FIX (pattern, 2nd independent occurrence): when a defect-class fix
would break comparability for zero bias benefit, the fix IS the note — record why it
stays so no future hand "fixes" it into an invalidation. (Their eval_rag_l2.py head-slice
kept for table comparability; writer's gold-arm non-edit on the 13th.)
Their sweep: Table 8m relabeled (first-500 was 100% sh_alias — single-hop mislabeled as
mix; internal comparisons valid); published Tables 8a-8l clean.

### SFT comparator on the diverse pool — chained (user directive: "run SFT compare")
sft-div45k: tuned config (lr 1e-4, b24, 1200 steps, ck/150) on pool_div45k_v3 (same 45k
head-slice as the RL arms; train_sft auto-filters to gold-bearing image-valid rows), full
per-seed best-ck estimator (evals ck450-1200). Launches on GPU 0 when the RL arms land.
Seed 0 tonight; PROTOCOL REMINDER: no RL-vs-SFT margin quoted until seeds s1/s2 match
the estimator on both legs. Note eval stays the InfoSeek benchmark — the div pool is a
TRAINING-side change; both legs share it, so the comparison is recipe-vs-recipe on
identical data.

### EMA three-arm verdict, seed 0 (div45k_v3 pool, corpus_div_v2, InfoSeek eval, 500 steps @ 2e-5)
All three arms trained healthily (reward/raw_mean ~0.20-0.22 and degenerate-group frac
~0.57 flat first-50 vs last-50 in every arm — the near-zero final-line snapshot was one
noisy step, not collapse). Eval chains re-run locally after the entry-script self-edit
crash (see recovery note below).
  div45kv3-v3pure  entR@5 76.83  ansR@5 64.57  acc 33.00   (baseline, fresh /100-step refresh)
  div45kv3-emaidx  entR@5 75.77  ansR@5 63.14  acc 32.93   (schemes 1+2: index-EMA m=0.9 write-back + boundary negs + 256/step stalest sweep)
  div45kv3-emaenc  entR@5 72.47  ansR@5 60.49  acc 32.20   (+ scheme 3: model-EMA doc tower m=0.99)
Reading, scoped: (a) index-EMA is parity-to-slightly-negative (-1.06 entR@5, -0.07 acc,
1 seed — inside the seed band; no evidence it helps at 45k-pool scale where full refresh
is still affordable; its case must come from BIG pools where fresh refresh is the thing
you can't pay for). (b) The model-EMA arm's -4.36 entR@5 is CONFOUNDED: these local
weights predate the ema_tower save fix, so training scored q(live)·d(EMA) but eval
encoded docs with LIVE weights — a train/eval geometry mismatch. The MLX duplicate
carries the fix; its two-way eval (live-doc vs EMA-doc encoding) is the clean read. No
scheme-3 verdict until it lands. (c) Diverse-pool v3pure vs original InfoSeek-pool
v3-pure: retrieval matches (76.83 vs 76.09), acc 33.00 vs 34.71±0.19 — plausibly the
diverse-training-distribution cost on an InfoSeek-only benchmark; 1 seed, not a claim.
final index/stale_mean ~205 in both EMA arms (T=500), consistent with the designed
coverage regime; scored candidates always fresh in all arms.

### Recovery note + CONVENTION (3rd shell-discipline entry)
The three local arms' eval stages all died at "mlx_entry_cell.sh: line 4" while training
survived: mlx_entry_cell.sh was edited in place (keep_gpu guard v2, ~15:30) while three
bash processes launched at ~15:05 were still executing it — bash reads scripts
incrementally, so an in-place edit corrupts the read offset of every running instance.
CONVENTION: never edit a shell script that live processes are executing; for running
launchers, copy-then-edit (new filename) or cp to a versioned name and exec that.
Training was unaffected (python already loaded); adapters + metrics complete; evals
re-run via eval_run.sh on GPUs 0-2, results identical protocol.

### Geometry probe (text-side exchange): the spread signature does NOT transfer
Mean pairwise cosine, fixed seeded 2,000-doc sample of corpus_small, per checkpoint
(geom_probe.py; results/geom_probe_div45kv3.json):
  base 0.0726 · v3pure 0.0486 · emaidx 0.0567 · emaenc 0.0895
Text side sees base 0.166 > frozen 0.114 >> dynamic 0.017-0.037 (dynamic arms spread
hard without buying recall). Tower side inverts the roles: the BARE RL loop is what
spreads (v3pure, -0.024 vs base); index-EMA damps that spread (0.0567, closer to base)
while also not buying recall; the model-EMA arm CONTRACTS above base (0.0895) — though
that cell is live-weight-encoded (same eval-geometry caveat as its recall number; EMA-
tower encoding may read differently). No near-zero collapse anywhere. So "index dynamics
buy spread without recall" is a text-side finding, not a mechanism truth: here the
spread belongs to the RL objective itself, and EMA write-back moderates it. 1 seed,
2,000 docs, absolute magnitudes not comparable across sides (different encoders/corpora)
— ordering is the claim, nothing else.

### Query-side geometry probe: the objective acts on the query tower
Same checkpoints, fixed seeded 1,000 image+question queries (geom_probe_q.py;
results/geom_probe_q_div45kv3.json). Query-side vs doc-side mean pairwise cosine:
  base    0.2212 | 0.0726
  v3pure  0.0594 | 0.0486
  emaidx  0.0612 | 0.0567
  emaenc  0.0976 | 0.0895
Four readings, scoped to 1 seed each:
(1) The RL spread lives on the QUERY side: v3pure moves queries -0.162 vs base against
-0.024 doc-side — consistent with the gradient structure (queries get direct gradient
every step; docs only via refresh/write-back). Matches the text side's corrected
query-side result: every annotation-free cell spreads queries below base, both
modalities. This is now a cross-side agreement on the comparable axis.
(2) The doc-side emaidx "damping" is a doc-tower-local effect: query-side emaidx is
indistinguishable from v3pure (0.0612 vs 0.0594) — as expected, index-EMA never touches
the query path.
(3) emaenc is systematically the most geometry-conservative arm on BOTH sides (queries
0.0976, docs 0.0895-above-base). NOTE the asymmetric caveat: emaenc's QUERY tower is
live in training (only docs use the EMA tower), so its query-side number is CLEAN —
unlike its doc-side and recall numbers, which carry the live-weight eval confound.
(4) Same encoder, healthy bar differs by side (base 0.2212 query vs 0.0726 doc) —
confirms the text side's "no single healthy number" point; only orderings travel.
Pending falsifier for the cross-side SFT-contraction claim: query-side probe of the
div45k SFT comparator's best checkpoint when it lands ("annotated contracts,
annotation-free spreads" needs the second modality's SFT leg, not just its RL legs).

### EMA convention pinned (cross-side blend-weight hypothesis)
train_rl.py ema_write (line ~317): index[p] <- l2norm(m*old + (1-m)*new), m = --index_ema.
So m=0.9 is RETENTION on the old row -> fresh weight 0.1. Text side's online_index_ema
0.5 is the weight ON FRESH. Common convention (fresh weight): mine 0.1, theirs 0.5 — a
5x fresh-injection difference, which sits right next to their ~6x doc-side geometry
delta (-0.141 vs my -0.016/-0.024 vs own base). Conventions differ but the 6x is
PHYSICAL, not definitional. Also pinned: my blend renormalizes (l2norm) after mixing —
if their write-back doesn't, unit-norm drift is a second divergence to rule out before
attributing everything to the blend weight. Their live hypothesis stands as the
best-posed unrun cell: doc-side spread set by blend weight, not refresh volume (their
coverage sweep already flat at fixed blend; my damped write-back barely moves docs).
Recorded as UNRUN on both sides; their allocation is committed to the 50k budget run.
Normalized-by-own-base query-side agreement also on record: unregularized annotation-
free loop = -0.16 to -0.17 vs base on both sides (their table, b7bf8cf8).

### Blend-weight hypothesis: both artifact explanations now ruled out (text side 6b22fe84)
Text side grepped their _refresh_rows: they DO renormalize after blending (unit-norm
asserted in tests/test_dynamic_index.py, re-run, passes). So the shrinkage confound I
raised is ruled out on their side, and the blend weight stands alone as the live knob
for doc-side geometry: fresh weight 0.1 (mine) -> docs ~static; 0.5 (theirs) -> -0.141.
Cell remains UNRUN on both sides by explicit allocation decision, with both artifact
explanations (coverage, norm shrinkage) affirmatively excluded rather than unexamined.

### 2026-08-28: div45k SFT comparator landed; loop page committed with per-row reproduce commands
[all entR@5 / ansR@5 / acc, InfoSeek, 7B eval reader, 1 seed each unless noted]
sft-div45k (tuned config, pool_div45k_v3, ck/150, 1200 steps; evaluated ck450..1050):
  ck450 73.67/59.02/31.67 · ck600 74.43/58.00/30.87 · ck750 75.87/60.41/32.07 (best) ·
  ck900 72.77/59.80/31.93 · ck1050 71.57/54.53/30.73. ck1200 NOT evaluated: launcher looked
  for checkpoint-1200, train_sft.py writes the final adapter at the run root (gap, not a loss).
  Curve shape: peaks at 750 then declines on both retrieval columns — same shape as the
  InfoSeek-only curves (seed 0 peaked at 750 too).
RL v3-pure on the same pool 76.83/64.57/33.00 → RL − SFT-best = +0.96 / +4.16 / +0.93.
  Recorded, NOT claimed as a margin: one seed per leg (protocol: matched seeds first).
  The answer-recall gap (+4.16) is the largest of the three and the one worth seeding.
Doc: mmrag/docs/vqa_retriever_loop.html now carries a "reproduce" toggle under every
  result row (exact CELL_TRAIN_ARGS / script path / result file). Corrections made while
  reconciling each row against args.json + results/:
  - v2 (+anchor) entity R@5 filled in: 80.13 (was "—" in the page; the tex already had it).
    v2 retrieves better than v3-pure on both recall columns and answers worse; the
    "retrieval is the primary result" sentence now says so. v2 mean = seeds 0/2/3; seed 1
    (plgrpo-gme2b-nogold-s1, 32.93 acc) is on disk and outside the mean — 4-seed 33.60 ± 0.70.
  - SigLIP2 anchored "3 seeds": seed 0 = siglip2-nogold (GRPO-8 + anchor), seeds 1-2 =
    PL-GRPO + anchor. Not estimator-matched replicates; retraction stands regardless.
  - CLIP-L anchored GRPO-8 @1e-4 (clip-nogold-base) 29.13 entR@5 now shown in the ladder
    cell, flagged 1-seed/unreplicated (same shape as the SigLIP2 lottery; not claimed).
  - GME-7B zero-shot 69.37 was on disk (gme7b-zeroshot); ladder said "not on disk".
  - VLM2Vec-4B: both MLX attempts stopped at step 250 (group_std 0 at 249); no result.
    Page says "unfinished" instead of "in flight".
  - Qwen2-VL-2B trained cells filled (0.00 at 5e-6 / 2e-5 / anchored).
  - The page's old appendix line "v2 = --algo grpo --num_candidates 8 + anchor" was wrong;
    v2 is the PL-GRPO command + --contrastive_coef 0.3. GRPO-8 + anchor is the v1/"nogold"
    family (rl-j2e5-nogold-det-repro, scale-rows*, *-nogold).
Scripts moved into the repo so the commands resolve: mmrag/scripts/{eval_run,launch_sft_div,
  sft_curve,run_repro,ladder_zs,local_cell_worker,resubmit_ema}.sh, mmrag/geom_probe{,_q}.py.
On hold per user ("Don't submit new job now"): SFT-best query-geometry probe, emaenc MLX
  duplicate, scheme 4, blend-weight cell.

### 2026-09-07: third site stood up (H200 shared box) — emaidx-1M-s3000 launched
Site: `fukushimadc-02-hgx-0011`, 8x H200 143GB, `MMRAG_DATA=/raid/icy/mmrag_data`,
no MLX and no bytenas mount. Non-MLX path from EMA_HANDOFF.md used throughout.
- **Class-1 data NOT rebuilt.** Pulled verbatim from the HF dataset `Icey444/mmrag-class1-built`
  (published by the prior session). sha256 verified against its manifest for all three cell-1
  files: pool_train_1M `c8ea97ed…`, corpus_small `2182002…`, queries_test `a7c9e11f…`. The
  training process then reported `train rows: 810276`, matching the recorded pool exactly.
  Rebuilding pool_train_1M would have been wrong even if the flags were known: train_rl.py
  draws rows by INDEX off a seeded RNG, and 3000 steps x batch 4 touches only ~12k of 810k
  rows, so file row order alone determines the training set.
- Class-2 images re-downloaded from HF `BByrneLab/M2KR_Images` (both InfoSeek tars, 54 GB);
  808,601 images indexed across the two tars; staging smoke test returned real sizes.
  Class-3 models 31 GB. Baseline `results/*.json` merged with `cp -n` (369 files).
- **Env deviations from the handoff pins, both recorded rather than hidden:** flash-attn
  **2.8.3** not 2.8.1 (upstream publishes no 2.8.1 wheel for torch 2.8; 2.8.3 is the version
  README.md:73 / RESUME.md:13 already ran on Isambard), and transformers 4.57.0 is now YANKED
  on PyPI — the exact pin still installs and imports, but new sites should prefer 4.57.1.
  Also `qwen-vl-utils` imports `torchvision` without declaring it, so a clean venv dies at the
  entry script's dependency check; fixed upstream in 5e1de0f.
- **Shared-GPU protocol** (a co-resident 8-GPU job owns 58-80 GB per card): built an isolated
  venv at `$MMRAG_DATA/venv` so the entry script never pip-installs into the shared system
  python; `CUDA_VISIBLE_DEVICES=<n>` maps the hardcoded `--device cuda:0` onto a headroom GPU;
  `--mem_frac 0.32` (~46 GB of 143 GB) caps training above its ~35 GiB working set and below
  the neighbour's peak. `hostname > $MMRAG_DATA/local_hostname.txt` written BEFORE the first
  launch, so the entry script's `pkill -x keep_gpu` guard never fired on this box.
  Note the cap does NOT reach the three eval stages (CELL_TRAIN_ARGS goes only to train_rl.py);
  left uncapped deliberately — ~63 GB free even at the neighbour's peak vs a ~30 GiB eval need,
  and an eval OOM costs minutes because the entry script skips training when the adapter exists.
  Eval batch sizes left at baseline: `encode_bs` is a free knob (pure forward passes) but
  `reader_batch_size` is NOT — reader_temperature 0.7 means changing it reorders RNG
  consumption and changes which rollouts are sampled.
- Launched `emaidx-1M-s3000` on GPU 1. Args = `runs/v3-rows1M-s3000/args.json` verified
  field-by-field (all 13 estimator-relevant fields identical, corpus and batch_size default
  to the same values) + `--index_ema 0.9 --index_refresh_extra 256` + `--mem_frac 0.32`.
  RL index = 173,755 passages. index_ema>0 means ONE full encode at init and NO periodic full
  refresh — instead 256 stalest docs + the step's own pool docs are EMA-written every step
  (train_rl.py:312-330), so this arm is cheaper than the baseline's 30 full refreshes, and the
  knob is verifiably live (unlike the confounded div45kv3-emaenc arm).
  Results pending; will be recorded curve-vs-curve against v3-rows1M-s3000
  (final entR@5 72.63 / ansR@5 61.59 / acc 33.8; ck2250 74.67 / 62.82 / 35.0), 1 seed = recorded.
- **`--index_ema` IS A COMPOUND TREATMENT — constrain the claim accordingly.** train_rl.py:354
  gates the periodic full refresh on `index_ema == 0`, so this arm differs from
  v3-rows1M-s3000 in TWO ways at once: (a) EMA blending at m=0.9 on write-back, and (b) a
  wholly different refresh schedule — incremental and continuous (368 docs every step)
  instead of 30 discrete full re-encodes. Whatever this cell shows, the honest claim is about
  the EMA refresh SCHEME as a whole; a win or a null CANNOT be attributed to the blending
  coefficient alone. Encoding cost differs by the same token: baseline 30 x 173,755 = 5.21M
  doc-encodes vs this arm's 173,755 + 1.10M = 1.28M. **That 4.1x is DOC-ENCODES, NOT WALL TIME —
  do not read it as a speedup** (a correction to this entry's first version, and to the same slip
  in the handoff). Measured decomposition instead: a full refresh of this 173,755-passage index
  takes 749 s clean (232 passages/s at encode_bs 64 — note REPRO_NEWCLUSTER:92's ~50/s is for a
  different setting), so the baseline's 30 refreshes = 6.24 h = **53% of its 11.77 h wall**, with
  residual training 6.63 s/step; this arm instead pays 368 docs/step = 1.59 s/step. Projected
  clean: 749 + 3000 x 8.22 s = 7.06 h, i.e. **1.67x wall / ~40% saving, not 4x**.
  SCOPE: the saving is bounded by how much of the baseline was refresh, so a smaller index or a
  longer `refresh_steps` shrinks it. This is not a general "EMA is faster" claim.
  This also retro-weakens the div45kv3-emaidx parity row: at 500 steps with refresh_steps 100
  the baseline paid only 5 full refreshes, so the schedule gap there was ~6x smaller than here,
  and "parity at 500 steps" was testing less of the treatment than it appeared to.
  Cheap decomposition available if this cell shows anything: rerun with `--index_ema 1e-8`,
  which still takes the `index_ema > 0` branch (keeping the incremental schedule) while making
  the write-back a pure fresh overwrite with no blending — isolating schedule from blending at
  the same cost. NOT launched; proposed for the user's decision after cell 1 lands.
- Init guard registered BEFORE results: if the coverage line prints PARTIAL rather than FULL,
  stop rather than run 12 h — PARTIAL means some passages keep their init embeddings all run,
  which is a stale-index confound, not an EMA result. Arithmetic predicts FULL at 6.35x.
- Shared-compute note: `nvidia-smi` aggregate `utilization.gpu` reads 0% on this box even under
  load; `nvidia-smi pmon` shows the truth (this run 19-79% SM alongside the neighbour's 14-28%).
  Wall-clock here is therefore NOT a clean throughput number for any other site.
- **Neighbour-safety fix, cell relaunched 17:22 UTC.** The co-resident verl job (8 cards, days
  of runtime) runs vLLM in SLEEP MODE: it releases ~64 GiB during actor training and RECLAIMS it
  when rollout starts, failing hard — killing ITS process, not ours — if anyone holds that memory
  at the moment it wakes. Measured phases per card: actor training 56.4 GiB, handover 18.3 GiB,
  vLLM rollout 80.6 GiB of 140.4 usable. So co-tenants must size against the 80.6 GiB ROLLOUT
  peak, not the training figure or a spot check; the swing is 24 GiB and a single nvidia-smi
  reading can mislead by that much either way (thanks to the h200-rag session for the sampling).
  `--mem_frac` covered training but NOT the three eval stages (~1 h of 7B reader, uncapped).
  Closed with a `.pth` hook in our own venv (`mmrag_memcap.py` + `zz_mmrag_memcap.pth`) that
  applies `set_per_process_memory_fraction(MMRAG_MEM_FRAC)` to every interpreter in that venv as
  soon as torch is importable — no edit to any estimator-path script. VERIFIED ENFORCING, not
  assumed: a deliberate 60 GiB allocation is refused with 117 GiB physically free. Worst case is
  now 80.6 + 44.7 = 125.3 of 140.4 GiB. Cell restarted to pick it up; cost was ~10 min of init
  encode and no checkpoint. NOTE for other sites: a venv `sitecustomize.py` does NOT work here —
  `/usr/lib/python3.10/sitecustomize.py` shadows it (that dir precedes venv site-packages on
  sys.path); the `.pth` is not shadowed.
- Scar, and the README already warned about it: `pgrep -f train_rl.py | while read p; kill $p`
  killed the issuing shell, because the pipeline's own command line contains the pattern. That is
  the fourth time this pattern-kill has bitten this project. Explicit PIDs from a listing, always.
  (No harm: all 16 neighbour processes verified alive afterwards, and no checkpoint existed yet.)

### 2026-09-07: the BASELINE's decline does not survive the project's own CI rule (pre-registered before the EMA arm lands)
Ran least-squares + 95% CI — the house rule, applied to the comparator rather than the new arm —
on `v3-rows1M-s3000`'s four on-disk checkpoints (ck750 / ck1500 / ck2250 / final), matched
estimator throughout (retrieval n=3000, VQA n=1500, 7B reader, top-5).

| fit | entR@5 | ansR@5 | acc |
|---|---|---|---|
| slope, pts/1k steps | −0.991 | −0.680 | −0.382 |
| 95% CI | [−3.711, +1.729] | [−2.438, +1.078] | [−2.303, +1.539] |
| verdict | spans 0 | spans 0 | spans 0 |

**On the matched 3000-step run alone, NO trend is distinguishable on any of the three metrics.**
EMA_HANDOFF.md's framing — "ends at entR@5 72.63 … does index-EMA prevent the decline?" — is an
ENDPOINT comparison (ck750 75.53 → 72.63), and the project's own rule says a CI spanning zero is
not distinguishable. Caveat both ways: 4 points, dof=2, t=4.303 — this is low power, so it is
"not established", NOT "established flat".

Merging in the 500-step run's two points (v3-rows1M-ck250 77.00, v3-rows1M 78.70) gives 6 points
over 250–3000 and flips two of the three: entR@5 −1.738 [−3.359, −0.118] and ansR@5 −1.032
[−1.990, −0.074] both exclude zero; acc −0.140 [−0.888, +0.608] still spans it.
**The merge is UNVERIFIED**: v3-rows1M and v3-rows1M-s3000 are separate runs (2k vs 12k draws)
and merging assumes a shared trajectory — same seed 0, same 810k pool, constant LR, warmup 20,
refresh_steps 100, differing only in max_steps. Plausible but not checked; results/ carries no
args.json, and there is no overlapping evaluated checkpoint to test it (s3000 saves every 250
steps but only ck750/1500/2250 were ever evaluated, so no ck500 exists to compare against
v3-rows1M's final). Asked the prior session to diff the two args.json files.

**The load-bearing weakness, true under either fit:** the entire declining signal lives in the
250- and 500-step points, and the 500-step point is exactly the one this ledger already records
as having FAILED seed validation (2026-08-23 11:20 — seed 0 entR@5 78.70 vs seed 1 74.37). That
is a **4.33-point seed spread carrying a 6.07-point endpoint "decline"**. Under the merge the
slope is anchored by the least trustworthy point on the curve, and a seed-1 version of the same
fit would be markedly flatter.

**PRE-REGISTERED, before emaidx-1M-s3000 produces any number:** "index-EMA prevents the decline"
is NOT a claim this design can support, because the decline it would prevent is not itself
established at the matched estimator. What the cell CAN support is (i) matched-checkpoint
comparison at 750/1500/2250/3000 and (ii) a trend-vs-trend contrast, both 1 seed = recorded, not
claimed. Registering this now so the reading is not chosen after seeing which way the numbers go.

### 2026-09-07 (cont.): merge VERIFIED, and anchor-sensitivity splits the answer BY METRIC
The unverified merge above is now verified. `runs/v3-rows1M/args.json` vs
`runs/v3-rows1M-s3000/args.json` differ in exactly four fields — `max_steps` (500 vs 3000),
`save_steps` (250 vs 750), `device`, `output_dir`. Seed 0 both, `lr_schedule constant` both,
`warmup_steps 20` both (a fixed step count, NOT a fraction of max_steps), pool/corpus/batch_size/
refresh_steps/pl_support/n_distractor_articles identical, `index_ema` None both. So max_steps
never enters the optimisation and a shared seed makes `rng.choice(rows)` draw the identical
sequence: **v3-rows1M is a genuine PREFIX of the s3000 trajectory, not a separate run.** Only
caveat is different physical GPUs = fp nondeterminism, not an estimator difference.

There is also a seed-1 early curve on disk (`v3-rows1M-s1-ck250`, `v3-rows1M-s1`; seed 1,
otherwise matched, 500 steps), so "the slope is anchored by the seed-failed point" is testable
rather than rhetorical. Refit with the early anchor swapped (slopes pts/1k steps, 95% CI):

| variant (early anchor) | entR@5 | ansR@5 | acc |
|---|---|---|---|
| A — none (s3000 only) | −0.991 [−3.711, +1.729] spans 0 | −0.680 [−2.438, +1.078] spans 0 | −0.382 spans 0 |
| B — seed 0 | −1.738 [−3.359, −0.118] **DECLINE** | −1.032 [−1.990, −0.074] **DECLINE** | −0.140 spans 0 |
| C — seed 1 | −0.794 [−1.774, +0.185] spans 0 | −0.939 [−1.659, −0.220] **DECLINE** | −0.090 spans 0 |
| D — seed-averaged | −1.266 [−2.243, −0.290] **DECLINE** | −0.986 [−1.695, −0.277] **DECLINE** | −0.115 spans 0 |

(Independently recomputed here from results/ and matching the prior session's fits to 3 dp.)

- **ansR@5 is the ONLY metric whose decline survives every anchor choice** — CI excludes zero
  under B, C and D alike, slope stable at −0.94 to −1.03 pts/1k steps.
- **entR@5's decline is anchor-fragile**: present under seed-0 or seed-averaged anchors, gone
  under seed-1. Visible in the raw points — seed 0 RISES 77.00 → 78.70 between steps 250 and 500
  while seed 1 FALLS 75.47 → 74.37. The 78.70 is a seed-0 excursion, not a curve feature.
- **acc has no trend under any variant.**

**REGISTERED (before emaidx-1M-s3000 produces a number):** primary metric = **ansR@5**, the only
one where "does index-EMA prevent the decline" is a well-posed question. entR@5 and acc are
secondary and explicitly scoped — on entR@5 the baseline trend is not robustly established, and
on acc there is no trend to prevent.

**Detectability bound, also registered in advance.** Observed between-seed spread at matched
steps (2 seeds — a crude bound, NOT an SD):

| step | entR@5 | ansR@5 | acc |
|---|---|---|---|
| 250 | 1.53 | 0.69 | 0.00 |
| 500 | 4.33 | 1.51 | 0.33 |

So a single-seed emaidx-vs-baseline gap below roughly **4 pts entR@5 or 1.5 pts ansR@5** is
inside seed noise and CANNOT be called. Written down before the numbers exist, deliberately.
EMA_HANDOFF.md's "declines to entR@5 72.63" framing is being corrected at source by its author.

### 2026-09-07: throughput investigation on the shared box — cause is reader generation under time-slicing
The EMA arm ran far slower than the baseline's 6.64 s/step and the cause was chased before
committing to a long run. Baseline rate is solid: recovered from checkpoint mtimes as four
independent 750-step intervals (6.63 / 7.38 / 5.75 / 6.80 s/step, mean 6.64, total 11.77 h),
so the recipe is not at fault and the problem is site-specific.

Ruled out by measurement, not argument:
- **Memory cap: NOT the cause.** The live trainer sits at 27,808 MiB against its 46,006 MiB
  ceiling — 60%, stable across every sample rather than pinned the way a thrashing allocator
  would be. A standalone reader benchmark under the identical `MMRAG_MEM_FRAC=0.32` AND
  `expandable_segments:True` reports `num_alloc_retries=0`, `num_ooms=0`, reserved 10.66 GiB.
  The allocator never approaches the limit, so the slow free-and-retry path is never entered.
- **Attention backend: NOT the cause.** Reader reports `_attn_implementation=flash_attention_2`,
  dtype bf16. The flash-attn 2.8.3 wheel genuinely engages against torch 2.8.0 — no eager fallback.
- **Warmup: NOT the cause.** Rate is unchanged when steps 0-12 are excluded.

**Cause: reward-reader generation, punished ~10x by GPU time-slicing.** `score_judge` on this
box costs 14-18 s for 16 triples; the baseline's ENTIRE step was 6.64 s. Encoding is punished far
less — the init corpus encode came in at 2.9x (2145 s here vs 749 s clean). **Those two factors
are NOT the same number, and "the site is uniformly ~3x slow" was an over-smoothing of two
different measurements.** Mechanistically sensible: decode is a long tail of tiny launch-bound
kernels (worst case for context-switching), encoding is a few large matmuls.

**CORRECTION — a wrong line-number claim I made and retract.** I reported the per-step reader
load as 96 triples, citing `train_rl.py:531`. That line is in the **`else:` (non-plgrpo) branch**.
The plgrpo path is `train_rl.py:456-467`, which scores one triple per **UNIQUE** doc across the
sampled lists (`sorted(set(pl_idx[b].flatten().tolist()))`, with an `owner` map scattering
per-document rewards back onto lists). `pl_idx` is (B, G, k) = (4, 4, 4) from `pl_sample_lists`,
Gumbel-top-k with INDEPENDENT noise per list (rl_core.py:39-42), so unique docs per query is
4-16 and **len(uniq) is 16-64 triples/step, not 96**. My 16-triple benchmark measured the
degenerate floor. At temperature 0.02 the scaled-sim gaps within the top-24 are comparable to
the Gumbel noise scale (~1.28), so lists diverge substantially and the realistic count is toward
the upper end (~45), which closes the step-time accounting without a second unexplained slowdown.
Lesson: a line number makes a claim *look* verified; trace which BRANCH consumes the variable.

Consequence for the campaign: `save_steps 250` means interim checkpoints arrive long before the
run ends, and the registered comparison is checkpoint-matched anyway — so ck750 can be evaluated
against the baseline's own ck750 without waiting out the full run.

### 2026-09-07: STOPPING RULE for emaidx-1M-s3000, registered before checkpoint 1 exists
The contended rate makes the full 3000 steps expensive, so an interim look is likely. Writing
the rule down NOW so it cannot be reverse-engineered from the numbers later.

- **Evaluate ck750 and ck1500** as they land (`mmrag/scripts`-style launcher `run_eval.sh GPU
  <result-name> runs/emaidx-1M-s3000/checkpoint-N`), same estimator as the in-cell eval:
  retrieval `--max-q 3000`, VQA 7B top-5 `--max-q 1500`.
- **Continue to 3000 unless ansR@5 is null at BOTH ck750 and ck1500** beyond the registered
  ~1.5-point between-seed floor. Baseline matched rows: ck750 entR@5 75.53 / ansR@5 63.55 /
  acc 35.13; ck1500 73.40 / 62.04 / 33.87.
- **If stopped early it is reported as an INTERIM-LOOK STOP, with the truncation named.** A
  truncated curve must NOT be fitted and presented as though its endpoint were pre-specified —
  the trend CIs above assume a fixed 4-point design, and an optional-stopping curve does not
  satisfy that. Any early-stop trend is descriptive only.
- A null at a single checkpoint is **uninformative**, not evidence of no effect: |delta| below
  ~1.5 pts ansR@5 / ~4 pts entR@5 is inside the observed between-seed spread on one seed.

Eval-stage budgeting note: the ~7-8x generation penalty hits `eval_vqa` (1500 queries through the
7B reader — the most contention-sensitive stage in the pipeline), while `encode_corpus` /
`eval_retrieval` track the ~2.9x encoding factor. Record the two halves' walltimes SEPARATELY;
a single "eval time" number would average two very different contention regimes.

### 2026-09-07: eval-stage cost — corrected DOWN by ~an order of magnitude (attributed, not verified here)
An earlier warning in this campaign that `eval_vqa` would cost "hours" under contention was wrong
and is retracted by its author. Baseline VQA-stage durations, recovered on the ORIGINAL cluster
from the gap between each result's `retrieval.metrics.json` and `vqa_top5.json` mtimes:
ck750 4.3 min, ck1500 7.0 min, ck2250 8.2 min, final 4.6 min — i.e. **1500 queries through the
7B reader cost 4-8 min uncontended, not hours.**

Reason the instinct was wrong, and it is a genuine regime difference worth remembering:
`eval_vqa` generates **greedily with no rollouts**, so `score_judge`'s `bs = batch_size // 2`
sampling halving (reader_vlm.py:135) does NOT apply — it runs 1500/8 = 188 full batches of short
generations, which is efficient even on a 7B. Training's 4-rollout sampled judge is a completely
different regime, and the ~7-8x generation penalty measured there should not be assumed to
transfer at the same magnitude.

**PROVENANCE — cannot be verified at this site.** Our `results/*.json` were pulled from the HF
bundle today, so every local mtime is the download timestamp (all identical, 17:09:27) and
carries no timing information. These four durations rest entirely on the prior session's copies
on the original cluster; recorded as attributed, not independently confirmed here.
The source-side mtimes DO self-validate, which is worth stating because it is checkable evidence
rather than assertion: `eval_retrieval` writes `retrieval.json` and `retrieval.metrics.json`
back-to-back, then `eval_vqa` writes `vqa_top5.json` a stage later. On the original cluster the
first pair is separated by 0.0 s and the second by minutes, for all four checkpoints. A bulk copy
cannot produce that structure — it collapses everything to one instant, which is exactly OUR
symptom. So the pattern itself distinguishes genuine write times from a flattened copy.
**Two caveats that travel WITH the attribution:** (1) single observations, one per checkpoint, no
repeats; (2) the spread is nearly 2x (4.3-8.2 min) for nominally identical work — 1500 queries,
same reader, same k — and the cause (content-dependent generation length vs unknown worker load)
cannot be distinguished. Treat as "single-digit minutes on an H100, order-of-magnitude reliable",
NOT a tight baseline. Consequence for our measurement: any contention ratio we compute against it
must be reported as a RANGE carrying that 2x baseline variance, never as a point estimate.
`eval_vqa.py` has no timing instrumentation, so no independent log exists to cross-check.

Projection for this box: ~30-70 min for the VQA half at the measured generation penalty. No
number is offered for the retrieval half — it is corpus-encode-bound (`--bs 128` over 173,755
passages + 3000 query encodes) and should track the ~2.9x ENCODING factor rather than the
generation one, but that has never been measured. Recording the two halves separately when
ck750 is evaluated will produce the first measurement of eval-stage contention for this campaign.

### 2026-09-07: I asserted a mid-run pull hazard, then my own test refuted it — narrowed
I claimed a `git pull` mid-run could corrupt the shell executing `mlx_entry_cell.sh`, because
bash reads a script incrementally from a saved byte offset. Half of that is true and half is not.

Measured here (git 2.34.1, XFS, /raid):
- The live shell DOES hold an open descriptor to the script: `/proc/<pid>/fd/255 ->
  mmrag/mlx_entry_cell.sh`. The incremental-read half is real.
- **`git checkout <ref> -- <file>` REPLACES the inode** — 3/3 trials, never in place. `git merge`
  likewise. So git writes a temp file and renames.

Those two facts together give the OPPOSITE of my claim: because git replaces the inode and bash
holds the OLD one open, the old file survives as long as that shell holds it, and the pull would
NOT have corrupted the run. **Corrected rule:** a shell executing from an open fd is vulnerable
only to writers that modify a file IN PLACE (same inode) — an editor or a truncating redirect.
Git is not such a writer here, though this may vary by git version/filesystem, which is itself a
reason not to state a flat rule. Check it in two minutes with `stat -c %i` before believing either
version.

The decision to defer the .sh files was still correct, but for only ONE of the two reasons given:
**a merge would have clobbered this site's uncommitted path rewiring.** That reason was sufficient
alone; leading with an untested mechanism was the error.

Process note, the sharpest of the day: the prior session initially believed the safe (correct)
version, talked itself out of it because my confident mechanism disagreed, ran its own test and
got a result AGREEING with me — and only my test on this box refuted it. **Two independent checks
agreeing is not the same as being right**, especially when one party's confidence seeded the
other's interpretation. Six claims were retracted across the two sites today; this is the only one
where a belief both parties had already converged on was overturned by a two-minute check.

### 2026-09-07 (cont.): pull-hazard settled on BOTH sites, and the right way to test it
Confirmed here with the definitive method (hard-link pin + an fd held open across the checkout),
XFS, git 2.34.1: inode 14301683 -> 14301686 (CHANGED); disk content afterwards `echo v1`; content
read through the descriptor opened BEFORE the checkout still `echo v2`. The old inode stays alive
behind the held descriptor. The prior session reproduced the same on its bytenas mount, where NFS
additionally left a `.nfs000...` silly-rename artifact — the filesystem announcing that the old
inode survived because something held it open. **Both sites: git REPLACES, a running shell is safe.**

**The inference asymmetry that produced the wrong rule, worth more than the result.** The prior
session's first test compared `stat -c %i` before/after, saw an unchanged number, and concluded
"rewrote in place". That is invalid: git unlinks then creates, and the filesystem hands back the
inode number it just freed, so an UNCHANGED number cannot distinguish "same file" from "new file,
recycled number". A CHANGED number, however, IS conclusive — an in-place rewrite cannot change the
inode. So our two `stat`-only tests were not equally good: mine landed on the sound side of the
asymmetry by luck, not method. **Use the open-fd / hard-link-pin test**, which discriminates in
both directions; record that as the method.

**Process lesson (the sharpest of the day).** Sequence was: prior session assumed replace+rename
(correct) -> I asserted in-place with no test -> their invalid test agreed with me -> both sites
converged on the wrong rule and committed it. An invalid test that agrees with a confident
assertion is worth LESS than no test, because it launders a hunch into a documented finding.
Independence is not about who ran the check but about whether it could have come out the other way.
**And a false safety rule is not free**: it tells a future site to skip a safe operation — here,
the eval-timing instrumentation, exactly the kind of improvement that would then be declined.

### 2026-09-07: MEASURED throughput for emaidx-1M-s3000 on the contended H200
First clean window (no benchmarks of ours running; an earlier window was discarded because our
own reader/encoder benchmarks were competing for CPU during it):

| quantity | value |
|---|---|
| window | 24 steps in 900 s (steps 28 -> 52) |
| rate | **37.5 s/step** |
| vs baseline 6.64 s/step | **5.6x slower** |
| projected full 3000 steps | 31.2 h |
| ck750 reached at | ~7.3 h from step 52 |
| ck1500 reached at | ~15.1 h |
| one-time init corpus encode | 2145 s for 173,755 passages = 81 passages/s (vs 232/s clean = 2.9x) |

**SECOND WINDOW (steps 53->76, 23 steps in 900 s): 39.1 s/step, 5.9x.** So the site rate is
**37.5-39.1 s/step (5.6-5.9x baseline)**, full run **31.2-32.6 h**, ck750 at ~7.3 h from either
window's start. Two consecutive 15-min windows spanning ~30 min — i.e. roughly two of the
neighbour's 12-15 min rollout cycles — agreeing within 4%. **Quote the range, not either point.**

**CAVEATS, and they matter more than the point estimate.** (i) Two windows, 47 steps total —
enough to show the rate is stable across neighbour cycles, not enough to characterise a tail. (ii) The neighbour verl job cycles between ~19 GB (vLLM asleep) and ~82 GB (rollout) on a
12-15 min period, so a 15-min window covers roughly ONE cycle and may not average it; the phase
transition is visible in gpu_watch.log at 18:32:21, where every card drops ~62 GB at once.
(iii) There is NO uncontended GPU on this box — all 8 cards host the neighbour — so no clean
control can be produced here and every number carries the contention caveat.

Note the measured 37.5 s/step is well BELOW the 45-60 s/step I was informally reading off coarse
row counts earlier, which is why that estimate was never quoted. The 5.6x aggregate also sits
between the ~2.9x measured on encoding and the ~7-8x inferred for generation, consistent with a
step that mixes both — and is a further reason not to collapse them into one site factor.

### 2026-09-07: trained top-24 geometry is unrecoverable from ANY results bundle (code property)
Question raised while building the walltime cost model: what is the within-query spread of
sim(q, d_i) across a query's top-24 under the TRAINED policy? It drives Gumbel-top-k overlap and
therefore `len(uniq)`, the per-step reader load.

- **Not answerable from the existing geometry probes.** `geom_probe_q.py` computes *mean pairwise
  cosine over a fixed seeded 1,000-query sample* — BETWEEN-query anisotropy (how spread the query
  cloud is), a different object from WITHIN-query top-24 spread. The div45kv3 numbers (query side
  base 0.2212 -> v3pure 0.0594 / emaidx 0.0612 / emaenc 0.0976; doc side barely moves) do say
  training acts on the query tower, which makes "zero-shot geometry may not transfer" well founded
  rather than pro-forma — but they are a different statistic, from a 500-step 45k-pool cell, and
  must not stand in for the measurement.
- **Not answerable from any retrieval dump, at any site, ever.** `eval_retrieval.py:110` does
  `sims.topk(...).indices` — the `.values` are discarded at the topk call and never reach the
  JSON, which stores only `topk` pids, `gold_pids`, `ans_pids` and metadata.
  **METHOD NOTE:** the prior session established this by inspecting four dump files on its
  cluster; checking the WRITER instead establishes it for every site and every past run,
  permanently. *Checking the producer generalises where checking the product does not* — worth
  reaching for whenever a claim is about what data exists rather than what a particular file holds.
- **Cheap path taken instead:** `bench_uniq.py` now reads its encoder checkpoint from `$CKPT`
  (default `__profile__`), so the identical script rerun against ck750 yields the trained geometry
  for one encode on our own lane — turning the caveat into a measured before/after.
- **DEFERRED, deliberately:** storing a few per-query top-k score summary statistics in the dump
  would make this answerable from any results bundle with no re-encoding. NOT changing the dump
  format mid-campaign — it would desynchronise our ck750 eval from the baseline dumps for a
  benefit already obtainable via `CKPT`. Post-campaign change, if ever.

### 2026-09-07 19:09-19:10: RUN KILLED — whole-box takeover, not OOM. emaidx-1M-s3000 lost at step 113
`train_rl.py` (pid 3496548) SIGKILLed at 19:10:31, step ~113 of 3000. No traceback, no CUDA error —
the shell recorded "Killed". A concurrent one-off benchmark of ours on GPU 3 died at the same moment.

**Cause: a whole-box vLLM serving deployment displaced every tenant.** Eight
`vllm.entrypoints.openai.api_server` processes started 19:09:20-19:10:13, ports 31020-31027, one
pinned per card via CUDA_VISIBLE_DEVICES, `--gpu-memory-utilization 0.85`, serving
`/raid/icy/tmp/gpt54emrl_v17/keep_step8`. The 8-GPU verl job that had been the resident tenant all
day is entirely gone (0 `ray::WorkerDict` remain), as are the h200-rag session's two trainers
(SIGKILLed 19:09:44, 47 s before ours). **Three unrelated workloads dying inside one minute while a
new claim on all eight cards appears is a takeover, not an OOM** — an OOM killer takes one process
at a time by size. Host RAM was and is 1.9 TB of 2 TB available; the 6.3/8 GB swap reading is a red
herring. Not a pattern-kill either: h200-rag issued none, and our own guard was untouched.

**Cannot restart into current conditions.** Verified independently: all 8 cards at 120.1 GiB used /
20.3 GiB free of 140.4. Our trainer's measured working set is 27.8 GiB, so it does not fit on ANY
card. `--mem_frac` cannot help — it caps, it does not shrink.

**Cost of the kill: total for this attempt.** `train_rl.py` has NO resume path (no `--resume`, no
`start_step`; the entry script's only idempotency is "skip training if the FINAL adapter exists").
save_steps 250 and we reached 113, so no checkpoint was written. The loss is the ~36 min init
corpus encode plus ~70 min of stepping. Restarting is scientifically clean — the run is seeded, so
a fresh start reproduces the identical row sequence — it is purely a time cost.
Written but NOT launched: `supervise_cell.sh`, which relaunches the idempotent entry script across
external kills so a future kill costs minutes of idle GPU rather than hours until someone notices.

**Self-inflicted lesson, separate from the takeover.** Earlier, patching `auto_eval.sh` with
Python's `open(p,"w")` truncated the file IN PLACE while bash was executing it and holding fd 255
at byte offset 1275. That is precisely the hazard this ledger recorded as FALSE for git — because
git renames (new inode) while a truncating writer does not. Proving one writer safe does not make
writers-in-general safe. The watcher was restarted; it had done no work.

### 2026-09-07: plgrpo per-step cost model — MEASURED (partial; killed before the reader trials)
From `mmrag/../bench_uniq.py` on GPU 3, zero-shot gme2b geometry, real queries/passages.
The 120k-passage encode finished; the reader trials were lost to the takeover above.

| quantity | measured |
|---|---|
| corpus encode, 120k passages | 1211 s = **99.1 passages/s** contended (232/s clean = 2.3x here) |
| real passage length | **560 chars** mean (our earlier filler was ~1440, i.e. 2.6x too long) |
| doc encode 96 (pl_support 24 x batch 4) | **0.71 s** |
| query encode 4 (with images) | **0.40 s** |
| EMA write-back, 256 docs | **1.85 s** |
| **len(uniq)** | **~33 triples/step -> 5 generate calls** |

**Subsample-bias question settled, and it was never load-bearing.** Same embeddings, growing index:
40k -> 8.43 unique/query, 80k -> 8.41, 120k -> 8.19. Direction is DOWN (the sign I originally
guessed) but the magnitude is ~3% across a 3x index change, i.e. negligible. Neither side's
argument mattered — which is itself the result, and cheaper to obtain than to debate.

**len(uniq) ~33 falls in the prior session's AMBIGUOUS band (30-40)**, so the pre-registered
scoring rule says score against the RECONSTRUCTION, not the band.
Reconstruction by subtraction, since the direct reader trials were lost: non-reader cost is
0.71 + 0.40 + 1.85 = **2.96 s**, against a measured step of 37.5-39.1 s, leaving **~34.5-36.1 s for
the reader over 5 calls = ~6.9-7.2 s per call.** That is at or barely below the 7.2-9.1 s/call
measured with the 2.6x-too-long filler context — i.e. **cutting prefill 2.6x bought almost
nothing, which points at DECODE, not prefill, as the dominant cost** (24 sequential decode steps
are unaffected by context length). If it holds, that REFUTES the prior session's firm prediction
of a clear per-call drop. **NOT SCORED**: this is inference by subtraction, not the direct
measurement, and the direct trials must be rerun before the prediction is called either way.

**STRONGER REASON NOT TO SCORE IT (prior session's, and it is decisive).** The subtraction mixes
TWO CONTENTION REGIMES: the 2.96 s non-reader terms were measured on GPU 3 in a standalone
process, while the 37.5-39.1 s step came from the live run on GPU 1 at a different time under a
different co-tenant. Because contention here is ~2.9x on encoding but ~7-8x on generation, the
mismatch does NOT cancel — it lands entirely inside the residual being attributed to the reader.
So ~6.9-7.2 s/call carries both measurement errors plus an unknown cross-card contention delta and
is **uninterpretable**, independently of it being subtraction rather than a direct trial. Rerun the
direct trials on the same card; do not let the subtraction stand in for them.

**PREDICTION SCORED: MISS (self-scored by its author).** Predicted len(uniq) 22-27, measured 33 —
the AMBIGUOUS band. But since the registered rule was "score against the reconstruction, not the
band", the verdict is that the decomposition FAILED, and the failure is diagnosable: it assumed
~13.1 s of non-reader cost (3.2 s EMA + 9.9 s encoding) against a measured 2.96 s, i.e. **4.4x too
high**. Two errors, both from deriving rather than measuring:
1. The EMA write cost was computed from the INIT-encode rate (81 passages/s). Steady-state
   write-back is ~138/s (256 docs in 1.85 s) — a cold full encode is not a valid proxy for warm
   incremental writes.
2. The baseline's 6.64 s/step was split into reader vs encoder by inference, then the encoder half
   scaled by 2.9x — two guesses stacked on each other.
**A wrong band is information about the model; a right band would have been luck.** Recorded
because the same author had stated that rule and then failed to apply it to their own estimate.

**WHAT IS SOLID AND TRANSFERS** (sampler-and-geometry, not contention): ~8.2 unique docs/query ->
~33 triples/step, near-invariant to index size (8.43 / 8.41 / 8.19 at 40k / 80k / 120k).

### 2026-09-07 21:00: relaunched after the takeover — and the rate improved because the box emptied
Restarted on GPU 1 under `supervise_cell.sh` (relaunches across external kills; the entry script is
idempotent and the run is seeded, so a restart reproduces the identical row sequence). Cap tightened
0.32 -> **0.25 (~35 GiB)**, still clearing the 27.8 GiB working set while leaving the co-tenant room.
Identical startup checks passed: `train rows: 810276`, `RL corpus: 173755 passages`, init encode
2185 s (vs 2145 s on the killed attempt), **coverage 6.35x FULL** — the registered stop-guard.

**New rate, third clean window: 30.0 s/step (30 steps / 900 s), 4.5x baseline.** Down from the
37.5-39.1 s/step measured before the takeover, and the reason is visible rather than mysterious:
GPU 1 now shows **114.3 GiB free of 140.4**, against ~20-60 GiB free under the departed verl job.
Less SM time-slicing, less penalty — which is the mix-shift point in reverse and further evidence
that a single "site factor" is not a stable quantity here.
Revised ETA from step 31: **ck750 ~6.0 h, ck1500 ~12.2 h, full run 25.0 h.**
CAVEAT: one window under conditions only minutes old. The prior figure was quoted as a range from
two windows; this is a single observation and should be treated as provisional until a second one
agrees. Do NOT retro-apply 30.0 s/step to the killed attempt — that ran under a different tenant.

### 2026-09-08 00:55 (step ~460): reward signal is ERODING — flagged early, registered falsifiably
Trend on binned means, **incomplete final bin dropped** (10 rows of 75 — a partial bin must not
anchor a slope; that is the same error as the seed-0 excursion anchoring the entR@5 decline):

| signal | bin means (75-step bins, steps 0-449) | slope/1k | 95% CI | verdict |
|---|---|---|---|---|
| judge reward | .190 .215 .197 .203 .200 .195 | −0.006 | [−0.092, +0.080] | flat |
| gold in pool | .750 .763 .723 .767 .750 .753 | +0.008 | [−0.143, +0.159] | flat |
| **group std** | .0478 .0427 .0456 .0386 .0361 .0334 | **−0.038** | **[−0.057, −0.019]** | **FALLING** |
| **degenerate frac** | .540 .570 .553 .583 .607 .640 | **+0.244** | **[+0.115, +0.373]** | **RISING** |

**Reading: not collapse, not improvement — the LEARNING SIGNAL is thinning.** group_std is the
within-group spread of judge reward; when it hits zero the advantage is zero and the step teaches
nothing. Degenerate fraction is the share of groups already in that state: **0.54 -> 0.64**, and
group_std==0 in **16/100 of the last steps** against 42/460 all-time. Two independent statistics
agreeing, both CIs excluding zero, both roughly monotone across six bins.

Meanwhile reward and gold-in-pool are **flat** — so relative to base this run is **not improving on
its own train-side proxies**, it is holding level while the gradient signal erodes. This is the
early form of the failure that killed the VLM2Vec-4B cells (group_std 0 at step 249), differing
only in degree: those hit 100%, this is at 16% and climbing.

**DO NOT over-read the extrapolation.** A naive linear fit puts group_std at zero near step 1300,
but a quantity bounded below by zero decays rather than crossing it — expect an asymptote, not a
cliff. The extrapolation is not a prediction and is recorded only to show the direction is
material within this run's remaining budget.

**REGISTERED THRESHOLDS, before ck750 exists** (so the reading cannot be chosen after the fact):
- **Signal effectively dead** if, at any 75-step complete bin, degenerate frac >= 0.85 AND
  group_std <= 0.015. Report as a signal-death stop, not as an EMA result.
- **Erosion confirmed but survivable** if both trends keep their sign at ck750 and ck1500 while
  entR@5/ansR@5 stay within the registered seed floor of the baseline.
- **Erosion was noise** if either CI crosses zero once 10+ complete bins exist (step ~750).
Note this is a property of the RECIPE under a thinning-index regime, and cannot be attributed to
index-EMA without the matched baseline arm — v3-rows1M-s3000's own metrics.jsonl is not on this
site, so the comparison is currently impossible here. Requested from the origin cluster.

Pace note: now **20.0 s/step** (GPU 1 has 113.9 GiB free) — ck750 in ~1.6 h, full run ~14 h.

### 2026-09-08 01:11: LOCAL CONTROL ARM launched — `v3pure-1M-s3000-loc` (GPU 3)
The erosion finding above is **uninterpretable without a matched arm**: group_std falling and
degenerate-fraction rising may simply be what v3-pure does on the 810k pool at 3000 steps, in
which case it is a recipe property and a null for index-EMA. The origin cluster's
`runs/v3-rows1M-s3000/metrics.jsonl` is NOT in the published Class-1 bundle (results/*.json only),
so the comparison cannot be made from anything on this site. Request sent; the peer is offline.

Launched the comparator locally instead. Args are **emaidx-1M-s3000's minus exactly
`--index_ema 0.9 --index_refresh_extra 256`** — everything else identical, seed 0, same pool,
same corpus, same 3000 steps, `--mem_frac 0.25`. Distinct name so it can never overwrite the
published `v3-rows1M-s3000` rows.

Justification for the scope addition: all eight cards were idle (GPUs 3/5/7 at 140.4 GiB free)
after the takeover ended, so this displaces nothing; it is reversible; and it converts the primary
observation from uninterpretable to interpretable. It also yields a **same-hardware** comparator,
which is strictly better than the cross-site comparison — it removes the fp-nondeterminism and
different-GPU caveats that the published baseline carries.

What it decides:
- Baseline erodes at the same rate -> erosion is a RECIPE property, null for EMA.
- Baseline holds its spread while the EMA arm thins -> a genuine EMA effect, and a mechanistically
  sensible one: a deliberately stale index makes a query's top-24 more homogeneous over time,
  which is exactly what would flatten within-group reward.
Watch: two concurrent cells share this box's CPU (image decode is CPU-bound). If cell 1's rate
degrades materially from its current 20.0 s/step, the control gets killed — cell 1 has priority.

### 2026-09-08 03:52: FIRST RESULT — emaidx-1M-s3000 ck750 vs baseline ck750
Estimator-matched (retrieval n=3000, n_ans=2450, corpus_small, same eval script and flags).

| metric | EMA arm ck750 | baseline ck750 | delta | registered seed floor | callable? |
|---|---|---|---|---|---|
| **ansR@5 (PRIMARY)** | **65.84** | 63.55 | **+2.29** | 1.51 | **YES** |
| entR@5 (secondary) | 77.23 | 75.53 | +1.70 | 4.33 | no — inside noise |
| entR@1 / entR@10 | 59.53 / 83.13 | — | — | — | (baseline not on disk at these k) |
| **acc (secondary)** | **34.47** | 35.13 | **−0.67** | 0.33 | **YES — and it is NEGATIVE** |

**ck750 COMPLETE, and the two callable metrics point OPPOSITE ways.** ansR@5 +2.29 (callable,
positive) against acc −0.67 (callable, negative; strictEM 0.2573, F1 0.3472, n=1500; by split
unseen-entity 0.3396 / unseen-question 0.3594). This is the campaign's own documented
anti-diagnostic decoupling — "acc and R@5 routinely move in opposite directions" — appearing again
at a matched checkpoint, and it is exactly why the house rule demands the metric be named in every
claim. **"emaidx beats the baseline at ck750" is not a sentence anyone may write.** It beats it on
answer-recall and loses on accuracy, both outside their respective seed floors, at one seed each.
Note also the acc floor (0.33) is by far the tightest of the three, being derived from a
two-seed spread of 0.00 and 0.33 — so "callable" here rests on the weakest floor of the set.

**The primary metric clears its pre-registered noise floor.** ansR@5 +2.29 against a floor of 1.51
that was written down BEFORE any EMA number existed. entR@5 moves the same direction but is well
inside its (much wider) 4.33 floor, exactly as the registration anticipated — entR@5 was scoped
secondary precisely because its baseline trend was anchor-fragile.

**WHAT THIS IS NOT.** One seed and ONE checkpoint. The registered reading is curve-vs-curve at
750/1500/2250/3000; a single point is not a curve and cannot show whether EMA changes the SHAPE,
which was the actual question. Do not quote +2.29 as the effect. The baseline's own ck750 -> 3000
ansR@5 trend is −0.680 pts/1k [−2.438, +1.078], i.e. not distinguishable from flat, so "prevents
the decline" still has no established decline to prevent at the matched estimator.
Cross-site caveat stands: this ck750 was produced on H200s here, the baseline's on the origin
cluster. That is what `v3pure-1M-s3000-loc` (running, step ~120) exists to remove.

### Eval-stage contention split — first measurement in this campaign
| stage | measured | note |
|---|---|---|
| `encode_corpus.py` (422,378 passages) | **3165 s = 133 passages/s** | vs 232/s clean = **1.75x** |
| `eval_retrieval.py` query search (3000 q) | **304 s** | |
| `corpus_seconds` in the metrics JSON | **1.5 s** | **cache LOAD, not the encode** |
| `eval_vqa.py` 7B reader | running | |

The `corpus_seconds` trap is real and confirmed: `mlx_entry_eval.sh` runs `encode_corpus.py --cache`
first, so the JSON field records a 1.5 s cache load. **Reporting it as the corpus encode cost would
have given a ~2000x wrong number and a nonsense contention ratio.** Read the encode cost from
`encode_corpus.py`'s DONE line. Note 1.75x here vs 2.9x measured earlier — the box emptied, so the
site factor is not stable in time either.

### 2026-09-08: WHY THE RECIPE STOPS IMPROVING — the reward is STARVED, and it is reader-limited
User's actual target restated: *a dynamic recipe significantly better than relevance-SFT that keeps
improving within 1 epoch.* Scored against that, from data on this site.

**(1) Level vs relevance-SFT — RL wins, ~+3 to +4 ansR@5 (1 seed each, recorded not claimed).**
Best SFT ansR@5 63.27 (`sft-curve-lr1e4-s1-ck1050`); the `sft-big-1M` scaling line averages ~61.6.
RL gme2b arms reach 65-66.5; `emaidx-1M-s3000-ck750` is 65.84; gme7b-v3pure 69.02.

**(2) Scaling — NEITHER recipe satisfies "keeps improving". This is the unmet requirement.**
`sft-big-1M`, 11 checkpoints over steps 2100-27300, least squares per 10k steps:
entR@5 +0.131 [−0.565, +0.827] FLAT · ansR@5 +0.399 [−0.517, +1.315] FLAT ·
acc +0.427 [+0.032, +0.822] rising, but only ~1.2 acc points across the whole line.
RL `v3-rows1M-s3000` ck750->3000: all three flat-to-declining, every CI spanning zero.
**Scale context that reframes the goal: 810k rows at batch 4 is ~202,500 steps = 1 epoch. Every
run on record is <=3000 steps = 12k draws = 1.5% OF AN EPOCH.** Turnover is observed at ~2-3k
draws, i.e. at ~1% of an epoch. So "keeps improving through 1 epoch" is not merely untested — the
recipe currently stops improving two orders of magnitude before that point.

**(3) MECHANISM (new, measured on emaidx-1M-s3000, 924 steps): the reward is STARVED, not saturated.**
| condition | n steps | degenerate frac | mean reward |
|---|---|---|---|
| gold in pool for all 4 queries | 294 | **0.532** | 0.255 |
| gold in pool for 3 of 4 | 379 | 0.613 | 0.200 |
| gold in pool for 2 of 4 | 215 | 0.666 | 0.155 |
| gold in pool for 1 of 4 | 35 | 0.700 | 0.097 |
Degeneracy-heavy steps (>=0.75) have **LOW** reward (median 0.125; **0% of them exceed 0.9**), not
high — so degenerate groups are all-WRONG, not all-RIGHT. **Even with gold retrieved for every
query, 53% of groups still carry zero advantage.** The 3B reward reader converts a correct
retrieval into a correct answer only ~20-25% of the time; when it fails on all 4 sampled lists the
rewards tie, the advantage is exactly zero, and the step teaches nothing.
Erosion over the run (CIs excluding zero): degenerate 0.586 -> 0.621, group_std 0.0403 -> 0.0376,
while gold_in_pool (0.748 -> 0.753) and reward (0.200 -> 0.205) stay FLAT.

**(4) THE CONSEQUENCE THAT MATTERS: more data cannot fix this.** The bottleneck sits DOWNSTREAM of
retrieval, in the reader's ability to discriminate. That predicts exactly what the campaign already
observed and could not explain: 25k / 200k / 810k pools all plateau in the same place; acc and R@5
peak at different draws; and the SFT rows-line is flat too. A recipe that "keeps improving" must
first stop wasting ~60% of its steps on zero-advantage groups.

**(5) LEVERS, all present in the code and NONE tested at scale in this campaign.**
- `--reward logit` — mean logP(answer | image, question, context), continuous and ALWAYS defined,
  so ties are essentially impossible and degeneracy should collapse toward zero. **It is
  train_rl.py's own default; the campaign fixed on `judge` throughout.** Highest leverage.
- `--reward gain` — judge(ctx) − judge(no ctx); removes reward on queries answerable without retrieval.
- `--reward_gate_noctx` — per-query DATA FILTER keeping only queries whose best candidate beats the
  no-context baseline. Tried once: `gate-gme2b-nogold` 79.70/64.90/33.67 vs `nogate-gme2b-nogold`
  79.40/63.96/33.33 → +0.94 ansR@5, +0.30 entR@5, +0.34 acc (1 seed, small, right direction).
- `--reward mix` — z-scored judge+logit. `--reward_gate_std` — zeroes no-signal pools.

### Eval-stage walltime, complete (first full measurement under contention)
| stage | measured | vs clean | note |
|---|---|---|---|
| `encode_corpus.py`, 422,378 passages | 3165 s (133/s) | 232/s clean = **1.75x** | read from the DONE line, NOT `corpus_seconds` |
| `eval_retrieval.py` query search, 3000 q | 304 s | — | |
| `eval_vqa.py` 7B reader, 1500 q | 824 s = 13.7 min | 4.3-8.2 min = **1.7-3.2x** | greedy, no rollouts |
| `corpus_seconds` field in the metrics JSON | 1.5 s | — | **cache LOAD — reporting it as encode cost is a ~2000x error** |
The VQA ratio is given as a RANGE because the baseline figure is four single observations spanning
nearly 2x; a point estimate would fake a precision the comparator does not have.

### 2026-09-08 04:32: ATTRIBUTION ANSWERED — the erosion is a RECIPE property, NOT an EMA effect
`v3pure-1M-s3000-loc` (identical args minus the two EMA flags, same seed, same hardware) reached
step 300. Matched over the first 300 steps of each arm:

| signal | EMA arm | control | delta |
|---|---|---|---|
| degenerate frac | 0.5617 | 0.5717 | −0.010 |
| group_std | 0.0437 | 0.0423 | +0.0014 |
| reward raw mean | 0.2012 | 0.2018 | −0.0006 |
| gold_in_pool | 0.7508 | 0.7075 | **+0.0433** |

**Three of four are indistinguishable.** The reward starvation — ~56-57% zero-advantage groups —
is present in the control at the same level, so it is a property of the v3-pure recipe under this
reward, **not** something index-EMA causes. This resolves the branch registered at 00:55 in favour
of "recipe property, null for EMA", and it means the erosion finding generalises beyond this arm:
any cell using `--reward judge` with a 3B reader on this pool is wasting ~60% of its steps.

**The one real difference is gold_in_pool, +0.043 in the EMA arm** — the EMA index puts the gold
passage into the top-24 more often. That is a coherent mechanism for the ck750 ansR@5 advantage
(+2.29) and is what index-EMA would be expected to do if it helps at all. 1 seed, 300 steps,
recorded not claimed; it needs the matched ck750 from the control before it is worth anything.

CAVEAT: 300 steps of overlap, one seed per arm, and the control is 700 steps behind. The
degeneracy comparison is the robust part (three statistics agreeing to within 1-2%); the
gold_in_pool gap is a single number and could still be noise.

### 2026-09-08 09:19: ck1500 — the two metrics separate in a way that speaks to the original question
| | ck750 | ck1500 | change |
|---|---|---|---|
| EMA entR@5 | 77.23 | 76.83 | **−0.40** |
| base entR@5 | 75.53 | 73.40 | **−2.13** |
| EMA ansR@5 | 65.84 | 64.29 | −1.55 |
| base ansR@5 | 63.55 | 62.04 | −1.51 |

**On entR@5 the EMA arm declines 1.73 points LESS than the baseline over the same interval; on
ansR@5 the two slopes are indistinguishable (−1.55 vs −1.51).** The advantage therefore behaves
differently by metric: ansR@5 is a near-CONSTANT OFFSET (+2.29 -> +2.24), while entR@5 WIDENS
(+1.70 -> +3.43) purely because the baseline falls faster.

Read against the original framing — "does index-EMA prevent the decline?" — this is the first
evidence that the answer may be **metric-specific and the opposite way round from the
registration**: the decline it appears to slow is **entR@5**, the metric scoped SECONDARY because
its baseline trend was anchor-fragile; on **ansR@5**, the registered PRIMARY and the only metric
whose baseline decline survived every seed anchor, EMA shifts the level but does **not** change the
slope. A recipe that raises the level without changing the slope does not "keep improving" — it
plateaus in parallel, one step higher.

**NOT CALLABLE, and the reasons are specific.** (i) The entR@5 deltas (+1.70, +3.43) are both
inside their 4.33 seed floor, so the very comparison carrying this reading is the one the seed
bound cannot resolve. (ii) Two points cannot be fitted with a CI (needs >=3), so "declines less" is
an arithmetic difference between two pairs, not a trend test. (iii) One seed per arm.
**REGISTERED to check at ck2250 and ck3000, before those numbers exist:** if the entR@5 gap keeps
widening while the ansR@5 gap stays flat near +2.2, the metric-specific reading holds; if the
ansR@5 gap also widens, EMA is changing the slope and the registration's primary/secondary scoping
was wrong; if the entR@5 gap collapses back, +3.43 was seed noise as the floor warns.

### 2026-09-08 09:40: ck1500 acc RETRACTS the ck750 accuracy verdict
ck1500 acc 34.13 vs baseline 33.87 = **+0.27**. Against ck750's **−0.67**, the accuracy delta
**flips sign between consecutive checkpoints**, and the two-point mean is −0.20, back inside the
floor and not callable.

**Retraction:** the ck750 line "acc −0.67, callable" should not be read as an accuracy cost of
index-EMA. A quantity that reverses sign between adjacent matched checkpoints is behaving like
noise, and the flip is direct evidence that the **0.33 acc floor is too tight** — it was derived
from a two-seed spread of 0.00 and 0.33 at steps 250/500, the smallest and least trustworthy of
the three floors. The honest current statement on accuracy is **no distinguishable difference**.

Standing after two matched points (1 seed each, recorded not claimed):

| metric | ck750 delta | ck1500 delta | mean | verdict |
|---|---|---|---|---|
| ansR@5 (PRIMARY) | +2.29 | +2.24 | **+2.27** | callable at both, and REMARKABLY stable |
| entR@5 | +1.70 | +3.43 | +2.57 | inside its 4.33 floor at both |
| acc | −0.67 | +0.27 | −0.20 | **sign flip -> noise** |

The ansR@5 offset is the one durable signal: two independent matched checkpoints, +2.29 and +2.24,
a spread of 0.05 against a floor of 1.51. That stability is itself evidence — a noise-driven gap
would not reproduce to within 0.05 across 750 steps of training.
**But stability of the OFFSET is precisely what says the slope is unchanged**, i.e. EMA lifts the
level and does not alter the trajectory. Against the user's target (a recipe that KEEPS improving)
a constant offset is the wrong shape, however real it is.

### 2026-09-08: SEED FLOORS RE-DERIVED from every multi-seed family on disk (supersedes the 2-seed bound)
The floors used all campaign were taken from ONE two-seed pair (v3-rows1M s0/s1 at steps 250/500).
Recomputed as 2x the pooled between-seed SD over every family with >=2 matched seeds:

| family | n seeds | entR@5 | ansR@5 | acc |
|---|---|---|---|---|
| plgrpo-gme2b-nogold (v2, +anchor) | 4 | 79.78 ± 0.77 | 64.48 ± 1.96 | 33.60 ± 0.70 |
| scale-rows25k | 3 | 78.98 ± 0.49 | 65.37 ± 0.56 | 34.62 ± 0.23 |
| v3-rows1M @2k draws | 2 | 76.53 ± 3.06 | 64.47 ± 1.07 | 34.57 ± 0.24 |
| plgrpo-t005 | 2 | 79.10 ± 0.47 | 63.92 ± 0.23 | 33.23 ± 0.05 |
| **pooled SD** | | **1.20** | **0.96** | **0.30** |

**New floors (2 SD): entR@5 2.40 · ansR@5 1.91 · acc 0.61** — replacing 4.33 / 1.51 / 0.33.
The old bound was wrong in BOTH directions: it overstated entR@5 by 1.8x (its 3.06 SD was the
seed-0 excursion, a single anomalous run standing in for a variance estimate) and understated acc
by 2x, which is exactly why the ck750 acc verdict flipped sign at ck1500. `compare_ema.py` updated.

**Effect on the standing result** (2 matched checkpoints, 1 seed each):
| metric | mean delta | old verdict | NEW verdict |
|---|---|---|---|
| ansR@5 (primary) | +2.27 | callable | callable (floor 1.91) |
| entR@5 | +2.57 | not callable | **now callable** (floor 2.40) |
| acc | −0.20 | not callable | not callable (floor 0.61) |
Note the ck750 acc −0.67 sits within rounding of the new 0.61 floor, so it was never the clean
"callable negative" I first reported; the retraction was right for a second, independent reason.

### 2026-09-08 14:05: ck2250 RESOLVES the registered 3-way check — the entR@5 widening was noise
| gap (EMA − baseline) | ck750 | ck1500 | ck2250 |
|---|---|---|---|
| entR@5 | +1.70 | +3.43 | **+1.67** |
| ansR@5 | +2.29 | +2.24 | +1.84 |

Registered at ck1500 (before these numbers existed): *"if the entR@5 gap collapses back, +3.43 was
seed noise as the floor warns."* **That is the branch that fired.** The gap returned to its ck750
value, so the ck1500 reading "EMA slows the entR@5 decline by 1.73 points" is **RETRACTED** — it
was one checkpoint's excursion, and the re-derived 2.40 floor covers all three points.

**Trend vs trend (3 points, dof=1, so t=12.706 and the CIs are near-useless by construction):**
entR@5 EMA −0.600 [−1.089, −0.111] vs baseline −0.578 [−17.206, +16.050]; ansR@5 EMA −0.789 vs
baseline −0.490, both spanning zero. The point estimates are nearly identical between arms
(−0.60 vs −0.58 on entR@5), which is the same conclusion the constant offset already implied:
**EMA moves the level, not the slope.** Do not quote these CIs — with one degree of freedom a
3-point fit cannot distinguish anything; ck3000 makes it dof=2, still weak.

**Standing after 3 matched checkpoints (1 seed each):**
| metric | mean delta | floor | verdict |
|---|---|---|---|
| ansR@5 (primary) | **+2.12** | 1.91 | callable — but ck2250 alone (+1.84) is BELOW the floor |
| entR@5 | +2.27 | 2.40 | not callable |
| acc | −0.20 | 0.61 | not callable |

The honest summary of the whole EMA-on-corpus_small line: **one metric, answer-recall, sits about
2 points above the baseline as a constant offset, with no evidence of a slope change.** Against a
target of "keeps improving", a level shift with an unchanged (negative) slope is the wrong shape.
This is the arm the mix2M grid now re-tests on a diverse index and, for the first time, on the
ANCHORED recipe.

### 2026-09-08 16:34: SAME-HARDWARE CONTROL LANDS — "EMA is better" does NOT survive it
`v3pure-1M-s3000-loc-ck750` = the EMA arm's args minus exactly `--index_ema 0.9
--index_refresh_extra 256`, same seed, same pool, same corpus, **same GPUs, same week**.

| ck750 | entR@5 | ansR@5 | acc |
|---|---|---|---|
| emaidx-1M-s3000 | 77.23 | 65.84 | 34.47 |
| **control, same hardware** | **75.63** | **64.78** | **33.87** |
| published baseline, origin cluster | 75.53 | 63.55 | 35.13 |

| delta | vs SAME-HARDWARE control | vs published baseline |
|---|---|---|
| entR@5 (floor 2.40) | +1.60 — not callable | +1.70 — not callable |
| **ansR@5 (floor 1.91)** | **+1.06 — NOT CALLABLE** | +2.29 — callable |
| acc (floor 0.61) | +0.60 — not callable | −0.67 — callable |

**The primary metric's advantage halves against the clean comparator and drops below the floor.**
+2.29 becomes +1.06. So roughly half of what was being read as an EMA effect was cross-site
difference — different GPUs, different week, fp nondeterminism — not the treatment.
**On the estimator-matched, hardware-matched comparison, NOTHING is callable at ck750.**

This retracts the standing headline. The earlier entry "ansR@5 +2.27 mean over 3 checkpoints,
callable" was measured against the published baseline only; its same-hardware counterpart exists
at ck750 alone and does not reproduce the margin. The correct statement is:
**at ck750, index-EMA is not distinguishable from no index-EMA on any of the three metrics.**

Note the acc column also flips sign between comparators (+0.60 vs −0.67), which is a second,
independent demonstration that the cross-site gap is the same size as the effect being measured —
and is exactly why the control was launched. Cost of having launched it: one GPU for ~4 h.
Cost of NOT having launched it: a published +2.29 that is roughly half artifact.

The control's training was stopped at step 1000 (the mix2M grid supersedes it), so there is no
matched ck1500/2250/3000. The mix2M grid is built the right way round — all four cells are
same-hardware, same-week, matched pairs by construction.

### 2026-09-08 18:50: the memory cap DID bite — but only on the in-cell eval path (encode --bs 192)
`emaidx-1M-s3000` finished 3000 steps, then its in-cell final eval died:
`torch.OutOfMemoryError: Tried to allocate 25.43 GiB ... 125.98 GiB is free ... 34.95 GiB allowed`.
The card had **126 GiB free**; our own `MMRAG_MEM_FRAC=0.25` ceiling (34.95 GiB of 139.80) refused
the allocation. The 25.43 GiB request is the `lm_head` projection over a batch of long doc
sequences against a 152k vocab.

**Why it never showed before.** The three interim points (ck750/1500/2250) all ran through
`mlx_entry_eval.sh`, which encodes at `--bs 128`; the in-cell final eval in `mlx_entry_cell.sh`
uses `--bs 192`. Same cap, 1.5x the batch, and only the larger one crosses it. So the earlier
conclusion "the cap is not the problem" was right *for the path it was tested on* and wrong as a
general statement — a reminder that a negative result inherits the scope of the configuration it
was measured in.

**Fix, and why this one rather than raising the cap.** Re-ran the final point through
`run_eval.sh` (the `--bs 128` path). That is not a workaround but the *more* consistent choice:
the other three curve points were produced by exactly that path, and `encode_bs` is a free knob
(pure forward passes, no sampling, no RNG), so the estimator is unchanged. Raising the cap would
have made the final point the only one encoded at a different batch size.

**Also stopped the cell supervisor**: `train_rl.py` had already written the final adapter, so the
supervisor's retry would skip training, re-run the same `--bs 192` encode and OOM again — an
infinite relaunch loop. Idempotency protects against a killed *trainer*, not against a stage that
fails deterministically.

**Standing procedure adopted:** produce final-point evals through `run_eval.sh`, not the in-cell
stage, whenever a memory cap is in force. **The four mix2M grid cells will hit this same OOM at
their final eval** (same entry script, same `--bs 192`, same 0.25 cap) — handle their step-1500
points the same way.

### 2026-09-08 21:00: emaidx-1M-s3000 COMPLETE — full 4-point curve vs the published baseline
Final point produced via `run_eval.sh` (encode `--bs 128`, the same path as the three interim
points) after the in-cell `--bs 192` stage OOMed against our own cap.

| step | entR@5 | ansR@5 | acc | Δ entR@5 | Δ ansR@5 | Δ acc |
|---|---|---|---|---|---|---|
| 750 | 77.23 | 65.84 | 34.47 | +1.70 | +2.29 | −0.67 |
| 1500 | 76.83 | 64.29 | 34.13 | +3.43 | +2.24 | +0.27 |
| 2250 | 76.33 | 64.65 | 35.27 | +1.67 | +1.84 | +0.27 |
| 3000 | 75.83 | 64.82 | 34.20 | +3.20 | +3.22 | +0.40 |
| **mean** | | | | **+2.50 callable** | **+2.40 callable** | +0.07 not callable |

Slopes per 1k steps (dof=2): entR@5 EMA −0.627 [−0.726, −0.527] vs baseline −0.991 [−3.711, +1.729];
ansR@5 EMA −0.359 [−2.137, +1.418] vs baseline −0.680 [−2.438, +1.078]; acc EMA +0.044 vs −0.382,
both spanning zero. Point estimates favour the EMA arm on every metric and **only the EMA arm's
entR@5 decline is itself distinguishable from zero** — its CI is 30x tighter than the baseline's,
which is a statement about the baseline's noisiness, not about EMA.

**Against the published (cross-cluster) baseline the arm looks good: +2.50 / +2.40, both callable.**
**Against the same-hardware control at the one checkpoint where both exist it does not: +1.60 /
+1.06, neither callable.** Both statements are in the record and the second is the one that
governs, because it is the only comparison in which the treatment is the sole difference. The gap
trajectory is also non-monotone on both recall metrics (entR@5 +1.70 / +3.43 / +1.67 / +3.20 —
alternating), which is what a ~1-point real effect plus ~1.2-point seed-scale noise looks like, not
a clean offset.

**Honest summary of this cell:** index-EMA on the 810k pool is worth on the order of 1 to 2 points
of answer recall, is not separable from noise against its only matched control, and does not change
the slope. It does not deliver a recipe that keeps improving.

### 2026-09-12: INTERVALS ON EVERY SEEDED CELL — two estimator defects found, floors re-derived
Task was to put SD/CI on every result that has seeds. Doing it mechanically from the raw per-seed
json (new `mmrag/seed_stats.py`, now the single source for every ± in paper.md) surfaced two defects
in the headline table that hand-typed intervals had hidden, plus an invalid floor derivation.

**DEFECT 1 — the relevance-SFT row was not one estimator.** Its acc (32.85) was the 3-seed mean over
each seed's best checkpoint, exactly as the row label said. Its two RETRIEVAL columns (73.63 / 59.67)
came from `sft-lr1e4-h0-repro`, a different **single-seed single-checkpoint** run whose own acc is
31.00. Scored with the row's declared estimator (per-seed acc-argmax over ck450-1200), all three
columns are:
  s0 ck750 75.17/62.04/33.00 · s1 ck1050 75.73/63.27/33.67 · s2 ck1050 74.57/60.57/31.87
  → **75.16 ± 0.58 / 61.96 ± 1.35 / 32.84 ± 0.91**, raising the SFT retrieval bar +1.53 / +2.29.

**CONSEQUENCE — "RL beats SFT on every column" is RETRACTED.** v3-pure − SFT, matched 3 seeds:
| metric | delta | Student dof=4 | Welch | exact rank-sum | seeds |
|---|---|---|---|---|---|
| entR@5 | +0.93 | [−0.95, +2.82] spans 0 | [−1.23, +3.10] spans 0 | p=0.100 | overlapping |
| ansR@5 | +1.50 | [−1.14, +4.14] spans 0 | [−1.14, +4.14] spans 0 | p=0.100 | overlapping |
| acc | +1.87 | **[+0.36, +3.37] excl 0** | [−0.47, +4.20] spans 0 | **p=0.050** | **complete separation** |
Both retrieval deltas are also inside the re-derived floors. **Accuracy is the only surviving column.**
The acc verdict flips on the dof convention (Welch drops dof to 2.3 because SFT's SD is 4x RL's),
so the claim rests on complete rank separation (p=0.050, the 3v3 minimum) and per-query McNemar
(2 of 3 seeds), not on the between-seed t. The "+1.86 = 3.5 SE" framing overstated what n=3 resolves;
the margin is pinned no better than roughly +0.4 to +3.4 acc.

**DEFECT 2 — the v2 row dropped a seed.** 80.13/65.46/33.82 was the mean of seeds 0/2/3;
`plgrpo-gme2b-nogold-s1` (weakest on all three metrics) was on disk and outside the mean with no
stated criterion. All 4 in: **79.78 ± 0.77 / 64.48 ± 1.96 / 33.60 ± 0.70**. Anchor verdict narrows:
+3.69 entR@5 clears the floor, +1.02 ansR@5 does not, −1.11 acc is now marginal (was comfortable
against the old 0.61 floor).

**FLOORS RE-DERIVED — the old ones pooled across EVAL SETS, which is invalid.** E-VQA is n=2701 /
n_ans=2696; InfoSeek is n=3000 / n_ans=2450. Different measurements. InfoSeek only, 9 families,
dof=15: **entR@5 2.03 · ansR@5 2.32 · acc 1.02** (was 2.40 / 1.91 / 0.61, from 4 families).
E-VQA separately, 1 family dof=2: 2.43 / 7.10 / 2.54 — its ansR@5 SD is 3.55, the largest in the
campaign, so **out-of-domain answer recall is not measurable at 3 seeds**. `compare_ema.py` updated.
Claims that move: the −2.20 entR@5 control drift (mix2m-v3pure 3k→6k) now CLEARS 2.03 and is a
callable decline; the cross-cluster index-EMA row's +2.50/+2.40 now clear theirs but that row is
retracted for being cross-site anyway. The same-hardware index-EMA row stays inside under both.
Everything else re-checked and unchanged.

**Also corrected:** the E-VQA v3-pure numbers in paper.md (81.34/72.18/45.87) were **seed 0 alone**,
best of three on two of three metrics. 3-seed means: 80.23 ± 1.21 / 68.48 ± 3.55 / 44.67 ± 1.27.

**Caveat the intervals exposed:** SFT seeds 1 and 2 both peak at ck1050, the LAST checkpoint
evaluated for them (seed 0 has ck1200 and peaks at ck750). The per-seed-best estimator is truncated
for 2 of 3 SFT seeds, so the SFT bar may still rise, which would shrink the acc margin.

**Registered-threshold note.** The `mix2m-partialwb` prediction ("turns over by more than 2.40 below
its own 3k point") keeps its 2.40 as registered; moving a pre-registered bound after re-deriving the
floor is the failure the registration exists to prevent. Both readings will be reported. Its 3k eval
was already in flight from `orchestrate.sh` when this was written.

**Shared figure republished** with ± on every seeded cell and the claim box rewritten to "exactly one
margin over the supervised arm survives: answer accuracy on image + text". Text-side per-seed values
requested from tth100-vlm2vec (offline, queued); their cells carry "SD pending" and are not called.

### 2026-09-12 (paper fork): v2's retrieval win recovered; TWO presentation defects shipped and fixed
Writing the paper from the per-seed files surfaced a result the ledger had been under-reporting, and
in presenting it I shipped two errors of the same class I had just corrected. Both are recorded here
because the standing rule is that corrections get written down rather than quietly overwritten.

**RECOVERED — the anchored arm beats relevance-SFT on retrieval, callably.** v2 − SFT = **+4.63
entR@5** (Student [+3.25,+6.01], Welch [+3.31,+5.94], dof 5.0) and **+4.57 entR@10**, both with
COMPLETE seed separation, 4 seeds vs 3. This is the largest and cleanest win over relevance
supervision in the campaign — larger than the +1.87 accuracy margin, with one more seed, and callable
under BOTH dof conventions where the accuracy margin is callable under only one. It was hidden by two
things at once: the v2 row was published with its weakest seed dropped, and it was compared against
the understated SFT retrieval bar (73.63 instead of 75.16). v2 is annotation-free — pure policy pools,
the anchor's positive is the policy's own top-1.

**Also new: entR@1 SIGN FLIP.** SFT 60.13 ± 0.67 vs v3-pure 59.42 ± 1.57 = −0.71 for RL. The old
table showed SFT at 59.13, i.e. behind. "Beats SFT on every retrieval column" was wrong in two
independent ways.

**And: no configuration wins both.** v3-pure − v2 = +1.11 acc / −3.69 entR@5, both callable with
complete separation. Each arm ties exactly where the other wins. A tally of "three wins over
supervision" is a UNION OVER CONFIGURATIONS, not a property of a method, and must not be quoted.

**DEFECT 1 (shipped ~20 min, fixed) — a composite row in the shared figure.** I published a single
"indirect reward" row carrying v2's retrieval (79.78, 4 seeds) beside v3-pure's accuracy (34.71,
3 seeds) under one heading. That row describes a configuration that does not exist. Disclosing it in
the caption did not fix it: a reader takes two numbers in one row as one system. Caught by the parent
session. Fixed by splitting into two per-configuration blocks, each carrying all its own metrics so
the reader sees each arm tie where the other wins. Same error class as the estimator mixing corrected
earlier the same day — one row, one system, one estimator.

**DEFECT 2 (same publish, fixed) — an inconsistent rule for a broken comparator.** On the text-side
in-domain cell I discounted the +10.87 retrieval margin because the supervised arm sits below its own
untrained base, then counted +3.55 accuracy from that SAME comparator as a win. The supervised arm is
below base on BOTH metrics (58.65 vs 69.3 retrieval; 73.00 vs 75.6 accuracy), so one rule has to apply
to both. That cell now referees nothing; the only defensible reading is RL vs base, **+0.95 accuracy**,
base at 1 seed so no interval. Also caught by the parent session.

**Text-side numbers received (h200-rag), including a correction of their own.** Their multi-source
"+1.3 over supervised" compared RL's seed mean to SFT at ONE checkpoint rather than per-seed best;
scored with RL's own estimator it is **+0.35, 95% CI [−1.57, +2.27], rank-sum p=0.69 — a tie.** What
survives on that row is the shape (RL monotone to 25k queries, SFT peaks by 6.4k rows then declines),
not the level. Their in-domain cEM margin is +3.55 [+2.52,+4.58] with complete separation, but see
DEFECT 2: it comes from the comparator that is below its own base. Their floor is not a pooled SD but
a config-identity replicate bound (two runs of one recipe at one seed diverging up to 3.2 R@5), and
they do not pool across eval sets.

**DISTRIBUTION DEFECT — the shared artifact's link is pinned to an old version.** The live page
carries every correction; anyone holding the share link sees a pinned earlier version, which still
shows the retracted "+2.46, clean" reading. Neither session can unpin it — there is no such action in
the tool, it is set from the page's share menu. This is a defect in distribution, not in the numbers:
the correction was made and published and the audience did not receive it. Standing note: any shared
artifact needs its pin re-pointed after a correction. Carried to the user as an action item.

**Paper destination.** The user named it mid-session: the Overleaf-backed repo Irisicy4/Overleaf_Indirect,
a LaTeX paper titled "Indirect Feedback for Reinforcement Learning" with three experiments (captioning,
RAG, skills). Migrated to the ICLR 2027 style (the .sty files are byte-identical to 2026 apart from the
year). `sec_mm_rag.tex` existed in that repo but was never \input, so it had never been compiled and
failed on an undefined \varnothing; it is now included and the document builds at 37 pages with all 66
citations resolving. Two undefined refs remain (app:mech, app:rl_budget), both text-side appendices,
left to that side rather than invented.

### 2026-09-12 (paper fork, cont.): depth profile = mechanism; R@1 overread and corrected
**CORRECTION to my own line from an hour ago.** I wrote "at entR@1 the sign flips" with SFT ahead.
The point estimate does favour SFT but both intervals span zero (v3: −0.71 [−3.45,+2.03];
v2: −0.87 [−2.68,+0.95]). **It is a TIE, not a supervised win.** Caught by the parent session. This
is the third time in one day that a point estimate got described as a direction without its interval.

**NEW RESULT — entity recall by depth, and it supplies a mechanism.** Same seeds, v2 − SFT:
| depth | v2 − SFT | Welch CI | separation |
|---|---|---|---|
| R@1 | −0.87 | [−2.68, +0.95] | overlapping |
| R@5 | +4.63 | [+3.31, +5.94] | COMPLETE |
| R@10 | +4.57 | [+2.88, +6.26] | COMPLETE |
| R@20 | +3.79 | [+2.30, +5.29] | COMPLETE |

Zero at rank 1, opens to +4.6 by rank 5, decays slowly. **Reading: the action is a list.** PL-GRPO
draws an ordered slate by Gumbel-top-k and the reward is computed on the whole slate, so the gradient
shapes the body of the ranking; a contrastive objective on relevance labels optimises the top-1
margin directly and matches it there. The R@1 tie is the one depth where the two objectives optimise
the same quantity. Also explains why acc tracks R@5 not R@1 — the reader is handed 5 passages.
Caveat stated in the paper: 1 dataset, 1 backbone, text-side depth sweep not yet run. Requested from
h200-rag; if their arm shows the same flat-at-1/wide-at-5 shape it is cross-modal.

**v2's annotation-free property verified to the code by the parent**, not just the cell name:
`infonce_loss` takes candidate 0 as positive; `cands = chosen if no_force_gold else [gold] + chosen`;
plgrpo asserts no_force_gold on entry; `git log -S` puts that assert in the same commit that
introduced plgrpo (a3b734e, 2026-08-10). So no plgrpo run has ever seen a labelled positive. Stated
in the paper as a runtime assertion present since the estimator was introduced, which is stronger
than a recorded flag — the v2 run dirs are not on this box, only their result json.

**TEXT-SIDE APPENDICES WRITTEN** (content from h200-rag's eval logs), resolving the last two
undefined refs. app:mech carries a cross-project convergence worth its own line:

| reader with gold / without | text | image+text |
|---|---|---|
| correct / wrong (usable) | 36.9% | 32.3% |
| correct / correct (saturated) | 21.4% | 18.1% |
| wrong / wrong (noise) | 39.2% | 48.5% |
| wrong / correct | 2.6% | 1.1% |

Measured independently, different modalities, readers and corpora, landing within a few points on
every cell. **This is the strongest evidence either project has that reward starvation is a property
of indirect feedback rather than of a dataset.** Their 30B swap reproduces our prediction exactly
(usable share FALLS 36.9→30.6). Their filter result (+2.75 mean over 4 seeds, frozen 1M @10k) is the
Q4 lever this project never launched.

**Their budget curve settles the scaling question on their side:** Rel-SFT is flat from its FIRST
checkpoint (62.8 at 640 labelled rows, 63.7 at 6.4k, declining to 60.6 by 51.2k — 80x more annotation
buys nothing and eventually costs 3 points), while RL rises monotonically to 25k queries. At the
ceiling they tie (+0.35, CI [−1.57,+2.27]). So the asymmetry is the SHAPE, not the level: RL needs
~13k unlabelled queries to reach what SFT reaches with 640 labelled rows.

**Their anchored axis does not exist** — every live-index cell of theirs runs contrastive_coef 0, and
the gold-positive anchor in their earlier record predates this program. So our pure-vs-anchored metric
split has no text-side analogue to confirm or deny, and no cross-project claim about "the anchor"
should be made without naming which anchor.

### 2026-09-12 (paper fork, cont. 2): the cross-project taxonomy is NOT a close replication
**CORRECTION, caught by the parent before it shipped.** I wrote the text/multimodal reader taxonomy
up as two independent measurements "landing within a few points on every cell". Arithmetic says
otherwise. Binomial SE of the difference (text n=60,000 training queries, MM n=1,500 eval queries):

| cell | img+text | text | diff | z | 95% CI |
|---|---|---|---|---|---|
| usable | 32.3% | 36.9% | +4.6 | 3.8 | [+2.2, +7.0] |
| saturated | 18.1% | 21.4% | +3.3 | 3.3 | [+1.3, +5.3] |
| **noise** | 48.5% | 39.2% | **−9.3** | **7.1** | [−11.9, −6.7] |
| harms | 1.1% | 2.6% | +1.5 | 5.4 | [+1.0, +2.0] |

**All four differ beyond sampling error**, and the noise bucket — the one the starvation argument
leans on hardest — differs at 7.1 SE. Worse, **the two are different populations**: mine is a
frozen-base pass over the held-out EVAL set, theirs partitions the TRAINING stream. A side-by-side
table implying one measurement is misleading regardless of the statistics.

**What actually replicates, written that way in the paper:** the ORDERING (noise > usable >
saturated > harms≈0) and the REGIME (~2/5 of rows yield no usable reward, ~1/5 need no retrieval,
~1/3 carry reward variance). Two independent settings landing in the same regime supports
"starvation is structural to indirect feedback"; it does NOT support quoting either column's values
as the other's. Bucket sizes are setting-specific and must be measured per deployment.

**CREDIT CORRECTION — the 30B prediction was NOT confirmed cross-modality.** paper.md:956-960
registered that swapping *our* reader for a 30B would shrink the useful training set. That is a
multimodal prediction and **we never ran it**. Their 36.9→30.6 swap confirms the mechanism in *their*
setting, i.e. their result reproducing their own mechanism. The paper now says: registered from their
earlier data, confirmed by them on text, **multimodal reader swap unrun**, multimodal numbers merely
consistent.

**STATED PLAINLY IN THE PAPER: the reward-variance filter is untested on multimodal.** Their result
is +2.75 R@5 over a matched random control at 4 seeds (on top of +4 to +7 at 2 seeds earlier). It is
the highest-value open experiment either project has identified and leaving it implicit would let a
reader assume we ran it. The MM taxonomy sizes the population it would act on (32.3% usable, 48.5%
noise); no filtered MM run exists.

**DEPTH SWEEP — text side does NOT replicate our shape.** Their RL−SFT gap is separated at rank 1 and
flat from R@5 outward at both sizes, because their supervised comparator is below base at EVERY depth
(8B: −3.4 at R@1 through −9.9 at R@15), so that difference measures the comparator. Against BASE at
4B the shape does appear: +0.8 at rank 1 → +2.2 at rank 5, flat after. Magnitude +1.4 rank1→rank5
against our +5.5. Recorded in the paper as a consistent-sign replication at one backbone, explicitly
NOT a second independent demonstration. At 8B their RL−base is ~0 at every depth.

**R@1 DROPPED FROM THE PAPER'S INFOSEEK TABLES** at the user's instruction: this is a retriever
feeding 5 passages to a reader, not a reranker, so R@1 is not the operating point. The full R@1
numbers stay here in the ledger (SFT 60.13 ± 0.67 / v3-pure 59.42 ± 1.57 / v2 59.27 ± 1.18 — a tie,
all intervals spanning zero). E-VQA keeps its R@1 column, where v3-pure leads.

**PROCESS FIX.** Three direction-without-interval errors in one day is a pattern, not three slips, and
the shape is always a delta computed in one place and written down in another. seed_stats.py already
refuses to print a delta without its interval; `paper_tables.py` now has `margin()`, the only path
that renders a difference, returning delta + CI + callable verdict + separation as one inseparable
string. Mechanical, not attentional.

### 2026-09-12: Q4 GATE WAS INERT ON plgrpo — code gap, not an oversight; now implemented + launched
`--reward_gate_noctx` (the Q4 no-context utility filter, the strongest lever either project has
found) lived ONLY inside the final `else:` reward branch of train_rl.py. plgrpo takes the first
branch, so **the flag was silently inert for v3-pure and v2** — the entire winning family. The one
recorded trial (+0.94 ansR@5, 1 seed, "right direction") can therefore only have been a grpo-family
run. That is why the lever was never tested on the recipe that matters, and "never launched" in the
earlier ledger understated it: it was not runnable.
Implemented on the plgrpo path with semantics copied verbatim from the other branch (keep a query iff
its best sampled list beats the reader with no context), plus the keep-given-gold vs
keep-given-no-gold contingency so we can tell a pool-quality detector from a query-difficulty one.
Launched `q4gate-s0` (GPU 5) and `q4gate-s1` (GPU 7) against the existing 2-seed `mix2m-v3pure`
ck750 control, one flag apart. Text side reports +2.75 R@5 at 4 seeds for the same idea.

### 2026-09-12: DEPTH PROFILE — the "decays past R@5" reading is a CEILING ARTEFACT (retracted)
v2−SFT on raw recall: +4.63 @5, +4.57 @10, +3.79 @20. Read as "widest at the operating depth,
decaying past it". **That reverses sign on a ceiling-free scale.** v2 is at 89.52 @20, so a raw
increment must compress. On miss rate (1−R):

| depth | raw margin | miss ratio v2/SFT | log-miss advantage | residual miss removed |
|---|---|---|---|---|
| @1 | −0.87 | 1.022 | +0.021 ± 0.048 spans 0 | −2.2% |
| @5 | +4.63 | 0.814 | −0.206 ± 0.064 EXCL | +18.6% |
| @10 | +4.57 | 0.760 | −0.274 ± 0.075 EXCL | +24.0% |
| @20 | +3.79 | 0.734 | −0.309 ± 0.110 EXCL | +26.6% |

The advantage **widens monotonically with depth**. Also tested the within-run form (R@5 − R@20 per
seed, paired inside each run): v2−SFT +0.83, CI [+0.20, +1.47], which looks like real narrowing and
is the same ceiling compression — v2 simply has less room left at 20.
**The real shape is a discontinuity, not a peak:** nothing at rank 1 on *either* scale, then a step
to +18.6%, then monotone widening. The rank-1 null survives the ceiling correction, which is what
makes it diagnostic.
**Mechanism, read off the loss rather than fitted to the curve:** the plgrpo branch scores each
unique sampled doc once and sets the list reward to the **mean over positions** — per-document
credit, no position weighting, pl_k 4. Rank *within* the slate carries exactly zero gradient signal
by construction, so the objective shapes slate membership and not slate order. Rank 1 is the one
quantity a position-uniform slate reward does not privilege; supervised contrastive training
optimises that margin directly and matches it there.
**Open tension with the user's instruction** to drop R@1 from the paper (sound for reporting: the
system feeds 5 passages, R@5 is the operating point). R@1 here is mechanism evidence, not a
performance metric. Recommendation sent: keep R@5 as the reported operating point and retain R@1
only inside the mechanism discussion, labelled diagnostic. If it comes out entirely, the mechanism
must be stated as a hypothesis, because a monotone curve with no inflection is much weaker evidence.
**Text-side replication is weak, not supporting:** appears at 4B at ~1/4 magnitude, flat at every
depth at 8B, and their 4B opens at +0.8 at rank 1 rather than showing the rank-1 null. So it does not
reproduce the feature the argument rests on.

### 2026-09-12 (paper fork, cont. 3): REWRITE RATE orders every index-refresh cell in both projects
h200-rag's rolling-1M program through 4k queries. Their schedule names are **steps per full cycle**,
so rewrite rate = C/name rows per step. Normalised as corpus fraction per optimizer step, all seven
cells either project has run fall into one order:

| rate | form | side | outcome |
|---|---|---|---|
| 0.10%/step | in-order (Sweep1000) | text | **holds** through 4k, +3.1 vs frozen parity |
| 0.10%/step | random (fullcorp-rand10) | MM | **holds**, −1.40 vs control, inside floor |
| 0.20%/step | in-order (Sweep500) | text | +4.0 at 2k then **COLLAPSES** −8.8 at 3k |
| 0.20%/step | random (Rand500) | text | declines from 2k: +1.0 / −1.8 / −4.2 |
| 0.40%/step | either (Sweep250 / Rand250) | text | collapses by 2k |

**TWO CONSEQUENCES, both of which weaken claims this project has been making.**

1. **"Random fails, in-order works" was confounded with rate.** Their Rand500 is 0.20%/step; my
   random arm is 0.10%. Matched on rate the verdicts agree, and the cross-project "disagreement" I
   recorded as open is dissolved. Their data DOES separate form from rate at 0.20% (Sweep500 is 5.8
   above Rand500 at 2k) so ordering genuinely helps — but both are gone by 3k. **Rate decides
   survival; form is second-order and visible only while the run is still alive.**
2. **The age-ceiling story is demoted from cause to contributor.** The in-order sweep has a bounded
   worst-case age at EVERY rate and still collapses at 0.20% and 0.40%. Bounded age is necessary and
   demonstrably not sufficient. The quantity that must stay small is how fast the index moves under
   the policy, not only how stale its oldest row may get. The figure's update-rule table gained a
   rate column (k/C per step) so any scheme can be placed on the axis.

**Caveat carried in the figure:** 1 seed past 2k on every text schedule, 1 seed on my arm. Seven
cells ordering consistently across two projects is worth stating; no pair clears a seed interval.
Written as the shape to design against, not a measured threshold.

**Their diagnostic finding, the most actionable thing here:** training-side reward std and degeneracy
stay FLAT through every collapse; only held-out retrieval and query pairwise cosine move
(Sweep1000 0.154→0.113→0.082→0.087; Sweep500 0.138→0.094→0.068 into its collapse). **An index
collapse is invisible from the training signals.** Anyone running this needs a held-out retrieval
probe; the loss will not tell them.

**Open, suggested to them:** Sweep1000 at 0.20% via halving the corpus rather than doubling rows,
to separate rate from absolute row count — my 422k arm cannot do that against their 1M. Also their
Sweep250Filt (0.528 / 0.530 at 1k, 2 seeds) sits well above Sweep250 (0.490), i.e. the reward-variance
filter appears to rescue a schedule that is otherwise too fast. If that holds at 2k it means data
quality buys back index-refresh headroom, which would be more interesting than the filter's headline.

**Scope note:** the paper was cut to v3-pure only at the user's instruction, so this rolling material
lives in the shared figure and this ledger, not in the paper.

### 2026-09-12: TEXT-SIDE SCHEDULE LADDER REFUTES THE "BOUNDED AGE" MEDIATOR (figure claim must change)
h200-rag's rolling-1M program, R@5 vs frozen-1M parity (4 seeds parity; **1 seed past 2k on every
rolling arm**), 1M multi-source corpus:

| schedule | rows rewritten/step | max row age | 1k | 2k | 3k | 4k |
|---|---|---|---|---|---|---|
| Sweep1000 | 1,000 | 1000 steps | −1.0 | +2.0 | +3.4 | **+3.1** |
| Sweep500 | 2,000 | 500 steps | +2.4 | +4.0 | **−8.8** | — |
| Sweep250 | 4,000 | 250 steps | +1.4 | −2.2 | — | — |
| Rand500 | 2,000 random | unbounded | +1.0 | −1.8 | −4.2 | — |
| Rand250 | 4,000 random | unbounded | −1.2 | −8.2 | — | — |
| **Sweep250 + Q4 filter** | 4,000 | 250 steps | **+5.2** | **+3.6** | — | — |

**All three sweeps have a bounded ceiling, and the TIGHTER the bound the WORSE the result.** So
boundedness is not the mediator, and the shared figure's claim ("refresh schemes with a bounded worst
case work; the one without a ceiling fails") is wrong as written. Unboundedness is still bad — both
random arms lose — but it is not the discriminator. Two knobs, not one: (a) does every row eventually
get refreshed, (b) **how much of the index is rewritten per step**. Best = bounded AND low churn.
Also, at matched rate 2,000/step, round-robin beats random at 1k/2k then collapses HARDER at 3k
(−8.8 vs −4.2), so "bounded beats random" does not hold either, on one seed.

**Their stated mechanism is backwards on its own arithmetic** ("the light schedule's index is mostly
one age"): 1,000 rows/step over a 1,000-step cycle leaves ages uniform on [0,1000] — the WIDEST
spread and the OLDEST mean. The most mixed-age schedule is the one that survives. Also: they credited
me with a "mixed-age co-drift mechanism" that appears in NONE of my documents. Attribution declined.

**Mechanism that does fit — non-stationarity, not age.** Reward std and degeneracy are FLAT through
every collapse while held-out retrieval dies. That is the signature of the train index tracking a
drifting encoder: train-time retrieval stays healthy by construction, and evaluation, which encodes
once with the final model, is where a contracted geometry becomes visible. **Every training-side
metric is structurally blind to this failure.** Higher churn = tighter encoder/index coupling =
faster contraction.

**They already hold the leading indicator, and it is not a reward metric.** Query mean pairwise
cosine: Sweep1000 0.154→0.113→0.082→**0.087** (stops falling between 3k and 4k — the arm that stopped
contracting is the arm that stopped degrading); Sweep500 0.138→0.094→0.068, still falling when it
broke. Available hundreds of steps before held-out retrieval moves.

**Their filter does not add points, it PREVENTS the collapse.** Same schedule (Sweep250): 1k +1.4→+5.2,
2k −2.2→+3.6. Worth +3.8 and +5.8, far above its standalone +2.75, so schedule and filter interact.
Unifying candidate, from our own gate measurement today: gold-free pools generate reward uncorrelated
with relevance, which is the drift driver; the gate removes them (keep 0.49 with gold vs 0.12–0.16
without). **Pre-registered test, from logs they already have: on Sweep250Filt the query cosine spread
should FLATTEN where unfiltered Sweep250's keeps falling.** If it falls just as fast while retrieval
holds, the contraction mechanism is wrong and it gets retracted.

**Honesty item that must reach the user.** Best rolling arm at 4k is 0.580 against their Rel-SFT
ceiling 0.630 and against the 0.6435 RL number in the shared figure (their 12.8k-row setting). So the
rolling full-corpus program is **−5.0 vs supervision and −6.4 vs their own non-rolling RL arm**. It
beats a frozen index and loses to both of their better arms. Sweep1000 is still climbing at 4k while
frozen parity flattens, so its 5k point decides it, but "the full-corpus idea works" is true only
relative to freezing.

### 2026-09-12 (paper fork, cont. 4): FRESHNESS HURTS — the age-ceiling reading is inverted, not just insufficient
My previous entry said a bounded age is "necessary and not sufficient". That was too generous to the
old claim. On h200-rag's round-robin ladder, **the tighter the ceiling the worse the result**:

| rows/step | rate | worst-case age | outcome |
|---|---|---|---|
| 4,000 | 0.40% | **250 steps** (tightest) | collapses by 2k |
| 2,000 | 0.20% | 500 steps | collapses at 3k, −8.8 |
| 1,000 | 0.10% | **1,000 steps** (loosest) | **no collapse** through 4k, +3.1 |

**The logical care that matters here:** on a round-robin ladder rate and ceiling are the same number
inverted (rate k/C, ceiling C/k steps), so the ladder ALONE cannot separate them. **The random arms
do** — they have no ceiling at all, and across them rate still orders the outcome (0.10% holds on MM,
0.20% declines, 0.40% collapses). With staleness unbounded by construction, rate still does the work.
And at matched 0.20% the BOUNDED sweep fails harder at 3k than the unbounded one (−8.8 vs −4.2),
which is the opposite of what the age story predicts. So: **rate is the mediator; the age ceiling is
second-order and its direction, as drawn, was backwards.** Corrected in the figure (v11). I had also
written "ordering does help" off the 2k point; it reverses by 3k, so that line is gone too.

**MECHANISM (hypothesis, not claimed).** Training signals are flat through every collapse — reward
spread and degeneracy show nothing — while held-out retrieval dies. Structural reason: the train index
is re-encoded by the drifting model, so train-time retrieval is healthy BY CONSTRUCTION, while eval
encodes once with the final model and is the only place a contracted geometry shows. Every
training-side metric is blind to it. Leading indicator they hold: query mean pairwise cosine falls
monotonically on failing schedules and STOPS falling between 3k and 4k on the surviving one, exactly
when degradation stops. A pre-registered falsification test is queued on their side; held as
hypothesis until it lands.

**GENUINE FINDING — filter × schedule interact.** The reward-variance filter does not add points, it
PREVENTS the collapse: on the worst schedule +3.8 at 1k and +5.8 at 2k, converting −2.2 into +3.6,
far above its +2.75 standalone. A filtered run tolerates a faster index than an unfiltered one. Ties
to the parent's measurement that the gate keeps gold-bearing pools ~4x as often as gold-free ones,
making gold-free pools the plausible drift driver.

**PLACEMENT — "the full-corpus idea works" is true only against freezing, and this needs saying.**
Best rolling arm at 4k = 0.580, against frozen parity 0.549, Rel-SFT ceiling 0.630, and their own
NON-rolling RL arm 0.6435. So **+3.1 over freezing, −5.0 vs supervision, −6.3 vs their better arm.**
It is still climbing where frozen parity has flattened, so the 5k point decides whether it closes.
Carried to the user explicitly; the paper must not say "works" unqualified.

### 2026-09-12 (paper fork, cont. 5): cosine early-warning WITHDRAWN; full rate ladder
**RETRACTED, one version after publishing it — query cosine is NOT a leading indicator of index
collapse.** I put it in the figure (v11) on the parent's suggestion, sourced from h200-rag's own
trace. h200-rag then checked it against their other cells and it fails three ways:
- the FILTERED Sweep250's cosine falls as fast as its unfiltered twin's and sits LOWER at every
  checkpoint, while its retrieval RISES;
- two cells share cosine 0.082 at 3k with OPPOSITE outcomes.
So the correlation I read off the surviving schedule does not generalise. Removed from the figure.
**What is safe is descriptive only:** every rolling cell spreads its queries 2-3x faster than a
frozen cell, survivor or not. **There is currently NO known training-time signal that predicts the
collapse** — reward spread and degeneracy are flat through all of them — which is a stronger and more
useful statement than the indicator would have been: a held-out retrieval probe is mandatory, not
advisable. They also withdrew an age-mixture mechanism of their own; the survivor is the schedule
with the WIDEST age spread, so a bounded age is not the mediator from that direction either.

**Full ladder as they now hold it** (rate = rows/step ÷ corpus, vs a frozen index):
| rate | form | outcome |
|---|---|---|
| 0.10% | in-order | +3.1 at 4k, no collapse |
| 0.10% | random (mine, MM) | holds, −1.40 vs control, inside floor |
| 0.20% | in-order | +4.0 / +4.6 at 2k (**2 seeds**), then −8.8 at 3k (1 seed) |
| 0.20% | random | +1.0 / −1.8 / −4.2 |
| 0.40% | in-order | −2.2 at 2k |
| 0.40% | random | −8.2 at 2k |
| 0.80% | in-order | **+3.0 at 1k**, 2k pending |

**CORRECTION to my own cont.4 entry:** I dropped the "ordering helps" line as unsupported. It is
supported, at two matched rates: in-order beats random by **+5.8 at 0.20% and +6.0 at 0.40%**, both at
2k while alive. What is true is that ordering does not decide SURVIVAL, and that a bounded schedule
collapses harder when it goes (−8.8 vs −4.2 at 3k). Restored in that form. Also note 0.80% is still
+3.0 at 1k, so a rate is not fatal per se — **rate sets how long a run lasts**, it is rate x duration.

**REGISTERED TEST (theirs), separating rate from absolute row count.** Rate-matched twins on the
200k corpus, which already has a 2-seed frozen-EMA stride comparator: dmix200kSweep500 = 400 rows/step
(0.20%, 500-step period — same rate AND period as the 1M Sweep500 at 5x fewer rows) and
dmix200kSweep1000 = 200 rows/step (0.10%). Pre-registered: **collapse at ~3k ⇒ rate/period is the knob
and absolute rows are not; holding through 5k ⇒ absolute rows or corpus size matter and the rate axis
is incomplete.** First points ~19:00.

**Placement wording aligned to theirs:** above the frozen index at matched budget ONLY; −5.0 vs the
Rel-SFT ceiling (0.630) and −6.4 vs frozen RL's own 25k endpoint (0.6435); 1 seed past 2k everywhere.
