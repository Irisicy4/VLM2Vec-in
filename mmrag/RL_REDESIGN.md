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
