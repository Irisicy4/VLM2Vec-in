---
name: haystack-collapse-is-oven-not-volume
description: The 2M-index collapse of the frozen multimodal retriever is attraction to OVEN entity summaries, not cross-domain distractor volume
metadata:
  type: project
---

On the 2M mixed-domain index (built/corpus_mix2M.jsonl), the frozen GME-2B base scores entR@5 = 14.27
against 68.53 on the 422k in-domain index. Attributing the stored top-5 by each passage's recorded
provenance (`mmrag/haystack_domains.py`, no GPU) shows the cause is NOT distractor volume:

- the nine BEIR text domains are 37% of the index and are retrieved at lift <= 0.05 — inert volume;
- 86.3% of the base's slate goes to OVEN entity-summary passages (34% of the index, lift 2.54), and
  96.6% of those are the WRONG entity;
- every RL arm suppresses OVEN to lift <= 0.49 and scores 67-70 on the same index — including
  `idx10k-full` and `emaidx-1M-s3000`, which trained only on the pure-Wikipedia 422k index and never
  saw OVEN or BEIR. That is transfer, not exposure.
- relaxing the metric to credit any passage of the right entity (OVEN summaries included) moves the
  base only 14.27 -> 26.03, so it is not a scoring artifact.

corpus_small (422k) is 100% Wikipedia and contains ZERO lead/section-less passages, so the rival
"prefers lead paragraphs" explanation cannot be tested in-domain — do not assert it. What IS testable
is that BEIR passages are also section-less and are never retrieved, which argues against a pure
format preference.

**Why:** the paper's earlier framing attributed +55.6 to "cross-domain distractors", and the user
pruned `tab:mm_indexsize` in c987e5a after objecting that the effect sounded like dataset-format
fine-tuning. This attribution is what adjudicates that objection.

**How to apply:** the open control is [[sft-control-decides-haystack-claim]] — if plain relevance-SFT
is equally immune, the robustness belongs to fine-tuning in general, not to the indirect reward.
Tables live in `tab_mm_haystack.tex` in /raid/icy/Overleaf_Indirect. See also
[[check-direction-before-claiming]].
