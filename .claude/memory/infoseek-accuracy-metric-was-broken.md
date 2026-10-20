---
name: infoseek-accuracy-metric-was-broken
description: InfoSeek accuracy in results/*.vqa_top5.json is wrong for 21.9% of queries; always read the corrected acc from *.vqa_rescored.json instead.
metadata: 
  node_type: memory
  type: project
  originSessionId: 4432a7aa-e1e1-4d8e-8bb0-e803fd5ea440
  modified: 2026-09-14T00:53:22.260Z
---

**Never read `acc` out of `$MMRAG_DATA/results/*.vqa_top5.json`.** It is wrong, silently, by about 8
points for every arm. Use `acc_rescored_any` from the matching `*.vqa_rescored.json`, produced by
`mmrag/rescore.py` (discovered and fixed 2026-09-12).

Cause: 15,414 of 71,335 InfoSeek queries (329 of the standard 1,500-query eval sample, 21.9%) carry a
numeric gold with an explicit tolerance, stored as a dict like
`{'wikidata': 176.0, 'range': [158.4, 193.6]}`. `answer_correct` does `_normalize(str(g))`, so the gold
becomes that literal text and is searched for as a token sequence in the reader's answer. Nothing can
match. A prediction of `"176 metres"` against that gold scored **0.0**. Across all 217 evals on disk the
number of such queries ever scored correct is exactly 0.

Consequences a future session will trip over:
* Every accuracy number in older ledger entries and in `paper.md` prose written before 2026-09-12 is on
  the broken metric. Levels shift ~+8.5; **orderings do not change**.
* The accuracy noise floor is **1.83** on the corrected metric, not 1.02. The old floor was deflated
  because 329 queries were a constant-zero block contributing no between-seed variance.
* Retrieval metrics (entR@*, ansR@*) are unaffected — the defect is reader-side only.
* `rescore.py --all --write` is cheap and idempotent; rerun it after any new eval lands.

The fix was free only because `eval_vqa.py` stores per-question predictions on purpose ("so metric fixes
can rescore without regeneration"). Keep that property in any new eval path.

Related: [[user-works-by-terse-directive]]
