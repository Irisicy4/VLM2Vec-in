# Full-corpus RL index + random re-embed — sweep design (2026-09-11)

RL index = the ENTIRE corpus_small (422,378 passages, all 21,143 non-gold articles + all golds), so
the training index IS the eval index — no train/eval gap, unlike every earlier cell (41% / 9.2%).
All cells: v3-pure recipe, 1500 steps = 6k queries, seed 0, eval on corpus_small. Endpoint and ck750.

| cell | mode | coverage | rows never refreshed | tests |
|---|---|---|---|---|
| fullcorp-rand05 | random 5% /100 | 0.75x | 46.3% | extreme staleness |
| fullcorp-rand10 | random 10% /100 | 1.50x | 20.6% | the user's proposal |
| fullcorp-rand25 | random 25% /100 | 3.75x | 1.3% | |
| fullcorp-rand50 | random 50% /100 | 7.50x | 0.0% | fresh end |
| **fullcorp-stale** | stalest-first, extra 310 | **1.50x** | **0.0%** | **matched coverage to rand10, max age bounded -> isolates WHICH rows, not how many** |
| fullcorp-rand10p50 | random 10% /50 | 3.00x | 4.2% | halves the period at similar coverage -> separates period from volume |

Reference points already measured on corpus_small at the 41%-subset index size:
emaidx-1M-s3000 (stalest-first, 2.99x) 4-point curve 77.23 / 76.83 / 76.33 / 75.83 entR@5;
v3pure-1M-s3000-loc-ck750 (full refresh) 75.63. Base 68.53/54.94/30.80. Floors 2.40/1.91/0.61.

Prediction registered before results: if the mediator is MAX age, fullcorp-stale > fullcorp-rand10
at identical coverage. If it is coverage alone, they match and the sweep is monotone in coverage.
