---
name: check-direction-before-claiming
description: "On this project I repeatedly asserted a direction, mediator or instrument before checking its arithmetic; the cheap checks that catch it each take one command."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4432a7aa-e1e1-4d8e-8bb0-e803fd5ea440
  modified: 2026-09-14T00:53:45.223Z
---

Across one long session on this project I retracted roughly eight claims, and every one had the same
shape: **a direction, a mediator, or an instrument asserted before its arithmetic was checked.** Two of
them had already been sent to a collaborator as advice. Cataloguing the shapes, because each has a
one-command check:

* **Instrument sign.** Published a falling pairwise-cosine as evidence of embeddings *contracting*. It
  means the opposite; contraction drives that statistic toward 1.0. → *Check an instrument's value
  against a known baseline before building on it* (base cosine was 0.257, right there on disk).
* **Instrument dynamic range.** Offered a zero policy-loss as a starvation diagnostic. It is float noise
  at every step — mean −1.7e-10 — because advantages are z-scored within a group so they sum to zero by
  construction. → *One `max(abs(column))` over the existing data.*
* **Rival mediators that are the same variable.** Framed "rewrite rate" as replacing "bounded staleness".
  For a covering round-robin sweep the period is exactly the reciprocal of the rate. → *Write the two
  quantities algebraically before calling them rivals.*
* **A classifier that silently lumps categories.** Claimed a training index was "89% Wikipedia" using a
  test for one URL prefix; the corpus had five sources and two of them fell through as Wikipedia. →
  *Enumerate the categories exhaustively once, don't test for the one you expect.*
* **A statistic that does not fit the data type.** Reported a McNemar delta on seed-averaged per-query
  scores, which are fractional; counting only the *direction* of each difference gave +3.93 where the
  level difference is +2.22. → *The reported delta and the test's own statistic must be the same number.*
* **Aggregates read as a join.** Inferred what a filter discarded from marginal rates; the Fréchet bounds
  spanned [32%, 100%]. → *Log the join.*
* **Operational pattern-matching.** Four "stalled run" alarms, all false: a CPU counter misread, a
  scheduled index refresh read as a hang, a PID exiting between two commands, and `grep -c` counting
  matches-per-line as processes. → *`nvidia-smi pmon -s u` settles a stall in seconds; reach for it
  first.*
* **Over-generalising from one pair.** "Collapses reproduce 14x more tightly than healthy states" held
  for one schedule and failed on the next seed.

**What actually worked, and is cheap enough to do always:** state a mechanism's falsifiable prediction
in the same breath as the mechanism, then run it against logs that already exist. That killed two
mechanisms at no compute cost, one before publication. Pre-registering a prediction *before* reading a
result also worked twice — both were then disconfirmed, which is the point.

Also: a wrong retraction is as costly as a wrong assertion. I once demanded a correct number be deleted
from a published figure without checking it. Corrections need the same verification as claims.

Related: [[user-works-by-terse-directive]], [[infoseek-accuracy-metric-was-broken]]
