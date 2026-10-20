---
name: user-works-by-terse-directive
description: This user gives short high-level directives and expects autonomous execution plus results; corrections arrive as brief mid-turn interventions.
metadata: 
  node_type: memory
  type: user
  originSessionId: 4432a7aa-e1e1-4d8e-8bb0-e803fd5ea440
  modified: 2026-09-14T00:53:11.619Z
---

This user directs work in a few words and expects the whole task executed without check-in questions.
Observed directives: "keep sweeping different data recipe", "add eval datasets so it can be a larger
eval table of what in domain/outdomain improves or no", "investgate our small questions", "yes test
this out as many way as you can in 24hrs", "make sure you have some new run result for me after
12hrs".

They do not want to be asked permission before work starts. They do want the *result*, with the
reasoning compressed.

**Corrections arrive as short interventions, often mid-turn, and are absolute.** Examples: "you cant
touch gpu 0-3", "dont talk to paper session wihout my ask from now, you brought too much noise",
"delete ckpt we dont need", "you shall stay on gpu 4-7's headrun". Treat each as a standing constraint
from that point on, not as a one-off. Prefer making such a constraint *mechanically enforced* over
remembering it — see the venv GPU guard in `mmrag/EMA_HANDOFF.md` rule 3d, which the user's card
restriction prompted.

They also ask sharp evaluative questions that expect a survey of everything on disk rather than a new
experiment: "Any more tuned combo that supports RL works better in this arch?", "summarize current
answer to the paper questions", "The dynamic rerank exp how did it fail". Answer those from the
existing results first; only run compute if the survey shows a real gap.

They enforce evidence standards hard and stated them up front: 1 seed = recorded, not claimed; no
margin without matched seeds AND a matched estimator; name the metric in every claim; least-squares
trends with 95% CI, never eyeballed; ledger everything including nulls.

Related: [[no-unprompted-peer-messaging]], [[infoseek-accuracy-metric-was-broken]]
