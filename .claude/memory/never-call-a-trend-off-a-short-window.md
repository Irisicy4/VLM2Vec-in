---
name: never-call-a-trend-off-a-short-window
description: A within-run trend called off one 80-step window reversed completely with more data; require the window to be a stated fraction of the run before claiming a trend
metadata:
  type: feedback
---

At 200 steps of `judge-alt2` the judge-vs-cover-EM dead-gradient gap looked like it decayed to zero:
steps 0-119 gave -0.1375 (callable) and steps 120-199 gave +0.0188, and the early-vs-late contrast
tested callable at z = -2.14. I reported the decay, with a mechanism attached (flip rate falling as the
policy improves).

At 613 steps it was gone. The windowed diffs are -0.1375 (0-119), -0.1036 (120-399), -0.1163 (400+),
all callable and mutually consistent; pooled -0.1145, 95% CI [-0.155, -0.074]; early-vs-late z = -0.37.
Flip rate is flat at 0.20-0.24, not declining. The "decay" was the single 100-199 window sitting at
+0.005 by chance.

**Why:** a z of -2.14 on an 80-step window is what noise looks like when you test one of several
possible split points on 13% of a run. The mechanism I attached made the artifact feel explanatory,
which is exactly what made it convincing.

**How to apply:** before claiming any within-run trend, state the window as a fraction of the planned
run; below roughly a third, report the pooled estimate and say the trend is untestable yet. Prefer
re-reading at a later checkpoint over publishing a split-point test. This is the ninth instance of the
shape catalogued in [[check-direction-before-claiming]].
