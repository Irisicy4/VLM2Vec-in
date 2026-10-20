#!/bin/bash
# MORE EVAL BENCHES, multimodal analogue of the text side's BEIR columns.
#
# The 2M-passage mixed-domain index is a genuinely harder bench than the 422k in-domain one. The
# domain attribution (mmrag/haystack_domains.py) showed the frozen base falls 68.53 -> 14.27 entity
# R@5 there because 86.3% of its slate goes to OVEN entity-summary passages (34% of the index, lift
# 2.54) of which 96.5% are the WRONG entity, while two RL arms that trained only on the pure-Wikipedia
# 422k index sit at 67-70 on it. That is transfer, not exposure.
#
# THE ORDER IS DELIBERATE. sft-local-s0 runs FIRST because it is the control that decides what the
# result means: if a plain relevance-SFT checkpoint is equally immune, haystack robustness belongs to
# fine-tuning in general and the arm labels are decoration; if it collapses like the base, the
# robustness belongs to the indirect reward. Every data-mixture cell behind it is a supporting column;
# this one is the claim. Do not reorder it behind them again.
#
# GUARD PATTERNS USE THE BRACKET TRICK ON PURPOSE. A plain pgrep -f pattern also matches the shell
# that launched this script, because that shell's command line contains the pattern text verbatim.
# That deadlocked the chain once already (2026-09-15, idle 4h52m). Do not 'simplify' the brackets.
#
# Each cell re-encodes 2M passages inline (no cache path exists for a non-default corpus), ~2.5 h at
# the measured 225 passages/s, so they run SEQUENTIALLY on one card rather than contending.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=${1:-7}

# never start a second encode next to one already in flight
while pgrep -f "[m]mrag/eval_retrieval.py" >/dev/null; do sleep 120; done

for ARM in sft-local-s0 frozen-idx q4pool-filt q4pool-rand mix2m-v3pure-s1 divmix-v3pure fullcorp-rand10p50 q4gate-s1; do
  NAME="$ARM-on-mix2M"
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME done"; continue; }
  # a training run still writing this dir would be cross-evaluated mid-flight
  while pgrep -f "[m]mrag/train_(rl|sft).py .*runs/$ARM( |\$)" >/dev/null; do
    echo "$(date +%T) $ARM still training, waiting"; sleep 300
  done
  [ -f "$D/runs/$ARM/adapter_model.safetensors" ] || { echo "$(date +%T) $ARM has no adapter, skip"; continue; }
  echo "$(date +%T) starting $NAME"
  bash "$D/cross_eval.sh" "$GPU" "$NAME" "$D/runs/$ARM" built/corpus_mix2M.jsonl \
       >> "$D/local_$NAME.log" 2>&1 || echo "$(date +%T) $NAME FAILED"
done
echo "$(date +%T) bench2m chain done"
