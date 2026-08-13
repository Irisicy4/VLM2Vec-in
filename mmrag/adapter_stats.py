"""lora_B abs-sum for run adapters — separates "didn't train" from "trained and it
didn't help", which neither recall nor the distinct-count can see. lora_B is
zero-initialised, so its post-training abs-sum measures how far the optimiser moved.
No GPU needed. Part of the standard harvest alongside acc/entR@5/distinct-top5.

usage: python3 mmrag/adapter_stats.py <run_dir_or_runs_root> [more dirs...]
"""
import os
import sys

from safetensors import safe_open


def stats(adapter_path):
    tot_b = tot_a = 0.0
    with safe_open(adapter_path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if "lora_B" in k:
                tot_b += float(t.abs().sum())
            elif "lora_A" in k:
                tot_a += float(t.abs().sum())
    return tot_b, tot_a


def main():
    targets = []
    for arg in sys.argv[1:]:
        direct = os.path.join(arg, "adapter_model.safetensors")
        if os.path.exists(direct):
            targets.append((os.path.basename(arg.rstrip("/")), direct))
        elif os.path.isdir(arg):  # runs root: scan children
            for d in sorted(os.listdir(arg)):
                p = os.path.join(arg, d, "adapter_model.safetensors")
                if os.path.exists(p):
                    targets.append((d, p))
    for name, p in targets:
        b, a = stats(p)
        print(f"{name:32s} lora_B {b:12.1f}   lora_A {a:12.1f}")


if __name__ == "__main__":
    main()
