"""Refuse to touch a forbidden physical GPU. Loaded via zz_mmrag_gpuguard.pth for every
interpreter in this venv.

WHY A MECHANISM AND NOT A RULE. The allocation on this box is cards 4-7 only, and within that,
5 and 7 are ours while 4 and 6 belong to the collaborating project. Cards 0-3 carry other people's
work: a ray job, another project's retrieval eval, and someone else's vLLM engine. Every launcher
here (run_cell.sh, run_eval.sh, run_q4filter.sh) pins CUDA_VISIBLE_DEVICES from an explicit
argument, so launched paths are already safe. The hazard is running a tool DIRECTLY -- eval_vqa.py,
mmeb_subset.py, a quick probe -- where torch's default `cuda:0` means PHYSICAL card 0 and lands on
a stranger's job. Discipline has already failed twice here on the adjacent problem (two cells placed
on the collaborator's cards), so this is enforced rather than remembered.

TWO CASES, both fail safe rather than fail open:

  CUDA_VISIBLE_DEVICES names a disallowed card  ->  os._exit(97) at import. A .pth hook swallows
      exceptions (Python prints "Remainder of file ignored" and carries on), so a raise here is
      NOT a refusal -- verified the hard way. Exit code 97 is the guard's signature.

  CUDA_VISIBLE_DEVICES is unset (i.e. every card visible, so cuda:0 == physical 0)  ->  pinned to
      "" so the process is CPU-only. A CPU tool is unaffected; a GPU tool fails with a plain
      "no CUDA device" instead of quietly taking card 0. No monkeypatching, so nothing that works
      today can break in a surprising way.

Override with MMRAG_GPU_ALLOW: a comma list ("5,7"), or "any" to disable. Allocation policy only;
touches no numerics. See also mmrag_memcap.py, which caps per-process memory on the same box.
"""
import os
import sys

_ALLOW_RAW = os.environ.get("MMRAG_GPU_ALLOW", "4,5,6,7")

if _ALLOW_RAW.strip().lower() not in ("any", "all"):
    _allow = {p.strip() for p in _ALLOW_RAW.split(",") if p.strip()}
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

    if _cvd is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.stderr.write(
            "[mmrag_gpuguard] CUDA_VISIBLE_DEVICES was unset, so cuda:0 would be PHYSICAL card 0, "
            "which belongs to another job.\n"
            f"                 Pinned to CPU-only. Allowed cards: {sorted(_allow)}. "
            "Set CUDA_VISIBLE_DEVICES=<card> to use a GPU,\n"
            "                 or launch via run_cell.sh / run_eval.sh, which pin it for you. "
            "MMRAG_GPU_ALLOW=any disables this.\n")
    elif _cvd.strip() != "":
        _bad = sorted({p.strip() for p in _cvd.split(",") if p.strip()} - _allow)
        if _bad:
            sys.stderr.write(
                f"\n[mmrag_gpuguard] REFUSING TO START: CUDA_VISIBLE_DEVICES names disallowed "
                f"GPU(s) {_bad}\n"
                f"  allowed physical GPUs : {sorted(_allow)}\n"
                f"  CUDA_VISIBLE_DEVICES  : {_cvd!r}\n"
                "  Cards 0-3 carry other people's jobs. Override with MMRAG_GPU_ALLOW=any.\n\n")
            sys.stderr.flush()
            # A .pth hook SWALLOWS exceptions -- Python prints "Remainder of file ignored" and the
            # interpreter continues, so `raise` here does not stop anything (verified: a raise let
            # CUDA_VISIBLE_DEVICES=0 run to completion). os._exit is the only refusal a .pth can
            # actually enforce. Hard by design: proceeding onto a stranger's card is the failure
            # this guard exists to prevent.
            os._exit(97)
