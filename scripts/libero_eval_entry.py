#!/usr/bin/env python
"""Run OpenVLA's LIBERO evaluator with SDPA attention instead of hard-coded flash-attn.

OpenVLA's stock ``experiments/robot/openvla_utils.py`` hard-codes
``attn_implementation="flash_attention_2"``.  The BTKD environment intentionally does
not require flash-attn, so this launcher patches that one argument at runtime and then
executes the upstream evaluator unchanged.

Put this file's arguments exactly where you would put ``run_libero_eval.py``'s args.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    openvla = root / "openvla"
    if not (openvla / "experiments/robot/libero/run_libero_eval.py").is_file():
        raise SystemExit(f"OpenVLA source not found under {openvla}")

    os.chdir(openvla)
    sys.path.insert(0, str(openvla))

    from transformers import AutoModelForVision2Seq

    original = AutoModelForVision2Seq.from_pretrained

    def from_pretrained(*args, **kwargs):
        kwargs["attn_implementation"] = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
        return original(*args, **kwargs)

    AutoModelForVision2Seq.from_pretrained = from_pretrained
    runpy.run_path(
        str(openvla / "experiments/robot/libero/run_libero_eval.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
