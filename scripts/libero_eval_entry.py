#!/usr/bin/env python
"""Run OpenVLA's LIBERO evaluator with SDPA attention instead of hard-coded flash-attn.

OpenVLA's stock ``experiments/robot/openvla_utils.py`` hard-codes
``attn_implementation="flash_attention_2"``.  The BTKD environment intentionally does
not require flash-attn, so this launcher patches that one argument at runtime and then
executes the upstream evaluator unchanged.

The upstream evaluator also does an unconditional ``import wandb`` even though W&B is
only touched when ``--use_wandb True``.  Weights & Biases is not required here, so a
no-op stand-in is installed unless ``BTKD_USE_WANDB=1`` is set.  All progress and the
final success rate still go to the local ``experiments/logs/EVAL-*.txt`` file.

Put this file's arguments exactly where you would put ``run_libero_eval.py``'s args.
"""

from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path


class _NoOp:
    """Absorbs any attribute access/call so ``wandb.xxx(...)`` becomes a no-op."""

    def __init__(self, name: str = "wandb") -> None:
        self._name = name

    def __call__(self, *args, **kwargs) -> "_NoOp":
        return self

    def __getattr__(self, item: str) -> "_NoOp":
        return _NoOp(f"{self._name}.{item}")

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<no-op {self._name}>"


def install_wandb_stub() -> None:
    """Make ``import wandb`` work without Weights & Biases.

    W&B is only used inside ``if cfg.use_wandb:`` blocks, so a stub loses nothing for
    offline evaluation.  Set ``BTKD_USE_WANDB=1`` to use the real library instead.
    """
    if os.environ.get("BTKD_USE_WANDB", "0") == "1":
        return
    try:
        import wandb  # noqa: F401
        note = "real wandb found but disabled (BTKD_USE_WANDB=0)"
    except Exception:
        note = "wandb not installed"
    module = types.ModuleType("wandb")
    module.__getattr__ = lambda name: _NoOp(f"wandb.{name}")  # PEP 562
    module.__all__ = []
    sys.modules["wandb"] = module
    print(f"[libero_eval_entry] {note} -> using a no-op stub; metrics go to the local EVAL log")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    openvla = root / "openvla"
    if not (openvla / "experiments/robot/libero/run_libero_eval.py").is_file():
        raise SystemExit(f"OpenVLA source not found under {openvla}")

    os.chdir(openvla)
    sys.path.insert(0, str(openvla))
    install_wandb_stub()

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
