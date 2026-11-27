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
import tempfile
from pathlib import Path


_WANDB_STUB_SRC = '''"""No-op stand-in for Weights & Biases (BTKD offline LIBERO evaluation)."""

__version__ = "0.0.0"
__all__ = ["init", "log", "save", "finish", "watch"]


class _NoOp:
    """Absorbs any attribute access or call so ``wandb.xxx(...)`` does nothing."""

    def __init__(self, name="wandb"):
        self._name = name

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, item):
        return _NoOp(self._name + "." + item)

    def __iter__(self):
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __bool__(self):
        return False

    def __repr__(self):
        return "<no-op %s>" % self._name


def __getattr__(name):  # PEP 562: any unknown attribute becomes a no-op
    return _NoOp("wandb." + name)


def init(*args, **kwargs):
    return None


def log(*args, **kwargs):
    return None


def save(*args, **kwargs):
    return None


def finish(*args, **kwargs):
    return None
'''


def install_wandb_stub() -> None:
    """Make ``import wandb`` work without Weights & Biases.

    The upstream evaluator imports wandb unconditionally but only uses it inside
    ``if cfg.use_wandb:`` blocks (default False).  A real module on disk is used as the
    stub so that ``__file__``/``__spec__`` exist -- tooling such as ``inspect.getfile``
    raises ``TypeError: ... is a built-in module`` for a bare ``ModuleType``.

    Set ``BTKD_USE_WANDB=1`` to keep the real library instead.
    """
    if os.environ.get("BTKD_USE_WANDB", "0") == "1":
        print("[libero_eval_entry] BTKD_USE_WANDB=1 -> using the real wandb library")
        return
    if "wandb" in sys.modules:
        return

    stub_dir = Path(tempfile.mkdtemp(prefix="btkd_wandb_stub_"))
    pkg = stub_dir / "wandb"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(_WANDB_STUB_SRC, encoding="utf-8")
    sys.path.insert(0, str(stub_dir))
    import wandb  # noqa: F401  (resolves to the stub because stub_dir comes first)

    print(f"[libero_eval_entry] wandb disabled (BTKD_USE_WANDB=0) -> no-op stub at {pkg}")


def install_action_debug() -> None:
    """BTKD_DEBUG_ACTIONS=1 -> print the first actions the policy emits, plus its stats.

    A frozen arm in LIBERO has two very different causes and this print separates them:

    * ``action`` is identical (or ~0) every step  -> the policy itself is degenerate
      (merged weights broken, or the student only learned the marginal action mode);
    * ``action`` changes but the arm does not move -> the simulator/harness is at fault
      (wrong unnorm_key, clipped actions, bad camera input).
    """
    if os.environ.get("BTKD_DEBUG_ACTIONS", "0") != "1":
        return

    import numpy as np

    from experiments.robot import robot_utils

    original = robot_utils.get_action
    n_dump = int(os.environ.get("BTKD_DEBUG_ACTIONS_N", "25"))
    seen = {"i": 0}

    def wrapped(cfg, model, obs, task_label, processor=None):
        action = original(cfg, model, obs, task_label, processor=processor)
        i = seen["i"]
        seen["i"] += 1
        if i < n_dump:
            print(
                f"[act {i:03d}] action={np.round(action, 4).tolist()} "
                f"eef={np.round(obs['state'][:3], 4).tolist()}",
                flush=True,
            )
            if i == 0:
                print(f"[act] checkpoint={cfg.pretrained_checkpoint}", flush=True)
                print(f"[act] unnorm_key={cfg.unnorm_key} center_crop={cfg.center_crop}", flush=True)
                try:
                    stats = model.get_action_stats(cfg.unnorm_key)
                    q01 = np.round(stats["q01"], 4).tolist()
                    q99 = np.round(stats["q99"], 4).tolist()
                    print(f"[act] q01={q01}", flush=True)
                    print(f"[act] q99={q99}", flush=True)
                    print(
                        "[act] q99-q01=" + str([round(b - a, 4) for a, b in zip(q01, q99)]),
                        flush=True,
                    )
                except Exception as exc:  # stats are only used for debugging
                    print(f"[act] could not read norm stats: {exc}", flush=True)
        return action

    robot_utils.get_action = wrapped
    print(f"[libero_eval_entry] action debug on: dumping the first {n_dump} actions")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    openvla = root / "openvla"
    if not (openvla / "experiments/robot/libero/run_libero_eval.py").is_file():
        raise SystemExit(f"OpenVLA source not found under {openvla}")

    os.chdir(openvla)
    sys.path.insert(0, str(openvla))
    install_wandb_stub()
    install_action_debug()

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
