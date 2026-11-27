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

Diagnostics
-----------
``BTKD_DEBUG_ACTIONS=1`` records the policy's raw output and prints a verdict when the
run ends: per-dimension action magnitude/variance, end-effector displacement, and the
fraction of generated tokens that fall outside the action vocabulary.  Combined with
``BTKD_DEBUG_STEPS=N`` the rollout is cut short after N model calls, so a frozen arm can
be diagnosed in ~2 minutes instead of a full episode.

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


# --------------------------------------------------------------------------------------
# Policy diagnostics (BTKD_DEBUG_ACTIONS=1)
# --------------------------------------------------------------------------------------
class DiagnoseStop(BaseException):
    """Abort the rollout once enough steps were sampled (``BTKD_DEBUG_STEPS``).

    Subclasses ``BaseException`` on purpose: the upstream eval loop wraps each step in
    ``except Exception`` and would otherwise swallow it.
    """


_DIAG = {
    "actions": [],
    "eef": [],
    "tokens": [],
    "last_tokens": None,
    "q01": None,
    "q99": None,
    "ckpt": None,
    "unnorm_key": None,
    "center_crop": None,
    "limit": 0,
    "vocab": None,
    "action_dim": 7,
}


def _diag_on() -> bool:
    return os.environ.get("BTKD_DEBUG_ACTIONS", "0") == "1"


def _wrap_generate(model) -> None:
    """Record the last ``n_action_dim`` generated token ids, then defer to the original."""
    original = model.generate

    def generate(*args, **kwargs):
        out = original(*args, **kwargs)
        try:
            n = int(_DIAG["action_dim"])
            _DIAG["last_tokens"] = [int(t) for t in out[0, -n:].tolist()]
        except Exception:  # never break evaluation because of bookkeeping
            _DIAG["last_tokens"] = None
        return out

    model.generate = generate


def install_action_debug() -> None:
    """Patch ``get_action`` so every policy call is recorded and can be summarised."""
    if not _diag_on():
        return

    import numpy as np

    from experiments.robot import robot_utils

    _DIAG["limit"] = int(os.environ.get("BTKD_DEBUG_STEPS", "30"))
    n_dump = int(os.environ.get("BTKD_DEBUG_ACTIONS_N", "10"))
    original = robot_utils.get_action

    def wrapped(cfg, model, obs, task_label, processor=None):
        if len(_DIAG["actions"]) >= _DIAG["limit"] > 0:
            raise DiagnoseStop()

        action = original(cfg, model, obs, task_label, processor=processor)
        action = np.asarray(action, dtype=np.float64)

        _DIAG["actions"].append(action.copy())
        _DIAG["eef"].append(np.asarray(obs["state"][:3], dtype=np.float64).copy())
        _DIAG["tokens"].append(_DIAG["last_tokens"])

        if _DIAG["ckpt"] is None:
            _DIAG["ckpt"] = str(cfg.pretrained_checkpoint)
            _DIAG["unnorm_key"] = str(getattr(cfg, "unnorm_key", "?"))
            _DIAG["center_crop"] = bool(getattr(cfg, "center_crop", False))
            try:
                stats = model.get_action_stats(cfg.unnorm_key)
                _DIAG["q01"] = np.asarray(stats["q01"], dtype=np.float64)
                _DIAG["q99"] = np.asarray(stats["q99"], dtype=np.float64)
                _DIAG["action_dim"] = len(stats["q01"])
            except Exception as exc:  # stats are only used for the verdict
                print(f"[diag] could not read norm stats: {exc}", flush=True)
            try:
                _DIAG["vocab"] = int(model.vocab_size)
            except Exception:
                _DIAG["vocab"] = None

        i = len(_DIAG["actions"]) - 1
        if i < n_dump:
            print(
                f"[act {i:03d}] action={np.round(action, 4).tolist()} "
                f"eef={np.round(obs['state'][:3], 4).tolist()} "
                f"tokens={_DIAG['last_tokens']}",
                flush=True,
            )
        return action

    robot_utils.get_action = wrapped
    print(
        f"[libero_eval_entry] action debug on: {_DIAG['limit']} steps, "
        f"dumping the first {n_dump}",
        flush=True,
    )


def print_verdict() -> None:
    """Summarise the recorded rollout and name the most likely failure mode."""
    import numpy as np

    acts = np.asarray(_DIAG["actions"])
    if acts.size == 0:
        print("[diag] no actions were recorded", flush=True)
        return
    eef = np.asarray(_DIAG["eef"])

    print("\n" + "=" * 78)
    print("BTKD policy diagnosis")
    print("=" * 78)
    print(f"checkpoint : {_DIAG['ckpt']}")
    print(f"unnorm_key : {_DIAG['unnorm_key']}   center_crop={_DIAG['center_crop']}")
    if _DIAG["q01"] is not None:
        print(f"q01        : {np.round(_DIAG['q01'], 3).tolist()}")
        print(f"q99        : {np.round(_DIAG['q99'], 3).tolist()}")
        print(f"q99-q01    : {np.round(_DIAG['q99'] - _DIAG['q01'], 3).tolist()}")
    print(f"steps      : {len(acts)}")

    mean_abs = np.abs(acts).mean(axis=0)
    std = acts.std(axis=0)
    print(f"mean |a|   : {np.round(mean_abs, 4).tolist()}")
    print(f"std(a)     : {np.round(std, 4).tolist()}")
    print(f"first      : {np.round(acts[0], 4).tolist()}")
    print(f"last       : {np.round(acts[-1], 4).tolist()}")
    uniq = len({tuple(np.round(a, 4)) for a in acts})
    print(f"distinct   : {uniq}/{len(acts)} distinct action vectors")

    move = float(np.linalg.norm(eef[-1] - eef[0])) if len(eef) > 1 else 0.0
    path = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()) if len(eef) > 1 else 0.0
    print(f"eef        : start={np.round(eef[0], 4).tolist()} end={np.round(eef[-1], 4).tolist()}")
    print(f"eef move   : net {move:.4f} m, path {path:.4f} m")

    oos = None
    toks = [t for t in _DIAG["tokens"] if t]
    vocab = _DIAG["vocab"]
    if toks and vocab:
        lo = vocab - 256
        bad = sum(1 for row in toks for t in row if not (lo <= t <= vocab))
        oos = bad / float(sum(len(row) for row in toks))
        print(f"tokens     : {len(toks)} steps, first={toks[0]}")
        print(f"out-of-space action tokens: {oos * 100:.1f}%  (valid range {lo}..{vocab})")

    print("-" * 78)

    xyz_abs = float(np.abs(acts[:, :3]).mean())
    xyz_std = float(acts[:, :3].std(axis=0).max())

    if oos is not None and oos > 0.02:
        print("[VERDICT] 模型生成了动作词表之外的 token（乱码）")
        print("  -> 词表外的 token 会被 clip 成极端 bin，策略等于瞎走；先确认 merged 权重没坏：")
        print("     python scripts/eval_btkd_student.py --cache_dir caches/libero_spatial_teacher \\")
        print("       --dataset_name libero_spatial_no_noops --limit 256 --device cuda:0 \\")
        print("       --merged_path merged/btkd_spatial_best")
        print("     若离线分数远低于 adapter（~0.48），用 --rebuild-merge 重做 merge。")
    elif xyz_std < 1e-3 and xyz_abs < 5e-3:
        print("[VERDICT] 策略输出的动作恒定且 ≈ 0（模型退化，不是仿真问题）")
        print("  -> 学生只学到了动作的边缘分布：每步都吐同一个“平均动作”，臂自然不动。")
        print("  -> 训练侧改：--temperature 1.0、--lambda_stu 0.7、--epochs 2；先试 final/。")
    elif move < 0.01 and xyz_abs > 0.05:
        print("[VERDICT] 动作幅度正常但末端几乎没动（仿真侧没执行动作）")
        print("  -> 查 unnorm_key / dataset_statistics.json / 控制器剪裁；")
        print("     也确认 --center-crop 与学生训练时一致（BTKD 用 false）。")
    elif xyz_std < 5e-3:
        print("[VERDICT] 动作几乎不随时间变化（准静态输出），臂只会缓慢漂移")
        print("  -> 同“模型退化”：回到训练侧（温度/CE 权重/epoch），或换 final/。")
    else:
        print("[VERDICT] 策略在动，动作随时间变化正常")
        print("  -> 冻结不是 bug：跑完整 10x50（--trials 50）看真实成功率；")
        print("     若成功率为 0，说明 token 级指标（0.48）没有转化为闭环能力。")
    print("=" * 78 + "\n", flush=True)


def _fix_relative_checkpoint(root: Path) -> None:
    """Resolve a relative --pretrained_checkpoint against the repo root.

    ``run_libero_sim.sh`` starts this file after ``cd openvla/``, so a checkpoint such as
    ``merged/btkd_spatial_final`` no longer resolves and Transformers reports
    "is not a local folder and is not a valid model identifier".
    """
    for i, tok in enumerate(sys.argv[:-1]):
        if tok != "--pretrained_checkpoint":
            continue
        val = sys.argv[i + 1]
        if not val or os.path.isabs(val) or os.path.exists(val):
            continue
        candidate = root / val
        if candidate.exists():
            sys.argv[i + 1] = str(candidate)
            print(f"[libero_eval_entry] resolved --pretrained_checkpoint -> {candidate}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    openvla = root / "openvla"
    if not (openvla / "experiments/robot/libero/run_libero_eval.py").is_file():
        raise SystemExit(f"OpenVLA source not found under {openvla}")

    _fix_relative_checkpoint(root)
    os.chdir(openvla)
    sys.path.insert(0, str(openvla))
    install_wandb_stub()
    install_action_debug()

    from transformers import AutoModelForVision2Seq

    original = AutoModelForVision2Seq.from_pretrained

    def from_pretrained(*args, **kwargs):
        kwargs["attn_implementation"] = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
        model = original(*args, **kwargs)
        if _diag_on():
            _wrap_generate(model)
        return model

    AutoModelForVision2Seq.from_pretrained = from_pretrained
    try:
        runpy.run_path(
            str(openvla / "experiments/robot/libero/run_libero_eval.py"),
            run_name="__main__",
        )
    except DiagnoseStop:
        pass
    finally:
        if _diag_on():
            print_verdict()


if __name__ == "__main__":
    main()
