"""BTKD loss on OpenVLA action tokens (pure PyTorch, no VLA dependency).

Paper route (as implemented in ``tool.py`` / ``dis_resnet18_cf100.py``):

    1. teacher forward       -> y_tea
    2. mask = argmax(y_tea) == label          # easy / hard split
    3. easy: KL(student || teacher)
    4. hard: KL(student || calibrate3(y_tea))
    5. b = cos^2(pi/2 * t / T);  loss = lam*CE + (1-lam)*T^2*(b*easy + (1-b)*hard)

This module keeps the arithmetic identical to ``calibrate3`` but vectorises it over
``[batch, n_dims, n_bins]`` action logits and makes every masked reduction
empty-safe (the original code yields NaN when a batch has no easy/hard samples).

Set semantics: we distinguish three cases per token
    * teacher argmax outside the action token range  -> always "hard"
      (an out-of-space argmax is by definition not a valid action)
    * teacher argmax inside range and == gt token    -> "easy"
    * teacher argmax inside range and != gt token    -> "hard"
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .spec import ActionSpec, BTKDConfig

__all__ = [
    "cosine_square_schedule",
    "cosine_square_schedule_hold",
    "split_easy_hard",
    "calibrate3_logits",
    "per_token_kl",
    "masked_mean",
    "branch_weights",
    "btkd_term",
    "teacher_token_accuracy",
]


# --------------------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------------------
def cosine_square_schedule(step: int, total_steps: int) -> float:
    """b(t) = cos^2(pi/2 * t / T).  1.0 at t=0 (all easy) -> 0.0 at t=T (all hard)."""
    if total_steps <= 0:
        return 1.0
    frac = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return math.cos((math.pi / 2.0) * frac) ** 2


def cosine_square_schedule_hold(
    step: int, total_steps: int, hold_frac: float = 0.0
) -> float:
    """``b(t)`` with an initial easy plateau.

    ``hold_frac=0`` reproduces :func:`cosine_square_schedule` (the paper's raw
    schedule).  ``hold_frac=0.4`` keeps ``b == 1`` for the first 40% of training -
    i.e. the student first has to actually absorb the teacher - and only then
    decays to ``b == 0`` (calibrate3 / hard branch) over the remaining 60%.

    Rationale: with a short budget the raw schedule spends the tail fitting
    ``calibrate3(teacher)``, which is *deliberately* off the ground-truth
    distribution, so a student that has not converged on the teacher yet loses
    ground-truth accuracy in the last third of the run.
    """
    if total_steps <= 0:
        return 1.0
    hold = min(max(float(hold_frac), 0.0), 0.95)
    frac = min(max(float(step) / float(total_steps), 0.0), 1.0)
    if frac <= hold:
        return 1.0
    span = 1.0 - hold
    if span <= 1e-9:
        return 0.0
    return math.cos((math.pi / 2.0) * ((frac - hold) / span)) ** 2


# --------------------------------------------------------------------------------------
# Easy / hard split
# --------------------------------------------------------------------------------------
def split_easy_hard(
    teacher_argmax_token: torch.Tensor,
    gt_token: torch.Tensor,
    spec: ActionSpec,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split token positions into easy / hard.

    Args:
        teacher_argmax_token: ``[B, D]`` int64, argmax over the *full* vocabulary.
        gt_token: ``[B, D]`` int64, ground-truth action token ids.

    Returns:
        easy_mask, hard_mask, gt_col, wrong_col, in_space -- all ``[B, D]``.
        ``wrong_col`` is only meaningful where ``easy_mask`` is False; where the
        teacher argmax fell outside the action subspace it is filled with the
        argmax over the action slice at call time (see :func:`calibrate3_logits`).
    """
    if teacher_argmax_token.shape != gt_token.shape:
        raise ValueError(
            f"shape mismatch: teacher_argmax_token {tuple(teacher_argmax_token.shape)} "
            f"vs gt_token {tuple(gt_token.shape)}"
        )

    in_space = spec.in_space(teacher_argmax_token)
    gt_col = spec.col_of_token(gt_token)
    wrong_col = torch.where(
        in_space, spec.col_of_token(teacher_argmax_token), torch.full_like(gt_token, -1)
    )

    if (gt_col < 0).any() or (gt_col >= spec.n_bins).any():
        raise ValueError(
            "ground-truth action tokens fall outside the action subspace "
            f"[{spec.offset}, {spec.vocab_size}); check vocabulary size / bins"
        )

    easy_mask = in_space & (wrong_col == gt_col)
    hard_mask = ~easy_mask
    return easy_mask, hard_mask, gt_col, wrong_col, in_space


# --------------------------------------------------------------------------------------
# calibrate3
# --------------------------------------------------------------------------------------
def calibrate3_logits(
    teacher_logits: torch.Tensor,
    gt_col: torch.Tensor,
    wrong_col: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Vectorised ``tool.calibrate3`` over the last action-logit dimension.

    Reference (per sample, on raw logits)::

        a, c = y[l_idx], y[y_idx]                 # c >= a, y_idx == teacher argmax
        sum1 = a + c
        a1, c1 = (a + 1) / 2, c / 2
        sum2 = a1 + c1
        scale = sum1 / sum2
        y[l_idx], y[y_idx] = a1 * scale, c1 * scale

    which is exactly::

        a2 = a + c / (a + c + 1)
        c2 = c - c / (a + c + 1)

    i.e. the gap is narrowed while ``a2 + c2 == a + c`` is preserved.  ``strength``
    scales the transfer (1.0 == paper, 0.0 == no calibration).

    Fully out-of-place: the input tensor is not modified.
    """
    if teacher_logits.dim() != 3 or gt_col.shape != teacher_logits.shape[:2] or wrong_col.shape != gt_col.shape:
        raise ValueError(
            f"gt_col/wrong_col must be [B, D] matching logits [B, D, n_bins]; got "
            f"logits {tuple(teacher_logits.shape)}, gt_col {tuple(gt_col.shape)}, "
            f"wrong_col {tuple(wrong_col.shape)}"
        )

    # Rows whose teacher argmax was outside the action subspace: use the slice argmax.
    fallback = teacher_logits.argmax(dim=-1)
    wrong_col = torch.where(wrong_col < 0, fallback, wrong_col)
    # A "wrong" column equal to the gt column cannot happen for hard rows, but guard anyway.
    wrong_col = torch.where(wrong_col == gt_col, (gt_col + 1) % teacher_logits.size(-1), wrong_col)

    dtype = teacher_logits.dtype
    work = teacher_logits.float()

    a = work.gather(-1, gt_col.unsqueeze(-1)).squeeze(-1)
    c = work.gather(-1, wrong_col.unsqueeze(-1)).squeeze(-1)

    denom = a + c + 1.0
    delta = torch.where(
        denom.abs() < 1e-6,
        torch.zeros_like(denom),
        float(strength) * c / denom,
    )
    a2 = a + delta
    c2 = c - delta

    out = work.clone()
    out.scatter_(-1, gt_col.unsqueeze(-1), a2.unsqueeze(-1))
    out.scatter_(-1, wrong_col.unsqueeze(-1), c2.unsqueeze(-1))
    return out.to(dtype)


# --------------------------------------------------------------------------------------
# Distillation primitives
# --------------------------------------------------------------------------------------
def _normalize_logits(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "center":
        return x - x.mean(dim=-1, keepdim=True)
    if mode == "std":
        return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    raise ValueError(f"unknown logit_norm mode: {mode}")


def per_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    logit_norm: str = "none",
    apply_t2: bool = True,
) -> torch.Tensor:
    """KL(teacher || student) per token position, shape ``[B, D]`` (no reduction)."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student logits {tuple(student_logits.shape)} vs teacher {tuple(teacher_logits.shape)}"
        )
    s = _normalize_logits(student_logits.float(), logit_norm)
    t_logits = _normalize_logits(teacher_logits.float(), logit_norm)
    tau = float(temperature)
    kl = F.kl_div(
        F.log_softmax(s / tau, dim=-1), F.softmax(t_logits / tau, dim=-1), reduction="none"
    )
    kl = kl.sum(dim=-1)
    if apply_t2:
        kl = kl * (tau**2)
    return kl


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over ``mask``; returns a *zero* (gradient-connected) tensor when empty."""
    if values.shape != mask.shape:
        raise ValueError(f"values {tuple(values.shape)} vs mask {tuple(mask.shape)}")
    n = int(mask.sum().item())
    if n == 0:
        return values.sum() * 0.0
    return values[mask].mean()


def branch_weights(
    n_easy: int,
    n_hard: int,
    b: float,
    renormalize: bool = False,
) -> Tuple[float, float]:
    """Weights for the easy / hard branches."""
    if not renormalize or (n_easy > 0 and n_hard > 0):
        return float(b), float(1.0 - b)
    if n_easy > 0:
        return 1.0, 0.0
    if n_hard > 0:
        return 0.0, 1.0
    return 0.0, 0.0


def btkd_term(
    student_action_logits: torch.Tensor,
    teacher_action_logits: torch.Tensor,
    gt_token: torch.Tensor,
    teacher_argmax_token: torch.Tensor,
    spec: ActionSpec,
    cfg: BTKDConfig,
    b: float,
) -> Dict[str, torch.Tensor]:
    """BTKD distillation term (without the student CE part).

    Args:
        student_action_logits: ``[B, D, n_bins]`` live logits of the student.
        teacher_action_logits: ``[B, D, n_bins]`` cached teacher logits (fp16 ok).
        gt_token: ``[B, D]`` ground-truth action token ids.
        teacher_argmax_token: ``[B, D]`` teacher argmax over the full vocabulary.

    Returns dict with ``loss`` (scalar), ``easy``, ``hard`` (scalar, weighted),
    ``n_easy``, ``n_hard``, ``easy_frac`` for logging.
    """
    if student_action_logits.size(-1) != spec.n_bins:
        raise ValueError(
            f"student action logits last dim {student_action_logits.size(-1)} != n_bins {spec.n_bins}"
        )
    if teacher_action_logits.size(-1) != spec.n_bins:
        raise ValueError(
            f"teacher action logits last dim {teacher_action_logits.size(-1)} != n_bins {spec.n_bins}"
        )

    easy_mask, hard_mask, gt_col, wrong_col, _ = split_easy_hard(teacher_argmax_token, gt_token, spec)

    # ---- easy: plain KD against the unmodified teacher -------------------------------
    kl_easy = per_token_kl(
        student_action_logits, teacher_action_logits, cfg.temperature, cfg.logit_norm, cfg.apply_t2
    )
    easy_raw = masked_mean(kl_easy, easy_mask)

    # ---- hard: KD against the calibrated teacher -------------------------------------
    calibrated = calibrate3_logits(teacher_action_logits.float(), gt_col, wrong_col, cfg.calibrate_strength)
    kl_hard = per_token_kl(
        student_action_logits, calibrated, cfg.temperature, cfg.logit_norm, cfg.apply_t2
    )
    hard_raw = masked_mean(kl_hard, hard_mask)

    n_easy = int(easy_mask.sum().item())
    n_hard = int(hard_mask.sum().item())
    w_easy, w_hard = branch_weights(n_easy, n_hard, b, cfg.renormalize_empty_branch)

    loss = w_easy * easy_raw + w_hard * hard_raw
    total = n_easy + n_hard
    return {
        "loss": loss,
        "easy": w_easy * easy_raw,
        "hard": w_hard * hard_raw,
        "easy_raw": easy_raw,
        "hard_raw": hard_raw,
        "n_easy": n_easy,
        "n_hard": n_hard,
        "easy_frac": (n_easy / total) if total else 0.0,
        "b": b,
    }


def teacher_token_accuracy(teacher_argmax_token: torch.Tensor, gt_token: torch.Tensor) -> float:
    """Diagnostic: fraction of action tokens the teacher predicts correctly."""
    if teacher_argmax_token.numel() == 0:
        return 0.0
    return float((teacher_argmax_token == gt_token).float().mean().item())


def total_loss(
    ce_loss: torch.Tensor,
    kd: Dict[str, torch.Tensor],
    cfg: BTKDConfig,
) -> torch.Tensor:
    """lambda * CE + (1 - lambda) * KD  (T^2 already folded into `per_token_kl`)."""
    return cfg.lambda_stu * ce_loss + (1.0 - cfg.lambda_stu) * kd["loss"]
