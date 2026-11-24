"""Shared offline metrics for BTKD-on-OpenVLA.

Both the standalone evaluator (``scripts/eval_btkd_student.py``) and the
in-training evaluator inside ``scripts/train_btkd_student.py`` use
:class:`MetricAccumulator`, so the numbers are directly comparable.

Everything here is teacher-free: the teacher distribution comes from the logit
cache, so evaluation only needs the student (one 7B model).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .spec import ActionSpec

__all__ = ["MetricAccumulator", "offline_eval_frames"]


class MetricAccumulator:
    """Accumulates action-token metrics over batches of ``[B, D, n_bins]`` logits."""

    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec
        self.n_seen = 0
        self.n_tokens = 0
        self.correct = 0
        self.teacher_correct = 0
        self.agree = 0
        self.agree_denom = 0
        self.teacher_out_of_space = 0
        self.easy_correct = 0
        self.easy_total = 0
        self.hard_correct = 0
        self.hard_total = 0
        self.ce_sum = 0.0
        self.dim_correct = torch.zeros(spec.n_dims, dtype=torch.long)
        self.dim_total = torch.zeros(spec.n_dims, dtype=torch.long)

    @torch.no_grad()
    def update(
        self,
        action_logits: torch.Tensor,
        gt_token: torch.Tensor,
        teacher_argmax_token: torch.Tensor,
        ce: Optional[float] = None,
    ) -> None:
        spec = self.spec
        gt_token = gt_token.detach().cpu()
        tea_tok = teacher_argmax_token.detach().cpu()
        stu_col = action_logits.detach().argmax(dim=-1).cpu()

        gt_col = spec.col_of_token(gt_token)
        tea_in = spec.in_space(tea_tok)
        tea_col = spec.col_of_token(tea_tok)

        matches_stu = stu_col == gt_col
        matches_tea = tea_tok == gt_token

        b = int(gt_token.size(0))
        d = int(gt_token.size(1))
        self.n_seen += b
        self.n_tokens += b * d
        self.correct += int(matches_stu.sum().item())
        self.teacher_correct += int(matches_tea.sum().item())
        self.teacher_out_of_space += int((~tea_in).sum().item())
        self.agree += int(((stu_col == tea_col) & tea_in).sum().item())
        self.agree_denom += int(tea_in.sum().item())

        self.easy_correct += int((matches_stu & matches_tea).sum().item())
        self.easy_total += int(matches_tea.sum().item())
        self.hard_correct += int((matches_stu & ~matches_tea).sum().item())
        self.hard_total += int((~matches_tea).sum().item())

        for dim in range(d):
            self.dim_correct[dim] += int(matches_stu[:, dim].sum().item())
            self.dim_total[dim] += b

        if ce is not None:
            self.ce_sum += float(ce) * b

    def result(self, **extra: Any) -> Dict[str, Any]:
        n_tokens = max(1, self.n_tokens)
        metrics: Dict[str, Any] = {
            "n_samples": self.n_seen,
            "n_tokens": self.n_tokens,
            "ce": self.ce_sum / max(1, self.n_seen),
            "student_token_acc": self.correct / n_tokens,
            "teacher_token_acc": self.teacher_correct / n_tokens,
            "student_teacher_agreement": (
                self.agree / self.agree_denom if self.agree_denom else 0.0
            ),
            "teacher_out_of_space_frac": self.teacher_out_of_space / n_tokens,
            "student_token_acc_easy": (
                self.easy_correct / self.easy_total if self.easy_total else 0.0
            ),
            "student_token_acc_hard": (
                self.hard_correct / self.hard_total if self.hard_total else 0.0
            ),
            "easy_frac": (self.easy_total / n_tokens) if self.n_tokens else 0.0,
            "per_dim_acc": [
                round(self.dim_correct[d].item() / max(1, self.dim_total[d].item()), 4)
                for d in range(self.spec.n_dims)
            ],
        }
        metrics.update(extra)
        return metrics


@torch.no_grad()
def offline_eval_frames(
    model,
    collator,
    store,
    cache,
    spec: ActionSpec,
    begin_idx: int,
    device: str,
    batch_size: int = 4,
    limit: int = 512,
    progress_every: int = 20,
) -> Dict[str, Any]:
    """Evaluate ``model`` on the first ``limit`` rows of a materialised frame store.

    Row ``i`` of the frame store and row ``i`` of the teacher cache are the same
    sample (both come from the deterministic RLDS order), so no replay or
    alignment check is needed here.
    """
    from .vla_glue import extract_action_logits, move_pixel_values

    available = min(len(store), len(cache))
    n = available if limit < 0 else min(int(limit), available)
    if n <= 0:
        raise ValueError("nothing to evaluate (frames/cache are empty)")

    acc = MetricAccumulator(spec)
    was_training = model.training
    model.eval()
    use_amp = str(device).startswith("cuda")
    try:
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            idx = torch.arange(start, stop, dtype=torch.long)
            batch = collator([store[int(i)] for i in idx])
            cached = cache.get_rows(idx)
            labels = batch["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=move_pixel_values(batch["pixel_values"], device, torch.bfloat16),
                    labels=labels,
                )
            action_logits, gt_token, _ = extract_action_logits(out.logits, labels, begin_idx, spec)
            acc.update(action_logits, gt_token, cached["teacher_argmax_token"], ce=float(out.loss.item()))
            if progress_every and start % (batch_size * progress_every) == 0:
                print(f"[eval] {stop}/{n} rows", flush=True)
    finally:
        if was_training:
            model.train()
    return acc.result()
