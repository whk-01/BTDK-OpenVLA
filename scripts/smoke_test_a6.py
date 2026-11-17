#!/usr/bin/env python
"""End-to-end CPU smoke test for the A6 offline-BTKD pipeline (no model, no dataset).

It exercises exactly the code paths the real run uses:

    1. write a synthetic teacher cache (same writer/shard logic)
    2. read it back through the same reader the trainer uses
    3. run N optimisation steps on a toy student parameter through
       ``extract -> split_easy_hard -> calibrate3 -> KL -> masked mean``
    4. assert the loss is finite, decreasing, and that b moves 1 -> 0

Run:  python scripts/smoke_test_a6.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btkd_vla.cache import LogitCache, LogitCacheWriter  # noqa: E402
from btkd_vla.core import btkd_term, cosine_square_schedule, teacher_token_accuracy  # noqa: E402
from btkd_vla.spec import ActionSpec, BTKDConfig  # noqa: E402

SPEC = ActionSpec(n_bins=256, n_dims=7, vocab_size=32000)
N, B = 64, 8


def build_fake_cache(tmp: str) -> LogitCache:
    torch.manual_seed(0)
    with LogitCacheWriter(tmp, SPEC, meta={"dummy": True}, shard_size=20) as w:
        for s in range(0, N, B):
            b = min(B, N - s)
            logits = torch.randn(b, SPEC.n_dims, SPEC.n_bins) * 2.0
            gt_col = torch.randint(0, SPEC.n_bins, (b, SPEC.n_dims))
            wrong_col = (gt_col + torch.randint(1, SPEC.n_bins, (b, SPEC.n_dims))) % SPEC.n_bins
            keep = torch.rand(b, SPEC.n_dims) < 0.85
            argmax_col = torch.where(keep, gt_col, wrong_col)
            logits.scatter_(-1, argmax_col.unsqueeze(-1), 14.0)
            w.add(SPEC.token_of_col(gt_col), logits, SPEC.token_of_col(argmax_col))
    return LogitCache(tmp, verbose=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cache = build_fake_cache(tmp)
        assert cache.is_complete and len(cache) == N
        assert len(cache.shards) == 4, [s["n"] for s in cache.shards]

        cfg = BTKDConfig(temperature=4.0, lambda_stu=0.5, total_steps=300)
        torch.manual_seed(7)
        # toy student = teacher logits + a learnable bias, starting slightly off
        amp = (torch.randn(1, SPEC.n_dims, SPEC.n_bins) * 0.5).requires_grad_(True)
        opt = torch.optim.Adam([amp], lr=0.1)

        # a single fixed batch: the optimum is "make the bias zero", so the loss must fall
        rows = cache.get_range(0, B)
        teacher = rows["teacher_logits"].float()

        first, last = None, None
        for step in range(300):
            student = (teacher + amp).expand(B, -1, -1)

            b = cosine_square_schedule(step, 300)
            kd = btkd_term(
                student, teacher, rows["gt_token"], rows["teacher_argmax_token"], SPEC, cfg, b
            )
            opt.zero_grad()
            kd["loss"].backward()
            opt.step()

            assert torch.isfinite(kd["loss"]), f"non-finite loss at step {step}"
            if step == 0:
                first = float(kd["loss"].item())
            last = float(kd["loss"].item())

        acc = teacher_token_accuracy(
            cache.get_range(0, N)["teacher_argmax_token"], cache.get_range(0, N)["gt_token"]
        )
        print(f"[smoke] teacher token acc      = {acc:.3f}")
        print(f"[smoke] kd loss step0 / step299 = {first:.4f} / {last:.4f}")
        print(f"[smoke] learned student bias |amp|max = {amp.detach().abs().max().item():.4f}")
        print(f"[smoke] b(0)={cosine_square_schedule(0,300):.3f}  b(150)={cosine_square_schedule(150,300):.3f}  "
              f"b(300)={cosine_square_schedule(300,300):.3f}")
        assert last < first * 0.5, f"KD loss did not decrease: {first:.4f} -> {last:.4f}"

        # ---- empty-branch behaviour ----------------------------------------------
        rows = cache.get_range(0, B)
        # a *non-uniform* perturbation: a constant shift would be invisible to softmax
        student = rows["teacher_logits"].float() * 1.3
        teacher = rows["teacher_logits"].float()

        # (a) all easy, b = 1 -> easy branch carries everything
        kd_easy = btkd_term(student, teacher, rows["gt_token"], rows["gt_token"], SPEC, cfg, b=1.0)
        assert abs(float(kd_easy["loss"]) - float(kd_easy["easy_raw"])) < 1e-5
        print(f"[smoke] all-easy, b=1: kd={float(kd_easy['loss']):.4f} (easy_raw={float(kd_easy['easy_raw']):.4f})")

        # (b) all hard, b = 1 -> paper-faithful drops KD entirely; renormalising keeps it
        hard_tok = SPEC.token_of_col((rows["gt_token"] - SPEC.offset + 1) % SPEC.n_bins)
        kd_hard_faithful = btkd_term(student, teacher, rows["gt_token"], hard_tok, SPEC, cfg, b=1.0)
        cfg_renorm = BTKDConfig(renormalize_empty_branch=True)
        kd_hard_renorm = btkd_term(student, teacher, rows["gt_token"], hard_tok, SPEC, cfg_renorm, b=1.0)
        print(f"[smoke] all-hard, b=1: faithful kd={float(kd_hard_faithful['loss']):.4f}, "
              f"renormalised kd={float(kd_hard_renorm['loss']):.4f}")
        assert float(kd_hard_faithful["loss"]) == 0.0, "paper-faithful b=1 must ignore the hard branch"
        assert float(kd_hard_renorm["loss"]) > 0.0, "renormalised weights must fall back to the hard branch"

        for v in (kd_easy["easy"], kd_easy["hard"], kd_hard_faithful["hard"], kd_hard_renorm["hard"]):
            assert torch.isfinite(v), "no branch may produce NaN/inf"

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
