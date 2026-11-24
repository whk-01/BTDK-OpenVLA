"""Unit tests for the BTKD action-token math.

Run with:  python tests/test_core.py        (plain asserts, no pytest required)
       or: python -m pytest tests/test_core.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from itertools import islice

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btkd_vla.cache import (  # noqa: E402
    LogitCache,
    LogitCacheWriter,
    cache_covers_full_epoch,
)
from btkd_vla.core import (  # noqa: E402
    btkd_term,
    calibrate3_logits,
    cosine_square_schedule,
    cosine_square_schedule_hold,
    masked_mean,
    per_token_kl,
    split_easy_hard,
    teacher_token_accuracy,
    total_loss,
)
from btkd_vla.evaluate import MetricAccumulator  # noqa: E402
from btkd_vla.frames import FrameStore, FrameStoreWriter  # noqa: E402
from btkd_vla.spec import ActionSpec, BTKDConfig  # noqa: E402
from btkd_vla.vla_glue import extract_action_logits  # noqa: E402
from scripts.train_btkd_student import frames_batches, resolve_plan  # noqa: E402

SPEC = ActionSpec(n_bins=256, n_dims=7, vocab_size=32000)
torch.manual_seed(0)


# ---------------------------------------------------------------------------------
# reference implementation straight from tool.py:calibrate3
# ---------------------------------------------------------------------------------
def reference_calibrate3(logits: torch.Tensor, gt_col: torch.Tensor, wrong_col: torch.Tensor) -> torch.Tensor:
    y = logits.clone().double()
    b, d, _ = y.shape
    for i in range(b):
        for j in range(d):
            l_idx = int(gt_col[i, j])
            y_idx = int(wrong_col[i, j])
            sum1 = y[i, j, l_idx] + y[i, j, y_idx]
            y[i, j, l_idx] = (y[i, j, l_idx].clone() + 1) / 2
            y[i, j, y_idx] = (y[i, j, y_idx].clone() + 0) / 2
            sum2 = y[i, j, l_idx] + y[i, j, y_idx]
            y[i, j, l_idx] = y[i, j, l_idx].clone() * (sum1 / sum2)
            y[i, j, y_idx] = y[i, j, y_idx].clone() * (sum1 / sum2)
    return y


def test_calibrate3_matches_reference():
    logits = torch.randn(4, 7, 256) * 3.0
    gt = torch.randint(0, 256, (4, 7))
    wrong = (gt + torch.randint(1, 256, (4, 7))) % 256
    got = calibrate3_logits(logits, gt, wrong, strength=1.0).double()
    want = reference_calibrate3(logits, gt, wrong)
    err = (got - want).abs().max().item()
    assert err < 1e-4, f"calibrate3 mismatch, max abs err = {err}"
    print(f"  ok  calibrate3 == reference loop (max err {err:.2e})")


def test_calibrate3_preserves_sum_and_narrows_gap():
    logits = torch.randn(2, 7, 256) * 2.0
    gt = torch.randint(0, 256, (2, 7))
    wrong = (gt + torch.randint(1, 256, (2, 7))) % 256
    # make 'wrong' the actual argmax, as in the paper
    logits.scatter_(-1, wrong.unsqueeze(-1), 15.0)
    out = calibrate3_logits(logits, gt, wrong, 1.0)

    a0 = logits.gather(-1, gt.unsqueeze(-1)).squeeze(-1)
    c0 = logits.gather(-1, wrong.unsqueeze(-1)).squeeze(-1)
    a1 = out.gather(-1, gt.unsqueeze(-1)).squeeze(-1)
    c1 = out.gather(-1, wrong.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(a0 + c0, a1 + c1, atol=1e-4), "sum of the two logits must be preserved"
    assert torch.all(a1 > a0), "true-class logit must increase"
    assert torch.all(c1 < c0), "wrong-class logit must decrease"
    assert torch.all((c1 - a1) < (c0 - a0)), "gap must be narrowed"
    print("  ok  calibrate3 preserves a+c, raises gt, lowers wrong, narrows gap")


def test_calibrate3_strength_zero_is_identity():
    logits = torch.randn(2, 7, 256)
    gt = torch.randint(0, 256, (2, 7))
    wrong = (gt + 1) % 256
    out = calibrate3_logits(logits, gt, wrong, strength=0.0)
    assert torch.allclose(out, logits, atol=1e-6)
    print("  ok  calibrate_strength=0 is a no-op")


def test_split_masks():
    tok = torch.tensor([[32000 - 1, 31744 + 5], [32000 - 3, 32000 - 7]])
    gt = torch.tensor([[32000 - 1, 31744 + 5], [32000 - 3, 32000 - 9]])
    easy, hard, gt_col, wrong_col, in_space = split_easy_hard(tok, gt, SPEC)
    assert easy.tolist() == [[True, True], [True, False]]
    assert hard.tolist() == [[False, False], [False, True]]
    assert in_space.all()

    # argmax outside the action subspace -> forced hard, wrong_col == -1 (fallback later)
    tok2 = torch.tensor([[1234] * 7])
    gt2 = torch.tensor([[32000 - 5] * 7])
    easy2, hard2, _, wrong2, in2 = split_easy_hard(tok2, gt2, SPEC)
    assert not easy2.any() and hard2.all()
    assert not in2.any() and (wrong2 == -1).all()
    print("  ok  easy/hard split incl. out-of-space argmax")


def test_masked_mean_empty_is_finite():
    v = torch.randn(3, 4, requires_grad=True)
    m = torch.zeros(3, 4, dtype=torch.bool)
    out = masked_mean(v, m)
    assert out.item() == 0.0 and torch.isfinite(out)
    out.backward()
    assert v.grad is not None and float(v.grad.abs().sum()) == 0.0
    print("  ok  masked_mean over empty mask -> 0.0, finite, zero grad")


def test_kl_identical_is_zero():
    x = torch.randn(2, 7, 256)
    kl = per_token_kl(x, x.clone(), temperature=4.0)
    # float32 softmax/log_softmax round-trip plus the T^2 factor leaves ~1e-6 of noise
    assert kl.abs().max().item() < 1e-4, f"KL(x||x) should be ~0, got {kl.abs().max().item():.3e}"
    assert kl.shape == (2, 7)
    print("  ok  KL(student==teacher) == 0")


def test_schedule():
    assert abs(cosine_square_schedule(0, 100) - 1.0) < 1e-9
    assert abs(cosine_square_schedule(100, 100) - 0.0) < 1e-9
    assert abs(cosine_square_schedule(50, 100) - 0.5) < 1e-9
    assert cosine_square_schedule(0, 0) == 1.0
    print("  ok  b(t): 1 -> 0.5 -> 0")


def _fake_batch(b=4, d=7, easy_frac=0.8):
    """Returns (student_logits, teacher_logits, gt_token, teacher_argmax_token)."""
    teacher = torch.randn(b, d, 256) * 2.0
    gt_col = torch.randint(0, 256, (b, d))
    wrong_col = (gt_col + torch.randint(1, 256, (b, d))) % 256
    keep = torch.rand(b, d) < easy_frac
    argmax_col = torch.where(keep, gt_col, wrong_col)
    teacher.scatter_(-1, argmax_col.unsqueeze(-1), 12.0)
    student = teacher + torch.randn_like(teacher) * 0.5
    return student, teacher, SPEC.token_of_col(gt_col), SPEC.token_of_col(argmax_col)


def test_btkd_term_is_finite_in_all_regimes():
    cfg = BTKDConfig(temperature=4.0, lambda_stu=0.5, total_steps=100)
    for easy_frac in (0.0, 0.5, 1.0):
        student, teacher, gt, tok = _fake_batch(easy_frac=easy_frac)
        for b in (1.0, 0.5, 0.0):
            kd = btkd_term(student, teacher, gt, tok, SPEC, cfg, b)
            assert torch.isfinite(kd["loss"]), f"non-finite KD at easy_frac={easy_frac}, b={b}"
            assert kd["n_easy"] + kd["n_hard"] == student.numel() // 256
    print("  ok  BTKD term finite for all-easy / mixed / all-hard and b in {0,0.5,1}")


def test_total_loss_weighting():
    cfg = BTKDConfig(temperature=4.0, lambda_stu=0.25)
    ce = torch.tensor(2.0, requires_grad=True)
    student, teacher, gt, tok = _fake_batch()
    kd = btkd_term(student, teacher, gt, tok, SPEC, cfg, 1.0)
    loss = total_loss(ce, kd, cfg)
    expect = 0.25 * 2.0 + 0.75 * kd["loss"].item()
    assert abs(loss.item() - expect) < 1e-5
    loss.backward()
    print("  ok  loss = lambda*CE + (1-lambda)*KD")


def test_teacher_accuracy():
    tok = torch.tensor([[1, 2, 3, 4]])
    gt = torch.tensor([[1, 2, 9, 9]])
    assert abs(teacher_token_accuracy(tok, gt) - 0.5) < 1e-9
    print("  ok  teacher token accuracy")


def test_extract_action_logits():
    b, l, v, d = 2, 20, 32000, 7
    logits = torch.randn(b, l, v)
    labels = torch.full((b, l), -100, dtype=torch.long)
    action_start = l - d  # last 7 positions carry the action tokens
    gt = torch.randint(SPEC.offset, SPEC.vocab_size, (b, d))
    labels[:, action_start:] = gt
    # the logit that predicts position t is at t-1
    for i in range(b):
        for j in range(d):
            logits[i, action_start + j - 1, int(gt[i, j])] = 50.0  # teacher is perfect

    action_logits, gt_tok, argmax_tok = extract_action_logits(logits, labels, SPEC.offset - 1, SPEC)
    assert action_logits.shape == (b, d, 256)
    assert torch.equal(gt_tok, gt)
    assert torch.equal(argmax_tok, gt), "teacher should be perfect here"
    # the slice must be the last 256 vocab entries
    assert torch.allclose(action_logits[0, 0], logits[0, action_start - 1, -256:])

    # a broken label layout must raise
    bad = labels.clone()
    bad[0, action_start] = -100
    try:
        extract_action_logits(logits, bad, SPEC.offset - 1, SPEC)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong number of action tokens")
    print("  ok  extract_action_logits aligns logits[t-1] with label[t]")


def test_extract_action_logits_with_visual_prefix():
    """The VLM inserts 256 visual tokens after BOS; labels must be shifted the same way."""
    b, text_l, n_visual, v, d = 2, 20, 256, 32000, 7
    logits = torch.randn(b, text_l + n_visual, v)
    labels = torch.full((b, text_l), -100, dtype=torch.long)
    action_start = text_l - d  # last 7 text positions are action tokens
    gt = torch.randint(SPEC.offset, SPEC.vocab_size, (b, d))
    labels[:, action_start:] = gt

    # multimodal layout: [BOS] + 256 visual tokens + text[1:]
    shifted_start = action_start + n_visual
    for i in range(b):
        for j in range(d):
            logits[i, shifted_start + j - 1, int(gt[i, j])] = 50.0

    action_logits, gt_tok, argmax_tok = extract_action_logits(
        logits, labels, SPEC.offset - 1, SPEC
    )
    assert action_logits.shape == (b, d, 256)
    assert torch.equal(gt_tok, gt)
    assert torch.equal(argmax_tok, gt)
    print("  ok  extract_action_logits handles OpenVLA's visual-token prefix")


def test_cache_roundtrip():
    torch.manual_seed(1234)
    with tempfile.TemporaryDirectory() as tmp:
        n, d = 23, 7
        gt = torch.randint(SPEC.offset, SPEC.vocab_size, (n, d))
        lg = torch.randn(n, d, 256)
        am = gt.clone()
        with LogitCacheWriter(tmp, SPEC, meta={"who": "test"}, shard_size=10) as w:
            for s in range(0, n, 7):
                w.add(gt[s : s + 7], lg[s : s + 7], am[s : s + 7])

        c = LogitCache(tmp, verbose=False)
        assert len(c) == n == c.n_samples
        assert c.is_complete
        assert c.manifest["meta"]["who"] == "test"
        assert len(c.shards) == 3  # 10 + 10 + 3

        rows = c.get_range(0, n)
        assert torch.equal(rows["gt_token"], gt)
        # logits are stored as fp16: half-ulp is ~1e-3 for |x| in [2, 4)
        assert torch.allclose(rows["teacher_logits"].float(), lg, rtol=1e-3, atol=1e-3)

        # cross-shard single fetch
        one = c.get_range(9, 12)
        assert torch.equal(one["gt_token"], gt[9:12])
        assert torch.equal(c[2]["gt_token"], gt[2])

        # random-access gather: unordered, duplicated and cross-shard indices
        pick = torch.tensor([22, 0, 10, 10, 5, 9])
        gathered = c.get_rows(pick)
        assert torch.equal(gathered["gt_token"], gt[pick])
        assert torch.equal(gathered["teacher_argmax_token"], am[pick])
        assert torch.allclose(
            gathered["teacher_logits"].float(), lg[pick], rtol=1e-3, atol=1e-3
        )

        # resume appends rather than restarting
        with LogitCacheWriter(tmp, SPEC, shard_size=10, resume=True) as w2:
            w2.add(gt[:3], lg[:3], am[:3])
        assert len(LogitCache(tmp, verbose=False)) == n + 3
        print(f"  ok  cache round-trip ({n} samples, 3 shards, resume, cross-shard reads)")


def test_cache_resume_after_abort():
    """An interrupted export must resume from the rows actually on disk (no duplicates)."""
    torch.manual_seed(99)
    n = 15
    gt = torch.randint(SPEC.offset, SPEC.vocab_size, (n, SPEC.n_dims))
    lg = torch.randn(n, SPEC.n_dims, SPEC.n_bins)
    am = gt.clone()
    with tempfile.TemporaryDirectory() as tmp:
        w = LogitCacheWriter(tmp, SPEC, shard_size=10, meta={"dataset": "dummy"})
        w.add(gt, lg, am)
        w.abort()  # crash mid-export: 15 rows are on disk, manifest incomplete
        assert not LogitCache(tmp, verbose=False).is_complete

        w2 = LogitCacheWriter(tmp, SPEC, shard_size=10, resume=True, meta={"dataset": "dummy"})
        assert w2.n_samples == n, w2.n_samples
        assert w2.existing_meta["dataset"] == "dummy"
        w2.add(gt[:5], lg[:5], am[:5])
        w2.close()

        c = LogitCache(tmp, verbose=False)
        assert c.is_complete and len(c) == n + 5
        rows = c.get_range(0, n + 5)
        assert torch.equal(rows["gt_token"][:n], gt)
        assert torch.equal(rows["gt_token"][n:], gt[:5])
        print(f"  ok  cache resume after abort ({n + 5} rows, no duplication)")


def test_cache_overwrite_removes_old_shards():
    torch.manual_seed(100)
    with tempfile.TemporaryDirectory() as tmp:
        gt = torch.randint(SPEC.offset, SPEC.vocab_size, (12, SPEC.n_dims))
        lg = torch.randn(12, SPEC.n_dims, SPEC.n_bins)
        with LogitCacheWriter(tmp, SPEC, shard_size=5) as w:
            w.add(gt, lg, gt)
        assert os.path.exists(os.path.join(tmp, "shard_00002.pt"))

        with LogitCacheWriter(tmp, SPEC, shard_size=5, overwrite=True) as w:
            w.add(gt[:3], lg[:3], gt[:3])
        c = LogitCache(tmp, verbose=False)
        assert len(c) == 3 and len(c.shards) == 1
        assert not os.path.exists(os.path.join(tmp, "shard_00001.pt"))
        assert not os.path.exists(os.path.join(tmp, "shard_00002.pt"))
        print("  ok  overwrite clears stale shards")


def test_cache_resume_rejects_missing_shard():
    torch.manual_seed(101)
    with tempfile.TemporaryDirectory() as tmp:
        gt = torch.randint(SPEC.offset, SPEC.vocab_size, (5, SPEC.n_dims))
        lg = torch.randn(5, SPEC.n_dims, SPEC.n_bins)
        with LogitCacheWriter(tmp, SPEC, shard_size=5) as w:
            w.add(gt, lg, gt)
        os.remove(os.path.join(tmp, "shard_00000.pt"))
        try:
            LogitCacheWriter(tmp, SPEC, shard_size=5, resume=True)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError for a missing shard")
        print("  ok  resume rejects missing shard")


def test_cache_resume_from_empty_manifest():
    torch.manual_seed(102)
    with tempfile.TemporaryDirectory() as tmp:
        LogitCacheWriter(tmp, SPEC, shard_size=5, meta={"dataset": "dummy"}).abort()
        c0 = LogitCache(tmp, verbose=False)
        assert len(c0) == 0 and not c0.is_complete

        gt = torch.randint(SPEC.offset, SPEC.vocab_size, (3, SPEC.n_dims))
        lg = torch.randn(3, SPEC.n_dims, SPEC.n_bins)
        with LogitCacheWriter(tmp, SPEC, shard_size=5, resume=True) as w:
            w.add(gt, lg, gt)
        c1 = LogitCache(tmp, verbose=False)
        assert len(c1) == 3 and c1.is_complete
        assert torch.equal(c1.get_range(0, 3)["gt_token"], gt)
        print("  ok  empty manifest resumes from row 0")


def test_cache_window_crosses_epoch_boundary():
    """A batch may straddle the dataset boundary as [tail..., head...]."""
    torch.manual_seed(103)
    with tempfile.TemporaryDirectory() as tmp:
        n = 11
        gt = torch.randint(SPEC.offset, SPEC.vocab_size, (n, SPEC.n_dims))
        lg = torch.randn(n, SPEC.n_dims, SPEC.n_bins)
        with LogitCacheWriter(tmp, SPEC, shard_size=5) as w:
            w.add(gt, lg, gt)
        c = LogitCache(tmp, verbose=False)

        rows, next_ptr, wrapped = c.get_window(9, 4, allow_wrap=True)
        assert wrapped and next_ptr == 2
        assert torch.equal(rows["gt_token"], torch.cat([gt[9:], gt[:2]], dim=0))

        rows, next_ptr, wrapped = c.get_window(6, 5, allow_wrap=True)
        assert not wrapped and next_ptr == n
        assert torch.equal(rows["gt_token"], gt[6:])

        try:
            c.get_window(9, 4, allow_wrap=False)
        except IndexError:
            pass
        else:
            raise AssertionError("expected IndexError when wrapping is disabled")
        print("  ok  cache window wraps as [tail, head] without losing alignment")


def test_cache_wrap_safety():
    full = {"dataset_length": 100, "target_samples": 100, "limit": -1}
    assert cache_covers_full_epoch(full, 100, train_limit=-1)
    assert not cache_covers_full_epoch(full, 100, train_limit=20)

    partial = {"dataset_length": 100, "target_samples": 50, "limit": 50}
    assert not cache_covers_full_epoch(partial, 50, train_limit=-1)

    # --limit larger than the dataset is capped to one full pass and may wrap.
    capped = {"dataset_length": 100, "target_samples": 100, "limit": 100000}
    assert cache_covers_full_epoch(capped, 100, train_limit=-1)
    print("  ok  cache wrap requires a full export and no training --limit")


def test_cosine_square_schedule_hold():
    total = 100
    assert abs(cosine_square_schedule_hold(0, total, 0.0) - 1.0) < 1e-12
    assert abs(cosine_square_schedule_hold(50, total, 0.0) - 0.5) < 1e-12
    assert abs(cosine_square_schedule_hold(100, total, 0.0)) < 1e-12

    # 40% easy plateau, then the same cosine-square decay over the remaining 60%.
    assert cosine_square_schedule_hold(0, total, 0.4) == 1.0
    assert cosine_square_schedule_hold(40, total, 0.4) == 1.0
    assert abs(cosine_square_schedule_hold(70, total, 0.4) - 0.5) < 1e-12
    assert abs(cosine_square_schedule_hold(100, total, 0.4)) < 1e-12
    print("  ok  easy-hold schedule: b=1 plateau then cos^2 decay to 0")


def test_frame_store_roundtrip():
    torch.manual_seed(77)
    n = 11
    with tempfile.TemporaryDirectory() as tmp:
        gt = torch.randint(SPEC.offset, SPEC.vocab_size, (n, SPEC.n_dims))
        ids = [torch.tensor([i, 1000 + i]) for i in range(n)]
        pix = [torch.rand(3, 8, 8) for _ in range(n)]
        with FrameStoreWriter(tmp, image_shape=(3, 8, 8), dataset="dummy", shard_size=4) as w:
            for i in range(n):
                w.add(ids[i], gt[i], pix[i])
        store = FrameStore(tmp, verbose=False)
        assert store.is_complete and len(store) == n and len(store.shards) == 3
        for i in (0, 3, 4, 10, 6):
            row = store[i]
            assert torch.equal(row["input_ids"], ids[i])
            assert torch.equal(row["labels"], gt[i])
            assert torch.allclose(row["pixel_values"], pix[i].half().float())
        print("  ok  frame-store round-trip across shards")


def test_frames_batches_share_one_index():
    """Frame row i and cache row i must be fetched from the same integer index."""
    torch.manual_seed(78)
    n, bs = 10, 4
    gt = torch.stack(
        [torch.arange(SPEC.offset + i, SPEC.offset + i + SPEC.n_dims) for i in range(n)]
    )
    lg = torch.randn(n, SPEC.n_dims, SPEC.n_bins)

    def toy_collator(rows):
        return {
            "input_ids": torch.stack([r["input_ids"] for r in rows]),
            "labels": torch.stack([r["labels"] for r in rows]),
            "attention_mask": torch.ones(len(rows), 2, dtype=torch.long),
            "pixel_values": torch.stack([r["pixel_values"] for r in rows]),
        }

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = os.path.join(tmp, "cache")
        frames_dir = os.path.join(tmp, "frames")
        with LogitCacheWriter(cache_dir, SPEC, shard_size=3) as w:
            w.add(gt, lg, gt)
        with FrameStoreWriter(frames_dir, image_shape=(3, 2, 2), shard_size=3) as w:
            for i in range(n):
                w.add(
                    torch.tensor([i, 100 + i]),
                    gt[i],
                    torch.zeros(3, 2, 2),
                )
        cache = LogitCache(cache_dir, verbose=False)
        store = FrameStore(frames_dir, verbose=False)

        for shuffle in (False, True):
            batches = list(
                islice(
                    frames_batches(store, cache, toy_collator, bs, n, shuffle=shuffle, seed=5),
                    3,
                )
            )
            seen = torch.cat([b["_idx"] for b, _ in batches])
            assert sorted(seen.tolist()) == list(range(n)), "one epoch must cover every row"
            assert [len(b["input_ids"]) for b, _ in batches] == [4, 4, 2]
            for batch, cached in batches:
                idx = batch["_idx"]
                assert torch.equal(batch["input_ids"][:, 0], idx)
                assert torch.equal(batch["labels"], gt[idx])
                assert torch.equal(cached["gt_token"], gt[idx])
                assert torch.allclose(cached["teacher_logits"], lg[idx].half())
    print("  ok  frames/cache use the same index (ordered, shuffled, tail batch)")


def test_metric_accumulator():
    gt = torch.stack(
        [torch.arange(SPEC.offset + 1, SPEC.offset + 1 + SPEC.n_dims)]
    )
    tea = gt.clone()
    logits = torch.full((1, SPEC.n_dims, SPEC.n_bins), -100.0)
    for d in range(SPEC.n_dims):
        logits[0, d, d + 1] = 100.0  # correct: gt column is d+1 here
    logits[0, 1, 0] = 200.0         # dim 1 deliberately wrong
    acc = MetricAccumulator(SPEC)
    acc.update(logits, gt, tea, ce=1.25)
    out = acc.result()
    assert abs(out["student_token_acc"] - (SPEC.n_dims - 1) / SPEC.n_dims) < 1e-12
    assert out["teacher_token_acc"] == 1.0
    assert abs(out["student_teacher_agreement"] - (SPEC.n_dims - 1) / SPEC.n_dims) < 1e-12
    assert out["per_dim_acc"] == [1.0, 0.0] + [1.0] * (SPEC.n_dims - 2)
    assert out["teacher_out_of_space_frac"] == 0.0
    print("  ok  shared metric accumulator")


def test_resolve_plan_epochs():
    # LIBERO spatial: 52970 samples, bs=1, accum=16, 2 epochs
    plan = resolve_plan(52970, 1, 2.0, 20000, 16)
    assert plan["steps_per_epoch"] == 52970
    assert plan["total_micro_batches"] == 105940
    assert plan["optimizer_updates"] == math.ceil(105940 / 16) == 6622
    assert abs(plan["epochs_done"] - 2.0) < 1e-12

    # tail batch must not be dropped: 101 samples / bs=4 -> 26 steps, not 25
    plan = resolve_plan(101, 4, 1.0, 20000, 1)
    assert plan["steps_per_epoch"] == 26
    assert plan["total_micro_batches"] == 26

    # legacy semantics: --max_steps counts micro-batches, and reproduces the old 0.38 epoch run
    legacy = resolve_plan(52970, 1, 0, 20000, 16)
    assert legacy["total_micro_batches"] == 20000
    assert legacy["optimizer_updates"] == 1250
    assert abs(legacy["epochs_done"] - 20000 / 52970) < 1e-12
    print("  ok  epoch/max_steps plan conversion")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print(f"running {len(ALL_TESTS)} test(s)")
    failures = 0
    for fn in ALL_TESTS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
