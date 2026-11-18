#!/usr/bin/env python
"""A6 step 3/3 -- train the OpenVLA student with BTKD using a cached teacher.

Only ONE model is resident here: the student (LoRA on ``openvla-7b`` by default).
The teacher distribution comes from the on-disk cache produced by
``scripts/export_teacher_logits.py``, so this fits comfortably in 24 GB (and in
~10 GB with ``--quantize_4bit``).

Loss (faithful to ``dis_resnet18_cf100.py``)::

    loss = lambda * CE + (1 - lambda) * [ b * KL(stu || tea) + (1-b) * KL(stu || calibrate3(tea)) ]
    b    = cos^2(pi/2 * step / total_steps)          # 1 (easy) -> 0 (hard)
    T^2 is folded into the KL term.

Example
-------
    python scripts/train_btkd_student.py \
        --cache_dir caches/bridge_orig_teacher \
        --vla_path openvla/openvla-7b \
        --data_root_dir /data/open-x-embodiment --dataset_name bridge_orig \
        --batch_size 4 --grad_accumulation_steps 4 --max_steps 20000 \
        --lora_rank 32 --learning_rate 5e-4 \
        --temperature 4.0 --lambda_stu 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btkd_vla.cache import DATA_ORDER, LogitCache, cache_covers_full_epoch  # noqa: E402
from btkd_vla.core import (  # noqa: E402
    btkd_term,
    cosine_square_schedule,
    teacher_token_accuracy,
    total_loss,
)
from btkd_vla.spec import BTKDConfig  # noqa: E402
from btkd_vla.vla_glue import extract_action_logits, move_pixel_values  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--cache_dir", required=True)
    p.add_argument("--vla_path", default="openvla/openvla-7b")
    p.add_argument("--data_root_dir", default="datasets/open-x-embodiment")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--run_root_dir", default="runs_btkd")

    # optimisation
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--max_steps", type=int, default=20_000)
    p.add_argument("--save_steps", type=int, default=2_000)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # LoRA / memory
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--quantize_4bit", action="store_true", help="QLoRA: ~10 GB total")
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager"])

    # BTKD
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--lambda_stu", type=float, default=0.5)
    p.add_argument("--schedule", default="cos2", choices=["cos2"])
    p.add_argument("--calibrate_strength", type=float, default=1.0)
    p.add_argument("--logit_norm", default="none", choices=["none", "center", "std"])
    p.add_argument("--renormalize_empty_branch", action="store_true")

    # data plumbing
    p.add_argument("--shuffle_buffer_size", type=int, default=1, help="must match the export run")
    p.add_argument("--image_aug", action="store_true", help="differs from the teacher's view; see docs")
    p.add_argument("--verify_batches", type=int, default=5, help="alignment checks against cached GT tokens")
    p.add_argument("--limit", type=int, default=-1, help="only use the first N cached samples")

    # logging
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--exp_name", default=None)
    p.add_argument(
        "--check_cache_only",
        action="store_true",
        help="print cache statistics and exit (no model / dataset needed)",
    )
    return p


def build_dataloader(args, processor, vla, n_bins: int):
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform

    from btkd_vla.rlds_dataset import DeterministicSingleRLDSDataset

    action_tokenizer = ActionTokenizer(processor.tokenizer, bins=n_bins)
    begin_idx = action_tokenizer.action_token_begin_idx

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder
        if "v01" not in args.vla_path
        else VicunaV15ChatPromptBuilder,
    )
    dataset = DeterministicSingleRLDSDataset(
        args.data_root_dir,
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=args.shuffle_buffer_size,
        train=True,
        image_aug=args.image_aug,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0
    )
    return loader, begin_idx, action_tokenizer


def check_alignment(cached: Dict[str, torch.Tensor], gt_token: torch.Tensor, batch_idx: int, cache_ptr: int) -> None:
    """Fail loudly if the cache and the dataset have drifted out of sync."""
    a = cached["gt_token"]
    b = gt_token.cpu()
    if a.shape != b.shape or not torch.equal(a, b):
        raise RuntimeError(
            f"cache/dataset misalignment at batch {batch_idx} (cache rows starting at {cache_ptr}).\n"
            f"  cached : {a[:2, :4].tolist()} ...\n"
            f"  dataset: {b[:2, :4].tolist()} ...\n"
            "Export and training must both use the deterministic A6 RLDS loader "
            "(TFDS shard shuffling off, one read/map worker) and the same dataset. "
            "Re-export the cache with the current code."
        )


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if args.image_aug:
        raise SystemExit("--image_aug is forbidden for offline distillation: it breaks image/logit alignment")
    if args.shuffle_buffer_size != 1:
        raise SystemExit("--shuffle_buffer_size must be 1 for offline distillation (dataset order contract)")

    try:
        cache = LogitCache(args.cache_dir)
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nRun scripts/export_teacher_logits.py first.") from None
    if not cache.is_complete:
        raise SystemExit(
            f"cache at {args.cache_dir} is marked incomplete (interrupted export). "
            "Resume the export with --resume, or delete the directory."
        )
    full_cache_len = len(cache)
    cache_meta = cache.manifest.get("meta") or {}
    cached_dataset = cache_meta.get("dataset")
    if cached_dataset and cached_dataset != args.dataset_name:
        raise SystemExit(
            f"cache was exported for dataset {cached_dataset!r}, but --dataset_name is "
            f"{args.dataset_name!r}; use the matching cache or re-export it"
        )
    cached_order = cache_meta.get("data_order")
    if cached_order != DATA_ORDER:
        raise SystemExit(
            f"cache data order is {cached_order!r}, expected {DATA_ORDER!r}. "
            "This cache predates deterministic RLDS ordering; re-export it with "
            "scripts/export_teacher_logits.py --overwrite."
        )
    # Decide before shrink_to(): a cache may cover the full dataset even when this
    # training run deliberately schedules only a prefix (--limit N).
    can_wrap_cache = cache_covers_full_epoch(cache_meta, full_cache_len, args.limit)
    if args.limit > 0:
        cache.shrink_to(min(args.limit, len(cache)))
    spec = cache.spec

    if args.check_cache_only:
        stats = cache.analysis()
        print("[check] " + json.dumps(stats, indent=2))
        bad = stats["nonfinite_logits"] > 0
        print(
            "[check] interpretation: teacher token accuracy drives the easy/hard split; "
            f"{stats['easy_frac'] * 100:.1f}% of action tokens go to the easy branch."
        )
        raise SystemExit(1 if bad else 0)

    # Only a real training run touches the dataset, so a CPU-only cache inspection
    # must remain usable after moving the cache to a machine with a different path.
    cached_root = cache_meta.get("data_root_dir")
    if cached_root and os.path.abspath(str(cached_root)) != os.path.abspath(args.data_root_dir):
        raise SystemExit(
            f"cache was exported from {cached_root!r}, but --data_root_dir is "
            f"{args.data_root_dir!r}; use the matching data root or re-export it"
        )

    exp = args.exp_name or (
        f"btkd+{Path(args.vla_path).name}+{args.dataset_name}"
        f"+b{args.batch_size * args.grad_accumulation_steps}"
        f"+lr{args.learning_rate}+T{args.temperature}+lam{args.lambda_stu}"
        + ("+q4bit" if args.quantize_4bit else "")
    )
    run_dir = Path(args.run_root_dir) / exp
    run_dir.mkdir(parents=True, exist_ok=True)

    from btkd_vla.vla_glue import load_openvla_model_and_processor

    model, processor = load_openvla_model_and_processor(
        args.vla_path,
        device=args.device,
        dtype="bfloat16",
        quantize_4bit=args.quantize_4bit,
        attn_implementation=args.attn_implementation,
        trainable_lora=True,
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    tokenizer = processor.tokenizer
    if int(tokenizer.vocab_size) != int(spec.vocab_size):
        raise SystemExit(
            f"tokenizer vocab_size={tokenizer.vocab_size} != cached spec vocab_size={spec.vocab_size}. "
            "Re-export the cache with the matching --vocab_size."
        )

    loader, begin_idx, action_tokenizer = build_dataloader(args, processor, model, spec.n_bins)
    if begin_idx != spec.vocab_size - spec.n_bins - 1:
        raise SystemExit(
            f"action_token_begin_idx={begin_idx} inconsistent with cached spec {spec.to_dict()}"
        )

    cfg = BTKDConfig(
        temperature=args.temperature,
        lambda_stu=args.lambda_stu,
        schedule=args.schedule,
        total_steps=args.max_steps,
        calibrate_strength=args.calibrate_strength,
        logit_norm=args.logit_norm,
        renormalize_empty_branch=args.renormalize_empty_branch,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    wandb = None
    if args.wandb_project:
        import wandb as _wandb

        wandb = _wandb
        wandb.init(project=args.wandb_project, name=exp, config=vars(args))

    device = args.device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    cache_ptr = 0
    wraps = 0
    ce_hist, kd_hist, easy_hist, hard_hist, acc_hist, loss_hist = [], [], [], [], [], []
    for step, batch in enumerate(loader):
        if step >= args.max_steps:
            break

        n = int(batch["input_ids"].size(0))
        wrapped = False
        if cache_ptr == len(cache):
            # Exact end of a pass. Restart at row 0 and verify the first batch of the
            # new pass before training on it.
            if not can_wrap_cache:
                raise RuntimeError(
                    f"cache exhausted at step {step} ({len(cache)} rows). The cache does "
                    "not cover one full dataset pass (exported with --limit, or the dataset "
                    "has more samples). Re-export the full dataset, or keep --limit small "
                    "enough for the scheduled steps."
                )
            cache_ptr = 0
            wraps += 1
            wrapped = True
            print(f"[data] cache wrapped to row 0 (pass {wraps}); verifying alignment", flush=True)

        if cache_ptr + n > len(cache) and not can_wrap_cache:
            raise RuntimeError(
                f"cache exhausted at step {step} (row {cache_ptr}+{n} > {len(cache)}). "
                "The cache does not cover one full dataset pass (exported with --limit, or the "
                "dataset has more samples). Re-export the full dataset, or keep --limit small "
                "enough for the scheduled steps."
            )
        window_start = cache_ptr
        try:
            cached, cache_ptr, crossed = cache.get_window(
                cache_ptr, n, allow_wrap=can_wrap_cache
            )
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"cannot read a cache window at step {step}: {exc}") from exc
        if crossed:
            # The RLDS stream repeats forever, so a batch may straddle the dataset
            # boundary as [tail..., head...]. get_window reproduces that exact order.
            wraps += 1
            wrapped = True
            print(f"[data] cache wrapped across row 0 (pass {wraps}); verifying alignment", flush=True)
        labels = batch["labels"]

        b = cosine_square_schedule(step, args.max_steps)
        use_amp = device.startswith("cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=move_pixel_values(batch["pixel_values"], device, torch.bfloat16),
                labels=labels.to(device),
            )
            ce = out.loss
            action_logits, gt_token, _ = extract_action_logits(out.logits, labels.to(device), begin_idx, spec)
            kd = btkd_term(
                action_logits,
                cached["teacher_logits"].to(device=device, dtype=torch.float32),
                gt_token,
                cached["teacher_argmax_token"].to(device),
                spec,
                cfg,
                b,
            )
            loss = total_loss(ce, kd, cfg)

        if step < args.verify_batches or wrapped:
            check_alignment(cached, gt_token, step, window_start)

        (loss / args.grad_accumulation_steps).backward()
        if (step + 1) % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_hist.append(float(loss.item()))
        ce_hist.append(float(ce.item()))
        kd_hist.append(float(kd["loss"].item()))
        easy_hist.append(float(kd["easy"].item()))
        hard_hist.append(float(kd["hard"].item()))
        acc_hist.append(teacher_token_accuracy(cached["teacher_argmax_token"], cached["gt_token"]))

        if step % args.log_steps == 0:
            window = slice(-args.log_steps, None)
            msg = (
                f"[step {step:6d}] loss={sum(loss_hist[window])/max(1,len(loss_hist[window])):.4f} "
                f"ce={sum(ce_hist[window])/max(1,len(ce_hist[window])):.4f} "
                f"kd={sum(kd_hist[window])/max(1,len(kd_hist[window])):.4f} "
                f"easy={sum(easy_hist[window])/max(1,len(easy_hist[window])):.4f} "
                f"hard={sum(hard_hist[window])/max(1,len(hard_hist[window])):.4f} "
                f"b={b:.3f} teacher_tok_acc={sum(acc_hist[window])/max(1,len(acc_hist[window])):.3f} "
                f"easy_frac={kd['easy_frac']:.3f}"
            )
            print(msg, flush=True)
            if wandb is not None:
                wandb.log(
                    {
                        "loss": float(loss.item()),
                        "ce": float(ce.item()),
                        "kd": float(kd["loss"].item()),
                        "kd_easy": float(kd["easy"].item()),
                        "kd_hard": float(kd["hard"].item()),
                        "b": b,
                        "easy_frac": kd["easy_frac"],
                        "lr": optimizer.param_groups[0]["lr"],
                        "step": step,
                    }
                )

        if args.save_steps > 0 and step > 0 and step % args.save_steps == 0:
            ckpt = run_dir / f"checkpoint-{step}"
            model.save_pretrained(ckpt)
            print(f"[save] {ckpt}", flush=True)

    final = run_dir / "final"
    model.save_pretrained(final)
    print(f"[done] adapter saved to {final}")


if __name__ == "__main__":
    main()
