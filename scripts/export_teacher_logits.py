#!/usr/bin/env python
"""A6 step 1/3 -- export OpenVLA teacher action-token logits to disk.

The teacher runs exactly once, in a single forward pass per batch (no ``generate()``,
no KV-cache growth): teacher-forcing on the ground-truth action tokens gives, at each
of the D action positions, a 256-way distribution over the action bins.  We store
that and nothing else, so the student can be distilled on a machine that cannot hold
two 7B models.

Usage (real teacher + RLDS dataset)
-----------------------------------
    python scripts/export_teacher_logits.py \
        --teacher_path openvla/openvla-7b \
        --data_root_dir /data/open-x-embodiment \
        --dataset_name bridge_orig \
        --cache_dir caches/bridge_orig_teacher \
        --batch_size 8

Usage (no model, no data -- synthetic cache for smoke tests)
------------------------------------------------------------
    python scripts/export_teacher_logits.py --dummy --limit 512 \
        --cache_dir caches/dummy --vocab_size 32000

Ordering contract
-----------------
Rows are appended in dataset order.  Both this script and the trainer must iterate
the dataset identically (``--shuffle_buffer_size 1`` -> identity order, image
augmentation OFF).  The trainer re-checks every batch against the cached ground-truth
tokens, so any drift fails loudly instead of silently corrupting the distillation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btkd_vla.cache import LogitCacheWriter  # noqa: E402
from btkd_vla.spec import ActionSpec  # noqa: E402
from btkd_vla.vla_glue import (  # noqa: E402
    extract_action_logits,
    load_openvla_model_and_processor,
    move_pixel_values,
)

PROGRESS = "progress.json"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--cache_dir", required=True, help="output directory for the logit cache")
    p.add_argument("--teacher_path", default="openvla/openvla-7b")
    p.add_argument("--data_root_dir", default="datasets/open-x-embodiment")
    p.add_argument("--dataset_name", default=None, help="OXE dataset / mixture name")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--shard_size", type=int, default=5000, help="samples per shard file")
    p.add_argument("--limit", type=int, default=-1, help="stop after N samples (-1 = all)")
    p.add_argument("--shuffle_buffer_size", type=int, default=1, help="keep at 1 to preserve order")
    p.add_argument("--image_aug", action="store_true", help="DO NOT use for offline KD (breaks alignment)")

    p.add_argument("--vocab_size", type=int, default=32000, help="LLM vocab size (Llama-2 = 32000)")
    p.add_argument("--n_bins", type=int, default=256)
    p.add_argument("--n_dims", type=int, default=7)

    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--quantize_4bit", action="store_true", help="saves memory, contaminates logits (see docs)")
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])

    p.add_argument("--resume", action="store_true", help="continue an interrupted export")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dummy", action="store_true", help="write a synthetic cache, no model/data needed")
    p.add_argument("--seed", type=int, default=42)
    return p


# --------------------------------------------------------------------------------------
# Dummy cache (CPU smoke test)
# --------------------------------------------------------------------------------------
def write_dummy_cache(args, spec: ActionSpec) -> None:
    torch.manual_seed(args.seed)
    n = args.limit if args.limit > 0 else 512
    teacher_acc = 0.82  # pretend the teacher is right 82% of the time

    with LogitCacheWriter(
        args.cache_dir,
        spec,
        meta={"dummy": True, "teacher": "synthetic", "teacher_token_acc": teacher_acc},
        shard_size=args.shard_size,
        overwrite=True,
    ) as w:
        for start in range(0, n, 256):
            b = min(256, n - start)
            # teacher logits: a smooth random distribution + a peak on a random token
            logits = torch.randn(b, spec.n_dims, spec.n_bins) * 2.0
            peak = torch.randint(0, spec.n_bins, (b, spec.n_dims))
            logits.scatter_(-1, peak.unsqueeze(-1), torch.rand(b, spec.n_dims, 1) * 6.0 + 6.0)

            gt = torch.randint(0, spec.n_bins, (b, spec.n_dims))
            keep = torch.rand(b, spec.n_dims) < teacher_acc
            wrong = (gt + torch.randint(1, spec.n_bins, (b, spec.n_dims))) % spec.n_bins
            teacher_col = torch.where(keep, gt, wrong)

            # emulate the real sign convention: token id -> column
            gt_token = spec.token_of_col(gt)
            argmax_token = spec.token_of_col(teacher_col)
            # make the logits consistent with the chosen argmax column
            logits.scatter_(-1, teacher_col.unsqueeze(-1), torch.full((b, spec.n_dims, 1), 20.0))
            w.add(gt_token, logits, argmax_token)
    print(f"[dummy] wrote {n} synthetic samples to {args.cache_dir}")


# --------------------------------------------------------------------------------------
# Real export
# --------------------------------------------------------------------------------------
def build_dataloader(args, processor, vla, spec: ActionSpec):
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

    action_tokenizer = ActionTokenizer(processor.tokenizer, bins=spec.n_bins)
    begin_idx = action_tokenizer.action_token_begin_idx
    if begin_idx != spec.vocab_size - spec.n_bins - 1:
        raise ValueError(
            f"action_token_begin_idx={begin_idx} does not match spec "
            f"(vocab_size={spec.vocab_size}, n_bins={spec.n_bins}); adjust --vocab_size/--n_bins"
        )

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder
        if "v01" not in args.teacher_path
        else VicunaV15ChatPromptBuilder,
    )
    dataset = RLDSDataset(
        args.data_root_dir,
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=max(1, args.shuffle_buffer_size),
        image_aug=args.image_aug,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0
    )
    return loader, begin_idx, dataset


def read_progress(cache_dir: str) -> Dict[str, int]:
    path = os.path.join(cache_dir, PROGRESS)
    if not os.path.exists(path):
        return {"batches_done": 0, "samples_done": 0}
    return json.load(open(path, "r", encoding="utf-8"))


def write_progress(cache_dir: str, batches_done: int, samples_done: int) -> None:
    path = os.path.join(cache_dir, PROGRESS)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"batches_done": batches_done, "samples_done": samples_done, "updated_at": time.strftime("%F %T")},
            f,
            indent=2,
        )
    os.replace(tmp, path)


def run_export(args, spec: ActionSpec) -> None:
    from tqdm import tqdm

    print(f"[export] loading teacher from {args.teacher_path} ({args.dtype})")
    vla, processor = load_openvla_model_and_processor(
        args.teacher_path,
        device=args.device,
        dtype=args.dtype,
        quantize_4bit=args.quantize_4bit,
        attn_implementation=args.attn_implementation,
        trainable_lora=False,
    )
    if not args.quantize_4bit:
        for prm in vla.parameters():
            prm.requires_grad_(False)
    vla.eval()

    loader, begin_idx, dataset = build_dataloader(args, processor, vla, spec)
    print(f"[export] dataset={args.dataset_name} length={getattr(dataset, 'dataset_length', '?')}")

    progress = read_progress(args.cache_dir) if args.resume else {"batches_done": 0, "samples_done": 0}
    meta = {
        "teacher": args.teacher_path,
        "teacher_dtype": args.dtype,
        "teacher_4bit": bool(args.quantize_4bit),
        "dataset": args.dataset_name,
        "data_root_dir": args.data_root_dir,
        "action_token_begin_idx": begin_idx,
        "image_aug": bool(args.image_aug),
        "shuffle_buffer_size": args.shuffle_buffer_size,
    }

    writer = LogitCacheWriter(
        args.cache_dir,
        spec,
        meta=meta,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    done_batches = int(progress["batches_done"])
    done_samples = int(progress["samples_done"])
    if done_batches:
        print(f"[export] resuming after {done_samples} samples ({done_batches} batches)")

    it = iter(loader)
    skipped = 0
    while skipped < done_batches:
        next(it)
        skipped += 1

    total_written = done_samples
    batches_written = done_batches
    scalar_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    pbar = tqdm(unit="samples", desc="export")
    ok = False
    try:
        for batch in it:
            n = int(batch["input_ids"].size(0))
            if args.limit > 0 and total_written + n > args.limit:
                n = args.limit - total_written
                if n <= 0:
                    break
                batch = {
                    k: (
                        v[:n]
                        if torch.is_tensor(v)
                        else {kk: vv[:n] for kk, vv in v.items()}
                        if isinstance(v, dict)
                        else v
                    )
                    for k, v in batch.items()
                }
            if args.limit > 0 and total_written >= args.limit:
                break

            use_amp = args.device.startswith("cuda") and scalar_dtype != torch.float32
            with torch.inference_mode(), torch.autocast("cuda", dtype=scalar_dtype, enabled=use_amp):
                out = vla(
                    input_ids=batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"].to(args.device),
                    pixel_values=move_pixel_values(batch["pixel_values"], args.device, scalar_dtype),
                )

            action_logits, gt_token, argmax_token = extract_action_logits(
                out.logits, batch["labels"].to(out.logits.device), begin_idx, spec
            )
            writer.add(gt_token.cpu(), action_logits.cpu(), argmax_token.cpu())
            del out, action_logits

            total_written += n
            batches_written += 1
            pbar.update(n)
            if batches_written % 50 == 0:
                write_progress(args.cache_dir, batches_written, total_written)
            if args.limit > 0 and total_written >= args.limit:
                break
        ok = True
    finally:
        pbar.close()
        if ok:
            writer.close()
        else:
            # keep the partial shards but mark the cache as incomplete so the trainer refuses it
            writer._flush()
            writer._write_manifest(complete=False)
        write_progress(args.cache_dir, batches_written, total_written)

    print(f"[export] done: {total_written} samples -> {args.cache_dir}")


def main() -> None:
    args = build_parser().parse_args()
    spec = ActionSpec(n_bins=args.n_bins, n_dims=args.n_dims, vocab_size=args.vocab_size)
    if args.dummy:
        write_dummy_cache(args, spec)
        return
    if not args.dataset_name:
        raise SystemExit("--dataset_name is required unless --dummy is given")
    run_export(args, spec)


if __name__ == "__main__":
    main()
