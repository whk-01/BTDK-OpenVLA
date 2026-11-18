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
        --batch_size 2  # 24 GB card; use 8 on 48 GB cards

Usage (no model, no data -- synthetic cache for smoke tests)
------------------------------------------------------------
    python scripts/export_teacher_logits.py --dummy --limit 512 \
        --cache_dir caches/dummy --vocab_size 32000

Ordering contract
-----------------
Rows are appended in dataset order.  This script and the trainer share
``btkd_vla.rlds_dataset.DeterministicSingleRLDSDataset``, which disables TFDS shard-file
shuffling and pins every RLDS read/map/flatten parallelism to 1.  A frame shuffle buffer
of 1 and image augmentation OFF are additional requirements.  The trainer re-checks every
batch against the cached ground-truth tokens, so any drift fails loudly instead of
silently corrupting the distillation.
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

from btkd_vla.cache import CACHE_VERSION, DATA_ORDER, LogitCacheWriter  # noqa: E402
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
    p.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=1,
        help="must stay 1; the A6 loader additionally disables TFDS shard-file shuffling",
    )
    p.add_argument("--image_aug", action="store_true", help="DO NOT use for offline KD (breaks alignment)")

    p.add_argument("--vocab_size", type=int, default=32000, help="LLM vocab size (Llama-2 = 32000)")
    p.add_argument("--n_bins", type=int, default=256)
    p.add_argument("--n_dims", type=int, default=7)

    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--quantize_4bit", action="store_true", help="saves memory, contaminates logits (see docs)")
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])

    cache_mode = p.add_mutually_exclusive_group()
    cache_mode.add_argument("--resume", action="store_true", help="continue an interrupted export")
    cache_mode.add_argument("--overwrite", action="store_true")
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
        meta={
            "dummy": True,
            "dataset": args.dataset_name or "dummy",
            "teacher": "synthetic",
            "teacher_token_acc": teacher_acc,
            "data_order": DATA_ORDER,
            "rlds_shard_shuffle": False,
            "rlds_num_parallel_reads": 1,
            "rlds_num_parallel_calls": 1,
        },
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
    from prismatic.vla.datasets import RLDSBatchTransform

    from btkd_vla.rlds_dataset import DeterministicSingleRLDSDataset

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


def _take_batch(batch, n: int):
    """Keep only the first ``n`` rows of a collated batch."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[:n]
        elif isinstance(v, dict):
            out[k] = {kk: vv[:n] for kk, vv in v.items()}
        else:
            out[k] = v
    return out


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
    dataset_length = int(getattr(dataset, "dataset_length", 0) or 0)
    print(f"[export] dataset={args.dataset_name} length={dataset_length}")

    # One pass over the dataset by default: the RLDS pipeline repeats forever, so an
    # explicit stop point is what makes "export the training set" well defined.
    if args.limit > 0:
        target = args.limit
    elif dataset_length > 0:
        target = dataset_length
    else:
        raise SystemExit("cannot determine the dataset length; pass --limit N explicitly")
    if args.limit > 0 and dataset_length > 0 and args.limit > dataset_length:
        print(f"[export] --limit {args.limit} > dataset length {dataset_length}; capping")
        target = dataset_length
    print(f"[export] target: {target} samples (one pass)")

    meta = {
        "teacher": args.teacher_path,
        "teacher_dtype": args.dtype,
        "teacher_4bit": bool(args.quantize_4bit),
        "dataset": args.dataset_name,
        "data_root_dir": args.data_root_dir,
        "action_token_begin_idx": begin_idx,
        "image_aug": bool(args.image_aug),
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "data_order": DATA_ORDER,
        "rlds_shard_shuffle": False,
        "rlds_num_parallel_reads": 1,
        "rlds_num_parallel_calls": 1,
        "dataset_length": dataset_length,
        "limit": args.limit,
        "target_samples": target,
    }

    cache_manifest = os.path.join(args.cache_dir, "manifest.json")
    if os.path.exists(cache_manifest) and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"{args.cache_dir} already contains a cache; pass --resume to continue it "
            "or --overwrite to rebuild it"
        )
    if args.resume and os.path.exists(cache_manifest):
        with open(cache_manifest, "r", encoding="utf-8") as f:
            man = json.load(f)
        if int(man.get("version", -1)) != CACHE_VERSION:
            raise SystemExit(
                f"cannot resume: cache version {man.get('version')} != expected {CACHE_VERSION}"
            )
        try:
            cached_spec = ActionSpec.from_dict(man["spec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"cannot resume: invalid cache spec: {exc}") from exc
        if cached_spec != spec:
            raise SystemExit(
                f"cannot resume: cached spec {cached_spec.to_dict()} != requested {spec.to_dict()}"
            )
        old = man.get("meta") or {}
        keys = (
            "dataset",
            "data_root_dir",
            "teacher",
            "teacher_dtype",
            "teacher_4bit",
            "action_token_begin_idx",
            "image_aug",
            "shuffle_buffer_size",
            "data_order",
            "rlds_shard_shuffle",
            "rlds_num_parallel_reads",
            "rlds_num_parallel_calls",
            "dataset_length",
            "limit",
            "target_samples",
        )
        mismatched = {
            k: (old.get(k), meta[k])
            for k in keys
            if k in old and old.get(k) != meta[k]
        }
        if mismatched:
            raise SystemExit(
                "cannot resume: cache was exported with different settings "
                f"{mismatched}. Re-export with --overwrite."
            )
        if bool(man.get("complete", False)):
            print(f"[export] cache already complete ({man.get('n_samples')} samples); nothing to do")
            return

    writer = LogitCacheWriter(
        args.cache_dir,
        spec,
        meta=meta,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    # Resume point comes from the manifest (rows actually on disk), never from
    # progress.json -- the latter is written every N batches and would duplicate rows.
    done_samples = writer.n_samples if args.resume else 0
    if done_samples > target:
        raise SystemExit(
            f"cache already holds {done_samples} samples > target {target}; "
            "re-export with --overwrite or raise --limit"
        )
    if done_samples:
        print(f"[export] resuming after {done_samples} samples already on disk")

    it = iter(loader)
    skipped = 0
    pending = None
    while skipped < done_samples:
        batch = next(it)
        k = int(batch["input_ids"].size(0))
        if skipped + k <= done_samples:
            skipped += k
        else:
            pending = _take_batch(batch, done_samples - skipped)
            skipped = done_samples
    if skipped != done_samples:
        raise RuntimeError(
            f"dataset ended after {skipped} samples while resuming at {done_samples}; "
            "the dataset order/content changed, re-export with --overwrite"
        )

    def batches():
        if pending is not None:
            yield pending
        yield from it

    total_written = done_samples
    batches_written = 0
    scalar_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    pbar = tqdm(total=target, initial=done_samples, unit="samples", desc="export")
    ok = False
    try:
        for batch in batches():
            if total_written >= target:
                break
            n = int(batch["input_ids"].size(0))
            if total_written + n > target:
                n = target - total_written
                batch = _take_batch(batch, n)

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
        ok = True
    finally:
        pbar.close()
        if ok:
            writer.close()
        else:
            writer.abort()
        write_progress(args.cache_dir, batches_written, total_written)

    print(f"[export] done: {total_written} samples -> {args.cache_dir}")


def main() -> None:
    args = build_parser().parse_args()
    spec = ActionSpec(n_bins=args.n_bins, n_dims=args.n_dims, vocab_size=args.vocab_size)
    if args.quantize_4bit:
        raise SystemExit(
            "--quantize_4bit is forbidden for teacher export: 4-bit noise corrupts the "
            "BTKD calibration target. Export the teacher in bf16/fp16."
        )
    if args.image_aug:
        raise SystemExit("--image_aug is forbidden for offline distillation: it breaks image/logit alignment")
    if args.shuffle_buffer_size != 1:
        raise SystemExit("--shuffle_buffer_size must be 1 for offline distillation (dataset order contract)")
    if args.dummy:
        write_dummy_cache(args, spec)
        return
    if not args.dataset_name:
        raise SystemExit("--dataset_name is required unless --dummy is given")
    run_export(args, spec)


if __name__ == "__main__":
    main()
