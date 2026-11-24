#!/usr/bin/env python
"""A6 step 2.5 -- materialise the training frames (removes TFDS from training).

The deterministic RLDS loader is required for cache alignment, but it decodes
every frame on the critical path (single read/map worker by design).  This script
walks that same loader *once* and writes exactly the tensors the model consumes
to a sharded frame store, in the same row order as the teacher cache:

    frames_dir/shard_00000.pt   input_ids / labels / pixel_values(fp16) / dataset_name

After that, training reads row ``i`` of the frame store together with row ``i`` of
the teacher cache: alignment is guaranteed by construction, so shuffling and
multiple epochs become safe and fast (no TFRecord, no JPEG decode on the GPU path).

Runs on CPU: only the tokenizer/image processor are loaded, never the 7B model.

Example
-------
    python scripts/materialize_frames.py \
        --cache_dir caches/libero_spatial_teacher \
        --out_dir caches/libero_spatial_teacher_frames \
        --dataset_name libero_spatial_no_noops \
        --vla_path openvla/openvla-7b
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btkd_vla.cache import DATA_ORDER, LogitCache  # noqa: E402
from btkd_vla.frames import MANIFEST, FrameStoreWriter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True, help="teacher-logit cache (defines row order)")
    p.add_argument("--out_dir", default="", help="frames dir (default: <cache_dir>_frames)")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--data_root_dir", default="datasets/open-x-embodiment")
    p.add_argument("--vla_path", default="openvla/openvla-7b", help="used for tokenizer + image size only")
    p.add_argument("--shard_size", type=int, default=1000)
    p.add_argument("--limit", type=int, default=-1, help="only materialise the first N rows")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_if_complete", action="store_true", help="exit 0 if the store is already complete")
    p.add_argument("--verify_every", type=int, default=1, help="check gt tokens against the cache every N samples (0=off)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir or (args.cache_dir.rstrip("/") + "_frames")

    if args.skip_if_complete and (Path(out_dir) / MANIFEST).is_file():
        try:
            from btkd_vla.frames import FrameStore

            store = FrameStore(out_dir, verbose=False)
            if store.is_complete:
                print(f"[frames] already complete: {out_dir} ({len(store)} samples); skipping")
                return
        except Exception:
            pass

    cache = LogitCache(args.cache_dir)
    if not cache.is_complete:
        raise SystemExit("cache is incomplete; finish the teacher export first")
    meta = cache.manifest.get("meta") or {}
    if meta.get("data_order") != DATA_ORDER:
        raise SystemExit(
            f"cache data order {meta.get('data_order')!r} != {DATA_ORDER!r}; re-export the cache first"
        )
    if meta.get("dataset") and meta["dataset"] != args.dataset_name:
        raise SystemExit(
            f"cache was exported for {meta['dataset']!r}, got --dataset_name {args.dataset_name!r}"
        )
    cached_root = meta.get("data_root_dir")
    if cached_root and os.path.abspath(str(cached_root)) != os.path.abspath(args.data_root_dir):
        raise SystemExit(
            f"cache data root {cached_root!r} != --data_root_dir {args.data_root_dir!r}"
        )

    spec = cache.spec
    n_target = len(cache) if args.limit < 0 else min(int(args.limit), len(cache))
    if n_target <= 0:
        raise SystemExit("nothing to materialise")

    # ---- processor only (no 7B weights) -------------------------------------------
    from transformers import AutoConfig, AutoProcessor

    from btkd_vla.vla_glue import register_openvla_classes

    register_openvla_classes()
    config = AutoConfig.from_pretrained(args.vla_path, trust_remote_code=True)
    image_sizes = tuple(getattr(config, "image_sizes", (224, 224)))
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform

    from btkd_vla.rlds_dataset import DeterministicSingleRLDSDataset

    action_tokenizer = ActionTokenizer(processor.tokenizer, bins=spec.n_bins)
    begin_idx = action_tokenizer.action_token_begin_idx
    if begin_idx != spec.vocab_size - spec.n_bins - 1:
        raise SystemExit(
            f"action_token_begin_idx={begin_idx} != cached spec "
            f"({spec.vocab_size} - {spec.n_bins} - 1)"
        )

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder
        if "v01" not in args.vla_path
        else VicunaV15ChatPromptBuilder,
    )
    dataset = DeterministicSingleRLDSDataset(
        Path(args.data_root_dir),
        args.dataset_name,
        batch_transform,
        resize_resolution=image_sizes,
        shuffle_buffer_size=1,
        train=True,
        image_aug=False,
    )
    if len(dataset) != len(cache):
        print(
            f"[frames] WARNING dataset_length={len(dataset)} != cache rows={len(cache)}; "
            f"materialising {n_target} rows",
            flush=True,
        )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    writer_cfg = dict(
        image_shape=(3, *image_sizes),
        dataset=args.dataset_name,
        shard_size=args.shard_size,
        meta={
            "data_order": DATA_ORDER,
            "data_root_dir": os.path.abspath(args.data_root_dir),
            "cache_dir": os.path.abspath(args.cache_dir),
            "n_dims": spec.n_dims,
            "n_bins": spec.n_bins,
            "vocab_size": spec.vocab_size,
            "began_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        overwrite=args.overwrite,
    )

    print(
        f"[frames] materialising {n_target} samples -> {out_dir} "
        f"(dataset={args.dataset_name}, image={image_sizes})",
        flush=True,
    )
    started = time.time()
    n_done = 0
    # The context manager marks the manifest complete only when the body exits
    # successfully.  A failed verification therefore cannot leave a half-built
    # store that looks train-ready.
    with FrameStoreWriter(out_dir, **writer_cfg) as writer:
        for instance in dataset:
            if n_done >= n_target:
                break
            batch = collator([instance])
            labels = batch["labels"][0]
            gt = labels[labels > begin_idx]
            if int(gt.numel()) != spec.n_dims:
                raise SystemExit(
                    f"sample {n_done}: expected {spec.n_dims} action tokens, got {int(gt.numel())}"
                )
            if args.verify_every and n_done % args.verify_every == 0:
                cached_gt = cache[n_done]["gt_token"]
                if not torch.equal(gt.long().cpu(), cached_gt.long()):
                    raise SystemExit(
                        "frames/cache misalignment at row "
                        f"{n_done}:\n  frames : {gt.tolist()}\n  cache  : {cached_gt.tolist()}\n"
                        "The frames must be materialised from the same deterministic loader "
                        "and the same dataset/cache pair."
                    )
            writer.add(
                batch["input_ids"][0],
                labels,
                batch["pixel_values"][0],
                dataset_name=instance.get("dataset_name", args.dataset_name),
            )
            n_done += 1
            if n_done % 500 == 0 or n_done == n_target:
                elapsed = time.time() - started
                rate = n_done / max(1e-6, elapsed)
                eta = (n_target - n_done) / max(1e-6, rate)
                print(
                    f"[frames] {n_done}/{n_target} rows | {rate:.2f} rows/s | "
                    f"elapsed {elapsed/60:.1f} min | eta {eta/60:.1f} min",
                    flush=True,
                )

    print(f"[frames] done: {out_dir} ({n_done} samples)")


if __name__ == "__main__":
    main()
