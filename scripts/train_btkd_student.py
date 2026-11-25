#!/usr/bin/env python
"""A6 step 3/4 -- train the OpenVLA student with BTKD using a cached teacher.

Only ONE model is resident here: the student (LoRA on ``openvla-7b``).  The
teacher distribution comes from the on-disk cache written by
``scripts/export_teacher_logits.py``, so a single 24 GB card is enough.

Loss (BTKD, with an optional easy plateau)::

    loss = lambda * CE + (1 - lambda) * [ b * KL(stu || tea) + (1-b) * KL(stu || calibrate3(tea)) ]
    b    = 1                                        if t < hold
         = cos^2(pi/2 * (t-hold)/(1-hold))          otherwise
    T^2 is folded into the KL term.

Data path
---------
* Preferred: a *materialised* frame store next to the cache
  (``caches/<name>_teacher_frames``, built by ``scripts/materialize_frames.py``).
  Row ``i`` of the frames == row ``i`` of the cache, so shuffling and multiple
  epochs are safe and there is no TFDS/JPEG decode on the training path.
* Fallback: the deterministic RLDS stream (requires ``--shuffle_buffer_size 1``),
  which replays the exact order the cache was exported in.

Steps
-----
``--epochs N`` is the recommended way to size a run: it converts to
``ceil(N * n_samples / batch_size)`` micro-batches.  ``--max_steps`` keeps the
old meaning (micro-batches) and is only used when ``--epochs 0``.

Example
-------
    python scripts/train_btkd_student.py \
        --cache_dir caches/libero_spatial_teacher \
        --dataset_name libero_spatial_no_noops \
        --epochs 2 --batch_size 2 --grad_accumulation_steps 8 \
        --gradient_checkpointing --learning_rate 3e-4 \
        --easy_hold_frac 0.4 --temperature 2.0 \
        --eval_steps 1000 --eval_limit 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btkd_vla.cache import DATA_ORDER, LogitCache, cache_covers_full_epoch  # noqa: E402
from btkd_vla.core import (  # noqa: E402
    btkd_term,
    cosine_square_schedule_hold,
    total_loss,
)
from btkd_vla.frames import MANIFEST as FRAMES_MANIFEST  # noqa: E402
from btkd_vla.frames import FrameStore  # noqa: E402
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
    p.add_argument("--epochs", type=float, default=0.0, help="number of dataset epochs (0 = use --max_steps)")
    p.add_argument("--max_steps", type=int, default=20_000, help="micro-batches; only used when --epochs 0")
    p.add_argument("--save_steps", type=int, default=2_000, help="optimizer updates between checkpoints")
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr_schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--lr_warmup_frac", type=float, default=0.03)

    # LoRA / memory
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--quantize_4bit", action="store_true", help="QLoRA: ~10 GB total")
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--gradient_checkpointing", action="store_true", help="trade compute for memory; lets bs>1 fit")

    # BTKD
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--lambda_stu", type=float, default=0.5)
    p.add_argument("--schedule", default="cos2", choices=["cos2"])
    p.add_argument("--easy_hold_frac", type=float, default=0.40,
                   help="keep b=1 (pure easy/teacher KD) for this fraction of training; 0 = paper's raw schedule")
    p.add_argument("--calibrate_strength", type=float, default=1.0)
    p.add_argument("--logit_norm", default="none", choices=["none", "center", "std"])
    p.add_argument("--renormalize_empty_branch", action="store_true")

    # data plumbing
    p.add_argument("--frames_dir", default="", help="materialised frames (default: <cache_dir>_frames)")
    p.add_argument("--no_shuffle_frames", action="store_true", help="disable shuffling on the frames path")
    p.add_argument(
        "--frame_shuffle",
        default="shard",
        choices=["shard", "global"],
        help=(
            "shard: randomise shard order, then rows inside each shard (each shard is "
            "read once per epoch -> fast); global: one permutation over all rows, which "
            "reloads a shard on almost every step when batch_size is small"
        ),
    )
    p.add_argument(
        "--frame_cache_shards",
        type=int,
        default=8,
        help="frame shards kept mmap-ed at once (0 = all; bigger = fewer torch.load calls)",
    )
    p.add_argument("--shuffle_buffer_size", type=int, default=1, help="must match the export run")
    p.add_argument("--image_aug", action="store_true", help="forbidden for offline distillation")
    p.add_argument("--verify_batches", type=int, default=5, help="alignment checks against cached GT tokens")
    p.add_argument(
        "--profile_steps",
        type=int,
        default=0,
        help="for the first N micro-batches, sync the GPU and report data/forward/backward seconds",
    )
    p.add_argument("--limit", type=int, default=-1, help="only use the first N cached samples")

    # in-training evaluation (frames path only)
    p.add_argument("--eval_steps", type=int, default=0, help="evaluate every N optimizer updates (0 = off)")
    p.add_argument("--eval_limit", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=1,
                   help="in-training eval batch (1 is safest together with a resident 7B)")

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


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def build_collator(processor):
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction

    return PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )


def build_rlds_loader(args, processor, vla, n_bins: int):
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
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
        Path(args.data_root_dir),
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=args.shuffle_buffer_size,
        train=True,
        image_aug=args.image_aug,
    )
    collator = build_collator(processor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0
    )
    return loader, begin_idx


def rlds_batches(loader, cache, n_batches: int, can_wrap: bool) -> Iterator[Tuple[Dict, Dict]]:
    """Sequential replay of the export order (fallback path)."""
    cache_ptr = 0
    wraps = 0
    for step, batch in enumerate(loader):
        if step >= n_batches:
            return
        n = int(batch["input_ids"].size(0))
        wrapped = False
        if cache_ptr == len(cache):
            if not can_wrap:
                raise RuntimeError(
                    f"cache exhausted at step {step} ({len(cache)} rows) and wrapping is disabled"
                )
            cache_ptr = 0
            wraps += 1
            wrapped = True
            print(f"[data] cache wrapped to row 0 (pass {wraps})", flush=True)
        if cache_ptr + n > len(cache) and not can_wrap:
            raise RuntimeError(
                f"cache exhausted at step {step} (row {cache_ptr}+{n} > {len(cache)})"
            )
        cached, cache_ptr, crossed = cache.get_window(cache_ptr, n, allow_wrap=can_wrap)
        if crossed:
            wraps += 1
            wrapped = True
            print(f"[data] cache wrapped across row 0 (pass {wraps})", flush=True)
        batch["_wrapped"] = wrapped
        yield batch, cached


def shard_local_permutation(store: FrameStore, n: int, gen: torch.Generator) -> torch.Tensor:
    """Shuffle *shards*, then rows inside each shard.

    A global ``randperm`` over 53k rows touches a random 300 MB shard on almost
    every ``batch_size=1`` step.  With only a handful of open shards that turns the
    input pipeline into ``torch.load`` + page-fault bound work and the GPU idles
    (the classic 0-30 % utilisation sawtooth).  Keeping the rows of one shard
    together means each shard is opened once per epoch while the sample order is
    still random.
    """
    n_shards = len(store.shards)
    if n_shards <= 1:
        return torch.randperm(n, generator=gen)
    shard_order = torch.randperm(n_shards, generator=gen).tolist()
    out = torch.empty(n, dtype=torch.long)
    pos = 0
    for si in shard_order:
        start, stop = store.shard_span(si)
        if start >= n:
            continue
        count = min(stop, n) - start
        out[pos : pos + count] = start + torch.randperm(count, generator=gen)
        pos += count
    if pos != n:  # pragma: no cover - manifest would be inconsistent
        raise RuntimeError(f"shard-local permutation covered {pos} rows, expected {n}")
    return out


def frames_batches(
    store: FrameStore,
    cache: LogitCache,
    collator,
    batch_size: int,
    n_samples: int,
    shuffle: bool,
    seed: int,
    shard_local: bool = True,
) -> Iterator[Tuple[Dict, Dict]]:
    """Random-access epochs over the materialised store (fast path).

    The generator is infinite: the caller stops at ``total_steps``.  Every batch
    draws ``idx`` once and uses it for *both* the frames and the teacher cache, so
    the two can never drift apart.
    """
    n = min(int(n_samples), len(store), len(cache))
    if batch_size <= 0:
        raise SystemExit(f"batch_size must be positive, got {batch_size}")
    if n <= 0:
        raise SystemExit("no samples available for the materialised-frame path")
    gen = torch.Generator().manual_seed(int(seed))
    epoch = 0
    while True:
        if not shuffle:
            perm = torch.arange(n)
        elif shard_local:
            perm = shard_local_permutation(store, n, gen)
        else:
            perm = torch.randperm(n, generator=gen)
        # Keep the tail batch (do not silently drop samples).  The step budget is
        # computed from steps_per_epoch=ceil(n/batch_size), so every epoch covers
        # exactly all available rows.
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = collator([store[int(i)] for i in idx])
            cached = cache.get_rows(idx)
            batch["_wrapped"] = False
            batch["_idx"] = idx
            yield batch, cached
        epoch += 1
        if epoch == 1:
            stats = getattr(store, "stats", {})
            print(
                f"[data] frames epoch 1 done ({n} rows, "
                f"{math.ceil(n / batch_size)} batches, batch_size={batch_size}, "
                f"shard loads={stats.get('loads', '?')}, hits={stats.get('hits', '?')})",
                flush=True,
            )
            if hasattr(store, "reset_stats"):
                store.reset_stats()


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------
def check_alignment_labels(
    cached: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    begin_idx: int,
    batch_idx: int,
    spec,
) -> None:
    """Cheap pre-forward check: action tokens in ``labels`` must equal cached GT."""
    mask = labels > int(begin_idx)
    per_row = mask.sum(dim=1)
    if not torch.all(per_row == spec.n_dims):
        raise RuntimeError(
            f"expected {spec.n_dims} supervised action tokens per row, got {per_row.tolist()} "
            f"at batch {batch_idx}"
        )
    b_idx, t_idx = mask.nonzero(as_tuple=False).unbind(dim=1)
    gt = labels[b_idx, t_idx].reshape(labels.size(0), spec.n_dims).long().cpu()
    want = cached["gt_token"].long().cpu()
    if gt.shape != want.shape or not torch.equal(gt, want):
        raise RuntimeError(
            f"cache/dataset misalignment at batch {batch_idx}.\n"
            f"  cached : {want[:2].tolist()} ...\n"
            f"  dataset: {gt[:2].tolist()} ...\n"
            "Rebuild the frames with scripts/materialize_frames.py from the same cache, "
            "or (RLDS path) re-export the cache with the deterministic loader."
        )


def lr_at(step: int, total_steps: int, base_lr: float, warmup_frac: float, schedule: str) -> float:
    if total_steps <= 0:
        return base_lr
    warm = max(1, int(round(max(0.0, warmup_frac) * total_steps)))
    if step < warm:
        return base_lr * float(step + 1) / float(warm)
    if schedule == "constant":
        return base_lr
    span = max(1, total_steps - warm)
    prog = min(1.0, max(0.0, float(step - warm) / float(span)))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * prog))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def resolve_plan(
    n_samples: int,
    batch_size: int,
    epochs: float,
    max_steps: int,
    grad_accumulation_steps: int,
) -> Dict[str, Any]:
    """Turn a dataset size + epoch budget into micro-batch / optimizer-update counts.

    ``--epochs > 0`` wins over ``--max_steps`` and each epoch covers *all* samples
    (ceil, so the tail batch is not dropped).  ``--max_steps`` keeps its legacy
    meaning: micro-batches, **not** optimizer updates - 20000 micro-batches with
    accum=16 is only 1250 updates, which is how the first A6 run ended up at
    ~0.38 epochs.
    """
    if n_samples <= 0:
        raise SystemExit("empty training plan: no samples available")
    steps_per_epoch = int(math.ceil(n_samples / max(1, int(batch_size))))
    accum = max(1, int(grad_accumulation_steps))
    if epochs and float(epochs) > 0:
        total_steps = int(math.ceil(float(epochs) * steps_per_epoch))
    else:
        total_steps = int(max_steps)
    if total_steps <= 0:
        raise SystemExit("empty training plan: set --epochs > 0 or --max_steps > 0")
    return {
        "steps_per_epoch": steps_per_epoch,
        "total_micro_batches": total_steps,
        "optimizer_updates": int(math.ceil(total_steps / accum)),
        "epochs_done": total_steps / float(steps_per_epoch),
    }


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
    if cache_meta.get("data_order") != DATA_ORDER:
        raise SystemExit(
            f"cache data order is {cache_meta.get('data_order')!r}, expected {DATA_ORDER!r}. "
            "This cache predates deterministic RLDS ordering; re-export it with "
            "scripts/export_teacher_logits.py --overwrite."
        )
    can_wrap_cache = cache_covers_full_epoch(cache_meta, full_cache_len, args.limit)
    spec = cache.spec

    if args.check_cache_only:
        stats = cache.analysis(
            max_samples=None if args.limit <= 0 else min(args.limit, full_cache_len)
        )
        print("[check] " + json.dumps(stats, indent=2))
        bad = stats["nonfinite_logits"] > 0
        print(
            "[check] interpretation: teacher token accuracy drives the easy/hard split; "
            f"{stats['easy_frac'] * 100:.1f}% of action tokens go to the easy branch."
        )
        raise SystemExit(1 if bad else 0)

    cached_root = cache_meta.get("data_root_dir")
    if cached_root and os.path.abspath(str(cached_root)) != os.path.abspath(args.data_root_dir):
        raise SystemExit(
            f"cache was exported from {cached_root!r}, but --data_root_dir is "
            f"{args.data_root_dir!r}; use the matching data root or re-export it"
        )

    # ---- materialised frames (fast path) ------------------------------------------
    frames_dir = args.frames_dir or (args.cache_dir.rstrip("/") + "_frames")
    store: Optional[FrameStore] = None
    expected_cache_rows = (
        full_cache_len if args.limit <= 0 else min(int(args.limit), full_cache_len)
    )
    if (Path(frames_dir) / FRAMES_MANIFEST).is_file():
        store = FrameStore(frames_dir, max_open_shards=args.frame_cache_shards)
        if not store.is_complete:
            raise SystemExit(f"frames at {frames_dir} are incomplete; re-run scripts/materialize_frames.py")
        frames_meta = store.manifest.get("meta") or {}
        frames_cache = frames_meta.get("cache_dir")
        if frames_cache and os.path.abspath(str(frames_cache)) != os.path.abspath(cache.cache_dir):
            raise SystemExit(
                f"frames were built from cache {frames_cache!r}, but --cache_dir is "
                f"{cache.cache_dir!r}; rebuild the frames from the matching cache"
            )
        if len(store) < expected_cache_rows:
            raise SystemExit(
                f"frames have {len(store)} rows but this run needs {expected_cache_rows}; "
                "rebuild them with scripts/materialize_frames.py (omit --limit for full training)"
            )
        if args.limit <= 0 and len(store) != full_cache_len:
            raise SystemExit(
                f"frames have {len(store)} rows but the cache has {full_cache_len}; "
                "they must be built from the same full export (scripts/materialize_frames.py)"
            )
        print(
            f"[data] fast path: materialised frames ({len(store)} rows), "
            f"using {expected_cache_rows}; no TFDS/JPEG decode in the loop",
            flush=True,
        )
    else:
        print(
            f"[data] {frames_dir} not found -> falling back to the deterministic RLDS stream "
            f"(slower: TFDS decode on the critical path).\n"
            f"       Build it once with: python scripts/materialize_frames.py --cache_dir {args.cache_dir} "
            f"--dataset_name {args.dataset_name}",
            flush=True,
        )

    # Logical truncation must happen *after* the frames/cache consistency check;
    # otherwise a full frame store plus --limit looks like a length mismatch.
    if args.limit > 0:
        cache.shrink_to(expected_cache_rows)
    n_avail = expected_cache_rows

    plan = resolve_plan(
        n_avail, args.batch_size, args.epochs, args.max_steps, args.grad_accumulation_steps
    )
    steps_per_epoch = plan["steps_per_epoch"]
    total_steps = plan["total_micro_batches"]
    updates = plan["optimizer_updates"]
    epochs_done = plan["epochs_done"]
    print(
        f"[plan] samples={n_avail} steps/epoch={steps_per_epoch} "
        f"batch={args.batch_size} accum={args.grad_accumulation_steps} "
        f"-> micro_batches={total_steps} optimizer_updates={updates} "
        f"({epochs_done:.2f} epochs), lr={args.learning_rate} "
        f"warmup={args.lr_warmup_frac:g} easy_hold={args.easy_hold_frac:g} T={args.temperature}",
        flush=True,
    )

    exp = args.exp_name or (
        f"btkd+{Path(args.vla_path).name}+{args.dataset_name}"
        f"+b{args.batch_size * args.grad_accumulation_steps}"
        f"+lr{args.learning_rate}+T{args.temperature}+lam{args.lambda_stu}"
        + (f"+ep{args.epochs:g}" if args.epochs > 0 else "")
        + (f"+hold{args.easy_hold_frac:g}" if args.easy_hold_frac > 0 else "")
        + ("+q4bit" if args.quantize_4bit else "")
    )
    run_dir = Path(args.run_root_dir) / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **vars(args),
        "resolved": {
            "cache_dir": cache.cache_dir,
            "frames_dir": str(Path(frames_dir).resolve()),
            "frames_found": store is not None,
            "n_available": n_avail,
            "steps_per_epoch": steps_per_epoch,
            "total_micro_batches": total_steps,
            "optimizer_updates": updates,
            "epochs_done": epochs_done,
            "exp_name": exp,
        },
    }
    (run_dir / "train_config.json").write_text(
        json.dumps(resolved_config, indent=2), encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")  # one run == one metrics file

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
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print("[model] gradient checkpointing enabled", flush=True)
        except Exception as exc:  # pragma: no cover - depends on peft version
            print(f"[model] WARNING could not enable gradient checkpointing: {exc}", flush=True)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    tokenizer = processor.tokenizer
    if int(tokenizer.vocab_size) != int(spec.vocab_size):
        raise SystemExit(
            f"tokenizer vocab_size={tokenizer.vocab_size} != cached spec vocab_size={spec.vocab_size}. "
            "Re-export the cache with the matching --vocab_size."
        )

    collator = build_collator(processor)
    if store is not None:
        from prismatic.vla.action_tokenizer import ActionTokenizer

        begin_idx = ActionTokenizer(tokenizer, bins=spec.n_bins).action_token_begin_idx
        loader = None
    else:
        loader, begin_idx, _ = build_rlds_loader(args, processor, model, spec.n_bins)
    if begin_idx != spec.vocab_size - spec.n_bins - 1:
        raise SystemExit(
            f"action_token_begin_idx={begin_idx} inconsistent with cached spec {spec.to_dict()}"
        )

    # ---- optimisation config --------------------------------------------------------
    cfg = BTKDConfig(
        temperature=args.temperature,
        lambda_stu=args.lambda_stu,
        schedule=args.schedule,
        total_steps=total_steps,
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

    if store is not None:
        source = frames_batches(
            store,
            cache,
            collator,
            args.batch_size,
            n_avail,
            shuffle=not args.no_shuffle_frames,
            seed=args.seed,
            shard_local=(args.frame_shuffle == "shard"),
        )
    else:
        source = rlds_batches(loader, cache, total_steps, can_wrap_cache)

    device = args.device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Keep these as GPU scalars: calling ``.item()`` once per micro-batch forces a
    # full stream synchronisation (with bs=1 x accum=16 that is ~8 syncs/step) and
    # stops the CPU from preparing the next sample while the GPU works.  One sync
    # per log window is enough.
    ce_hist: List[torch.Tensor] = []
    kd_hist: List[torch.Tensor] = []
    easy_hist: List[torch.Tensor] = []
    hard_hist: List[torch.Tensor] = []
    acc_hist: List[torch.Tensor] = []
    loss_hist: List[torch.Tensor] = []

    def _window_mean(hist: List[torch.Tensor], window: slice) -> float:
        vals = hist[window]
        if not vals:
            return 0.0
        return float(torch.stack([v.detach().float().reshape(()) for v in vals]).mean().item())

    best_acc = -1.0
    best_step = -1
    n_updates = 0
    started = time.time()
    step = -1
    source_iter = iter(source)
    profiling = bool(args.profile_steps) and device.startswith("cuda")
    prof_data = prof_fwd = prof_bwd = 0.0
    prof_n = 0

    while step + 1 < total_steps:
        step += 1
        if profiling and step < args.profile_steps:
            torch.cuda.synchronize()
        t_data0 = time.perf_counter()
        try:
            batch, cached = next(source_iter)
        except StopIteration:  # pragma: no cover - the generators are infinite
            break
        t_data1 = time.perf_counter()

        labels = batch["labels"]
        if step < args.verify_batches or batch.get("_wrapped"):
            check_alignment_labels(cached, labels, begin_idx, step, spec)

        b = cosine_square_schedule_hold(step, total_steps, args.easy_hold_frac)
        use_amp = device.startswith("cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=move_pixel_values(batch["pixel_values"], device, torch.bfloat16),
                labels=labels.to(device),
            )
            ce = out.loss
            action_logits, gt_token, _ = extract_action_logits(
                out.logits, labels.to(device), begin_idx, spec
            )
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

        if profiling and step < args.profile_steps:
            torch.cuda.synchronize()
            t_fwd1 = time.perf_counter()

        did_update = (
            (step + 1) % args.grad_accumulation_steps == 0
            or (step + 1) >= total_steps
        )
        (loss / args.grad_accumulation_steps).backward()
        if did_update:
            lr = lr_at(step, total_steps, args.learning_rate, args.lr_warmup_frac, args.lr_schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_updates += 1

        if profiling and step < args.profile_steps:
            torch.cuda.synchronize()
            t_bwd1 = time.perf_counter()
            prof_data += t_data1 - t_data0
            prof_fwd += t_fwd1 - t_data1
            prof_bwd += t_bwd1 - t_fwd1
            prof_n += 1

        loss_hist.append(loss.detach())
        ce_hist.append(ce.detach())
        kd_hist.append(kd["loss"].detach())
        easy_hist.append(kd["easy"].detach())
        hard_hist.append(kd["hard"].detach())
        acc_hist.append(
            (cached["teacher_argmax_token"] == cached["gt_token"]).float().mean().detach()
        )

        if step % args.log_steps == 0:
            window = slice(-args.log_steps, None)
            progress = (step + 1) / max(1, total_steps)
            elapsed = time.time() - started
            eta = elapsed / max(1e-9, progress) - elapsed
            record = {
                "step": step,
                "update": n_updates,
                "progress": round(progress, 4),
                "epoch": round((step + 1) * args.batch_size / max(1, n_avail), 3),
                "loss": _window_mean(loss_hist, window),
                "ce": _window_mean(ce_hist, window),
                "kd": _window_mean(kd_hist, window),
                "kd_easy": _window_mean(easy_hist, window),
                "kd_hard": _window_mean(hard_hist, window),
                "b": round(b, 4),
                "easy_frac": round(float(kd["easy_frac"]), 4),
                "teacher_tok_acc": _window_mean(acc_hist, window),
                "lr": optimizer.param_groups[0]["lr"],
                "eta_min": round(eta / 60.0, 1),
            }
            if prof_n:
                record["prof_data_s"] = round(prof_data / prof_n, 3)
                record["prof_fwd_s"] = round(prof_fwd / prof_n, 3)
                record["prof_bwd_s"] = round(prof_bwd / prof_n, 3)
            print(
                f"[step {step:6d} upd {n_updates:5d}] loss={record['loss']:.4f} ce={record['ce']:.4f} "
                f"kd={record['kd']:.4f} easy={record['kd_easy']:.4f} hard={record['kd_hard']:.4f} "
                f"b={record['b']:.3f} easy_frac={record['easy_frac']:.3f} lr={record['lr']:.2e} "
                f"ep={record['epoch']:.2f} eta={record['eta_min']:.0f}min"
                + (
                    f" | prof data={prof_data / prof_n:.2f}s fwd={prof_fwd / prof_n:.2f}s "
                    f"bwd={prof_bwd / prof_n:.2f}s (first {prof_n} steps)"
                    if prof_n
                    else ""
                ),
                flush=True,
            )
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "train", **record}) + "\n")
            if wandb is not None:
                wandb.log(record)

        # ---- in-training evaluation (frames path only) ---------------------------
        if args.eval_steps > 0 and did_update and n_updates % args.eval_steps == 0:
            if store is None:
                print("[eval] skipped: in-training eval needs a materialised frame store", flush=True)
            else:
                from btkd_vla.evaluate import offline_eval_frames

                t0 = time.time()
                metrics = offline_eval_frames(
                    model,
                    collator,
                    store,
                    cache,
                    spec,
                    begin_idx,
                    device,
                    batch_size=args.eval_batch_size,
                    limit=args.eval_limit,
                    progress_every=0,
                )
                acc = metrics["student_token_acc"]
                mean6 = sum(metrics["per_dim_acc"][:6]) / 6.0
                print(
                    f"[eval @ update {n_updates} (step {step})] student={acc:.4f} dim0-5={mean6:.4f} "
                    f"teacher={metrics['teacher_token_acc']:.4f} "
                    f"agreement={metrics['student_teacher_agreement']:.4f} "
                    f"({time.time() - t0:.0f}s, {args.eval_limit} rows)",
                    flush=True,
                )
                with metrics_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"type": "eval", "step": step, "update": n_updates, **metrics}) + "\n")
                if wandb is not None:
                    wandb.log({"eval_step": step, "eval_update": n_updates, **{f"eval_{k}": v for k, v in metrics.items()
                                                     if isinstance(v, (int, float))}})
                if acc > best_acc:
                    best_acc = acc
                    best_step = step
                    model.save_pretrained(run_dir / "best")
                    print(
                        f"[eval] new best -> {run_dir / 'best'} "
                        f"(update {n_updates}, step {step}, acc {acc:.4f})",
                        flush=True,
                    )
                model.train()

        if args.save_steps > 0 and did_update and n_updates % args.save_steps == 0:
            ckpt = run_dir / f"checkpoint-{n_updates}"
            model.save_pretrained(ckpt)
            print(f"[save] {ckpt} (update {n_updates}, step {step})", flush=True)

    final = run_dir / "final"
    model.save_pretrained(final)
    elapsed = time.time() - started
    print(
        f"[done] adapter saved to {final} ({elapsed/3600:.2f} h, "
        f"{step + 1} micro-batches, {n_updates} optimizer updates)"
    )
    if best_acc >= 0:
        print(f"[done] best eval acc {best_acc:.4f} at step {best_step} -> {run_dir / 'best'}")
    else:
        print("[done] no in-training eval ran; use eval mode to compare final/ and checkpoints")


if __name__ == "__main__":
    main()
