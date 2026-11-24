#!/usr/bin/env python
"""A6 step 4/4 -- validate a BTKD student (base OpenVLA + LoRA adapter) offline.

This runs the student on the same deterministic RLDS stream used for training and
compares its action-token predictions against

* the cached teacher argmax (how much of the teacher's behaviour was absorbed),
* the ground-truth action tokens (the task metric),
* the cached teacher's own accuracy (the ceiling of this offline check).

No simulator and no teacher model are needed: only the student (one 7B model) and
the existing teacher-logit cache.  Use it as a sanity check before running the
much heavier LIBERO simulation evaluation.

Examples
--------
    # final adapter vs cached spatial teacher, 512 samples (default)
    python scripts/eval_btkd_student.py \
        --cache_dir caches/libero_spatial_teacher \
        --dataset_name libero_spatial_no_noops \
        --adapter_path runs_btkd/btkd+openvla-7b+libero_spatial_no_noops+b16+lr0.0005+T4.0+lam0.5/final

    # baseline: same command without --adapter_path
    python scripts/eval_btkd_student.py \
        --cache_dir caches/libero_spatial_teacher \
        --dataset_name libero_spatial_no_noops

    # full cache
    python scripts/eval_btkd_student.py ... --limit -1 --batch_size 8

Caveat: rows come from the cache, which is the training set.  These numbers are a
fit/sanity check, not a held-out generalisation estimate; only LIBERO rollouts give
the real success rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btkd_vla.cache import DATA_ORDER, LogitCache  # noqa: E402
from btkd_vla.vla_glue import (  # noqa: E402
    extract_action_logits,
    load_openvla_model_and_processor,
    move_pixel_values,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--cache_dir", required=True, help="teacher-logit cache used for training")
    p.add_argument("--adapter_path", default="", help="LoRA adapter dir (final/ or checkpoint-XXXX); empty = base model")
    p.add_argument("--vla_path", default="openvla/openvla-7b", help="base OpenVLA checkpoint")
    p.add_argument("--data_root_dir", default="datasets/open-x-embodiment")
    p.add_argument("--dataset_name", required=True)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--limit", type=int, default=512, help="number of cached rows to evaluate (-1 = all)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--quantize_4bit", action="store_true")
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--output_json", default="", help="optional path for the metrics JSON")
    p.add_argument("--seed", type=int, default=0)
    return p


def build_dataloader(args, processor, model, spec):
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform

    from btkd_vla.rlds_dataset import DeterministicSingleRLDSDataset

    action_tokenizer = ActionTokenizer(processor.tokenizer, bins=spec.n_bins)
    begin_idx = action_tokenizer.action_token_begin_idx
    if begin_idx != spec.vocab_size - spec.n_bins - 1:
        raise SystemExit(
            f"action_token_begin_idx={begin_idx} does not match cached spec "
            f"(vocab_size={spec.vocab_size}, n_bins={spec.n_bins})"
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
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=1,
        train=True,
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0
    )
    return loader, begin_idx


def _take_batch(batch: Dict, n: int) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[:n]
        elif isinstance(v, dict):
            out[k] = {kk: vv[:n] for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def check_alignment(cached: Dict[str, torch.Tensor], gt_token: torch.Tensor, batch_idx: int) -> None:
    a = cached["gt_token"]
    b = gt_token.cpu()
    if a.shape != b.shape or not torch.equal(a, b):
        raise RuntimeError(
            f"cache/dataset misalignment at batch {batch_idx}.\n"
            f"  cached : {a[:2, :4].tolist()} ...\n"
            f"  dataset: {b[:2, :4].tolist()} ...\n"
            "Use the cache exported by the current deterministic loader, with the same "
            "--dataset_name / --data_root_dir."
        )


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    cache = LogitCache(args.cache_dir)
    if not cache.is_complete:
        raise SystemExit(f"cache at {args.cache_dir} is incomplete; finish the export first")
    meta = cache.manifest.get("meta") or {}
    cached_dataset = meta.get("dataset")
    if cached_dataset and cached_dataset != args.dataset_name:
        raise SystemExit(
            f"cache was exported for {cached_dataset!r}, but --dataset_name is {args.dataset_name!r}"
        )
    if meta.get("data_order") != DATA_ORDER:
        raise SystemExit(
            f"cache data order is {meta.get('data_order')!r}, expected {DATA_ORDER!r}; re-export it"
        )
    cached_root = meta.get("data_root_dir")
    if cached_root and os.path.abspath(str(cached_root)) != os.path.abspath(args.data_root_dir):
        print(
            f"[eval] warning: cache data root {cached_root!r} != --data_root_dir "
            f"{args.data_root_dir!r}; row order must still match"
        )

    spec = cache.spec
    n_eval = len(cache) if args.limit < 0 else min(args.limit, len(cache))
    if n_eval <= 0:
        raise SystemExit("nothing to evaluate")

    adapter_dir = None
    if args.adapter_path:
        adapter_dir = Path(args.adapter_path)
        if not (adapter_dir / "adapter_config.json").is_file():
            raise SystemExit(
                f"{adapter_dir} has no adapter_config.json; pass final/ or checkpoint-XXXX/"
            )

    print(f"[eval] model={'base' if adapter_dir is None else str(adapter_dir)}")
    model, processor = load_openvla_model_and_processor(
        args.vla_path,
        device=args.device,
        dtype="bfloat16",
        quantize_4bit=args.quantize_4bit,
        attn_implementation=args.attn_implementation,
        trainable_lora=False,
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()

    if int(processor.tokenizer.vocab_size) != int(spec.vocab_size):
        raise SystemExit(
            f"tokenizer vocab_size={processor.tokenizer.vocab_size} != cached {spec.vocab_size}"
        )

    loader, begin_idx = build_dataloader(args, processor, model, spec)
    print(f"[eval] rows={n_eval} batch_size={args.batch_size} dataset={args.dataset_name}")

    device = args.device
    ptr = 0
    n_seen = 0
    ce_sum = 0.0
    correct = 0
    teacher_correct = 0
    agree = 0
    agree_denom = 0
    teacher_out_of_space = 0
    easy_correct = easy_total = 0
    hard_correct = hard_total = 0
    dim_correct = torch.zeros(spec.n_dims, dtype=torch.long)
    dim_total = torch.zeros(spec.n_dims, dtype=torch.long)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if ptr >= n_eval:
                break
            b = min(int(batch["input_ids"].size(0)), n_eval - ptr)
            batch = _take_batch(batch, b)
            cached, ptr, _ = cache.get_window(ptr, b, allow_wrap=False)

            labels = batch["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=move_pixel_values(batch["pixel_values"], device, torch.bfloat16),
                    labels=labels,
                )

            if batch_idx == 0:
                _, gt0, _ = extract_action_logits(out.logits, labels, begin_idx, spec)
                check_alignment(cached, gt0, batch_idx)

            action_logits, gt_token, _ = extract_action_logits(out.logits, labels, begin_idx, spec)
            gt_token = gt_token.cpu()
            gt_col = spec.col_of_token(gt_token)
            stu_col = action_logits.argmax(dim=-1).cpu()

            tea_tok = cached["teacher_argmax_token"]
            tea_in = spec.in_space(tea_tok)
            teacher_out_of_space += int((~tea_in).sum().item())
            tea_col = spec.col_of_token(tea_tok)

            matches_stu = stu_col == gt_col
            matches_tea = tea_tok == gt_token
            correct += int(matches_stu.sum().item())
            teacher_correct += int(matches_tea.sum().item())

            # Agreement only where the teacher's argmax is a legal action token.
            agree += int(((stu_col == tea_col) & tea_in).sum().item())
            agree_denom += int(tea_in.sum().item())

            easy = matches_tea
            easy_correct += int((matches_stu & easy).sum().item())
            easy_total += int(easy.sum().item())
            hard = ~easy
            hard_correct += int((matches_stu & hard).sum().item())
            hard_total += int(hard.sum().item())

            for d in range(spec.n_dims):
                dim_correct[d] += int(matches_stu[:, d].sum().item())
                dim_total[d] += matches_stu.size(0)

            ce_sum += float(out.loss.item()) * b
            n_seen += b
            if batch_idx % 10 == 0 or n_seen >= n_eval:
                print(f"[eval] {n_seen}/{n_eval} rows", flush=True)

    if n_seen == 0:
        raise SystemExit("no rows evaluated")

    n_tokens = n_seen * spec.n_dims
    metrics = {
        "n_samples": n_seen,
        "n_tokens": n_tokens,
        "ce": ce_sum / n_seen,
        "student_token_acc": correct / n_tokens,
        "teacher_token_acc": teacher_correct / n_tokens,
        "student_teacher_agreement": (agree / agree_denom) if agree_denom else 0.0,
        "teacher_out_of_space_frac": teacher_out_of_space / n_tokens,
        "student_token_acc_easy": (easy_correct / easy_total) if easy_total else 0.0,
        "student_token_acc_hard": (hard_correct / hard_total) if hard_total else 0.0,
        "easy_frac": (easy_total / n_tokens) if n_tokens else 0.0,
        "per_dim_acc": [round(dim_correct[d].item() / max(1, dim_total[d].item()), 4) for d in range(spec.n_dims)],
        "adapter": args.adapter_path or None,
        "cache_dir": os.path.abspath(args.cache_dir),
        "dataset": args.dataset_name,
    }

    print("\n[eval] summary")
    print(json.dumps(metrics, indent=2))
    gap = metrics["teacher_token_acc"] - metrics["student_token_acc"]
    print(
        f"\n[eval] student={metrics['student_token_acc']:.4f} "
        f"teacher={metrics['teacher_token_acc']:.4f} gap={gap:+.4f} "
        f"agreement={metrics['student_teacher_agreement']:.4f}"
    )
    if gap <= 0.03:
        print("[eval] 结论：学生已基本贴近缓存 teacher 的动作 token 分布。")
    elif gap <= 0.10:
        print("[eval] 结论：已明显学到，但与 teacher 仍有差距；可继续训练或增大 LoRA rank。")
    else:
        print("[eval] 结论：差距较大，检查是否训够步数、adapter 是否是最终版本。")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[eval] metrics written to {args.output_json}")


if __name__ == "__main__":
    main()
