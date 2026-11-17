"""Thin glue between OpenVLA models/datasets and :mod:`btkd_vla`.

Everything that imports ``prismatic`` / ``transformers`` lives here and is imported
lazily, so that the pure-math parts of the package stay importable on a CPU-only box
without the OpenVLA dependency stack.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .spec import ActionSpec

__all__ = [
    "register_openvla_classes",
    "load_openvla_model_and_processor",
    "extract_action_logits",
    "count_action_positions",
    "move_pixel_values",
]


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------
def register_openvla_classes() -> None:
    """Register OpenVLA's custom config/processor with HF AutoClasses (idempotent)."""
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    for auto_cls, model_cls, handler in (
        (AutoConfig, OpenVLAConfig, None),
        (AutoImageProcessor, OpenVLAConfig, PrismaticImageProcessor),
        (AutoProcessor, OpenVLAConfig, PrismaticProcessor),
        (AutoModelForVision2Seq, OpenVLAConfig, OpenVLAForActionPrediction),
    ):
        try:
            if handler is None:
                auto_cls.register("openvla", model_cls)
            else:
                auto_cls.register(model_cls, handler)
        except ValueError:
            # already registered in this process
            pass


def load_openvla_model_and_processor(
    vla_path: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    quantize_4bit: bool = False,
    attn_implementation: str = "sdpa",
    trainable_lora: bool = False,
    lora_rank: int = 32,
    lora_dropout: float = 0.0,
):
    """Load an OpenVLA checkpoint (teacher: frozen; student: optional LoRA)."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    register_openvla_classes()

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    kwargs: Dict[str, Any] = dict(
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device}
    else:
        kwargs["device_map"] = {"": device}

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(vla_path, **kwargs)

    if trainable_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if quantize_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=min(lora_rank, 16),
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_cfg)

    if trainable_lora:
        model.train()
    else:
        model.eval()
    return model, processor


# --------------------------------------------------------------------------------------
# Action logits
# --------------------------------------------------------------------------------------
def count_action_positions(labels: torch.Tensor, action_token_begin_idx: int) -> torch.Tensor:
    """Number of supervised action tokens per row: ``labels > action_token_begin_idx``."""
    return (labels > int(action_token_begin_idx)).sum(dim=1)


def extract_action_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_token_begin_idx: int,
    spec: ActionSpec,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the logits that *predict* the supervised action tokens.

    ``labels`` uses -100 for non-action positions (OpenVLA's collator convention), so
    the action region is ``labels > action_token_begin_idx``.  A logit at position
    ``t - 1`` predicts the token at position ``t``; that is exactly the alignment used
    by ``vla-scripts/finetune.py``.

    Returns:
        action_logits       [B, D, n_bins]  (fp32, last-n_bins slice, ascending token id)
        gt_token            [B, D]          int64
        teacher_argmax_token[B, D]          int64, argmax over the *full* vocabulary
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, L, V], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"labels {tuple(labels.shape)} vs logits {tuple(logits.shape)}")

    mask = labels > int(action_token_begin_idx)
    per_row = mask.sum(dim=1)
    d = spec.n_dims
    if not torch.all(per_row == d):
        bad = (per_row != d).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"expected exactly {d} supervised action tokens per sample, got {per_row.tolist()} "
            f"(offending rows: {bad[:8]}). "
            "Check that the collator/labels match OpenVLA's action-token convention."
        )

    b_idx, t_idx = mask.nonzero(as_tuple=False).unbind(dim=1)
    if bool((t_idx == 0).any()):
        raise ValueError("action token found at position 0 -- cannot shift by one")

    flat = logits[b_idx, t_idx - 1, :]  # [B*D, V]
    n_bins = spec.n_bins
    action_logits = flat[:, spec.logit_slice].reshape(logits.size(0), d, n_bins).float()
    gt_token = labels[b_idx, t_idx].reshape(logits.size(0), d).long()
    teacher_argmax_token = flat.argmax(dim=-1).reshape(logits.size(0), d).long()
    return action_logits, gt_token, teacher_argmax_token


def move_pixel_values(pixel_values, device, dtype=None):
    """Move ``pixel_values`` to a device, tolerating both tensor and dict layouts."""
    if isinstance(pixel_values, dict):
        return {k: move_pixel_values(v, device, dtype) for k, v in pixel_values.items()}
    out = pixel_values.to(device)
    if dtype is not None:
        out = out.to(dtype)
    return out
