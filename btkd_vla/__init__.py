"""BTKD (Bridging-Teacher-Knowledge Distillation) adapted to OpenVLA action tokens.

Entry points:
    btkd_vla.core        pure-tensor BTKD math (calibrate3, easy/hard, schedule)
    btkd_vla.cache       sharded fp16 teacher-logit cache (the "A6" offline route)
    btkd_vla.vla_glue    OpenVLA model loading + action-logit extraction
"""

from .spec import ActionSpec, BTKDConfig
from .core import (
    btkd_term,
    calibrate3_logits,
    cosine_square_schedule,
    masked_mean,
    per_token_kl,
    split_easy_hard,
    teacher_token_accuracy,
    total_loss,
)
from .cache import LogitCache, LogitCacheWriter

__version__ = "0.1.0"

__all__ = [
    "ActionSpec",
    "BTKDConfig",
    "LogitCache",
    "LogitCacheWriter",
    "btkd_term",
    "calibrate3_logits",
    "cosine_square_schedule",
    "masked_mean",
    "per_token_kl",
    "split_easy_hard",
    "teacher_token_accuracy",
    "total_loss",
]
