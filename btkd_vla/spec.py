"""BTKD-for-OpenVLA: action-token space and loss configuration.

OpenVLA discretises each action dimension into ``n_bins`` (default 256) bins and
maps bin ``d`` (``d in [1, n_bins]``) to token id ``vocab_size - d``.  All action
tokens therefore live in the *last* ``n_bins`` entries of the vocabulary, i.e.

    token id in [vocab_size - n_bins, vocab_size - 1]

We keep logits in **ascending token-id order** (a plain ``logits[..., -n_bins:]``
slice).  Column ``col`` of that slice corresponds to token
``vocab_size - n_bins + col`` and to bin ``n_bins - col``.  The exact ordering is
irrelevant for BTKD (calibration only touches two columns), but keeping one
convention everywhere avoids off-by-one bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class ActionSpec:
    """Static description of the action-token subspace of the vocabulary."""

    n_bins: int = 256
    n_dims: int = 7
    vocab_size: int = 32000

    def __post_init__(self) -> None:
        if self.n_bins <= 1:
            raise ValueError(f"n_bins must be > 1, got {self.n_bins}")
        if self.n_dims <= 0:
            raise ValueError(f"n_dims must be > 0, got {self.n_dims}")
        if self.vocab_size <= self.n_bins + 1:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must exceed n_bins ({self.n_bins}) + 1"
            )

    @property
    def offset(self) -> int:
        """First token id of the action subspace."""
        return self.vocab_size - self.n_bins

    @property
    def logit_slice(self) -> slice:
        return slice(self.offset, self.vocab_size)

    def col_of_token(self, token_id):
        """token id -> column index inside the ``[-n_bins:]`` logit slice."""
        import torch

        if isinstance(token_id, torch.Tensor):
            return token_id - self.offset
        return token_id - self.offset

    def token_of_col(self, col):
        return col + self.offset

    def in_space(self, token_id):
        import torch

        return (token_id >= self.offset) & (token_id < self.vocab_size)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActionSpec":
        return cls(n_bins=int(d["n_bins"]), n_dims=int(d["n_dims"]), vocab_size=int(d["vocab_size"]))


@dataclass
class BTKDConfig:
    """Hyper-parameters of the BTKD term.

    Mirrors ``dis_resnet18_cf100.py`` unless documented otherwise.
    """

    #: Distillation temperature (paper default 4.0).
    temperature: float = 4.0
    #: Weight of the student's own cross-entropy term (paper default 0.5).
    lambda_stu: float = 0.5
    #: "cos2" -> b = cos^2(pi/2 * step / total_steps)  (1 -> 0, easy -> hard)
    schedule: str = "cos2"
    #: Total optimiser steps, used by the schedule. Must be set before training.
    total_steps: int = 100_000
    #: alpha in the calibrated transfer  a2 = a + alpha*c/(a+c+1); 1.0 == BTKD's calibrate3.
    calibrate_strength: float = 1.0
    #: BTKD always uses b on easy and 1-b on hard.  With token-level masks one branch
    #: can be empty for a whole batch; if True we renormalise to the surviving branch
    #: instead of silently dropping the KD term.  Default False == paper-faithful.
    renormalize_empty_branch: bool = False
    #: "none" | "center" | "std" -- optional per-token logit re-scaling before KL.
    #: Use "none" when teacher and student are the same architecture (LoRA case).
    logit_norm: str = "none"
    #: Multiply the KL by T^2 (standard KD).  BTKD's script applies T*T outside.
    apply_t2: bool = True
