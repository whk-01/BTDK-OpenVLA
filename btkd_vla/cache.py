"""Sharded on-disk cache for BTKD teacher logits (the "A6" offline route).

Layout::

    <cache_dir>/
        manifest.json          # spec + meta + shard table
        shard_00000.pt         # {"gt_token", "teacher_logits", "teacher_argmax_token", "start"}
        shard_00001.pt
        ...

Each shard is a plain ``torch.save`` blob so it can be memory-mapped
(``torch.load(..., mmap=True)``) instead of being fully resident in RAM.

Per sample we store:
    gt_token            int32 [D]      ground-truth action token ids
    teacher_logits      fp16  [D, 256] teacher logits restricted to the action slice
    teacher_argmax_token int32 [D]     teacher argmax over the *full* vocabulary

3.5 KB/sample -> 100k samples ~= 350 MB (fp16), which is why this route fits an
8 GB laptop while two 7B models never will.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .spec import ActionSpec

CACHE_VERSION = 1
MANIFEST = "manifest.json"

__all__ = ["LogitCacheWriter", "LogitCache", "CACHE_VERSION"]


def _as_1d_long(x, name: str) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.dim() != 2:
        raise ValueError(f"{name} must be 2-D [N, D], got shape {tuple(t.shape)}")
    return t


class LogitCacheWriter:
    """Accumulates rows and flushes fixed-size shards."""

    def __init__(
        self,
        cache_dir: str,
        spec: ActionSpec,
        meta: Optional[Dict[str, Any]] = None,
        shard_size: int = 2000,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self.spec = spec
        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.meta: Dict[str, Any] = dict(meta or {})

        if os.path.exists(os.path.join(self.cache_dir, MANIFEST)):
            if not (overwrite or resume):
                raise FileExistsError(
                    f"{self.cache_dir} already contains a cache; pass overwrite=True or resume=True"
                )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.shards: List[Dict[str, Any]] = []
        self.n_samples = 0
        self._buf_gt: List[torch.Tensor] = []
        self._buf_logits: List[torch.Tensor] = []
        self._buf_argmax: List[torch.Tensor] = []
        self._buf_n = 0

        if resume and os.path.exists(os.path.join(self.cache_dir, MANIFEST)):
            self._load_existing()

    # ---------------------------------------------------------------- internals
    def _load_existing(self) -> None:
        man = json.load(open(os.path.join(self.cache_dir, MANIFEST), "r", encoding="utf-8"))
        if ActionSpec.from_dict(man["spec"]) != self.spec:
            raise ValueError(
                f"cannot resume: cached spec {man['spec']} != requested spec {self.spec.to_dict()}"
            )
        self.shards = list(man.get("shards", []))
        self.n_samples = int(man.get("n_samples", 0))
        merged = dict(man.get("meta") or {})
        merged.update(self.meta)
        self.meta = merged

    def _manifest(self, complete: bool) -> Dict[str, Any]:
        return {
            "version": CACHE_VERSION,
            "spec": self.spec.to_dict(),
            "meta": self.meta,
            "shard_size": self.shard_size,
            "n_samples": self.n_samples,
            "shards": self.shards,
            "complete": complete,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _write_manifest(self, complete: bool) -> None:
        tmp = os.path.join(self.cache_dir, MANIFEST + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._manifest(complete), f, indent=2, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.cache_dir, MANIFEST))

    def _write_shard(self, gt: torch.Tensor, lg: torch.Tensor, am: torch.Tensor) -> None:
        idx = len(self.shards)
        fname = f"shard_{idx:05d}.pt"
        start = self.n_samples
        torch.save(
            {
                "gt_token": gt.to(torch.int32),
                "teacher_logits": lg.to(torch.float16),
                "teacher_argmax_token": am.to(torch.int32),
                "start": start,
            },
            os.path.join(self.cache_dir, fname),
        )
        self.shards.append({"file": fname, "n": int(gt.size(0)), "start": start})
        self.n_samples += int(gt.size(0))

    def _flush(self, full_only: bool = False) -> None:
        """Write complete shards; on close (``full_only=False``) also the remainder."""
        if self._buf_n == 0:
            return
        if full_only and self._buf_n < self.shard_size:
            return

        gt = torch.cat(self._buf_gt, dim=0)
        lg = torch.cat(self._buf_logits, dim=0)
        am = torch.cat(self._buf_argmax, dim=0)

        n_full = self._buf_n // self.shard_size
        rem = self._buf_n % self.shard_size
        for i in range(n_full):
            sl = slice(i * self.shard_size, (i + 1) * self.shard_size)
            self._write_shard(gt[sl], lg[sl], am[sl])

        if rem and not full_only:
            self._write_shard(gt[-rem:], lg[-rem:], am[-rem:])
            rem = 0

        if rem:
            self._buf_gt, self._buf_logits, self._buf_argmax = [gt[-rem:]], [lg[-rem:]], [am[-rem:]]
            self._buf_n = rem
        else:
            self._buf_gt, self._buf_logits, self._buf_argmax = [], [], []
            self._buf_n = 0

        if n_full:
            self._write_manifest(complete=False)

    # ------------------------------------------------------------------- public
    def add(
        self,
        gt_token: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_argmax_token: torch.Tensor,
    ) -> None:
        """Append a batch. ``teacher_logits`` must be ``[N, D, n_bins]``."""
        gt_token = _as_1d_long(gt_token, "gt_token")
        teacher_argmax_token = _as_1d_long(teacher_argmax_token, "teacher_argmax_token")
        teacher_logits = torch.as_tensor(teacher_logits)
        if teacher_logits.dim() != 3:
            raise ValueError(
                f"teacher_logits must be 3-D [N, D, n_bins], got {tuple(teacher_logits.shape)}"
            )
        n, d, b = teacher_logits.shape
        if d != self.spec.n_dims or b != self.spec.n_bins:
            raise ValueError(
                f"teacher_logits shape {tuple(teacher_logits.shape)} incompatible with spec "
                f"(n_dims={self.spec.n_dims}, n_bins={self.spec.n_bins})"
            )
        if gt_token.size(0) != n or teacher_argmax_token.size(0) != n:
            raise ValueError("batch dimension mismatch between gt_token / logits / argmax")

        self._buf_gt.append(gt_token.detach().cpu())
        self._buf_logits.append(teacher_logits.detach().cpu())
        self._buf_argmax.append(teacher_argmax_token.detach().cpu())
        self._buf_n += n
        self._flush(full_only=True)

    def close(self) -> None:
        self._flush()
        self._write_manifest(complete=True)

    def __enter__(self) -> "LogitCacheWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            # keep whatever made it to disk, but mark the cache as incomplete
            self._flush()
            self._write_manifest(complete=False)


class LogitCache:
    """Read-only accessor, with contiguous-range batching for sequential training."""

    def __init__(self, cache_dir: str, verbose: bool = True) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        path = os.path.join(self.cache_dir, MANIFEST)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no {MANIFEST} in {self.cache_dir}")
        self.manifest = json.load(open(path, "r", encoding="utf-8"))
        if int(self.manifest.get("version", -1)) != CACHE_VERSION:
            raise ValueError(
                f"cache version {self.manifest.get('version')} != expected {CACHE_VERSION}"
            )
        self.spec = ActionSpec.from_dict(self.manifest["spec"])
        self.shards: List[Dict[str, Any]] = list(self.manifest["shards"])
        self.n_samples = int(self.manifest["n_samples"])
        if self.n_samples > 0 and not self.shards:
            raise ValueError(f"{self.cache_dir}: manifest claims {self.n_samples} samples but lists no shards")
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        if verbose:
            print(
                f"[btkd] loaded cache {self.cache_dir}: {self.n_samples} samples, "
                f"{len(self.shards)} shard(s), spec={self.spec.to_dict()}"
            )

    # ------------------------------------------------------------------ helpers
    def _shard(self, i: int) -> Dict[str, torch.Tensor]:
        f = self.shards[i]["file"]
        if f not in self._cache:
            blob = torch.load(
                os.path.join(self.cache_dir, f), map_location="cpu", mmap=True, weights_only=False
            )
            self._cache[f] = blob
        return self._cache[f]

    def _locate(self, idx: int) -> (int, int):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"index {idx} out of range [0, {self.n_samples})")
        lo, hi = 0, len(self.shards) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if int(self.shards[mid]["start"]) <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, idx - int(self.shards[lo]["start"])

    # ------------------------------------------------------------------- public
    def __len__(self) -> int:
        return self.n_samples

    @property
    def is_complete(self) -> bool:
        return bool(self.manifest.get("complete", False))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si, off = self._locate(idx)
        blob = self._shard(si)
        return {
            "gt_token": blob["gt_token"][off].long(),
            "teacher_logits": blob["teacher_logits"][off],
            "teacher_argmax_token": blob["teacher_argmax_token"][off].long(),
        }

    def get_range(self, start: int, stop: int) -> Dict[str, torch.Tensor]:
        """Contiguous slice ``[start, stop)`` -- one disk pass, used by the trainer."""
        if stop <= start:
            raise ValueError(f"empty range [{start}, {stop})")
        if start < 0 or stop > self.n_samples:
            raise IndexError(f"range [{start}, {stop}) outside [0, {self.n_samples})")
        gts, lgs, ams = [], [], []
        cursor = start
        while cursor < stop:
            si, off = self._locate(cursor)
            blob = self._shard(si)
            take = min(stop, int(self.shards[si]["start"]) + int(self.shards[si]["n"])) - cursor
            gts.append(blob["gt_token"][off : off + take].long())
            lgs.append(blob["teacher_logits"][off : off + take])
            ams.append(blob["teacher_argmax_token"][off : off + take].long())
            cursor += take
        return {
            "gt_token": torch.cat(gts, dim=0),
            "teacher_logits": torch.cat(lgs, dim=0),
            "teacher_argmax_token": torch.cat(ams, dim=0),
        }

    def shrink_to(self, n: int) -> None:
        """Logical truncation, used to align with a ``--limit`` on the dataset side."""
        if n > self.n_samples:
            raise ValueError(f"cannot shrink to {n}, cache only has {self.n_samples}")
        self.n_samples = int(n)

    def analysis(self, max_samples: Optional[int] = None, chunk: int = 4096) -> Dict[str, Any]:
        """Cheap sanity statistics computed from the cache alone.

        ``teacher_token_acc`` is the quantity that drives the easy/hard split, so it is
        worth checking before launching a long training run: a suspiciously low value
        usually means the cache was exported with mismatched normalisation statistics.
        """
        n = self.n_samples if max_samples is None else min(int(max_samples), self.n_samples)
        correct = easy = oos = total = nan = 0
        lo, hi = float("inf"), float("-inf")
        for start in range(0, n, chunk):
            rows = self.get_range(start, min(n, start + chunk))
            gt = rows["gt_token"]
            am = rows["teacher_argmax_token"]
            in_space = (am >= self.spec.offset) & (am < self.spec.vocab_size)
            correct += int((am == gt).sum().item())
            easy += int(((am == gt) & in_space).sum().item())
            oos += int((~in_space).sum().item())
            total += int(gt.numel())
            lg = rows["teacher_logits"].float()
            nan += int((~torch.isfinite(lg)).sum().item())
            lo = min(lo, float(lg.min().item()))
            hi = max(hi, float(lg.max().item()))
        return {
            "n_samples": n,
            "n_tokens": total,
            "teacher_token_acc": correct / max(1, total),
            "easy_frac": easy / max(1, total),
            "out_of_space_frac": oos / max(1, total),
            "nonfinite_logits": nan,
            "logit_min": lo,
            "logit_max": hi,
        }

    def summary(self) -> str:
        return (
            f"LogitCache(dir={self.cache_dir}, n={self.n_samples}, shards={len(self.shards)}, "
            f"complete={self.is_complete}, spec={self.spec.to_dict()})"
        )
