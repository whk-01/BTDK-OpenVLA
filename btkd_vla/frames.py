"""Materialised frame store: removes TFDS/RLDS from the training loop.

The deterministic RLDS pipeline (``rlds_dataset.DeterministicSingleRLDSDataset``)
is *correct* but slow: one read/map worker by design, so every training step pays
a full TFRecord + JPEG decode on the critical path (~1.5-3 s/step for 7B LoRA,
which caps a single epoch at dozens of hours).

This module stores exactly the tensors the model consumes - token ids, labels and
the already-transformed ``pixel_values`` - in shards.  Training then becomes a
plain random-access read (~0.3 MB/sample) with no decoding at all, and the row
order is the *same* order the teacher cache was exported in, so cache/dataset
misalignment becomes structurally impossible once indices are shared.

Layout::

    <frames_dir>/
        manifest.json
        shard_00000.pt     # input_ids / labels / pixel_values / dataset_name
        shard_00001.pt

``pixel_values`` is stored as fp16 ``[n, 3, H, W]`` so ``torch.load(..., mmap=True)``
memory-maps it and a random access only touches one sample's pages.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

FRAMES_VERSION = 1
MANIFEST = "manifest.json"
__all__ = ["FrameStoreWriter", "FrameStore", "FRAMES_VERSION", "MANIFEST"]


class FrameStoreWriter:
    """Append samples to a sharded frame store."""

    def __init__(
        self,
        out_dir: str,
        image_shape: Optional[Any] = None,
        dataset: str = "",
        shard_size: int = 1000,
        meta: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        if self.out_dir.exists():
            has_content = any(self.out_dir.iterdir())
            if has_content and not overwrite:
                raise FileExistsError(
                    f"{self.out_dir} already exists and is not empty; pass --overwrite to rebuild"
                )
            if has_content:
                shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = dataset
        self.dataset_name = dataset
        self.shard_size = max(1, int(shard_size))
        self.meta = dict(meta or {})
        self.image_shape = list(image_shape) if image_shape is not None else None

        self.shards: List[Dict[str, Any]] = []
        self.n_written = 0
        self._ids: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []
        self._pixels: List[torch.Tensor] = []
        self._write_manifest(complete=False)

    # ------------------------------------------------------------------ internals
    def _write_manifest(self, complete: bool) -> None:
        manifest = {
            "version": FRAMES_VERSION,
            "complete": bool(complete),
            "dataset": self.dataset,
            "n_samples": int(self.n_written + len(self._ids)),
            "image_shape": self.image_shape,
            "shard_size": self.shard_size,
            "shards": list(self.shards),
            "meta": dict(self.meta),
        }
        tmp = self.out_dir / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.out_dir / MANIFEST)

    def _flush(self) -> None:
        if not self._ids:
            return
        index = len(self.shards)
        fname = f"shard_{index:05d}.pt"
        blob = {
            "input_ids": [t.clone() for t in self._ids],
            "labels": [t.clone() for t in self._labels],
            "pixel_values": torch.stack(self._pixels, dim=0).to(torch.float16),
            "dataset_name": self.dataset_name,
        }
        torch.save(blob, self.out_dir / fname)
        self.shards.append({"file": fname, "n": len(self._ids), "start": self.n_written})
        self.n_written += len(self._ids)
        self._ids, self._labels, self._pixels = [], [], []
        self._write_manifest(complete=False)

    # --------------------------------------------------------------------- public
    def add(self, input_ids, labels, pixel_values, dataset_name: str = "") -> None:
        self._ids.append(torch.as_tensor(input_ids).to(torch.int32).reshape(-1).cpu())
        self._labels.append(torch.as_tensor(labels).to(torch.int32).reshape(-1).cpu())
        self._pixels.append(torch.as_tensor(pixel_values).to(torch.float16).cpu())
        if dataset_name:
            self.dataset_name = dataset_name
        if len(self._ids) >= self.shard_size:
            self._flush()

    def close(self) -> None:
        self._flush()
        self._write_manifest(complete=True)

    def __enter__(self) -> "FrameStoreWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._write_manifest(complete=False)


class FrameStore(Dataset):
    """Random-access view over a materialised frame store."""

    def __init__(self, root: str, verbose: bool = True, max_open_shards: int = 8) -> None:
        self.root = Path(root)
        path = self.root / MANIFEST
        if not path.is_file():
            raise FileNotFoundError(f"no {MANIFEST} in {self.root}")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if int(self.manifest.get("version", -1)) != FRAMES_VERSION:
            raise ValueError(
                f"{self.root}: frames version {self.manifest.get('version')} != expected {FRAMES_VERSION}"
            )
        self.dataset = self.manifest.get("dataset", "")
        self.n_samples = int(self.manifest["n_samples"])
        self.shards: List[Dict[str, Any]] = list(self.manifest["shards"])
        self.image_shape = self.manifest.get("image_shape")
        if self.n_samples > 0 and not self.shards:
            raise ValueError(f"{self.root}: manifest claims {self.n_samples} samples but has no shards")
        # ``<= 0`` keeps every shard mapped for the whole run.  ``mmap=True`` only
        # maps pages, so this costs virtual address space + one small tensor header
        # per sample, not 16 GB of RSS.
        self.max_open_shards = int(max_open_shards)
        self._open: Dict[int, Dict[str, Any]] = {}
        self._order: List[int] = []
        self.shard_loads = 0
        self.shard_hits = 0
        if verbose:
            print(
                f"[btkd] loaded frames {self.root}: {self.n_samples} samples, "
                f"{len(self.shards)} shard(s), image={self.image_shape}"
            )

    # ------------------------------------------------------------------ internals
    @property
    def is_complete(self) -> bool:
        return bool(self.manifest.get("complete", False))

    def _shard(self, index: int) -> Dict[str, Any]:
        blob = self._open.get(index)
        if blob is not None:
            self.shard_hits += 1
            # True LRU: a hit must move the shard to the back, otherwise the
            # eviction order degenerates to FIFO and the shard we just used can be
            # dropped on the very next (different) access.
            if not self._order or self._order[-1] != index:
                try:
                    self._order.remove(index)
                except ValueError:  # pragma: no cover - defensive
                    pass
                self._order.append(index)
            return blob
        fname = self.shards[index]["file"]
        self.shard_loads += 1
        blob = torch.load(
            self.root / fname, map_location="cpu", mmap=True, weights_only=False
        )
        self._open[index] = blob
        self._order.append(index)
        if self.max_open_shards > 0:
            while len(self._order) > self.max_open_shards:
                evict = self._order.pop(0)
                self._open.pop(evict, None)
        return blob

    @property
    def stats(self) -> Dict[str, int]:
        """Diagnostics: how many shard loads vs. cache hits since the last reset."""
        return {
            "open": len(self._open),
            "loads": self.shard_loads,
            "hits": self.shard_hits,
        }

    def reset_stats(self) -> None:
        self.shard_loads = 0
        self.shard_hits = 0

    def shard_span(self, index: int) -> Tuple[int, int]:
        """``[start, start + n)`` global row range covered by shard ``index``."""
        shard = self.shards[int(index)]
        start = int(shard["start"])
        return start, start + int(shard["n"])

    def _locate(self, idx: int):
        if idx < 0:
            idx += self.n_samples
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

    # --------------------------------------------------------------------- public
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        si, off = self._locate(int(idx))
        blob = self._shard(si)
        return {
            "input_ids": blob["input_ids"][off].long(),
            "labels": blob["labels"][off].long(),
            "pixel_values": blob["pixel_values"][off].float(),
            "dataset_name": blob.get("dataset_name") or self.dataset,
        }
