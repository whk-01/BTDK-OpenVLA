"""Deterministic single-dataset RLDS pipeline for offline teacher-logit caches.

OpenVLA's stock ``RLDSDataset`` is optimised for online training: it shuffles
TFRecord shard-file order and lets TensorFlow AUTOTUNE the read/map parallelism.
A frame-level shuffle buffer of 1 therefore does *not* make two independent
processes start from the same sample.  Offline KD needs that stronger contract:
export and training must replay exactly the same row order.

This module intentionally supports one dataset only.  Multi-dataset mixtures use
random frame-level sampling in upstream OpenVLA and cannot provide a stable row
contract without cache keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from torch.utils.data import IterableDataset


class DeterministicSingleRLDSDataset(IterableDataset):
    """OpenVLA-compatible iterable dataset with stable, single-pass row order.

    The config mirrors ``prismatic.vla.datasets.RLDSDataset`` for a single OXE
    dataset, but forces:

    * ``shuffle=False`` for TFDS shard files (not just the frame buffer);
    * one trajectory-read worker;
    * one trajectory/frame map worker;
    * one flatten worker;
    * no ``sample_from_datasets`` random choice stream.
    """

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: Any,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 1,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        if shuffle_buffer_size != 1:
            raise ValueError(
                "offline cache alignment requires shuffle_buffer_size=1, "
                f"got {shuffle_buffer_size}"
            )
        if image_aug:
            raise ValueError("offline cache alignment requires image_aug=False")

        # Import lazily: ``--check_cache_only`` must remain usable on a CPU-only
        # machine without TensorFlow / dlimp installed.
        from prismatic.vla.datasets.rlds.dataset import (
            apply_frame_transforms,
            apply_trajectory_transforms,
            make_dataset_from_rlds,
        )
        from prismatic.vla.datasets.rlds.oxe import (
            OXE_NAMED_MIXTURES,
            get_oxe_dataset_kwargs_and_weights,
        )
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.batch_transform = batch_transform

        if data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[data_mix]
        else:
            mixture_spec = [(data_mix, 1.0)]
        if len(mixture_spec) != 1:
            raise ValueError(
                "deterministic offline KD supports single-dataset mixes only; "
                f"{data_mix!r} expands to {len(mixture_spec)} datasets"
            )

        per_dataset_kwargs, _ = get_oxe_dataset_kwargs_and_weights(
            data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        dataset_kwargs: Dict[str, Any] = dict(per_dataset_kwargs[0])
        # This controls TFDS shard-file order.  It is the key setting the stock
        # OpenVLA pipeline leaves at True and is what broke export/train replay.
        dataset_kwargs["shuffle"] = False

        traj_transform_kwargs = dict(
            window_size=1,
            future_action_window_size=0,
            skip_unlabeled=True,
            goal_relabeling_strategy="uniform",
        )
        frame_transform_kwargs = dict(resize_size=resize_resolution)

        dataset, dataset_statistics = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_reads=1,
            num_parallel_calls=1,
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            train=train,
            num_parallel_calls=1,
            **traj_transform_kwargs,
        )
        dataset = dataset.flatten(num_parallel_calls=1)
        # Buffer size 1 is an explicit no-op; kept to mirror upstream structure and
        # to guarantee that ordering never depends on a larger shuffle buffer.
        dataset = dataset.shuffle(1)
        dataset = apply_frame_transforms(
            dataset,
            train=train,
            num_parallel_calls=1,
            **frame_transform_kwargs,
        )
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.dataset_length = int(dataset_statistics["num_transitions"])
        self.dataset_statistics = dataset_statistics

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length


__all__ = ["DeterministicSingleRLDSDataset"]
