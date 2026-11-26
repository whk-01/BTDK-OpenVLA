#!/usr/bin/env python
"""Merge a BTKD LoRA adapter into the base OpenVLA checkpoint.

The LIBERO evaluation script expects a complete Hugging Face checkpoint directory
(``config.json`` + model shards + processor/tokenizer files + ``dataset_statistics.json``).
A BTKD training output such as ``runs_btkd/<exp>/best`` only contains a PEFT adapter,
so it cannot be passed to ``run_libero_eval.py`` directly.

Example
-------
    CUDA_VISIBLE_DEVICES=3 python scripts/merge_btkd_adapter.py \
        --base openvla/openvla-7b \
        --adapter runs_btkd/<exp>/best \
        --out merged/btkd_spatial_best \
        --stats-from openvla/openvla-7b-finetuned-libero-spatial \
        --device cuda:0

``--stats-from`` may be either a ``dataset_statistics.json`` file or a directory
containing one.  These statistics are required to un-normalize the 7-D actions
inside LIBERO.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base OpenVLA checkpoint directory, e.g. openvla/openvla-7b")
    p.add_argument("--adapter", required=True, help="BTKD LoRA adapter directory, e.g. runs_btkd/<exp>/best")
    p.add_argument("--out", required=True, help="output directory for the merged full checkpoint")
    p.add_argument("--stats-from", default="", help="teacher checkpoint dir or dataset_statistics.json path")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda:0; CUDA_VISIBLE_DEVICES selects the physical GPU")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--overwrite", action="store_true", help="delete/rebuild --out if it already exists")
    return p.parse_args()


def _looks_like_repo_id(spec: str) -> bool:
    return (
        spec.count("/") == 1
        and not spec.startswith((".", "/", "~"))
        and " " not in spec
    )


def find_stats(spec: str) -> Path | None:
    """Resolve --stats-from as a JSON file, a local checkpoint dir, or an HF repo id."""
    if not spec:
        return None
    path = Path(spec).expanduser()
    if path.exists():
        if path.is_file():
            return path
        candidate = path / "dataset_statistics.json"
        return candidate if candidate.is_file() else None

    if _looks_like_repo_id(spec):
        # Teacher checkpoints are often only in the Hugging Face cache, not on disk as a
        # plain directory. Pull just the one statistics file we need.
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(repo_id=spec, filename="dataset_statistics.json")
            print(f"[merge] fetched dataset_statistics.json from HF repo {spec}", flush=True)
            return Path(downloaded)
        except Exception as exc:  # network/proxy/auth problems
            print(f"[merge] WARNING could not fetch dataset_statistics.json from {spec}: {exc}", flush=True)
            return None
    return None


def main() -> None:
    args = parse_args()
    base = Path(args.base).expanduser().resolve()
    adapter = Path(args.adapter).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not (base / "config.json").is_file():
        raise SystemExit(f"base checkpoint is missing config.json: {base}")
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"adapter is missing adapter_config.json: {adapter}")

    if out.exists():
        if not args.overwrite:
            # Allow an idempotent rerun when a complete merged checkpoint already exists.
            has_config = (out / "config.json").is_file()
            has_weights = any(out.glob("*.safetensors")) or any(out.glob("*.bin"))
            if has_config and has_weights:
                print(f"[merge] already exists, skipping: {out}")
                return
            raise SystemExit(f"output exists but is incomplete: {out} (pass --overwrite to rebuild)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    stats_src = find_stats(args.stats_from)
    if stats_src is None:
        for candidate in (adapter / "dataset_statistics.json", base / "dataset_statistics.json"):
            if candidate.is_file():
                stats_src = candidate
                print(
                    f"[merge] WARNING using fallback statistics {candidate}; these are only correct "
                    "if the adapter was trained with that exact dataset mixture.",
                    flush=True,
                )
                break
    if stats_src is None:
        print(
            "[merge] WARNING: no dataset_statistics.json found. LIBERO evaluation will fail at "
            "action un-normalization. Re-run with --stats-from <teacher checkpoint or repo id>.",
            flush=True,
        )

    # Import OpenVLA's custom HF classes before AutoModelForVision2Seq is used.
    from btkd_vla.vla_glue import register_openvla_classes

    register_openvla_classes()

    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device_map = None if args.device == "cpu" else {"": args.device}
    print(f"[merge] base={base}", flush=True)
    print(f"[merge] adapter={adapter}", flush=True)
    print(f"[merge] out={out} device={args.device} dtype={args.dtype}", flush=True)

    model = AutoModelForVision2Seq.from_pretrained(
        str(base),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    merged = model.merge_and_unload()
    merged.eval()

    merged.save_pretrained(str(out), safe_serialization=True)
    print("[merge] model shards written", flush=True)

    # Save processor/tokenizer files.  The custom .py files are also copied because
    # OpenVLA checkpoints use auto_map / trust_remote_code.
    try:
        processor = AutoProcessor.from_pretrained(str(base), trust_remote_code=True)
        processor.save_pretrained(str(out))
    except Exception as exc:  # pragma: no cover - depends on local checkpoint contents
        print(f"[merge] WARNING could not save processor via AutoProcessor: {exc}", flush=True)

    for src in base.iterdir():
        if not src.is_file():
            continue
        if src.suffix in {".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack", ".ot"}:
            continue
        if src.name == "config.json":
            continue  # keep the config written by save_pretrained()
        dst = out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    if stats_src is not None:
        shutil.copy2(stats_src, out / "dataset_statistics.json")
        with open(stats_src, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(f"[merge] dataset_statistics.json keys: {list(stats)}", flush=True)

    print(f"[merge] done: {out}", flush=True)


if __name__ == "__main__":
    main()
