#!/usr/bin/env python3
"""Build RE10K split indices (scene -> chunk file).

MVSplat's RE10K loader expects:
  dataset/re10k/train/index.json
  dataset/re10k/test/index.json

Those files are a mapping:
  { "<scene_id>": "<chunk_filename>.torch", ... }

This script regenerates that mapping by scanning `*.torch` chunk files and
collecting each example's `key`.

Examples:
  # Write canonical copies under assets/
  python scripts/indices/build_re10k_dataset_index.py --stage train
  python scripts/indices/build_re10k_dataset_index.py --stage eval

  # Also write into the dataset folder (what the dataloader uses)
  python scripts/indices/build_re10k_dataset_index.py --stage train --write-dataset-index
  python scripts/indices/build_re10k_dataset_index.py --stage eval  --write-dataset-index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_index(stage_dir: Path) -> dict[str, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required to build the dataset index (it loads *.torch chunk files)."
        ) from exc

    chunk_paths = sorted(stage_dir.glob("*.torch"))
    if not chunk_paths:
        raise FileNotFoundError(f"No *.torch files found under {stage_dir}")

    mapping: dict[str, str] = {}
    for chunk_path in chunk_paths:
        chunk = torch.load(chunk_path, map_location="cpu")
        if not isinstance(chunk, list):
            raise TypeError(f"Expected list in {chunk_path}, got {type(chunk)}")
        for example in chunk:
            if not isinstance(example, dict) or "key" not in example:
                raise KeyError(f"Expected dict with 'key' in {chunk_path}")
            scene = str(example["key"])
            prev = mapping.get(scene)
            if prev is not None and prev != chunk_path.name:
                raise ValueError(f"Scene {scene} appears in multiple chunks: {prev} and {chunk_path.name}")
            mapping[scene] = chunk_path.name

    return mapping


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Build RE10K split index.json from *.torch chunks")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset" / "re10k",
        help="Path to dataset/re10k (contains train/ and test/)",
    )
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "test"],
        required=True,
        help="Which split to index (this repo uses train/eval; eval maps to dataset/re10k/test for MVSplat compatibility)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the generated index JSON (defaults to assets/indices/re10k/dataset_index_<train|eval>.json)",
    )
    parser.add_argument(
        "--write-dataset-index",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write to <dataset-root>/<stage>/index.json (what the dataloader expects)",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="If set, compares the generated mapping to an existing index JSON and fails on mismatch.",
    )
    args = parser.parse_args()

    dataset_stage = args.stage
    out_stage = args.stage
    if args.stage in {"eval", "test"}:
        out_stage = "eval"
        dataset_stage = "eval" if (args.dataset_root / "eval").exists() else "test"

    stage_dir = args.dataset_root / dataset_stage
    out = (
        args.output
        if args.output is not None
        else repo_root / "assets" / "indices" / "re10k" / f"dataset_index_{out_stage}.json"
    )

    mapping = _build_index(stage_dir)
    _dump_json(out, mapping)

    print(f"Wrote: {out}")
    print(f"Scenes: {len(mapping):,}")

    if args.write_dataset_index:
        dataset_index_path = stage_dir / "index.json"
        _dump_json(dataset_index_path, mapping)
        print(f"Wrote: {dataset_index_path}")

    if args.verify_against is not None:
        expected = _load_json(args.verify_against)
        if expected != mapping:
            raise SystemExit(f"Mismatch vs {args.verify_against} (dicts not equal)")
        print(f"Verified match vs: {args.verify_against}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
