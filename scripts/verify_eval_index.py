#!/usr/bin/env python3
"""Verify RE10K evaluation-index JSONs.

This script has two layers:
1) Lightweight structural stats (always).
2) Optional deep validation against the dataset chunks (recommended before benchmarking).

Examples:
  # Quick stats
  python scripts/verify_eval_index.py assets/indices/re10k/evaluation_index_re10k.json

  # Full verification (checks that every referenced frame index exists)
  python scripts/verify_eval_index.py assets/indices/re10k/evaluation_index_re10k.json --check-dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int_list(value: Any, *, scene: str, key: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise TypeError(f"{scene}: expected '{key}' to be list[int], got {type(value)}")
    return value


def _print_stats(index: dict[str, Any], *, path: Path) -> None:
    num_scenes = len(index)
    non_null = [v for v in index.values() if v is not None]
    num_non_null = len(non_null)
    num_targets = sum(len(v.get("target", [])) for v in non_null)
    num_context = sum(len(v.get("context", [])) for v in non_null)

    print(f"Index file : {path}")
    print(f"Scenes     : {num_scenes:,} (non-null: {num_non_null:,})")
    print(f"Context    : {num_context:,} (total)")
    print(f"Targets    : {num_targets:,} (total)")
    if num_non_null:
        print(f"Targets/sc : {num_targets / num_non_null:.2f} (avg over non-null scenes)")
        print(f"Context/sc : {num_context / num_non_null:.2f} (avg over non-null scenes)")


def _verify_against_dataset(
    *,
    index: dict[str, Any],
    dataset_root: Path,
    split: str,
    limit_chunks: int | None,
) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for --check-dataset") from exc

    stage_dir = dataset_root / split
    dataset_index_path = stage_dir / "index.json"
    if not dataset_index_path.exists():
        raise FileNotFoundError(dataset_index_path)
    dataset_index = _load_json(dataset_index_path)

    # Group scenes by chunk file so we only load each *.torch once.
    by_chunk: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    missing_scenes: list[str] = []
    malformed: list[str] = []

    for scene, entry in index.items():
        if entry is None:
            continue
        if scene not in dataset_index:
            missing_scenes.append(scene)
            continue
        if not isinstance(entry, dict):
            malformed.append(scene)
            continue
        by_chunk[str(dataset_index[scene])].append((scene, entry))

    if missing_scenes:
        raise SystemExit(
            f"{len(missing_scenes)} scenes in eval index are missing from {dataset_index_path} "
            f"(e.g. {missing_scenes[0]})."
        )
    if malformed:
        raise SystemExit(f"{len(malformed)} non-null scenes have non-dict entries (e.g. {malformed[0]}).")

    bad_indices: list[str] = []
    checked_scenes = 0

    chunk_items = list(by_chunk.items())
    if limit_chunks is not None:
        chunk_items = chunk_items[:limit_chunks]

    is_tty = sys.stdout.isatty() or sys.stderr.isatty()
    use_rich = False
    try:
        if is_tty:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            use_rich = True
    except Exception:
        use_rich = False

    def verify_chunk(chunk_name: str, scenes: list[tuple[str, dict[str, Any]]]) -> None:
        nonlocal checked_scenes
        chunk_path = stage_dir / chunk_name
        if not chunk_path.exists():
            raise FileNotFoundError(chunk_path)

        chunk = torch.load(chunk_path, map_location="cpu")
        if not isinstance(chunk, list):
            raise TypeError(f"Expected list in {chunk_path}, got {type(chunk)}")
        examples_by_key = {str(ex.get("key")): ex for ex in chunk if isinstance(ex, dict)}

        for scene, entry in scenes:
            example = examples_by_key.get(scene)
            if example is None:
                bad_indices.append(f"{scene}: missing from chunk {chunk_name}")
                continue

            images = example.get("images")
            if not isinstance(images, list):
                bad_indices.append(f"{scene}: missing/invalid 'images' list in chunk {chunk_name}")
                continue
            num_views = len(images)

            ctx = _as_int_list(entry.get("context"), scene=scene, key="context")
            tgt = _as_int_list(entry.get("target"), scene=scene, key="target")

            if len(ctx) != 2:
                bad_indices.append(f"{scene}: expected 2 context views, got {len(ctx)}")
                continue

            all_indices = ctx + tgt
            if any(x < 0 or x >= num_views for x in all_indices):
                bad_indices.append(
                    f"{scene}: index out of range for num_views={num_views} "
                    f"(context={ctx}, target={tgt})"
                )
                continue

            checked_scenes += 1

    if use_rich:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
        )
        task = progress.add_task("Checking chunks", total=len(chunk_items))
        with progress:
            for i, (chunk_name, scenes) in enumerate(chunk_items, start=1):
                verify_chunk(chunk_name, scenes)
                progress.advance(task, 1)
                if i % 10 == 0 or i == len(chunk_items):
                    progress.update(task, description=f"Checking chunks | scenes verified: {checked_scenes:,}")
    else:
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None  # type: ignore

        iterator = chunk_items
        if tqdm is not None and is_tty:
            iterator = tqdm(chunk_items, total=len(chunk_items), desc="Checking chunks")  # type: ignore

        for i, (chunk_name, scenes) in enumerate(iterator, start=1):
            verify_chunk(chunk_name, scenes)
            if tqdm is None and (i % 10 == 0 or i == len(chunk_items)):
                print(f"Checked chunks: {i}/{len(chunk_items)} | scenes verified: {checked_scenes:,}")

    if bad_indices:
        preview = "\n".join(bad_indices[:20])
        raise SystemExit(f"Found {len(bad_indices)} invalid entries:\n{preview}")

    print(f"OK: verified {checked_scenes:,} scenes against {dataset_root}/{split}/")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Verify RE10K evaluation index JSON")
    parser.add_argument(
        "index_path",
        type=Path,
        nargs="?",
        default=repo_root / "assets" / "indices" / "re10k" / "evaluation_index_re10k.json",
    )
    parser.add_argument(
        "--check-dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deep-verify that every referenced frame index exists in the dataset chunks.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset" / "re10k",
        help="Path to dataset/re10k (contains train/ and test/).",
    )
    parser.add_argument(
        "--split",
        choices=["test", "eval"],
        default="test",
        help="Which dataset split directory to validate against (eval aliases test if you use that naming).",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Debug: only scan the first N chunk files referenced by the index.",
    )
    args = parser.parse_args()

    index_path = args.index_path
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    index = _load_json(index_path)
    _print_stats(index, path=index_path)

    if args.check_dataset:
        split = "test" if args.split == "eval" else args.split
        _verify_against_dataset(
            index=index,
            dataset_root=args.dataset_root,
            split=split,
            limit_chunks=args.limit_chunks,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
