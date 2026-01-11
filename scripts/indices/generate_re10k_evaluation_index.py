#!/usr/bin/env python3
"""Generate the RE10K evaluation view-sampler index (optional).

This script uses MVSplat's upstream evaluation-index generator logic, but runs it
in-process so we can default to CPU (and avoid hard-coded GPU requirements).

It writes the resulting JSON to:
  `assets/indices/re10k/evaluation_index_re10k.json`

Notes:
- Determinism: RE10K is an `IterableDataset`; multi-worker iteration can reorder
  scenes and therefore change the RNG stream. For a reproducible canonical index,
  use `--num-workers 0` (default).
- For the conference repo, treat the committed index as canonical and only
  regenerate if you intentionally change generator parameters.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_mvsplat_to_path(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "third_party" / "mvsplat"))


def main() -> int:
    repo_root = _repo_root()
    _add_mvsplat_to_path(repo_root)

    parser = argparse.ArgumentParser(description="Generate RE10K evaluation_index_re10k.json (MVSplat protocol)")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset" / "re10k",
        help="Path to dataset/re10k (contains train/ and test/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "assets" / "indices" / "re10k" / "evaluation_index_re10k.json",
        help="Where to write the final evaluation index JSON.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=repo_root / "assets" / "indices" / "re10k" / "_generated_evaluation_index_re10k",
        help="Temporary output directory used by the generator (contains evaluation_index.json + optional previews).",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Lightning accelerator to use (default: auto).",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices for Lightning (only relevant for CUDA).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for test split (default: 0 for determinism on IterableDataset).",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Debug: only process the first N scenes.",
    )
    parser.add_argument(
        "--save-previews",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save overlap preview PNGs under <work-dir>/previews (useful for sanity checks).",
    )
    args = parser.parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)

    from hydra import compose, initialize_config_dir

    import torch
    from pytorch_lightning import Trainer

    from src.config import load_typed_config
    from src.dataset import DatasetCfg
    from src.dataset.data_module import DataLoaderCfg, DataModule
    from src.evaluation.evaluation_index_generator import (
        EvaluationIndexGenerator,
        EvaluationIndexGeneratorCfg,
    )
    from src.global_cfg import set_cfg

    from dataclasses import dataclass

    @dataclass
    class RootCfg:
        dataset: DatasetCfg
        data_loader: DataLoaderCfg
        index_generator: EvaluationIndexGeneratorCfg
        seed: int

    config_dir = repo_root / "third_party" / "mvsplat" / "config"
    if not config_dir.exists():
        raise FileNotFoundError(config_dir)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"dataset.roots=[{args.dataset_root.resolve()}]",
        f"index_generator.output_path={args.work_dir.resolve()}",
        f"index_generator.save_previews={str(args.save_previews).lower()}",
        f"data_loader.test.num_workers={args.num_workers}",
        "data_loader.test.batch_size=1",
    ]

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg_dict = compose(config_name="generate_evaluation_index", overrides=overrides)

    set_cfg(cfg_dict)
    cfg = load_typed_config(cfg_dict, RootCfg)
    torch.manual_seed(int(cfg.seed))

    accelerator = args.accelerator
    if accelerator == "auto":
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --accelerator=cuda but CUDA is not available.")

    limit_test_batches = 1.0 if args.limit_scenes is None else int(args.limit_scenes)

    trainer = Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=(args.devices if accelerator == "cuda" else 1),
        logger=False,
        enable_checkpointing=False,
        limit_test_batches=limit_test_batches,
    )

    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker=None)
    evaluation_index_generator = EvaluationIndexGenerator(cfg.index_generator)
    trainer.test(evaluation_index_generator, datamodule=data_module)
    evaluation_index_generator.save_index()

    generated = args.work_dir / "evaluation_index.json"
    if not generated.exists():
        raise FileNotFoundError(generated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(generated, args.output)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
