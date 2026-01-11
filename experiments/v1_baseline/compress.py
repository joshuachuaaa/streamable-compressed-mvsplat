#!/usr/bin/env python3
"""V1 baseline preprocessing: compress RE10K context frames with ELIC.

This script compresses only the *transmitted* inputs: the 2 context views per scene
from a fixed evaluation index.

It produces:
  experiments/v1_baseline/compressed/lambda_<λ>/recon/<scene>/<frame>.png
  experiments/v1_baseline/compressed/lambda_<λ>/manifest.csv

The `manifest.csv` is the canonical source of truth for bpp (computed from ELIC
bitstreams, not PNG file size).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_third_party_to_path(repo_root: Path) -> None:
    # MVSplat is an implicit namespace package called `src`.
    sys.path.insert(0, str(repo_root / "third_party" / "mvsplat"))
    # ELIC reimplementation uses top-level modules (e.g., `Network.py`).
    sys.path.insert(0, str(repo_root / "third_party" / "ELiC-ReImplemetation"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_lambda(value: float) -> str:
    # Keep folder names stable and human-readable.
    return ("%.3f" % value).rstrip("0").rstrip(".")


def _lambda_to_checkpoint_name(lmbda: float) -> str:
    # Repo-provided checkpoints follow this convention (see checkpoints/vanilla/ELIC).
    mapping = {
        0.004: "0004",
        0.008: "0008",
        0.016: "0016",
        0.032: "0032",
        0.15: "0150",
        0.45: "0450",
    }
    # Float keys are safe here because these values come from CLI literals and are
    # representable exactly in binary for the ones we use? Not always. Use a tolerance.
    for k, v in mapping.items():
        if math.isclose(lmbda, k, rel_tol=0.0, abs_tol=1e-12):
            return f"ELIC_{v}_ft_3980_Plateau.pth.tar"
    raise ValueError(f"Unsupported lambda={lmbda}; expected one of {sorted(mapping.keys())}")


def _sum_bytes(obj: Any) -> int:
    """Recursively sum lengths of byte strings in `compress()` output."""
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_sum_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_sum_bytes(v) for v in obj.values())
    return 0


def _load_mvsplat_cfg(
    *,
    dataset_root: Path,
    index_path: Path,
) -> tuple[Any, Any]:
    """Compose the upstream MVSplat config (+experiment=re10k) for dataset iteration."""
    from hydra import compose, initialize_config_dir

    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg

    repo_root = _repo_root()
    config_dir = repo_root / "third_party" / "mvsplat" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                "+experiment=re10k",
                "mode=test",
                "wandb.mode=disabled",
                "dataset/view_sampler=evaluation",
                f"dataset.view_sampler.index_path={index_path.resolve()}",
                f"dataset.roots=[{dataset_root.resolve()}]",
                "test.save_image=false",
                "test.save_video=false",
                "test.compute_scores=false",
            ],
        )

    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    return cfg_dict, cfg


def _iter_eval_context_batches(cfg: Any) -> Iterable[dict[str, Any]]:
    from src.dataset.data_module import DataModule

    # Use the upstream datamodule for correctness (same shims + JPEG decoding).
    dm = DataModule(cfg.dataset, cfg.data_loader, step_tracker=None, global_rank=0)
    loader = dm.test_dataloader()
    yield from loader


def _load_elic_model(checkpoint_path: Path, device: str, entropy_coder: str) -> Any:
    import torch
    import compressai
    from compressai.zoo import load_state_dict

    # From the vendored ELIC reimplementation.
    from Network import TestModel

    compressai.set_entropy_coder(entropy_coder)

    # The provided checkpoints are plain OrderedDict state_dicts.
    state_dict = load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = TestModel().from_state_dict(state_dict).eval()
    return model.to(device)


def _pad_to_multiple(x_bchw: Any, multiple: int) -> tuple[Any, tuple[int, int, int, int]]:
    import torch.nn.functional as F

    _, _, h, w = x_bchw.shape
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
    return F.pad(x_bchw, padding, mode="constant", value=0.0), padding


def _unpad(x_bchw: Any, padding: tuple[int, int, int, int]) -> Any:
    left, right, top, bottom = padding
    if any(v != 0 for v in padding):
        return x_bchw[..., top : x_bchw.shape[-2] - bottom, left : x_bchw.shape[-1] - right]
    return x_bchw


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_existing_manifest(manifest_path: Path) -> set[tuple[str, int]]:
    if not manifest_path.exists():
        return set()
    processed: set[tuple[str, int]] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                processed.add((row["scene"], int(row["frame"])))
            except Exception:
                continue
    return processed


def main() -> int:
    repo_root = _repo_root()
    _add_third_party_to_path(repo_root)

    parser = argparse.ArgumentParser(description="Compress RE10K eval context frames with ELIC (V1 baseline)")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset" / "re10k",
        help="Path to dataset/re10k (contains train/ and test/).",
    )
    parser.add_argument(
        "--index-path",
        "--index_path",
        type=Path,
        default=repo_root / "assets" / "indices" / "re10k" / "evaluation_index_re10k.json",
        help="Evaluation index JSON (2 context + 3 target).",
    )
    parser.add_argument(
        "--split",
        choices=["test"],
        default="test",
        help="Dataset split to compress (only 'test' is supported for the fixed evaluation index).",
    )
    parser.add_argument(
        "--elic-checkpoints",
        type=Path,
        default=None,
        help="Directory containing ELIC_*.pth.tar (defaults to checkpoints/vanilla/ELIC if present).",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        required=True,
        help="One or more ELIC lambda values (e.g. 0.004 0.008 0.016 0.032 0.15 0.45).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "experiments" / "v1_baseline" / "compressed",
        help="Where to write reconstructions + manifests.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run ELIC compression on.",
    )
    parser.add_argument(
        "--entropy-coder",
        default="ans",
        help="CompressAI entropy coder (default: ans).",
    )
    parser.add_argument(
        "--pad-multiple",
        type=int,
        default=64,
        help="Pad H,W to a multiple of this value before ELIC compression.",
    )
    parser.add_argument(
        "--skip-existing",
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip (scene,frame) pairs already present in the manifest.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Debug: stop after processing N scenes per lambda.",
    )
    parser.add_argument(
        "--save-bitstreams",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save serialized bitstreams under bitstreams/<scene>/<frame>.bin (optional, can be large in file count).",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise SystemExit("Requested --device=cuda but CUDA is not available.")
        except ImportError as exc:
            raise SystemExit(f"torch import failed: {exc}") from exc

    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    if not args.index_path.exists():
        raise FileNotFoundError(args.index_path)

    elic_ckpt_dir = args.elic_checkpoints
    if elic_ckpt_dir is None:
        candidate = repo_root / "checkpoints" / "vanilla" / "ELIC"
        elic_ckpt_dir = candidate if candidate.exists() else (repo_root / "checkpoints" / "elic")
    if not elic_ckpt_dir.exists():
        raise FileNotFoundError(elic_ckpt_dir)

    # Load MVSplat config once (dataset iteration settings) and reuse across lambdas.
    _, cfg = _load_mvsplat_cfg(dataset_root=args.dataset_root, index_path=args.index_path)

    # Basic sanity: index stats.
    eval_index = _load_json(args.index_path)
    num_non_null = sum(v is not None for v in eval_index.values())
    print(f"Evaluation index scenes (non-null): {num_non_null:,}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "_meta").mkdir(parents=True, exist_ok=True)
    (args.output_root / "_meta" / "run_args.json").write_text(
        json.dumps({**vars(args), "dataset_root": str(args.dataset_root)}, indent=2, default=str),
        encoding="utf-8",
    )

    # Run each lambda independently to avoid holding multiple 150MB models in memory.
    for lmbda in args.lambdas:
        lmbda_str = _format_lambda(lmbda)
        out_dir = args.output_root / f"lambda_{lmbda_str}"
        recon_root = out_dir / "recon"
        bitstream_root = out_dir / "bitstreams"
        out_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = _lambda_to_checkpoint_name(lmbda)
        ckpt_path = elic_ckpt_dir / checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for lambda={lmbda}: {ckpt_path}")

        print(f"\n==> Lambda {lmbda_str} | checkpoint: {ckpt_path}")
        model = _load_elic_model(ckpt_path, device=args.device, entropy_coder=args.entropy_coder)

        manifest_path = out_dir / "manifest.csv"
        processed = _read_existing_manifest(manifest_path) if args.skip_existing else set()
        if args.skip_existing:
            print(f"Skip-existing enabled; already in manifest: {len(processed):,} images")

        rows_to_append: list[dict[str, Any]] = []
        scenes_done = 0

        # Iterate through evaluation dataset once for this lambda.
        for batch in _iter_eval_context_batches(cfg):
            scene = batch["scene"][0]
            # batch_size=1 always for test loader.
            context_indices = [int(x) for x in batch["context"]["index"][0].tolist()]
            context_images = batch["context"]["image"][0]  # [2,3,H,W]

            for view_slot, frame in enumerate(context_indices):
                key = (scene, frame)
                if key in processed:
                    continue

                # Run ELIC (expects BCHW).
                x = context_images[view_slot].to(args.device)  # [3,H,W]
                x = x.unsqueeze(0)
                x_padded, padding = _pad_to_multiple(x, args.pad_multiple)

                out_enc = model.compress(x_padded)
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                x_hat = _unpad(out_dec["x_hat"], padding).detach().clamp(0, 1).cpu()

                # Bit accounting (true entropy-coded bytes).
                num_pixels = int(x.shape[0] * x.shape[2] * x.shape[3])  # B*H*W
                num_bytes = _sum_bytes(out_enc.get("strings"))
                num_bits = 8 * num_bytes
                bpp = num_bits / num_pixels

                # Save reconstruction (for downstream MVSplat eval).
                recon_path = recon_root / scene / f"{frame:0>6}.png"
                _ensure_parent(recon_path)
                try:
                    import torchvision

                    torchvision.utils.save_image(x_hat, recon_path, nrow=1)
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to save PNG reconstruction. "
                        "Install torchvision and ensure PIL support is available."
                    ) from exc

                # Optionally save bitstreams for archival/debug.
                if args.save_bitstreams:
                    bit_path = bitstream_root / scene / f"{frame:0>6}.bin"
                    _ensure_parent(bit_path)
                    import torch

                    torch.save({"strings": out_enc["strings"], "shape": out_enc["shape"]}, bit_path)

                rows_to_append.append(
                    {
                        "scene": scene,
                        "frame": frame,
                        "lambda": lmbda_str,
                        "num_pixels": num_pixels,
                        "num_bytes": num_bytes,
                        "num_bits": num_bits,
                        "bpp": bpp,
                        "recon_png": str(recon_path),
                        "ckpt": str(ckpt_path),
                    }
                )

            scenes_done += 1
            if args.max_scenes is not None and scenes_done >= args.max_scenes:
                break

            if scenes_done % 200 == 0:
                print(f"  processed scenes: {scenes_done:,} | new images: {len(rows_to_append):,}")

        # Write/append manifest.
        is_new = not manifest_path.exists()
        with manifest_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "scene",
                "frame",
                "lambda",
                "num_pixels",
                "num_bytes",
                "num_bits",
                "bpp",
                "recon_png",
                "ckpt",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            for row in rows_to_append:
                writer.writerow(row)

        print(f"==> Wrote {len(rows_to_append):,} rows to {manifest_path}")

        # Free GPU memory before the next lambda.
        del model
        if args.device == "cuda":
            import torch

            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
