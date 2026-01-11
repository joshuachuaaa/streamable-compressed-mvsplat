#!/usr/bin/env python3
"""Fair MVSplat evaluation for V1 (optionally with compressed context RGB).

This script evaluates a pretrained MVSplat checkpoint on a fixed evaluation index
(`assets/indices/re10k/evaluation_index_re10k.json`).

Two modes:
  - Vanilla: uses the dataset's RGB context frames.
  - V1: replaces the 2 context frames with ELIC reconstructions produced by
        `experiments/v1_baseline/compress.py` (true bitstream bpp from manifest.csv).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_mvsplat_to_path(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "third_party" / "mvsplat"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _move_to_device(x: Any, device: str) -> Any:
    import torch

    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, device) for v in x)
    return x


def _load_mvsplat_cfg(
    *,
    dataset_root: Path,
    index_path: Path,
) -> tuple[Any, Any]:
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


def _load_bpp_manifest(manifest_path: Path) -> dict[tuple[str, int], float]:
    import csv

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    bpp_by_key: dict[tuple[str, int], float] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row.get("scene")
            frame = row.get("frame")
            bpp = row.get("bpp")
            if scene is None or frame is None or bpp is None:
                continue
            bpp_by_key[(scene, int(frame))] = float(bpp)
    return bpp_by_key


def main() -> int:
    repo_root = _repo_root()
    _add_mvsplat_to_path(repo_root)

    parser = argparse.ArgumentParser(description="Evaluate MVSplat on RE10K fixed eval index (fair protocol)")
    parser.add_argument("--tag", required=True, help="Row label (e.g., vanilla or v1_lambda_0.032).")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset" / "re10k",
        help="Path to dataset/re10k.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=repo_root / "assets" / "indices" / "re10k" / "evaluation_index_re10k.json",
        help="Evaluation index JSON (2 context + 3 target).",
    )
    parser.add_argument(
        "--mvsplat-ckpt",
        type=Path,
        default=(
            repo_root / "checkpoints" / "vanilla" / "MVSplat" / "re10k.ckpt"
            if (repo_root / "checkpoints" / "vanilla" / "MVSplat" / "re10k.ckpt").exists()
            else (repo_root / "checkpoints" / "mvsplat" / "re10k.ckpt")
        ),
        help="Pretrained MVSplat checkpoint (Lightning .ckpt).",
    )
    parser.add_argument(
        "--compressed-root",
        type=Path,
        default=None,
        help="If set, use reconstructions from <root>/recon/<scene>/<frame>.png and bpp from <root>/manifest.csv.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for MVSplat inference.",
    )
    parser.add_argument(
        "--vanilla-bpp",
        type=float,
        default=24.0,
        help="What bpp to report for the vanilla (uncompressed) point.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append a row to an existing CSV (writes header only if missing).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Debug: stop after N scenes.",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("Requested --device=cuda but CUDA is not available.")

    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    if not args.index_path.exists():
        raise FileNotFoundError(args.index_path)
    if not args.mvsplat_ckpt.exists():
        raise FileNotFoundError(
            f"{args.mvsplat_ckpt} not found. Place the pretrained checkpoint under checkpoints/ "
            "(see checkpoints/README.md and docs/INSTALL.md)."
        )

    # Load config + build model.
    _, cfg = _load_mvsplat_cfg(dataset_root=args.dataset_root, index_path=args.index_path)
    from src.dataset.data_module import DataModule, get_data_shim
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
    from src.loss import get_losses
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper, OptimizerCfg, TestCfg, TrainCfg

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)

    model_kwargs = {
        "optimizer_cfg": OptimizerCfg(lr=0.0, warm_up_steps=0, cosine_lr=False),
        "test_cfg": TestCfg(
            output_path=Path("outputs/unused"),
            compute_scores=False,
            save_image=False,
            save_video=False,
            eval_time_skip_steps=0,
        ),
        "train_cfg": TrainCfg(depth_mode=None, extended_visualization=False, print_log_every_n_steps=999999),
        "encoder": encoder,
        "encoder_visualizer": None,
        "decoder": decoder,
        "losses": get_losses([]),
        "step_tracker": None,
    }

    model = ModelWrapper.load_from_checkpoint(str(args.mvsplat_ckpt), strict=True, **model_kwargs)
    model = model.eval().to(args.device)
    data_shim = get_data_shim(model.encoder)

    # Dataset loader (fixed eval index).
    dm = DataModule(cfg.dataset, cfg.data_loader, step_tracker=None, global_rank=0)
    loader = dm.test_dataloader()

    # Optional compressed context config.
    bpp_by_key: dict[tuple[str, int], float] | None = None
    if args.compressed_root is not None:
        manifest_path = args.compressed_root / "manifest.csv"
        bpp_by_key = _load_bpp_manifest(manifest_path)
        recon_root = args.compressed_root / "recon"
        if not recon_root.exists():
            raise FileNotFoundError(recon_root)

        from src.misc.image_io import load_image

        def load_recon(scene: str, frame: int) -> Any:
            path = recon_root / scene / f"{frame:0>6}.png"
            if not path.exists():
                raise FileNotFoundError(path)
            return load_image(path)

    # Evaluate.
    psnr_list: list[float] = []
    ssim_list: list[float] = []
    lpips_list: list[float] = []
    bpp_list: list[float] = []

    import torch
    from einops import rearrange

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # DataLoader yields CPU tensors; keep scene as python string.
            scene = batch["scene"][0]
            context_frames = [int(x) for x in batch["context"]["index"][0].tolist()]

            # Replace context images if requested.
            if args.compressed_root is not None:
                assert bpp_by_key is not None
                recon = torch.stack([load_recon(scene, fr) for fr in context_frames], dim=0)  # [2,3,H,W]
                batch["context"]["image"][0] = recon

                bpp_scene = sum(bpp_by_key[(scene, fr)] for fr in context_frames) / len(context_frames)
                bpp_list.append(bpp_scene)

            # Move to device and apply model-required shims.
            batch = _move_to_device(batch, args.device)
            batch = data_shim(batch)

            _, _, _, h, w = batch["target"]["image"].shape

            gaussians = model.encoder(batch["context"], global_step=0, deterministic=False)
            output = model.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )

            rgb = output.color[0]  # [V,3,H,W]
            rgb_gt = batch["target"]["image"][0]  # [V,3,H,W]

            # Treat view dimension as batch dimension for metric computation.
            psnr_list.append(compute_psnr(rgb_gt, rgb).mean().item())
            ssim_list.append(compute_ssim(rgb_gt, rgb).mean().item())
            lpips_list.append(compute_lpips(rgb_gt, rgb).mean().item())

            if args.max_scenes is not None and (batch_idx + 1) >= args.max_scenes:
                break
            if (batch_idx + 1) % 200 == 0:
                print(f"evaluated scenes: {batch_idx + 1:,}")

    num_scenes = len(psnr_list)
    if num_scenes == 0:
        raise SystemExit("No scenes evaluated (check dataset + index).")

    psnr = sum(psnr_list) / num_scenes
    ssim = sum(ssim_list) / num_scenes
    lpips = sum(lpips_list) / num_scenes
    bpp = (
        (sum(bpp_list) / len(bpp_list))
        if args.compressed_root is not None
        else float(args.vanilla_bpp)
    )

    row = {
        "tag": args.tag,
        "bpp": bpp,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "num_scenes": num_scenes,
        "index_path": str(args.index_path),
        "mvsplat_ckpt": str(args.mvsplat_ckpt),
        "compressed_root": "" if args.compressed_root is None else str(args.compressed_root),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = True
    if args.append and args.output.exists() and args.output.stat().st_size > 0:
        write_header = False

    with args.output.open("a" if args.append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("Wrote:", args.output)
    print("Row:", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
