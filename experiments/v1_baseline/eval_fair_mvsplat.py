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
import re
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_mvsplat_to_path(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "third_party" / "mvsplat"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_filename(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("._-")
    return safe or "run"


def _resolve_output_csv_path(output: Path, *, tag: str) -> Path:
    output = output.expanduser()
    if output.exists() and output.is_dir():
        return output / f"fair_{_safe_filename(tag)}.csv"
    if not output.exists() and output.suffix == "":
        return output.with_suffix(".csv")
    return output


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
    data_loader_num_workers: int | None,
    data_loader_persistent_workers: bool | None,
    data_loader_batch_size: int | None,
) -> tuple[Any, Any]:
    from hydra import compose, initialize_config_dir

    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg

    repo_root = _repo_root()
    config_dir = repo_root / "third_party" / "mvsplat" / "config"
    overrides = [
        "+experiment=re10k",
        "mode=test",
        "wandb.mode=disabled",
        "dataset/view_sampler=evaluation",
        f"dataset.view_sampler.index_path={index_path.resolve()}",
        f"dataset.roots=[{dataset_root.resolve()}]",
        "test.save_image=false",
        "test.save_video=false",
        "test.compute_scores=false",
    ]
    if data_loader_num_workers is not None:
        overrides.append(f"data_loader.test.num_workers={int(data_loader_num_workers)}")
    if data_loader_persistent_workers is not None:
        overrides.append(
            f"data_loader.test.persistent_workers={str(bool(data_loader_persistent_workers)).lower()}"
        )
    if data_loader_batch_size is not None:
        overrides.append(f"data_loader.test.batch_size={int(data_loader_batch_size)}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg_dict = compose(
            config_name="main",
            overrides=overrides,
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


def _load_bpp_metrics_csv(metrics_path: Path) -> dict[tuple[str, int], float]:
    """Load (scene, frame)->bpp from a legacy ELIC baseline metrics.csv.

    Expected columns:
      - scene
      - image_idx (or frame)
      - bpp
    """
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    bpp_by_key: dict[tuple[str, int], float] = {}
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row.get("scene")
            frame = row.get("image_idx") or row.get("frame")
            bpp = row.get("bpp")
            if scene is None or frame is None or bpp is None:
                continue
            bpp_by_key[(scene, int(frame))] = float(bpp)
    return bpp_by_key


def _count_non_null_scenes(index_path: Path) -> int | None:
    try:
        index = _load_json(index_path)
    except Exception:
        return None
    if not isinstance(index, dict):
        return None
    return sum(v is not None for v in index.values())


def _sum_bytes(obj: Any) -> int:
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_sum_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_sum_bytes(v) for v in obj.values())
    return 0


def _find_frame_file(scene_dir: Path, frame: int, suffix: str) -> Path:
    candidates = [
        scene_dir / f"{frame}{suffix}",
        scene_dir / f"{frame:03d}{suffix}",
        scene_dir / f"{frame:06d}{suffix}",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {suffix} for frame={frame} under {scene_dir}")


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
        help=(
            "Use compressed context frames from a directory. Supported layouts:\n"
            "  (1) Repo-native: <root>/recon/<scene>/<frame>.png + <root>/manifest.csv\n"
            "  (2) Legacy: <root>/<scene>/<frame>.png (+ optional .bin) + <root>/metrics.csv"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for MVSplat inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override MVSplat dataloader workers (data_loader.test.num_workers).",
    )
    parser.add_argument(
        "--persistent-workers",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override MVSplat dataloader persistence (data_loader.test.persistent_workers).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Test batch size (only 1 is supported by this script).",
    )
    parser.add_argument(
        "--vanilla-bpp",
        type=float,
        default=24.0,
        help="What bpp to report for the vanilla (uncompressed) point.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (if a directory is provided, writes fair_<tag>.csv inside).",
    )
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
    if args.batch_size != 1:
        raise SystemExit("This script currently supports --batch-size=1 only.")

    raw_output = args.output
    args.output = _resolve_output_csv_path(args.output, tag=args.tag)
    if args.output != raw_output:
        print(f"[output] {raw_output} -> {args.output}")
        if raw_output == Path("."):
            print(
                "[output] Note: '.' usually means you passed an empty shell variable; "
                "prefer passing an explicit file like --output outputs/fair_vanilla.csv"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Preflight check: fail fast on unwritable output paths (and avoid running ~6k scenes then crashing).
        with args.output.open("a", newline="", encoding="utf-8"):
            pass
    except OSError as exc:
        raise SystemExit(f"Cannot write --output={args.output}: {exc}") from exc

    expected_scenes = _count_non_null_scenes(args.index_path)
    is_tty = sys.stdout.isatty() or sys.stderr.isatty()

    # Load config + build model.
    persistent_workers: bool | None = None
    if args.persistent_workers != "auto":
        persistent_workers = args.persistent_workers == "true"

    _, cfg = _load_mvsplat_cfg(
        dataset_root=args.dataset_root,
        index_path=args.index_path,
        data_loader_num_workers=args.num_workers,
        data_loader_persistent_workers=persistent_workers,
        data_loader_batch_size=args.batch_size,
    )
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
        # Two supported layouts:
        # 1) Repo-native: <root>/manifest.csv + <root>/recon/<scene>/<frame:06d>.png
        # 2) Legacy baseline_ELIC: <root>/metrics.csv + <root>/<scene>/<frame:03d>.png (+ .bin)
        recon_root = None

        manifest_path = args.compressed_root / "manifest.csv"
        metrics_path = args.compressed_root / "metrics.csv"
        if manifest_path.exists():
            bpp_by_key = _load_bpp_manifest(manifest_path)
            recon_root = args.compressed_root / "recon"
        elif metrics_path.exists():
            bpp_by_key = _load_bpp_metrics_csv(metrics_path)
            recon_root = args.compressed_root
        else:
            raise FileNotFoundError(
                f"Unsupported compressed-root layout: expected {manifest_path} or {metrics_path}"
            )

        if recon_root is None or not recon_root.exists():
            raise FileNotFoundError(recon_root if recon_root is not None else args.compressed_root)

        from src.misc.image_io import load_image

        is_manifest_layout = manifest_path.exists()

        def load_recon(
            scene: str, frame: int, *, target_hw: tuple[int, int]
        ) -> tuple[Any, tuple[int, int]]:
            if is_manifest_layout:
                path = recon_root / scene / f"{frame:0>6}.png"
            else:
                path = _find_frame_file(recon_root / scene, frame, ".png")
            img = load_image(path)
            orig_hw = tuple(img.shape[-2:])
            if orig_hw != target_hw:
                from src.dataset.shims.crop_shim import rescale_and_crop

                img, _ = rescale_and_crop(img, intrinsics=img.new_zeros(3, 3), shape=target_hw)
            return img, orig_hw

    # Evaluate.
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    bpp_sum = 0.0
    bpp_count = 0
    num_scenes = 0

    import torch

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

    def fmt_metrics() -> str:
        if num_scenes == 0:
            return "psnr=? ssim=? lpips=?"
        return (
            f"psnr={psnr_sum / num_scenes:.2f} "
            f"ssim={ssim_sum / num_scenes:.4f} "
            f"lpips={lpips_sum / num_scenes:.4f}"
        )

    def process_batch(batch: dict[str, Any]) -> tuple[float, float, float, float | None]:
        scene = batch["scene"][0]
        context_frames = [int(x) for x in batch["context"]["index"][0].tolist()]
        target_hw = tuple(batch["context"]["image"].shape[-2:])

        bpp_scene: float | None = None
        if args.compressed_root is not None:
            assert bpp_by_key is not None
            recon_views: list[torch.Tensor] = []
            bpps: list[float] = []

            for fr in context_frames:
                img, orig_hw = load_recon(scene, fr, target_hw=target_hw)
                recon_views.append(img)

                key = (scene, fr)
                if key in bpp_by_key:
                    bpps.append(bpp_by_key[key])
                    continue
                if is_manifest_layout:
                    raise KeyError(
                        f"Missing bpp for {key} in {manifest_path} "
                        "(and no *.bin fallback in manifest layout)."
                    )
                bin_path = _find_frame_file(Path(args.compressed_root) / scene, fr, ".bin")
                try:
                    payload = torch.load(bin_path, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(bin_path, map_location="cpu")

                strings = payload.get("strings") if isinstance(payload, dict) else payload
                num_bits = 8 * _sum_bytes(strings)
                num_pixels = int(orig_hw[0] * orig_hw[1])
                bpps.append(num_bits / float(num_pixels))

            recon = torch.stack(recon_views, dim=0)  # [2,3,H,W]
            batch["context"]["image"][0] = recon
            bpp_scene = float(sum(bpps) / len(bpps))

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
        psnr_val = float(compute_psnr(rgb_gt, rgb).mean().item())
        ssim_val = float(compute_ssim(rgb_gt, rgb).mean().item())
        lpips_val = float(compute_lpips(rgb_gt, rgb).mean().item())
        return psnr_val, ssim_val, lpips_val, bpp_scene

    with torch.no_grad():
        desc_base = f"Eval {args.tag}"
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
            task = progress.add_task(desc_base, total=expected_scenes)
            with progress:
                for batch_idx, batch in enumerate(loader, start=1):
                    psnr_val, ssim_val, lpips_val, bpp_scene = process_batch(batch)
                    psnr_sum += psnr_val
                    ssim_sum += ssim_val
                    lpips_sum += lpips_val
                    num_scenes += 1
                    if bpp_scene is not None:
                        bpp_sum += bpp_scene
                        bpp_count += 1

                    progress.advance(task, 1)
                    if num_scenes % 10 == 0:
                        progress.update(task, description=f"{desc_base} | {fmt_metrics()}")

                    if args.max_scenes is not None and batch_idx >= args.max_scenes:
                        break
        else:
            try:
                from tqdm import tqdm
            except Exception:
                tqdm = None  # type: ignore

            iterator = loader
            pbar = None
            if tqdm is not None and is_tty:
                pbar = tqdm(loader, total=expected_scenes, desc=desc_base)
                iterator = pbar

            for batch_idx, batch in enumerate(iterator, start=1):
                psnr_val, ssim_val, lpips_val, bpp_scene = process_batch(batch)
                psnr_sum += psnr_val
                ssim_sum += ssim_val
                lpips_sum += lpips_val
                num_scenes += 1
                if bpp_scene is not None:
                    bpp_sum += bpp_scene
                    bpp_count += 1

                if pbar is not None and num_scenes % 10 == 0:
                    pbar.set_postfix_str(fmt_metrics())

                if args.max_scenes is not None and batch_idx >= args.max_scenes:
                    break
                if pbar is None and batch_idx % 200 == 0:
                    print(f"evaluated scenes: {batch_idx:,}")

    if num_scenes == 0:
        raise SystemExit("No scenes evaluated (check dataset + index).")

    psnr = psnr_sum / num_scenes
    ssim = ssim_sum / num_scenes
    lpips = lpips_sum / num_scenes
    bpp = (
        (bpp_sum / bpp_count)
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
