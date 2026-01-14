#!/usr/bin/env python3
"""Fair MVSplat evaluation for the thesis baselines (fixed protocol).

This script evaluates a pretrained MVSplat checkpoint on the fixed evaluation index:
  `assets/indices/re10k/evaluation_index_re10k.json`

Two modes:
  - Vanilla: uses the dataset's RGB context frames.
  - Baseline (ELIC→MVSplat): replaces the 2 context frames with decoded context
    reconstructions from `--compressed-root` and reports bpp from `manifest.csv`.

This is the *only* evaluator used for thesis results (baselines + E2E), to keep
the protocol identical across methods.
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


def _count_non_null_scenes(index_path: Path) -> int:
    index = _load_json(index_path)
    if not isinstance(index, dict):
        return 0
    return sum(v is not None for v in index.values())


def _load_bpp_manifest(manifest_path: Path) -> dict[tuple[str, int], float]:
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


def _sum_bytes(obj: Any) -> int:
    """Recursively sum lengths of byte strings in `compress()` output."""
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_sum_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_sum_bytes(v) for v in obj.values())
    return 0


def _find_frame_file(scene_dir: Path, frame: int, ext: str) -> Path:
    """Find frame file using <frame:06d><ext> or raw <frame><ext>."""
    candidates = [
        scene_dir / f"{frame:0>6}{ext}",
        scene_dir / f"{frame}{ext}",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Missing frame file for frame={frame}: tried {candidates}")


def _add_third_party_to_path(repo_root: Path) -> None:
    # Back-compat alias (older name).
    _add_mvsplat_to_path(repo_root)


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
        cfg_dict = compose(config_name="main", overrides=overrides)
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    return cfg_dict, cfg


def _move_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    def move(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        if isinstance(x, list):
            return [move(v) for v in x]
        if isinstance(x, tuple):
            return tuple(move(v) for v in x)
        return x

    return move(batch)


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
        default=repo_root / "checkpoints" / "vanilla" / "MVSplat" / "re10k.ckpt",
        help="Pretrained MVSplat checkpoint (Lightning .ckpt).",
    )
    parser.add_argument(
        "--compressed-root",
        type=Path,
        default=None,
        help=(
            "Optional baseline context reconstructions directory. Supported layout:\n"
            "  <root>/manifest.csv + <root>/recon/<scene>/<frame:06d>.png\n"
            "If provided, bpp is loaded from manifest.csv and contexts are replaced by recon PNGs."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--persistent-workers", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1, help="Only 1 is supported.")
    parser.add_argument("--vanilla-bpp", type=float, default=24.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    if args.batch_size != 1:
        raise SystemExit("This script supports --batch-size=1 only.")

    raw_output = args.output
    args.output = _resolve_output_csv_path(args.output, tag=args.tag)
    if args.output != raw_output:
        print(f"[output] {raw_output} -> {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("a", newline="", encoding="utf-8"):
            pass
    except OSError as exc:
        raise SystemExit(f"Cannot write --output={args.output}: {exc}") from exc

    expected_scenes = _count_non_null_scenes(args.index_path)
    is_tty = sys.stdout.isatty() or sys.stderr.isatty()

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

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
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
        "losses": get_losses(cfg.loss),
        "step_tracker": None,
    }
    model = ModelWrapper.load_from_checkpoint(
        str(args.mvsplat_ckpt),
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        **model_kwargs,
    )
    model = model.to(args.device)
    model.eval()

    dm = DataModule(cfg.dataset, cfg.data_loader, step_tracker=model.step_tracker, global_rank=0)
    loader = dm.test_dataloader()
    data_shim = get_data_shim(model.encoder)

    bpp_by_key: dict[tuple[str, int], float] | None = None
    recon_root: Path | None = None
    if args.compressed_root is not None:
        manifest_path = args.compressed_root / "manifest.csv"
        recon_root = args.compressed_root / "recon"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        if not recon_root.exists():
            raise FileNotFoundError(recon_root)
        bpp_by_key = _load_bpp_manifest(manifest_path)

    def load_recon(scene: str, frame: int, target_hw: tuple[int, int]):
        import PIL.Image
        import torch
        import torchvision.transforms.functional as TF

        assert recon_root is not None
        p = _find_frame_file(recon_root / scene, frame, ".png")
        img = PIL.Image.open(p).convert("RGB")
        # Safety: if recon PNGs were generated elsewhere, resize to the dataset tensor H,W.
        if img.size != (target_hw[1], target_hw[0]):
            img = img.resize((target_hw[1], target_hw[0]), resample=PIL.Image.BILINEAR)
        x = TF.to_tensor(img)  # [3,H,W] in [0,1]
        return x

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    num_scenes = 0
    bpp_sum = 0.0
    bpp_count = 0

    def process_batch(batch: dict[str, Any]) -> tuple[float, float, float, float | None]:
        import torch

        scene = batch["scene"][0]
        context_frames = [int(x) for x in batch["context"]["index"][0].tolist()]
        target_hw = tuple(batch["context"]["image"].shape[-2:])

        bpp_scene: float | None = None
        if args.compressed_root is not None:
            assert bpp_by_key is not None
            recon_views: list[torch.Tensor] = []
            bpps: list[float] = []
            for fr in context_frames:
                recon_views.append(load_recon(scene, fr, target_hw=target_hw))
                bpps.append(bpp_by_key[(scene, fr)])
            # Replace contexts.
            batch["context"]["image"][0] = torch.stack(recon_views, dim=0)
            # Report average over the 2 context views (not sum).
            bpp_scene = float(sum(bpps) / len(bpps))

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

        rgb = output.color[0]
        rgb_gt = batch["target"]["image"][0]
        psnr_val = float(compute_psnr(rgb_gt, rgb).mean().item())
        ssim_val = float(compute_ssim(rgb_gt, rgb).mean().item())
        lpips_val = float(compute_lpips(rgb_gt, rgb).mean().item())
        return psnr_val, ssim_val, lpips_val, bpp_scene

    use_rich = False
    try:
        if is_tty:
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

            use_rich = True
    except Exception:
        use_rich = False

    def fmt_metrics() -> str:
        if num_scenes == 0:
            return "psnr=? ssim=? lpips=?"
        return f"psnr={psnr_sum/num_scenes:.2f} ssim={ssim_sum/num_scenes:.4f} lpips={lpips_sum/num_scenes:.4f}"

    with __import__("torch").no_grad():
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
            for batch_idx, batch in enumerate(loader, start=1):
                psnr_val, ssim_val, lpips_val, bpp_scene = process_batch(batch)
                psnr_sum += psnr_val
                ssim_sum += ssim_val
                lpips_sum += lpips_val
                num_scenes += 1
                if bpp_scene is not None:
                    bpp_sum += bpp_scene
                    bpp_count += 1
                if args.max_scenes is not None and batch_idx >= args.max_scenes:
                    break
                if (not is_tty) and batch_idx % 200 == 0:
                    print(f"evaluated scenes: {batch_idx:,}")

    if num_scenes == 0:
        raise SystemExit("No scenes evaluated (check dataset + index).")

    row = {
        "tag": args.tag,
        "bpp": (bpp_sum / bpp_count) if args.compressed_root is not None else float(args.vanilla_bpp),
        "psnr": psnr_sum / num_scenes,
        "ssim": ssim_sum / num_scenes,
        "lpips": lpips_sum / num_scenes,
        "num_scenes": num_scenes,
        "index_path": str(args.index_path),
        "mvsplat_ckpt": str(args.mvsplat_ckpt),
        "compressed_root": "" if args.compressed_root is None else str(args.compressed_root),
    }

    # Safe CSV writes even when multiple eval processes append concurrently.
    # (Useful when users run one process per lambda in parallel.)
    if args.append:
        mode = "a"
    else:
        mode = "w"

    try:
        import fcntl  # type: ignore
    except Exception:  # pragma: no cover
        fcntl = None  # type: ignore

    with args.output.open(mode, newline="", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            write_header = True
            if mode == "a" and args.output.exists() and args.output.stat().st_size > 0:
                write_header = False
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    print("Wrote:", args.output)
    print("Row:", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
