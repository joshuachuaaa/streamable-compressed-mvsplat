#!/usr/bin/env python3
"""Fast eval for V1 E2E checkpoints (ELIC forward pass, no bitstream export).

This evaluates:
  context (2 views) -> ELIC (eval/rounding) -> recon context -> MVSplat -> target (3 views)

and reports PSNR/SSIM/LPIPS plus an *estimated* bpp computed from ELIC likelihoods.

Use this for:
  - checkpoint selection curves during training (cheap; does not write recon PNGs)

For conference/appendix "fair" RD curves with *true entropy-coded bytes*, use:
  `experiments/v1_e2e/export_eval_fair.py`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_third_party_to_path(repo_root: Path) -> None:
    # MVSplat is an implicit namespace package called `src`.
    sys.path.insert(0, str(repo_root / "third_party" / "mvsplat"))
    # ELIC reimplementation uses top-level modules (e.g., `Network.py`).
    sys.path.insert(0, str(repo_root / "third_party" / "ELiC-ReImplemetation"))


def _format_lambda(value: float) -> str:
    return ("%.3f" % value).rstrip("0").rstrip(".")


def _lambda_to_elic_ckpt_name(lmbda: float) -> str:
    mapping = {
        0.004: "0004",
        0.008: "0008",
        0.016: "0016",
        0.032: "0032",
        0.15: "0150",
        0.45: "0450",
    }
    for k, v in mapping.items():
        if math.isclose(lmbda, k, rel_tol=0.0, abs_tol=1e-12):
            return f"ELIC_{v}_ft_3980_Plateau.pth.tar"
    raise ValueError(f"Unsupported --lambda={lmbda}; expected one of {sorted(mapping.keys())}")


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


def _compute_bpp_from_likelihoods(likelihoods: dict[str, Any], *, num_pixels: int) -> Any:
    import torch

    denom = float(-math.log(2) * num_pixels)
    bpp = torch.zeros((), dtype=torch.float32, device=next(iter(likelihoods.values())).device)
    for lik in likelihoods.values():
        bpp = bpp + torch.log(lik).sum() / denom
    return bpp


def _resolve_output_path(output: Path, tag: str) -> Path:
    if output.exists() and output.is_dir():
        return output / f"fast_{tag}.csv"
    if str(output) in ("", "."):
        return Path(".") / f"fast_{tag}.csv"
    return output


def _infer_global_step(mvsplat_ckpt: Path) -> int | None:
    import torch

    try:
        payload = torch.load(mvsplat_ckpt, map_location="cpu")
        step = payload.get("global_step") if isinstance(payload, dict) else None
        return int(step) if step is not None else None
    except Exception:
        return None


def _parse_step_from_path(path: Path) -> int | None:
    m = re.search(r"step_(\\d+)", str(path))
    return int(m.group(1)) if m else None


def _load_mvsplat_cfg(
    *,
    dataset_root: Path,
    index_path: Path,
    data_loader_num_workers: int | None,
    data_loader_persistent_workers: bool | None,
    data_loader_batch_size: int | None,
) -> tuple[Any, Any]:
    """Compose the upstream MVSplat config (+experiment=re10k) for dataset iteration."""
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


def _load_elic_model(checkpoint_path: Path, device: str, entropy_coder: str) -> Any:
    import compressai
    import torch
    from compressai.zoo import load_state_dict

    from Network import TestModel

    compressai.set_entropy_coder(entropy_coder)
    state_dict = load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = TestModel().from_state_dict(state_dict).eval()
    return model.to(device)


def main() -> int:
    repo_root = _repo_root()
    _add_third_party_to_path(repo_root)

    parser = argparse.ArgumentParser(description="V1-E2E fast eval: ELIC forward + MVSplat (RE10K)")
    parser.add_argument("--tag", required=True, help="Row label to write into the output CSV.")
    parser.add_argument("--lambda", dest="lmbda", type=float, required=True, help="Lambda value (e.g., 0.032).")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing mvsplat_finetuned.ckpt and ELIC_*.pth.tar (e.g., a step_* snapshot).",
    )
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
        help="Fixed evaluation index JSON (2 context → 3 target).",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device for evaluation.")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs" / "v1_e2e" / "results",
        help="CSV output path, or a directory to write into.",
    )
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a row to --output (disable to overwrite).",
    )
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap on evaluated scenes.")
    parser.add_argument(
        "--entropy-coder",
        default="ans",
        help="CompressAI entropy coder (default: ans).",
    )
    parser.add_argument(
        "--progress",
        choices=["auto", "rich", "tqdm", "none"],
        default="auto",
        help="Progress UI backend (auto prefers rich on TTY, else tqdm).",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Override MVSplat dataloader workers.")
    parser.add_argument(
        "--persistent-workers",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override MVSplat dataloader persistence.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Test batch size (only 1 is supported).")
    args = parser.parse_args()

    if args.batch_size != 1:
        raise SystemExit("--batch-size must be 1 for deterministic, comparable evaluation.")
    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    if not args.index_path.exists():
        raise FileNotFoundError(args.index_path)

    out_path = _resolve_output_path(args.output, args.tag)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", newline="", encoding="utf-8"):
            pass
    except OSError as exc:
        raise SystemExit(f"Cannot write --output={out_path}: {exc}") from exc

    mvsplat_ckpt = args.run_dir / "mvsplat_finetuned.ckpt"
    if not mvsplat_ckpt.exists():
        raise FileNotFoundError(mvsplat_ckpt)
    elic_ckpt = args.run_dir / _lambda_to_elic_ckpt_name(args.lmbda)
    if not elic_ckpt.exists():
        raise FileNotFoundError(elic_ckpt)

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
    model = ModelWrapper.load_from_checkpoint(str(mvsplat_ckpt), strict=True, **model_kwargs)
    model = model.eval().to(args.device)
    data_shim = get_data_shim(model.encoder)

    elic = _load_elic_model(elic_ckpt, device=args.device, entropy_coder=args.entropy_coder)

    # Dataset loader (fixed eval index).
    dm = DataModule(cfg.dataset, cfg.data_loader, step_tracker=None, global_rank=0)
    loader = dm.test_dataloader()

    expected_scenes = None
    try:
        index = _load_json(args.index_path)
        expected_scenes = sum(1 for v in index.values() if v is not None)
    except Exception:
        expected_scenes = None

    is_tty = os.isatty(1) or os.isatty(2)
    use_rich = False
    use_tqdm = False
    tqdm = None  # type: ignore[assignment]
    if args.progress in ("auto", "rich"):
        try:
            if args.progress == "rich" or is_tty:
                from rich.progress import (  # type: ignore[import-not-found]
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
    if not use_rich and args.progress in ("auto", "tqdm"):
        try:
            from tqdm import tqdm as _tqdm  # type: ignore[import-not-found]

            tqdm = _tqdm
            use_tqdm = True
        except Exception:
            use_tqdm = False
    if args.progress == "none":
        use_rich = False
        use_tqdm = False

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    bpp_sum = 0.0
    num_scenes = 0

    start_time = time.time()

    def process_batch(batch: dict[str, Any]) -> tuple[float, float, float, float]:
        import torch

        batch = _move_to_device(batch, args.device)
        batch = data_shim(batch)

        ctx = batch["context"]["image"]  # [B,2,3,H,W]
        b, v, c, h, w = ctx.shape
        if b != 1 or v != 2 or c != 3:
            raise RuntimeError(f"Expected context shape [1,2,3,H,W], got {tuple(ctx.shape)}")

        with torch.no_grad():
            ctx_flat = ctx.reshape(b * v, c, h, w)
            out_elic = elic(ctx_flat, noisequant=False)
            ctx_hat = out_elic["x_hat"].reshape(b, v, c, h, w).clamp(0, 1)

            num_pixels = int(ctx_flat.shape[0] * h * w)
            bpp_est = _compute_bpp_from_likelihoods(out_elic["likelihoods"], num_pixels=num_pixels)

        batch["context"]["image"] = ctx_hat

        _, _, _, th, tw = batch["target"]["image"].shape
        gaussians = model.encoder(batch["context"], global_step=0, deterministic=False)
        output = model.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (th, tw),
            depth_mode=None,
        )

        rgb = output.color[0]  # [V,3,H,W]
        rgb_gt = batch["target"]["image"][0]  # [V,3,H,W]
        psnr_val = float(compute_psnr(rgb_gt, rgb).mean().item())
        ssim_val = float(compute_ssim(rgb_gt, rgb).mean().item())
        lpips_val = float(compute_lpips(rgb_gt, rgb).mean().item())
        return psnr_val, ssim_val, lpips_val, float(bpp_est.detach().cpu().item())

    def should_stop(batch_idx_1b: int) -> bool:
        if args.max_scenes is None:
            return False
        return batch_idx_1b >= int(args.max_scenes)

    def fmt_metrics() -> str:
        if num_scenes == 0:
            return "psnr=? ssim=? lpips=?"
        return (
            f"psnr={psnr_sum / num_scenes:.2f} "
            f"ssim={ssim_sum / num_scenes:.4f} "
            f"lpips={lpips_sum / num_scenes:.4f} "
            f"bpp_est={bpp_sum / num_scenes:.4f}"
        )

    import torch

    with torch.no_grad():
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
            task = progress.add_task(f"eval {args.tag}", total=expected_scenes)
            with progress:
                for batch_idx, batch in enumerate(loader, start=1):
                    psnr_val, ssim_val, lpips_val, bpp_est = process_batch(batch)
                    psnr_sum += psnr_val
                    ssim_sum += ssim_val
                    lpips_sum += lpips_val
                    bpp_sum += bpp_est
                    num_scenes += 1
                    if num_scenes % 10 == 0:
                        progress.update(task, description=f"eval {args.tag} | {fmt_metrics()}")
                    progress.advance(task, 1)
                    if should_stop(batch_idx):
                        break
        elif use_tqdm and tqdm is not None:
            with tqdm(total=expected_scenes, desc=f"eval {args.tag}", dynamic_ncols=True) as pbar:
                for batch_idx, batch in enumerate(loader, start=1):
                    psnr_val, ssim_val, lpips_val, bpp_est = process_batch(batch)
                    psnr_sum += psnr_val
                    ssim_sum += ssim_val
                    lpips_sum += lpips_val
                    bpp_sum += bpp_est
                    num_scenes += 1
                    if num_scenes % 10 == 0:
                        pbar.set_postfix_str(fmt_metrics())
                    pbar.update(1)
                    if should_stop(batch_idx):
                        break
        else:
            for batch_idx, batch in enumerate(loader, start=1):
                psnr_val, ssim_val, lpips_val, bpp_est = process_batch(batch)
                psnr_sum += psnr_val
                ssim_sum += ssim_val
                lpips_sum += lpips_val
                bpp_sum += bpp_est
                num_scenes += 1
                if num_scenes % 200 == 0:
                    print(f"evaluated scenes: {num_scenes:,} | {fmt_metrics()}")
                if should_stop(batch_idx):
                    break

    if num_scenes == 0:
        raise SystemExit("No scenes evaluated (check dataset + index).")

    global_step = _infer_global_step(mvsplat_ckpt)
    if global_step is None:
        global_step = _parse_step_from_path(args.run_dir)

    row = {
        "tag": args.tag,
        "lambda": float(args.lmbda),
        "global_step": "" if global_step is None else int(global_step),
        "bpp_est": bpp_sum / num_scenes,
        "psnr": psnr_sum / num_scenes,
        "ssim": ssim_sum / num_scenes,
        "lpips": lpips_sum / num_scenes,
        "num_scenes": num_scenes,
        "elapsed_s": float(time.time() - start_time),
        "index_path": str(args.index_path),
        "mvsplat_ckpt": str(mvsplat_ckpt),
        "elic_ckpt": str(elic_ckpt),
        "run_dir": str(args.run_dir),
    }

    write_header = True
    if args.append and out_path.exists() and out_path.stat().st_size > 0:
        write_header = False

    with out_path.open("a" if args.append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("Wrote:", out_path)
    print("Row:", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
