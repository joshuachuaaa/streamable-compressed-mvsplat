#!/usr/bin/env python3
"""V1 end-to-end fine-tuning: ELIC (context codec) + MVSplat (renderer).

This script implements the *joint* optimization variant:

  x_ctx (2 views) -> ELIC -> x̂_ctx -> MVSplat -> Î_tgt (1 view)

and optimizes a rate–distortion objective where the "distortion" is *novel view
synthesis error* on the target view(s), and the "rate" is the entropy-model
estimate (bpp) for the transmitted context images.

Notes:
  - Training uses the upstream MVSplat RE10K pipeline with the bounded sampler
    (2 context views, 1 target view).
  - This script does not run validation; evaluate with the fixed eval index via
    `experiments/v1_baseline/eval_fair_mvsplat.py` after exporting bitstreams.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_third_party_to_path(repo_root: Path) -> None:
    import sys

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
    """Return entropy-model bpp estimate from CompressAI likelihood tensors."""
    import torch

    denom = float(-math.log(2) * num_pixels)
    bpp = torch.zeros((), dtype=torch.float32, device=next(iter(likelihoods.values())).device)
    for lik in likelihoods.values():
        bpp = bpp + torch.log(lik).sum() / denom
    return bpp


def _split_elic_parameters_for_optim(elic: Any) -> tuple[list[Any], list[Any]]:
    """Return (main_params, aux_params) following CompressAI conventions."""
    params: dict[str, Any] = dict(elic.named_parameters())
    main_names = {n for n in params.keys() if (not n.endswith(".quantiles")) and params[n].requires_grad}
    aux_names = {n for n in params.keys() if n.endswith(".quantiles") and params[n].requires_grad}
    if main_names & aux_names:
        raise RuntimeError("ELIC parameter split has an unexpected intersection.")
    return [params[n] for n in sorted(main_names)], [params[n] for n in sorted(aux_names)]


def _load_mvsplat_cfg_train(
    *,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    seed: int,
    unimatch_weights_path: Path | None,
) -> tuple[Any, Any]:
    """Compose the upstream MVSplat config (+experiment=re10k) for training."""
    from hydra import compose, initialize_config_dir

    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg

    repo_root = _repo_root()
    config_dir = repo_root / "third_party" / "mvsplat" / "config"
    overrides = [
        "+experiment=re10k",
        "mode=train",
        "wandb.mode=disabled",
        "dataset/view_sampler=bounded",
        f"dataset.roots=[{dataset_root.resolve()}]",
        f"data_loader.train.batch_size={int(batch_size)}",
        f"data_loader.train.num_workers={int(num_workers)}",
        f"data_loader.train.persistent_workers={str(bool(persistent_workers)).lower()}",
        f"data_loader.train.seed={int(seed)}",
    ]
    # The upstream config points to an external Unimatch/GmDepth weights file used
    # only to initialize the multiview backbone when training from scratch. For
    # fine-tuning from a full MVSplat checkpoint, we disable this by default to
    # avoid requiring that extra artifact.
    if unimatch_weights_path is None:
        overrides.append("model.encoder.unimatch_weights_path=null")
    else:
        overrides.append(f"model.encoder.unimatch_weights_path={unimatch_weights_path.resolve()}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    return cfg_dict, cfg


def _maybe_git_rev(repo_root: Path) -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
        return out or None
    except Exception:
        return None


def main() -> int:
    repo_root = _repo_root()
    _add_third_party_to_path(repo_root)

    parser = argparse.ArgumentParser(description="V1 E2E fine-tuning: ELIC + MVSplat (RE10K)")
    parser.add_argument("--tag", required=True, help="Run name (used for output folder naming).")
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        required=True,
        help="RD trade-off weight (matched to the ELIC checkpoint family: 0.004/0.008/0.016/0.032/0.15/0.45).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset" / "re10k",
        help="Path to dataset/re10k.",
    )
    parser.add_argument(
        "--mvsplat-init-ckpt",
        type=Path,
        default=repo_root / "checkpoints" / "vanilla" / "MVSplat" / "re10k.ckpt",
        help="Pretrained MVSplat checkpoint to fine-tune from.",
    )
    parser.add_argument(
        "--elic-checkpoints",
        type=Path,
        default=repo_root / "checkpoints" / "vanilla" / "ELIC",
        help="Directory containing ELIC_*.pth.tar checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "checkpoints" / "v1_e2e",
        help="Directory to write fine-tuned checkpoints (gitignored).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Training device.",
    )
    parser.add_argument(
        "--unimatch-weights-path",
        type=Path,
        default=None,
        help=(
            "Optional Unimatch/GmDepth checkpoint used only for scratch initialization of the MVSplat "
            "multiview backbone. When fine-tuning from --mvsplat-init-ckpt, leave unset (default) "
            "to avoid requiring this extra file."
        ),
    )
    parser.add_argument(
        "--progress",
        choices=["auto", "rich", "tqdm", "none"],
        default="auto",
        help="Progress UI backend (auto prefers rich on TTY, else tqdm).",
    )
    parser.add_argument("--max-steps", type=int, default=10_000, help="Number of training steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=1, help="Train batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Train dataloader workers.")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use persistent dataloader workers (recommended when num_workers>0).",
    )
    parser.add_argument("--lr-mvsplat", type=float, default=1e-5, help="LR for MVSplat parameters.")
    parser.add_argument("--lr-elic", type=float, default=1e-5, help="LR for ELIC parameters (excluding aux).")
    parser.add_argument("--lr-elic-aux", type=float, default=1e-3, help="LR for ELIC aux parameters (.quantiles).")
    parser.add_argument(
        "--ctx-reg-beta",
        type=float,
        default=0.0,
        help="Optional context reconstruction regularizer weight (0 disables).",
    )
    parser.add_argument(
        "--elic-noisequant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ELIC noise-quant training path (matches the ELIC reimplementation training default).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Global norm gradient clipping (0 disables).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log to stdout / CSV every N steps.",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("Requested --device=cuda but CUDA is not available.")

    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    if not args.mvsplat_init_ckpt.exists():
        raise FileNotFoundError(args.mvsplat_init_ckpt)
    if not args.elic_checkpoints.exists():
        raise FileNotFoundError(args.elic_checkpoints)
    if args.unimatch_weights_path is not None and not args.unimatch_weights_path.exists():
        raise FileNotFoundError(args.unimatch_weights_path)

    # Determinism / reproducibility.
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except Exception:
        pass
    import torch

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Config + data.
    _, cfg = _load_mvsplat_cfg_train(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
        unimatch_weights_path=args.unimatch_weights_path,
    )
    if args.unimatch_weights_path is None:
        print(
            "==> Unimatch/GmDepth backbone init: disabled (fine-tuning loads backbone weights from "
            f"{args.mvsplat_init_ckpt})"
        )
    from src.dataset.data_module import DataModule, get_data_shim
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper, OptimizerCfg, TestCfg, TrainCfg

    step_tracker = StepTracker()
    dm = DataModule(cfg.dataset, cfg.data_loader, step_tracker=step_tracker, global_rank=0)
    train_loader = dm.train_dataloader()

    # Build MVSplat and load pretrained weights.
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
        "train_cfg": TrainCfg(
            depth_mode=cfg.train.depth_mode,
            extended_visualization=False,
            print_log_every_n_steps=999999,
        ),
        "encoder": encoder,
        "encoder_visualizer": None,
        "decoder": decoder,
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }
    mvsplat = ModelWrapper.load_from_checkpoint(str(args.mvsplat_init_ckpt), strict=True, **model_kwargs)
    mvsplat = mvsplat.train().to(args.device)

    # The encoder expects a data shim (e.g., normalize intrinsics layout).
    data_shim = get_data_shim(mvsplat.encoder)

    # Load ELIC.
    import compressai
    from compressai.zoo import load_state_dict
    from Network import TestModel

    elic_ckpt_name = _lambda_to_elic_ckpt_name(args.lmbda)
    elic_init_path = args.elic_checkpoints / elic_ckpt_name
    if not elic_init_path.exists():
        raise FileNotFoundError(elic_init_path)

    compressai.set_entropy_coder("ans")
    elic_state = load_state_dict(torch.load(elic_init_path, map_location="cpu"))
    elic = TestModel().from_state_dict(elic_state).train().to(args.device)

    # Optimizers (main + aux) following CompressAI conventions.
    elic_main_params, elic_aux_params = _split_elic_parameters_for_optim(elic)
    mvsplat_params = [p for p in mvsplat.parameters() if p.requires_grad]
    optim_main = torch.optim.Adam(
        [
            {"params": mvsplat_params, "lr": float(args.lr_mvsplat)},
            {"params": elic_main_params, "lr": float(args.lr_elic)},
        ],
        betas=(0.9, 0.999),
    )
    optim_aux = torch.optim.Adam(elic_aux_params, lr=float(args.lr_elic_aux), betas=(0.9, 0.999))

    # Outputs.
    lmbda_str = _format_lambda(args.lmbda)
    run_dir = args.output_dir / f"{args.tag}_lambda_{lmbda_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    args_path = run_dir / "run_args.json"
    args_path.write_text(
        json.dumps(
            {
                "git_rev": _maybe_git_rev(repo_root),
                **vars(args),
                "dataset_root": str(args.dataset_root),
                "mvsplat_init_ckpt": str(args.mvsplat_init_ckpt),
                "elic_init_ckpt": str(elic_init_path),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Progress UI.
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

    # Training loop.
    header = [
        "step",
        "loss_total",
        "rate_bpp",
        "dist_nvs",
        "dist_nvs_scaled",
        "ctx_mse",
        "aux_loss",
        "time_s",
    ]
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    start_time = time.time()

    def write_row(row: dict[str, float]) -> None:
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writerow({k: row.get(k, "") for k in header})

    def fmt(row: dict[str, float]) -> str:
        return (
            f"loss={row['loss_total']:.4f} "
            f"bpp={row['rate_bpp']:.4f} "
            f"dist={row['dist_nvs']:.4f} "
            f"aux={row['aux_loss']:.4f}"
        )

    iterator = iter(train_loader)

    def next_batch() -> dict[str, Any]:
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            return next(iterator)

    def run_step(batch: dict[str, Any], global_step: int) -> dict[str, Any]:
        # Keep view sampling schedules consistent with MVSplat.
        step_tracker.set_step(global_step)

        batch = _move_to_device(batch, args.device)
        batch = data_shim(batch)

        ctx = batch["context"]["image"]  # [B,Vc,3,H,W]
        b, v, c, h, w = ctx.shape
        if v != 2 or c != 3:
            raise RuntimeError(f"Expected context shape [B,2,3,H,W], got {tuple(ctx.shape)}")

        ctx_flat = ctx.reshape(b * v, c, h, w)
        out_elic = elic(ctx_flat, noisequant=bool(args.elic_noisequant))
        ctx_hat = out_elic["x_hat"].reshape(b, v, c, h, w).clamp(0, 1)

        num_pixels = int(ctx_flat.shape[0] * h * w)
        rate_bpp = _compute_bpp_from_likelihoods(out_elic["likelihoods"], num_pixels=num_pixels)

        ctx_mse = (ctx_hat - ctx).pow(2).mean()

        # Replace transmitted inputs with decoded reconstructions.
        batch["context"]["image"] = ctx_hat

        # Render the target view(s).
        _, _, _, th, tw = batch["target"]["image"].shape
        gaussians = mvsplat.encoder(batch["context"], global_step, False, scene_names=batch["scene"])
        output = mvsplat.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (th, tw),
            depth_mode=mvsplat.train_cfg.depth_mode,
        )

        # Distortion is the same loss family used by vanilla MVSplat training.
        dist_nvs = torch.zeros((), dtype=torch.float32, device=rate_bpp.device)
        for loss_fn in mvsplat.losses:
            dist_nvs = dist_nvs + loss_fn.forward(output, batch, gaussians, global_step)

        # Match ELIC's scale convention (ELIC uses MSE * 255^2).
        dist_nvs_scaled = dist_nvs * (255.0**2)

        loss_total = rate_bpp + float(args.lmbda) * dist_nvs_scaled
        if args.ctx_reg_beta > 0:
            loss_total = loss_total + float(args.ctx_reg_beta) * (ctx_mse * (255.0**2))

        return {
            "loss_total": loss_total,
            "rate_bpp": rate_bpp.detach(),
            "dist_nvs": dist_nvs.detach(),
            "dist_nvs_scaled": dist_nvs_scaled.detach(),
            "ctx_mse": ctx_mse.detach(),
        }

    def train_one_step(global_step: int) -> dict[str, float]:
        batch = next_batch()

        optim_main.zero_grad(set_to_none=True)
        optim_aux.zero_grad(set_to_none=True)

        out = run_step(batch, global_step)
        out["loss_total"].backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(mvsplat.parameters(), float(args.grad_clip))
            torch.nn.utils.clip_grad_norm_(elic.parameters(), float(args.grad_clip))
        optim_main.step()

        aux_loss = elic.aux_loss()
        aux_loss.backward()
        optim_aux.step()

        return {
            "step": float(global_step),
            "loss_total": float(out["loss_total"].detach().cpu().item()),
            "rate_bpp": float(out["rate_bpp"].cpu().item()),
            "dist_nvs": float(out["dist_nvs"].cpu().item()),
            "dist_nvs_scaled": float(out["dist_nvs_scaled"].cpu().item()),
            "ctx_mse": float(out["ctx_mse"].cpu().item()),
            "aux_loss": float(aux_loss.detach().cpu().item()),
            "time_s": float(time.time() - start_time),
        }

    max_steps = int(args.max_steps)
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
        task = progress.add_task(f"train {args.tag} (λ={lmbda_str})", total=max_steps)
        with progress:
            for step in range(max_steps):
                row = train_one_step(step)
                if (step % int(args.log_every)) == 0 or step == max_steps - 1:
                    write_row(row)
                    progress.update(task, description=f"train {args.tag} | {fmt(row)}")
                progress.advance(task, 1)
    elif use_tqdm and tqdm is not None:
        desc = f"train {args.tag} (λ={lmbda_str})"
        with tqdm(total=max_steps, desc=desc, dynamic_ncols=True) as pbar:
            for step in range(max_steps):
                row = train_one_step(step)
                if (step % int(args.log_every)) == 0 or step == max_steps - 1:
                    write_row(row)
                    pbar.set_postfix_str(fmt(row))
                pbar.update(1)
    else:
        for step in range(max_steps):
            row = train_one_step(step)
            if (step % int(args.log_every)) == 0 or step == max_steps - 1:
                write_row(row)
                print(f"[{step:>6}] {fmt(row)}")

    elapsed = time.time() - start_time

    # Save checkpoints.
    mvsplat_ckpt_out = run_dir / "mvsplat_finetuned.ckpt"
    elic_ckpt_out = run_dir / elic_ckpt_name

    # Ensure entropy-model buffers are up to date for real bitstream export.
    elic = elic.eval()
    try:
        elic.update(force=True)
    except Exception:
        # Some CompressAI versions use update(force=...) while others accept update().
        try:
            elic.update()
        except Exception:
            pass

    import pytorch_lightning as pl

    torch.save(
        {
            "epoch": 0,
            "global_step": int(args.max_steps),
            "pytorch-lightning_version": pl.__version__,
            "state_dict": mvsplat.state_dict(),
            "loops": {},
        },
        mvsplat_ckpt_out,
    )
    torch.save(elic.state_dict(), elic_ckpt_out)

    print(f"Done in {elapsed/60:.1f} min")
    print("Wrote:", mvsplat_ckpt_out)
    print("Wrote:", elic_ckpt_out)
    print("Log:", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
