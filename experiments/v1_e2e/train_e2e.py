#!/usr/bin/env python3
"""V1 end-to-end fine-tuning: ELIC (context codec) + MVSplat (renderer).

This script implements the *joint* optimization variant:

  x_ctx (2 views) -> ELIC -> x̂_ctx -> MVSplat -> Î_tgt (N views)

and optimizes a rate–distortion objective where the "distortion" is *novel view
synthesis error* on the target view(s), and the "rate" is the entropy-model
estimate (bpp) for the transmitted context images.

Notes:
  - Training uses the upstream MVSplat RE10K pipeline with the bounded sampler
    (2 context views, and `num_target_views` targets; for RE10K the upstream
    dataset-specific config uses 4 target views).
  - This script does not run validation; evaluate with the fixed eval index via
    `experiments/v1_e2e/export_eval_fair.py` (which uses `experiments/eval/eval_fair_mvsplat.py`).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
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
    raise ValueError(
        f"Unsupported --lambda={lmbda}; expected one of {sorted(mapping.keys())}. "
        "Note: --lambda selects the ELIC checkpoint family; use --rd-lambda to change the E2E RD weight."
    )


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
        help=(
            "ELIC checkpoint family λ (0.004/0.008/0.016/0.032/0.15/0.45). "
            "By default this also sets the E2E RD trade-off weight unless --rd-lambda is provided."
        ),
    )
    parser.add_argument(
        "--rd-lambda",
        dest="rd_lambda",
        type=float,
        default=None,
        help=(
            "Optional E2E RD trade-off weight in L = R(bpp_est) + rd_lambda * D_nvs_scaled. "
            "If unset, defaults to --lambda."
        ),
    )
    parser.add_argument(
        "--nvs-mse-scale",
        "--nvs-dist-scale",
        dest="nvs_mse_scale",
        type=float,
        default=65025.0,
        help=(
            "Scale applied to the *MSE component* of the MVSplat novel-view loss (LPIPS is unscaled). "
            "Default 65025 (=255^2) matches ELIC/CompressAI MSE-in-8bit-units convention; "
            "use 1.0 to keep everything in MVSplat's native [0,1] image scale. "
            "Note: `--nvs-dist-scale` is a deprecated alias (kept for compatibility)."
        ),
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
    parser.add_argument(
        "--max-epochs",
        type=float,
        default=None,
        help=(
            "Optional epoch budget. If set, overrides --max-steps by converting epochs -> steps "
            "using the *nominal* train scene count (len(dataset)/batch_size)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--view-sampler-step-offset",
        type=int,
        default=None,
        help=(
            "Offset added to the shared global step used by MVSplat's view sampler schedules and other "
            "step-conditioned components (e.g., LPIPS apply_after_step). "
            "If unset, defaults to the dataset view sampler warm_up_steps (i.e., start at end-of-warmup) "
            "so fine-tuning from a pretrained MVSplat checkpoint uses the same view-sampling regime. "
            "Set 0 to reproduce warmup-from-scratch behavior."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Train batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Train dataloader workers.")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use persistent dataloader workers (recommended when num_workers>0).",
    )
    parser.add_argument("--lr-mvsplat", type=float, default=1e-5, help="LR for MVSplat parameters.")
    parser.add_argument("--lr-elic", type=float, default=3e-5, help="LR for ELIC parameters (excluding aux).")
    parser.add_argument("--lr-elic-aux", type=float, default=1e-3, help="LR for ELIC aux parameters (.quantiles).")
    parser.add_argument(
        "--lr-schedule",
        choices=["onecycle", "none"],
        default="onecycle",
        help=(
            "LR schedule for the *main* optimizer (MVSplat + ELIC main params). "
            "`onecycle` matches upstream MVSplat's default cosine OneCycle schedule; "
            "`none` keeps constant learning rates."
        ),
    )
    parser.add_argument(
        "--onecycle-pct-start",
        type=float,
        default=0.01,
        help="OneCycleLR warmup fraction (pct_start).",
    )
    parser.add_argument(
        "--onecycle-div-factor",
        type=float,
        default=25.0,
        help="OneCycleLR div_factor (initial_lr = max_lr/div_factor).",
    )
    parser.add_argument(
        "--onecycle-final-div-factor",
        type=float,
        default=10_000.0,
        help="OneCycleLR final_div_factor (min_lr = max_lr/final_div_factor).",
    )
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
        default=0.5,
        help="Global norm gradient clipping (0 disables). Default matches upstream MVSplat.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log to stdout / CSV every N steps.",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Save intermediate checkpoints every N optimizer steps (0 disables).",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=0,
        help="Save intermediate checkpoints every N *nominal* epochs (0 disables).",
    )
    args = parser.parse_args()

    rd_lambda = float(args.rd_lambda) if args.rd_lambda is not None else float(args.lmbda)
    if rd_lambda <= 0:
        raise SystemExit("--rd-lambda must be > 0.")
    if args.nvs_mse_scale <= 0:
        raise SystemExit("--nvs-mse-scale must be > 0.")

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
    if args.save_every_steps < 0:
        raise SystemExit("--save-every-steps must be >= 0.")
    if args.save_every_epochs < 0:
        raise SystemExit("--save-every-epochs must be >= 0.")
    if args.save_every_steps > 0 and args.save_every_epochs > 0:
        raise SystemExit("Specify only one of --save-every-steps or --save-every-epochs (not both).")

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

    # Epoch accounting (for reporting + checkpoint cadence). Note: DatasetRE10k.__len__ is a
    # nominal upper bound (it does not account for skipped examples), so treat this as an
    # *approximate* epoch mapping.
    train_num_scenes_nominal: int | None = None
    steps_per_epoch_nominal: int | None = None
    try:
        train_num_scenes_nominal = int(len(train_loader.dataset))
    except Exception:
        train_num_scenes_nominal = None

    if train_num_scenes_nominal is not None:
        steps_per_epoch_nominal = int(math.ceil(train_num_scenes_nominal / max(1, int(args.batch_size))))

    max_steps = int(args.max_steps)
    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise SystemExit("--max-epochs must be > 0.")
        if steps_per_epoch_nominal is None:
            raise SystemExit("Cannot use --max-epochs because the train dataloader has no valid __len__.")
        max_steps = int(math.ceil(float(args.max_epochs) * float(steps_per_epoch_nominal)))

    if steps_per_epoch_nominal is not None:
        approx_epochs = float(max_steps) / float(steps_per_epoch_nominal)
        print(
            "==> Train epoch mapping (nominal): "
            f"{train_num_scenes_nominal} scenes | batch_size={args.batch_size} "
            f"-> ~{steps_per_epoch_nominal} steps/epoch | max_steps={max_steps} (~{approx_epochs:.2f} epochs)"
        )
    else:
        print(f"==> max_steps={max_steps} (epoch mapping unavailable; dataset has no __len__)")

    view_sampler_warmup_steps: int | None = None
    try:
        view_sampler_warmup_steps = int(getattr(cfg.dataset.view_sampler, "warm_up_steps", 0))
    except Exception:
        view_sampler_warmup_steps = None

    view_sampler_step_offset = args.view_sampler_step_offset
    if view_sampler_step_offset is None:
        if view_sampler_warmup_steps is not None and view_sampler_warmup_steps > 0:
            view_sampler_step_offset = int(view_sampler_warmup_steps)
        else:
            view_sampler_step_offset = 0
    if int(view_sampler_step_offset) < 0:
        raise SystemExit("--view-sampler-step-offset must be >= 0.")

    if view_sampler_warmup_steps is not None and view_sampler_warmup_steps > 0:
        print(
            "==> View sampler schedule: "
            f"warm_up_steps={int(view_sampler_warmup_steps)} | "
            f"step_offset={int(view_sampler_step_offset)} "
            f"({'post-warmup' if int(view_sampler_step_offset) >= int(view_sampler_warmup_steps) else 'in-warmup'})"
        )
    else:
        print(f"==> View sampler schedule: warm_up_steps=0 | step_offset={int(view_sampler_step_offset)}")

    # Ensure dataloader worker prefetching starts from the intended schedule step.
    step_tracker.set_step(int(view_sampler_step_offset))

    save_every_steps: int | None = None
    if args.save_every_steps > 0:
        save_every_steps = int(args.save_every_steps)
    elif args.save_every_epochs > 0:
        if steps_per_epoch_nominal is None:
            raise SystemExit("Cannot use --save-every-epochs because the train dataloader has no valid __len__.")
        save_every_steps = int(args.save_every_epochs) * int(steps_per_epoch_nominal)
        if save_every_steps <= 0:
            raise SystemExit("--save-every-epochs produced a non-positive step interval.")

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

    # Print + record upstream loss config (critique-resistance: avoid "hidden schedules" ambiguity).
    mvsplat_losses_summary: list[dict[str, Any]] = []
    lpips_apply_after_step: int | None = None
    for loss_fn in mvsplat.losses:
        cfg_obj = getattr(loss_fn, "cfg", None)
        cfg_payload: Any
        if cfg_obj is not None and is_dataclass(cfg_obj):
            cfg_payload = asdict(cfg_obj)
        else:
            cfg_payload = str(cfg_obj)
        mvsplat_losses_summary.append(
            {
                "name": getattr(loss_fn, "name", type(loss_fn).__name__),
                "class": type(loss_fn).__name__,
                "cfg": cfg_payload,
            }
        )
        if type(loss_fn).__name__ == "LossLpips":
            try:
                lpips_apply_after_step = int(getattr(loss_fn.cfg, "apply_after_step", 0))
            except Exception:
                lpips_apply_after_step = None

    if mvsplat_losses_summary:
        parts: list[str] = []
        for item in mvsplat_losses_summary:
            name = str(item.get("name", "loss"))
            cfg = item.get("cfg", {})
            if isinstance(cfg, dict) and cfg:
                cfg_str = ", ".join(f"{k}={cfg[k]}" for k in sorted(cfg.keys()))
                parts.append(f"{name}({cfg_str})")
            else:
                parts.append(name)
        print("==> MVSplat loss config:", " + ".join(parts))
        if lpips_apply_after_step is not None:
            start_step = int(view_sampler_step_offset)
            if start_step >= int(lpips_apply_after_step):
                print(f"==> MVSplat LPIPS: active from step 0 (effective_global_step starts at {start_step})")
            else:
                print(
                    "==> MVSplat LPIPS: inactive at step 0 "
                    f"(apply_after_step={int(lpips_apply_after_step)}, effective_global_step starts at {start_step}). "
                    "Set --view-sampler-step-offset accordingly if you intend to enable LPIPS during fine-tuning."
                )

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
    scheduler_main = None
    if args.lr_schedule != "none":
        if args.lr_schedule == "onecycle":
            scheduler_main = torch.optim.lr_scheduler.OneCycleLR(
                optim_main,
                max_lr=[float(args.lr_mvsplat), float(args.lr_elic)],
                total_steps=int(max_steps),
                pct_start=float(args.onecycle_pct_start),
                cycle_momentum=False,
                anneal_strategy="cos",
                div_factor=float(args.onecycle_div_factor),
                final_div_factor=float(args.onecycle_final_div_factor),
            )
        else:
            raise RuntimeError(f"Unhandled --lr-schedule={args.lr_schedule}")

    # Outputs.
    lmbda_str = _format_lambda(args.lmbda)
    rd_lmbda_str = _format_lambda(rd_lambda)
    # Build a stable run directory name.
    #
    # Convention:
    #   <tag>_lambda_<elic_lambda>[_rd_<rd_lambda>][_s_<nvs_mse_scale>]
    #
    # Avoid accidental duplication when users include these suffixes in --tag.
    run_name = str(args.tag)
    if "_lambda_" not in run_name:
        run_name = f"{run_name}_lambda_{lmbda_str}"
    if not math.isclose(rd_lambda, float(args.lmbda), rel_tol=0.0, abs_tol=1e-12) and "_rd_" not in run_name:
        run_name = f"{run_name}_rd_{rd_lmbda_str}"
    if (
        not math.isclose(float(args.nvs_mse_scale), 65025.0, rel_tol=0.0, abs_tol=1e-12)
        and "_s_" not in run_name
    ):
        run_name = f"{run_name}_s_{float(args.nvs_mse_scale):g}"
    run_dir = args.output_dir / run_name
    if (run_dir / "train_log.csv").exists() or (run_dir / "run_args.json").exists():
        # Avoid silently mixing logs/checkpoints when reusing a tag.
        i = 2
        while True:
            candidate = args.output_dir / f"{run_name}_run{i}"
            if not (candidate / "train_log.csv").exists() and not (candidate / "run_args.json").exists():
                print(f"==> Output dir exists; writing to: {candidate} (use a unique --tag to avoid this)")
                run_dir = candidate
                break
            i += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    epoch_log_path = run_dir / "train_epoch_log.csv"
    args_path = run_dir / "run_args.json"
    args_path.write_text(
        json.dumps(
            {
                "git_rev": _maybe_git_rev(repo_root),
                "train_num_scenes_nominal": train_num_scenes_nominal,
                "steps_per_epoch_nominal": steps_per_epoch_nominal,
                "run_name_effective": str(run_dir.name),
                "mvsplat_losses": mvsplat_losses_summary,
                "max_steps_effective": max_steps,
                "save_every_steps_effective": save_every_steps,
                "rd_lambda_effective": float(rd_lambda),
                "nvs_mse_scale_effective": float(args.nvs_mse_scale),
                "view_sampler_step_offset_effective": int(view_sampler_step_offset),
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
                    MofNCompleteColumn,
                    Progress,
                    ProgressColumn,
                    SpinnerColumn,
                    TaskProgressColumn,
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
        "lr_mvsplat",
        "lr_elic",
        "lr_elic_aux",
        "loss_total",
        "rate_bpp",
        "dist_nvs",
        "dist_nvs_mse",
        "dist_nvs_other",
        "dist_nvs_scaled",
        "ctx_mse",
        "aux_loss",
        "grad_norm",
        "time_s",
    ]
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    epoch_header = [
        "epoch",
        "step_start",
        "step_end",
        "num_steps",
        "loss_total_mean",
        "rate_bpp_mean",
        "dist_nvs_mean",
        "dist_nvs_mse_mean",
        "dist_nvs_other_mean",
        "dist_nvs_scaled_mean",
        "ctx_mse_mean",
        "aux_loss_mean",
        "time_s_end",
    ]
    if steps_per_epoch_nominal is not None and not epoch_log_path.exists():
        with epoch_log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(epoch_header)

    start_time = time.time()

    def write_row(row: dict[str, float]) -> None:
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writerow({k: row.get(k, "") for k in header})

    def fmt(row: dict[str, float]) -> str:
        ctx_psnr = _psnr_from_mse(row["ctx_mse"])
        grad_norm = row.get("grad_norm")
        grad_norm_str = f"{float(grad_norm):.2f}" if grad_norm is not None and math.isfinite(float(grad_norm)) else "NA"
        lr_m = row.get("lr_mvsplat")
        lr_e = row.get("lr_elic")
        lr_m_str = f"{float(lr_m):.1e}" if lr_m is not None and float(lr_m) > 0 else "NA"
        lr_e_str = f"{float(lr_e):.1e}" if lr_e is not None and float(lr_e) > 0 else "NA"
        return (
            f"loss={row['loss_total']:.4f} "
            f"bpp={row['rate_bpp']:.4f} "
            f"dist={row['dist_nvs']:.4f} "
            f"dist*={row['dist_nvs_scaled']:.4f} "
            f"ctx_psnr={ctx_psnr:.2f} "
            f"aux={row['aux_loss']:.4f} "
            f"g={grad_norm_str} "
            f"lr={lr_m_str}/{lr_e_str}"
        )

    def _psnr_from_mse(mse: float, *, eps: float = 1e-10) -> float:
        # Images are in [0,1], so PSNR = 10*log10(1/mse) = -10*log10(mse).
        return -10.0 * math.log10(max(float(mse), float(eps)))

    def write_epoch_row(row: dict[str, float]) -> None:
        if steps_per_epoch_nominal is None:
            return
        with epoch_log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=epoch_header)
            w.writerow({k: row.get(k, "") for k in epoch_header})

    iterator = iter(train_loader)

    def next_batch() -> dict[str, Any]:
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            return next(iterator)

    def run_step(batch: dict[str, Any], global_step: int) -> dict[str, Any]:
        mvs_global_step = int(view_sampler_step_offset) + int(global_step)
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
        gaussians = mvsplat.encoder(batch["context"], mvs_global_step, False, scene_names=batch["scene"])
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
        #
        # Important: ELIC/CompressAI convention scales *only the MSE term* by 255^2.
        # LPIPS is unitless and should NOT be scaled.
        dist_nvs = torch.zeros((), dtype=torch.float32, device=rate_bpp.device)
        dist_mse: torch.Tensor | None = None
        dist_other = torch.zeros((), dtype=torch.float32, device=rate_bpp.device)
        for loss_fn in mvsplat.losses:
            term = loss_fn.forward(output, batch, gaussians, mvs_global_step)
            dist_nvs = dist_nvs + term
            if type(loss_fn).__name__ == "LossMse":
                dist_mse = term
            else:
                dist_other = dist_other + term

        if dist_mse is None:
            raise RuntimeError(
                "Expected a LossMse term in mvsplat.losses, but none was found. "
                "Check the underlying MVSplat config (loss=[mse, lpips])."
            )

        dist_term = dist_mse * float(args.nvs_mse_scale) + dist_other

        loss_total = rate_bpp + float(rd_lambda) * dist_term
        if args.ctx_reg_beta > 0:
            loss_total = loss_total + float(args.ctx_reg_beta) * (ctx_mse * (255.0**2))

        return {
            "loss_total": loss_total,
            "rate_bpp": rate_bpp.detach(),
            "dist_nvs": dist_nvs.detach(),
            "dist_nvs_mse": dist_mse.detach(),
            "dist_nvs_other": dist_other.detach(),
            # Historical name: in earlier versions we logged dist_nvs * 255^2.
            # Now this is the *distortion term used in the RD loss*:
            #   dist_nvs_scaled = (mse * nvs_mse_scale) + (lpips unscaled)
            "dist_nvs_scaled": dist_term.detach(),
            "ctx_mse": ctx_mse.detach(),
        }

    def train_one_step(global_step: int) -> dict[str, float]:
        # Ensure the dataloader's view sampling schedules see the intended global step.
        step_tracker.set_step(int(view_sampler_step_offset) + int(global_step))
        batch = next_batch()

        lr_mvsplat = float(optim_main.param_groups[0]["lr"]) if len(optim_main.param_groups) > 0 else math.nan
        lr_elic = float(optim_main.param_groups[1]["lr"]) if len(optim_main.param_groups) > 1 else math.nan
        lr_elic_aux = float(optim_aux.param_groups[0]["lr"]) if len(optim_aux.param_groups) > 0 else math.nan

        optim_main.zero_grad(set_to_none=True)
        optim_aux.zero_grad(set_to_none=True)

        out = run_step(batch, global_step)
        out["loss_total"].backward()
        grad_norm = math.nan
        if args.grad_clip and args.grad_clip > 0:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                [p for p in list(mvsplat.parameters()) + list(elic.parameters()) if p.requires_grad],
                float(args.grad_clip),
            )
            try:
                grad_norm = float(grad_norm_t.detach().cpu().item())
            except Exception:
                grad_norm = float(grad_norm_t)
        optim_main.step()
        if scheduler_main is not None:
            scheduler_main.step()

        aux_loss = elic.aux_loss()
        aux_loss.backward()
        optim_aux.step()

        return {
            "step": float(global_step),
            "lr_mvsplat": float(lr_mvsplat),
            "lr_elic": float(lr_elic),
            "lr_elic_aux": float(lr_elic_aux),
            "loss_total": float(out["loss_total"].detach().cpu().item()),
            "rate_bpp": float(out["rate_bpp"].cpu().item()),
            "dist_nvs": float(out["dist_nvs"].cpu().item()),
            "dist_nvs_mse": float(out["dist_nvs_mse"].cpu().item()),
            "dist_nvs_other": float(out["dist_nvs_other"].cpu().item()),
            "dist_nvs_scaled": float(out["dist_nvs_scaled"].cpu().item()),
            "ctx_mse": float(out["ctx_mse"].cpu().item()),
            "aux_loss": float(aux_loss.detach().cpu().item()),
            "grad_norm": float(grad_norm),
            "time_s": float(time.time() - start_time),
        }

    epoch_sums: dict[str, float] = {}
    epoch_count = 0
    epoch_idx_0b: int | None = None
    epoch_step_start_1b: int | None = None

    def epoch_accumulate(row: dict[str, float], global_step_1b: int) -> None:
        nonlocal epoch_sums, epoch_count, epoch_idx_0b, epoch_step_start_1b
        if steps_per_epoch_nominal is None:
            return
        current_epoch_0b = int((global_step_1b - 1) // int(steps_per_epoch_nominal))
        if epoch_idx_0b is None:
            epoch_idx_0b = current_epoch_0b
            epoch_step_start_1b = global_step_1b
        if current_epoch_0b != epoch_idx_0b:
            # This should only happen if steps_per_epoch_nominal changes mid-run (it shouldn't).
            epoch_idx_0b = current_epoch_0b
            epoch_step_start_1b = global_step_1b
            epoch_sums = {}
            epoch_count = 0

        for k in (
            "loss_total",
            "rate_bpp",
            "dist_nvs",
            "dist_nvs_mse",
            "dist_nvs_other",
            "dist_nvs_scaled",
            "ctx_mse",
            "aux_loss",
        ):
            epoch_sums[k] = epoch_sums.get(k, 0.0) + float(row[k])
        epoch_count += 1

    def epoch_maybe_flush(row: dict[str, float], *, global_step_1b: int, is_last_step: bool) -> None:
        nonlocal epoch_sums, epoch_count, epoch_idx_0b, epoch_step_start_1b
        if steps_per_epoch_nominal is None or epoch_count <= 0:
            return
        is_epoch_end = (global_step_1b % int(steps_per_epoch_nominal)) == 0
        if not is_epoch_end and not is_last_step:
            return
        assert epoch_idx_0b is not None and epoch_step_start_1b is not None
        write_epoch_row(
            {
                "epoch": float(epoch_idx_0b + 1),
                "step_start": float(epoch_step_start_1b),
                "step_end": float(global_step_1b),
                "num_steps": float(epoch_count),
                "loss_total_mean": epoch_sums["loss_total"] / epoch_count,
                "rate_bpp_mean": epoch_sums["rate_bpp"] / epoch_count,
                "dist_nvs_mean": epoch_sums["dist_nvs"] / epoch_count,
                "dist_nvs_mse_mean": epoch_sums["dist_nvs_mse"] / epoch_count,
                "dist_nvs_other_mean": epoch_sums["dist_nvs_other"] / epoch_count,
                "dist_nvs_scaled_mean": epoch_sums["dist_nvs_scaled"] / epoch_count,
                "ctx_mse_mean": epoch_sums["ctx_mse"] / epoch_count,
                "aux_loss_mean": epoch_sums["aux_loss"] / epoch_count,
                "time_s_end": float(row["time_s"]),
            }
        )
        epoch_sums = {}
        epoch_count = 0
        epoch_idx_0b = None
        epoch_step_start_1b = None

    def save_snapshot(global_step_1b: int) -> None:
        """Save a minimal, self-contained checkpoint folder for fair export/eval."""
        ckpt_dir = run_dir / "checkpoints" / f"step_{global_step_1b:07d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mvsplat_ckpt_path = ckpt_dir / "mvsplat_finetuned.ckpt"
        elic_ckpt_path = ckpt_dir / elic_ckpt_name

        import pytorch_lightning as pl

        torch.save(
            {
                "epoch": 0,
                "global_step": int(global_step_1b),
                "pytorch-lightning_version": pl.__version__,
                "state_dict": mvsplat.state_dict(),
                "loops": {},
            },
            mvsplat_ckpt_path,
        )
        torch.save(elic.state_dict(), elic_ckpt_path)

        approx_epoch = (
            (float(global_step_1b) / float(steps_per_epoch_nominal))
            if steps_per_epoch_nominal is not None
            else None
        )
        (ckpt_dir / "checkpoint_meta.json").write_text(
            json.dumps(
                {
                    "global_step": int(global_step_1b),
                    "approx_epoch_nominal": approx_epoch,
                    "elapsed_s": float(time.time() - start_time),
                    "mvsplat_ckpt": str(mvsplat_ckpt_path),
                    "elic_ckpt": str(elic_ckpt_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if use_rich:
        try:
            from rich import box  # type: ignore[import-not-found]
            from rich.console import Console, Group  # type: ignore[import-not-found]
            from rich.layout import Layout  # type: ignore[import-not-found]
            from rich.live import Live  # type: ignore[import-not-found]
            from rich.panel import Panel  # type: ignore[import-not-found]
            from rich.progress_bar import ProgressBar  # type: ignore[import-not-found]
            from rich.table import Table  # type: ignore[import-not-found]
            from rich.text import Text  # type: ignore[import-not-found]
        except Exception:
            use_rich = False

    if use_rich:
        def _fmt_float(value: Any, fmt: str, *, default: str = "--") -> str:
            try:
                x = float(value)
            except Exception:
                return default
            if not math.isfinite(x):
                return default
            return fmt.format(x)

        def _fmt_time(seconds: Any) -> str:
            try:
                s = float(seconds)
            except Exception:
                return "--:--"
            if not math.isfinite(s) or s < 0:
                return "--:--"
            t = int(s + 0.5)
            m, sec = divmod(t, 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:d}:{sec:02d}"

        def _kv_table(rows: list[tuple[str, str]]) -> Table:
            t = Table.grid(expand=True, padding=(0, 1))
            t.add_column(justify="right", style="bright_black", no_wrap=True)
            t.add_column(justify="left", no_wrap=True)
            for k, v in rows:
                t.add_row(k, v)
            return t

        def _panel(title: str, body: Any, *, style: str) -> Panel:
            return Panel(body, title=title, border_style=style, box=box.ROUNDED, padding=(0, 1))

        console = Console()

        if args.device == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        ema_sps: float | None = None
        last_wall = time.time()
        last_log_step_1b: int | None = None
        last_log_msg = ""
        last_snapshot_step_1b: int | None = None

        layout = Layout()
        layout.split(
            Layout(name="header", size=5),
            Layout(name="main", ratio=1),
        )
        layout["main"].split(
            Layout(name="top", size=11),
            Layout(name="bottom", ratio=1),
        )
        layout["top"].split_row(
            Layout(name="progress"),
            Layout(name="rd"),
            Layout(name="optim"),
        )
        layout["bottom"].split_row(
            Layout(name="dist"),
            Layout(name="system"),
        )

        with Live(layout, console=console, refresh_per_second=10, transient=False):
            for step in range(max_steps):
                row = train_one_step(step)
                global_step_1b = int(step) + 1
                effective_step = int(view_sampler_step_offset) + int(step)

                now = time.time()
                dt = now - last_wall
                last_wall = now
                if dt > 0:
                    sps = 1.0 / dt
                    ema_sps = sps if ema_sps is None else (0.9 * ema_sps + 0.1 * sps)

                eta_s = (
                    (float(max_steps - global_step_1b) / float(ema_sps))
                    if ema_sps is not None and ema_sps > 0
                    else math.nan
                )
                epoch_nominal = (
                    (1.0 + (float(global_step_1b - 1) / float(steps_per_epoch_nominal)))
                    if steps_per_epoch_nominal is not None
                    else math.nan
                )

                if (step % int(args.log_every)) == 0 or step == max_steps - 1:
                    write_row(row)
                    last_log_step_1b = global_step_1b
                    last_log_msg = fmt(row)

                epoch_accumulate(row, global_step_1b)
                epoch_maybe_flush(row, global_step_1b=global_step_1b, is_last_step=step == max_steps - 1)

                if save_every_steps is not None and (global_step_1b % int(save_every_steps)) == 0:
                    save_snapshot(global_step_1b)
                    last_snapshot_step_1b = global_step_1b

                # Header tile.
                header_left = _kv_table(
                    [
                        ("run", str(run_dir.name)),
                        ("device", str(args.device)),
                        ("bs/workers", f"{int(args.batch_size)}/{int(args.num_workers)}"),
                        ("seed", str(int(args.seed))),
                    ]
                )
                header_right = _kv_table(
                    [
                        ("λ_elic/λ_rd", f"{lmbda_str}/{rd_lmbda_str}"),
                        ("s_mse", f"{float(args.nvs_mse_scale):g}"),
                        ("sched", str(args.lr_schedule)),
                        ("offset", str(int(view_sampler_step_offset))),
                    ]
                )
                header_grid = Table.grid(expand=True)
                header_grid.add_column(ratio=1)
                header_grid.add_column(ratio=1)
                header_grid.add_row(header_left, header_right)
                layout["header"].update(_panel("Run", header_grid, style="cyan"))

                # Progress tile.
                bar = ProgressBar(total=float(max_steps), completed=float(global_step_1b), width=None)
                prog_lines = Group(
                    Text(
                        f"step {global_step_1b:,}/{max_steps:,} "
                        f"({_fmt_float(100.0 * global_step_1b / max(1, max_steps), '{:.2f}')}%)"
                    ),
                    Text(f"epoch (nominal): {_fmt_float(epoch_nominal, '{:.2f}', default='--')}"),
                    Text(f"effective global step: {effective_step:,}"),
                    bar,
                    Text(
                        f"sps={_fmt_float(ema_sps, '{:.2f}', default='--')} | "
                        f"elapsed={_fmt_time(row.get('time_s'))} | "
                        f"eta={_fmt_time(eta_s)}"
                    ),
                )
                layout["progress"].update(_panel("Progress", prog_lines, style="bright_black"))

                # RD tile.
                loss_total = row.get("loss_total", math.nan)
                rate_bpp = row.get("rate_bpp", math.nan)
                dist_raw = row.get("dist_nvs", math.nan)
                dist_scaled = row.get("dist_nvs_scaled", math.nan)
                ctx_mse = row.get("ctx_mse", math.nan)
                rd_term = float(rd_lambda) * float(dist_scaled) if math.isfinite(float(dist_scaled)) else math.nan
                ctx_reg_term = (
                    float(args.ctx_reg_beta) * (float(ctx_mse) * (255.0**2))
                    if args.ctx_reg_beta > 0 and math.isfinite(float(ctx_mse))
                    else 0.0
                )
                rd_table = _kv_table(
                    [
                        ("L_total", _fmt_float(loss_total, "{:.4f}")),
                        ("R (bpp_est)", _fmt_float(rate_bpp, "{:.4f}")),
                        ("λ·D_scaled", _fmt_float(rd_term, "{:.4f}")),
                        ("D_scaled", _fmt_float(dist_scaled, "{:.4f}")),
                        ("D_raw", _fmt_float(dist_raw, "{:.4f}")),
                        ("ctx_psnr", _fmt_float(_psnr_from_mse(float(ctx_mse)), "{:.2f} dB")),
                        ("β·ctx_mse", _fmt_float(ctx_reg_term, "{:.4f}")),
                    ]
                )
                layout["rd"].update(_panel("RD Objective", rd_table, style="green"))

                # Optimization tile.
                lr_table = _kv_table(
                    [
                        ("lr_mvsplat", _fmt_float(row.get("lr_mvsplat"), "{:.2e}")),
                        ("lr_elic", _fmt_float(row.get("lr_elic"), "{:.2e}")),
                        ("lr_aux", _fmt_float(row.get("lr_elic_aux"), "{:.2e}")),
                        ("grad_norm", _fmt_float(row.get("grad_norm"), "{:.2f}")),
                        ("aux_loss", _fmt_float(row.get("aux_loss"), "{:.4f}")),
                        ("noisequant", str(bool(args.elic_noisequant))),
                    ]
                )
                layout["optim"].update(_panel("Optimization", lr_table, style="blue"))

                # Distortion breakdown tile.
                dist_mse = row.get("dist_nvs_mse", math.nan)
                dist_other = row.get("dist_nvs_other", math.nan)
                dist_mse_scaled = (
                    float(dist_mse) * float(args.nvs_mse_scale) if math.isfinite(float(dist_mse)) else math.nan
                )
                dist_check = (
                    (float(dist_mse_scaled) + float(dist_other) - float(dist_scaled))
                    if math.isfinite(float(dist_mse_scaled))
                    and math.isfinite(float(dist_other))
                    and math.isfinite(float(dist_scaled))
                    else math.nan
                )
                lpips_status = "n/a"
                if lpips_apply_after_step is not None:
                    lpips_status = "on" if effective_step >= int(lpips_apply_after_step) else "off"
                dist_table = _kv_table(
                    [
                        ("MSE", _fmt_float(dist_mse, "{:.4f}")),
                        ("Other", _fmt_float(dist_other, "{:.4f}")),
                        ("s_mse·MSE", _fmt_float(dist_mse_scaled, "{:.4f}")),
                        ("D_scaled", _fmt_float(dist_scaled, "{:.4f}")),
                        ("(sum−D*)", _fmt_float(dist_check, "{:.2e}")),
                        ("LPIPS", lpips_status),
                    ]
                )
                layout["dist"].update(_panel("Distortion", dist_table, style="magenta"))

                # System / I/O tile.
                gpu_name = "cpu"
                mem_alloc = mem_reserved = mem_peak = "--"
                if args.device == "cuda":
                    try:
                        gpu_name = torch.cuda.get_device_name(0)
                        mem_alloc = _fmt_float(torch.cuda.memory_allocated() / (1024.0**3), "{:.2f} GiB")
                        mem_reserved = _fmt_float(torch.cuda.memory_reserved() / (1024.0**3), "{:.2f} GiB")
                        mem_peak = _fmt_float(torch.cuda.max_memory_allocated() / (1024.0**3), "{:.2f} GiB")
                    except Exception:
                        gpu_name = "cuda"
                system_table = _kv_table(
                    [
                        ("gpu", str(gpu_name)),
                        ("mem", f"{mem_alloc} alloc | {mem_reserved} rsvd | {mem_peak} peak"),
                        ("log", f"{str(last_log_step_1b) if last_log_step_1b is not None else '--'} @ {log_path.name}"),
                        (
                            "ckpt",
                            f"{str(last_snapshot_step_1b) if last_snapshot_step_1b is not None else '--'} "
                            f"@ {('checkpoints/' if save_every_steps else 'disabled')}",
                        ),
                    ]
                )
                sys_body = Group(system_table, Text(last_log_msg, style="bright_black") if last_log_msg else Text(""))
                layout["system"].update(_panel("System", sys_body, style="yellow"))
    elif use_tqdm and tqdm is not None:
        desc = f"train {args.tag} (λ_elic={lmbda_str}, λ_rd={rd_lmbda_str}, s_mse={args.nvs_mse_scale:g})"
        with tqdm(total=max_steps, desc=desc, dynamic_ncols=True) as pbar:
            for step in range(max_steps):
                row = train_one_step(step)
                if (step % int(args.log_every)) == 0 or step == max_steps - 1:
                    write_row(row)
                    pbar.set_postfix_str(fmt(row))
                global_step_1b = int(step) + 1
                epoch_accumulate(row, global_step_1b)
                epoch_maybe_flush(row, global_step_1b=global_step_1b, is_last_step=step == max_steps - 1)
                if save_every_steps is not None and (global_step_1b % int(save_every_steps)) == 0:
                    save_snapshot(global_step_1b)
                pbar.update(1)
    else:
        for step in range(max_steps):
            row = train_one_step(step)
            if (step % int(args.log_every)) == 0 or step == max_steps - 1:
                write_row(row)
                print(f"[{step:>6}] {fmt(row)}")
            global_step_1b = int(step) + 1
            epoch_accumulate(row, global_step_1b)
            epoch_maybe_flush(row, global_step_1b=global_step_1b, is_last_step=step == max_steps - 1)
            if save_every_steps is not None and (global_step_1b % int(save_every_steps)) == 0:
                save_snapshot(global_step_1b)

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
            "global_step": int(max_steps),
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
