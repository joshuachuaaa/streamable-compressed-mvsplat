#!/usr/bin/env python3
"""Plot V1-E2E training curves (raw + smoothed).

Inputs:
  - train_log.csv: produced by `experiments/v1_e2e/train_e2e.py`

Notes:
  - Output naming is automatic: even if you pass `--out some/path.png`, the
    script appends the run id to avoid overwriting when sweeping runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import deque
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _try_load_run_args(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_args.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _format_lambda(value: float) -> str:
    return ("%.3f" % float(value)).rstrip("0").rstrip(".")


def _try_get_float(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    v = payload.get(key)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _run_key(run_dir: Path, run_args: dict[str, Any] | None) -> str:
    # Prefer explicit args, but fall back to parsing the folder name.
    lmbda = _try_get_float(run_args, "lmbda")
    rd_lambda = _try_get_float(run_args, "rd_lambda_effective")
    if rd_lambda is None:
        rd_lambda = _try_get_float(run_args, "rd_lambda")

    if lmbda is not None and rd_lambda is not None:
        return f"lambda_{_format_lambda(lmbda)}_rd_{_format_lambda(rd_lambda)}"

    m = re.search(r"(?:^|[/_])rd_([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)", run_dir.name)
    if m:
        return f"rd_{m.group(1)}"
    return run_dir.name


def _try_load_steps_per_epoch(run_dir: Path) -> int | None:
    payload = _try_load_run_args(run_dir)
    try:
        v = payload.get("steps_per_epoch_nominal") if payload else None
        return int(v) if v is not None else None
    except Exception:
        return None


def _parse_step_from_str(text: str) -> int | None:
    m = re.search(r"step_(\d+)", text)
    return int(m.group(1)) if m else None


def _resolve_out_path(out: Path, *, run_dir: Path, run_key: str) -> Path:
    # Treat suffix-less paths as directories (even if they don't exist yet).
    if str(out) in ("", "."):
        out_dir = Path(".")
        return out_dir / f"curves_{run_key}.png"

    if out.suffix == "" or (out.exists() and out.is_dir()):
        out_dir = out
        return out_dir / f"curves_{run_key}.png"

    # Otherwise treat as a file template and append the run id.
    if run_key in out.stem:
        return out
    return out.with_name(f"{out.stem}_{run_key}{out.suffix}")


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    q: deque[float] = deque()
    s = 0.0
    out: list[float] = []
    for v in values:
        q.append(float(v))
        s += float(v)
        if len(q) > window:
            s -= q.popleft()
        out.append(s / float(len(q)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot V1-E2E training curves")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory (contains train_log.csv and run_args.json).",
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=None,
        help="Optional override for train_log.csv (defaults to --run-dir/train_log.csv).",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="(Unused) Kept for backward compatibility.",
    )
    parser.add_argument(
        "--train-mode",
        choices=["step", "epoch"],
        default="step",
        help="Use per-step train_log.csv or epoch-averaged train_epoch_log.csv (if present).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Moving-average window for the smoothed row (<=1 disables smoothing).",
    )
    parser.add_argument(
        "--x-axis",
        choices=["step", "epoch"],
        default="step",
        help="X axis for plots (epoch uses steps_per_epoch_nominal from run_args.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/v1_e2e/results"),
        help="Output PNG path, or a directory to write into.",
    )
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Show the figure.")
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)

    train_log = args.train_log if args.train_log is not None else (args.run_dir / "train_log.csv")
    if not train_log.exists():
        raise FileNotFoundError(train_log)

    if args.smooth_window < 0:
        raise SystemExit("--smooth-window must be >= 0.")

    steps_per_epoch = _try_load_steps_per_epoch(args.run_dir)
    if args.x_axis == "epoch" and (steps_per_epoch is None or steps_per_epoch <= 0):
        raise SystemExit("Cannot use --x-axis=epoch because run_args.json lacks steps_per_epoch_nominal.")

    if args.train_mode == "epoch":
        train_log = args.run_dir / "train_epoch_log.csv"
        if not train_log.exists():
            raise FileNotFoundError(train_log)

    train_rows = _read_csv(train_log)
    if not train_rows:
        raise SystemExit(f"No rows in {train_log}")

    def x_from_step(step: int) -> float:
        if args.x_axis == "step":
            return float(step)
        assert steps_per_epoch is not None
        return float(step) / float(steps_per_epoch)

    train_step: list[int] = []
    train_loss: list[float] = []
    train_bpp: list[float] = []
    train_dist: list[float] = []
    dist_key_step = "dist_nvs_scaled"
    dist_key_epoch = "dist_nvs_scaled_mean"
    for row in train_rows:
        try:
            if args.train_mode == "epoch":
                # epoch log uses step_end and *_mean columns.
                s = int(float(row["step_end"]))
                train_step.append(s)
                train_loss.append(float(row["loss_total_mean"]))
                train_bpp.append(float(row["rate_bpp_mean"]))
                if dist_key_epoch in row:
                    train_dist.append(float(row[dist_key_epoch]))
                else:
                    train_dist.append(float(row.get("dist_nvs_mean", "nan")))
            else:
                s = int(float(row["step"]))
                train_step.append(s)
                train_loss.append(float(row["loss_total"]))
                train_bpp.append(float(row["rate_bpp"]))
                if dist_key_step in row:
                    train_dist.append(float(row[dist_key_step]))
                else:
                    train_dist.append(float(row.get("dist_nvs", "nan")))
        except Exception:
            continue

    smooth_window = int(args.smooth_window)
    train_loss_s = _moving_average(train_loss, smooth_window)
    train_bpp_s = _moving_average(train_bpp, smooth_window)
    train_dist_s = _moving_average(train_dist, smooth_window)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in your env, e.g. `pip install matplotlib`."
        ) from exc

    x_train = [x_from_step(s) for s in train_step]
    x_label = "step" if args.x_axis == "step" else "epoch (nominal)"

    run_args = _try_load_run_args(args.run_dir)
    run_key = _run_key(args.run_dir, run_args)
    rd_lambda = _try_get_float(run_args, "rd_lambda_effective")
    if rd_lambda is None:
        rd_lambda = _try_get_float(run_args, "rd_lambda")
    nvs_mse_scale = _try_get_float(run_args, "nvs_mse_scale_effective")
    if nvs_mse_scale is None:
        nvs_mse_scale = _try_get_float(run_args, "nvs_mse_scale")

    rd_str = _format_lambda(rd_lambda) if rd_lambda is not None else "?"
    s_str = f"{float(nvs_mse_scale):g}" if nvs_mse_scale is not None else "?"
    formula = (
        "Objective:  L = rate_bpp + λ_rd · dist_nvs_scaled\n"
        "dist_nvs_scaled = s_mse · dist_nvs_mse + dist_nvs_other\n"
        f"λ_rd={rd_str},  s_mse={s_str}"
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x_train, train_loss, linewidth=1.2)
    ax.set_title("total loss (raw)")
    ax.set_xlabel(x_label)

    ax = axes[0, 1]
    ax.plot(x_train, train_bpp, linewidth=1.2)
    ax.set_title("rate R (bpp estimate, raw)")
    ax.set_xlabel(x_label)

    ax = axes[0, 2]
    ax.plot(x_train, train_dist, linewidth=1.2)
    ax.set_title("distortion D (raw)")
    ax.set_xlabel(x_label)

    ax = axes[1, 0]
    ax.plot(x_train, train_loss_s, linewidth=1.2)
    ax.set_title(f"total loss (MA{smooth_window})" if smooth_window > 1 else "total loss (smoothed)")
    ax.set_xlabel(x_label)

    ax = axes[1, 1]
    ax.plot(x_train, train_bpp_s, linewidth=1.2)
    ax.set_title(f"rate R (MA{smooth_window})" if smooth_window > 1 else "rate R (smoothed)")
    ax.set_xlabel(x_label)

    ax = axes[1, 2]
    ax.plot(x_train, train_dist_s, linewidth=1.2)
    ax.set_title(f"distortion D (MA{smooth_window})" if smooth_window > 1 else "distortion D (smoothed)")
    ax.set_xlabel(x_label)

    for ax in axes.ravel():
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.25)

    title = args.run_dir.name
    if rd_lambda is not None:
        title = f"{title} | λ_rd={rd_str}"
    fig.suptitle(title)
    fig.text(0.01, 0.01, formula, ha="left", va="bottom", fontsize=9)

    out_path = _resolve_out_path(args.out, run_dir=args.run_dir, run_key=run_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print("Wrote:", out_path)

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
