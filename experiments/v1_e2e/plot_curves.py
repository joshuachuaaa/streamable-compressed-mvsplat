#!/usr/bin/env python3
"""Plot V1-E2E training + (optional) eval curves.

Inputs:
  - train_log.csv: produced by `experiments/v1_e2e/train_e2e.py`
  - eval CSV (optional): produced by `experiments/v1_e2e/eval_fast_e2e.py`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import deque
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _try_load_steps_per_epoch(run_dir: Path) -> int | None:
    args_path = run_dir / "run_args.json"
    if not args_path.exists():
        return None
    try:
        payload = json.loads(args_path.read_text(encoding="utf-8"))
        v = payload.get("steps_per_epoch_nominal")
        return int(v) if v is not None else None
    except Exception:
        return None


def _parse_step_from_str(text: str) -> int | None:
    m = re.search(r"step_(\\d+)", text)
    return int(m.group(1)) if m else None


def _resolve_out_path(out: Path, run_dir: Path) -> Path:
    if out.exists() and out.is_dir():
        return out / f"curves_{run_dir.name}.png"
    if str(out) in ("", "."):
        return Path(".") / f"curves_{run_dir.name}.png"
    return out


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
    parser = argparse.ArgumentParser(description="Plot V1-E2E train/eval curves")
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
        help="Optional eval CSV from eval_fast_e2e.py (plots PSNR/SSIM/LPIPS).",
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
        default=0,
        help="Optional moving-average window for per-step curves (0 disables).",
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
    if args.train_mode == "epoch" and args.smooth_window > 0:
        raise SystemExit("--smooth-window applies only to --train-mode=step (epoch mode is already averaged).")

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
    for row in train_rows:
        try:
            if args.train_mode == "epoch":
                # epoch log uses step_end and *_mean columns.
                s = int(float(row["step_end"]))
                train_step.append(s)
                train_loss.append(float(row["loss_total_mean"]))
                train_bpp.append(float(row["rate_bpp_mean"]))
                train_dist.append(float(row["dist_nvs_mean"]))
            else:
                s = int(float(row["step"]))
                train_step.append(s)
                train_loss.append(float(row["loss_total"]))
                train_bpp.append(float(row["rate_bpp"]))
                train_dist.append(float(row["dist_nvs"]))
        except Exception:
            continue

    if args.train_mode == "step" and args.smooth_window > 1:
        train_loss = _moving_average(train_loss, int(args.smooth_window))
        train_bpp = _moving_average(train_bpp, int(args.smooth_window))
        train_dist = _moving_average(train_dist, int(args.smooth_window))

    eval_points: list[dict[str, float]] = []
    if args.eval_csv is not None:
        if not args.eval_csv.exists():
            raise FileNotFoundError(args.eval_csv)
        eval_rows = _read_csv(args.eval_csv)
        run_dir_str = str(args.run_dir.resolve())
        for row in eval_rows:
            ckpt = row.get("mvsplat_ckpt", "")
            if run_dir_str not in ckpt:
                continue
            step_raw = row.get("global_step", "")
            step = None
            try:
                step = int(float(step_raw))
            except Exception:
                step = _parse_step_from_str(ckpt)
            if step is None:
                continue
            try:
                eval_points.append(
                    {
                        "step": float(step),
                        "psnr": float(row["psnr"]),
                        "ssim": float(row["ssim"]),
                        "lpips": float(row["lpips"]),
                        "bpp_est": float(row.get("bpp_est", math.nan)),
                    }
                )
            except Exception:
                continue
        eval_points.sort(key=lambda d: d["step"])

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in your env, e.g. `pip install matplotlib`."
        ) from exc

    x_train = [x_from_step(s) for s in train_step]
    x_label = "step" if args.x_axis == "step" else "epoch (nominal)"

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x_train, train_loss, linewidth=1.2)
    title = "train: total loss"
    if args.train_mode == "epoch":
        title += " (epoch avg)"
    elif args.smooth_window > 1:
        title += f" (MA{int(args.smooth_window)})"
    ax.set_title(title)
    ax.set_xlabel(x_label)

    ax = axes[0, 1]
    ax.plot(x_train, train_bpp, linewidth=1.2)
    title = "train: bpp (estimate)"
    if args.train_mode == "epoch":
        title += " (epoch avg)"
    elif args.smooth_window > 1:
        title += f" (MA{int(args.smooth_window)})"
    ax.set_title(title)
    ax.set_xlabel(x_label)

    ax = axes[0, 2]
    ax.plot(x_train, train_dist, linewidth=1.2)
    title = "train: distortion (NVS loss)"
    if args.train_mode == "epoch":
        title += " (epoch avg)"
    elif args.smooth_window > 1:
        title += f" (MA{int(args.smooth_window)})"
    ax.set_title(title)
    ax.set_xlabel(x_label)

    ax = axes[1, 0]
    if eval_points:
        x_eval = [x_from_step(int(p["step"])) for p in eval_points]
        ax.plot(x_eval, [p["psnr"] for p in eval_points], marker="o", linewidth=1.2)
    ax.set_title("eval: PSNR")
    ax.set_xlabel(x_label)

    ax = axes[1, 1]
    if eval_points:
        x_eval = [x_from_step(int(p["step"])) for p in eval_points]
        ax.plot(x_eval, [p["ssim"] for p in eval_points], marker="o", linewidth=1.2)
    ax.set_title("eval: SSIM")
    ax.set_xlabel(x_label)

    ax = axes[1, 2]
    if eval_points:
        x_eval = [x_from_step(int(p["step"])) for p in eval_points]
        ax.plot(x_eval, [p["lpips"] for p in eval_points], marker="o", linewidth=1.2)
    ax.set_title("eval: LPIPS")
    ax.set_xlabel(x_label)

    fig.suptitle(args.run_dir.name)

    out_path = _resolve_out_path(args.out, args.run_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print("Wrote:", out_path)

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
