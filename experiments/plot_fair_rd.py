#!/usr/bin/env python3
"""Plot fair rate–distortion curves from CSV metrics.

Expected CSV columns (minimum):
  - tag
  - bpp
  - psnr
  - ssim
  - lpips

This intentionally stays lightweight (no pandas dependency).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    return rows


def _as_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _group_key(tag: str) -> str:
    # Heuristic grouping: v1_lambda_0.032 -> v1, v2_* -> v2, etc.
    if "_lambda_" in tag:
        return tag.split("_lambda_", 1)[0]
    return tag


def _plot_metric(
    *,
    rows: list[dict[str, str]],
    metric: str,
    out_path: Path,
    note: str | None,
    title: str | None,
) -> None:
    import matplotlib.pyplot as plt

    groups: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for r in rows:
        tag = r.get("tag", "")
        bpp = _as_float(r.get("bpp", "nan"))
        y = _as_float(r.get(metric, "nan"))
        if not tag or bpp != bpp or y != y:  # NaN checks
            continue
        groups[_group_key(tag)].append((bpp, y, tag))

    if not groups:
        raise SystemExit(f"No valid rows to plot for metric={metric}.")

    plt.figure(figsize=(6.0, 4.0))
    for group, pts in sorted(groups.items()):
        pts = sorted(pts, key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=2, label=group)

    plt.xlabel("Bits per pixel (bpp)")
    ylabel = {
        "psnr": "PSNR (dB)",
        "ssim": "SSIM",
        "lpips": "LPIPS",
    }.get(metric, metric)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if note:
        plt.figtext(0.01, 0.01, note, ha="left", va="bottom", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Plot fair RD curves from CSV metrics")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[repo_root / "experiments" / "v1_baseline" / "results" / "fair_val_metrics.csv"],
        help="One or more CSV files to read (rows are concatenated).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "experiments" / "v1_baseline" / "results" / "plots",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--all-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot PSNR + SSIM + LPIPS (otherwise PSNR only).",
    )
    parser.add_argument("--note", type=str, default=None, help="Optional note to place on plots.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    metrics = ["psnr", "ssim", "lpips"] if args.all_metrics else ["psnr"]

    for metric in metrics:
        stem = f"fair_rd_{metric}"
        _plot_metric(
            rows=rows,
            metric=metric,
            out_path=args.outdir / f"{stem}.pdf",
            note=args.note,
            title=args.title,
        )
        _plot_metric(
            rows=rows,
            metric=metric,
            out_path=args.outdir / f"{stem}.png",
            note=args.note,
            title=args.title,
        )
        print("Wrote:", args.outdir / f"{stem}.pdf")
        print("Wrote:", args.outdir / f"{stem}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

