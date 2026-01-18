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
import hashlib
import re
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


def _format_lambda(value: float) -> str:
    return ("%.3f" % float(value)).rstrip("0").rstrip(".")


def _extract_lambda(tag: str) -> float | None:
    m = re.search(r"(?:^|[/_])lambda_([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)", tag)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_lambda_from_row(row: dict[str, str]) -> float | None:
    # Prefer parsing from the tag, but fall back to paths if tags are not descriptive.
    for key in ("tag", "compressed_root", "run_dir", "mvsplat_ckpt", "elic_ckpt", "ckpt"):
        text = str(row.get(key, "")).strip()
        if not text:
            continue
        lmbda = _extract_lambda(text)
        if lmbda is not None:
            return lmbda
    return None


def _extract_rd_lambda(text: str) -> float | None:
    m = re.search(r"(?:^|[/_])rd_([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_rd_lambda_from_row(row: dict[str, str]) -> float | None:
    for key in ("tag", "compressed_root", "run_dir", "mvsplat_ckpt", "elic_ckpt", "ckpt"):
        text = str(row.get(key, "")).strip()
        if not text:
            continue
        rd = _extract_rd_lambda(text)
        if rd is not None:
            return rd
    return None


def _pretty_group_label(group: str) -> str:
    if group == "v1":
        return "Baseline (ELIC → MVSplat)"
    if "e2e" in group:
        # e.g., v1_e2e_noscale -> "E2E fine-tuned (noscale)"
        suffix = group
        if suffix.startswith("v1_e2e"):
            suffix = suffix.removeprefix("v1_e2e").lstrip("_")
        if suffix in ("", "v1_e2e"):
            return "E2E fine-tuned"
        return f"E2E fine-tuned ({suffix.replace('_', ' ')})"
    return group.replace("_", " ")


def _stable_index(key: str, n: int) -> int:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(1, int(n))


def _plot_metric(
    *,
    rows: list[dict[str, str]],
    metric: str,
    out_path: Path,
    note: str | None,
    title: str | None,
    label_points: bool,
    figsize: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tag = r.get("tag", "")
        bpp = _as_float(r.get("bpp", "nan"))
        y = _as_float(r.get(metric, "nan"))
        if not tag or bpp != bpp or y != y:  # NaN checks
            continue
        group = _group_key(tag)
        lmbda = _extract_lambda_from_row(r)
        rd = _extract_rd_lambda_from_row(r)
        if rd is None:
            rd = lmbda
        groups[group].append(
            {
                "bpp": bpp,
                "y": y,
                "tag": tag,
                "row": r,
                "lmbda": lmbda,
                "rd": rd,
            }
        )

    if not groups:
        raise SystemExit(f"No valid rows to plot for metric={metric}.")

    fig, ax = plt.subplots(figsize=figsize)
    baseline_key = "vanilla"
    baseline_pts = groups.get(baseline_key, [])
    non_baseline_pts = [pt for group, pts in groups.items() if group != baseline_key for pt in pts]
    zoom_xs = [float(pt["bpp"]) for pt in non_baseline_pts]
    zoom_xlim: tuple[float, float] | None = None
    if baseline_pts and zoom_xs:
        x_min = min(zoom_xs)
        x_max = max(zoom_xs)
        if x_max > x_min:
            pad = (x_max - x_min) * 0.05
        else:
            pad = x_min * 0.05 if x_min > 0 else 0.05
        zoom_xlim = (x_min - pad, x_max + pad)

    palette = list(plt.get_cmap("tab10").colors)

    # Color encodes the ELIC checkpoint family λ.
    lambda_values: dict[str, float] = {}
    for pt in non_baseline_pts:
        lmbda = pt.get("lmbda")
        if lmbda is None:
            continue
        lambda_values[_format_lambda(float(lmbda))] = float(lmbda)
    lambda_keys_sorted = sorted(lambda_values.keys(), key=lambda s: float(s))
    lambda_color = {k: palette[i % len(palette)] for i, k in enumerate(lambda_keys_sorted)}

    has_e2e = any("e2e" in group for group in groups.keys())
    label_note = None
    if label_points and has_e2e:
        label_note = "Colors: ELIC λ. Point labels: E2E shows λ_rd."
    elif label_points:
        label_note = "Colors: ELIC λ."
    if note and label_note:
        note = f"{note}\n{label_note}"
    elif label_note:
        note = label_note

    def annotate_point(
        *,
        x: float,
        y: float,
        text: str,
        color: Any,
        base_offset: tuple[int, int],
    ) -> None:
        # Collision-avoidance heuristic: keep labels close to the point, but nudge when needed.
        #
        # We avoid new dependencies (e.g., adjustText) but still get good readability.
        radii = [0, 10, 18, 26]
        directions = [
            (1, 1),
            (1, 0),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (0, -1),
            (1, -1),
            (-1, -1),
        ]
        offset_candidates_raw: list[tuple[int, int]] = []
        for r in radii:
            for dx_sign, dy_sign in directions:
                offset_candidates_raw.append((int(base_offset[0] + dx_sign * r), int(base_offset[1] + dy_sign * r)))
        # De-dupe, but keep close-by offsets first.
        offset_candidates = sorted(
            set(offset_candidates_raw),
            key=lambda xy: (abs(int(xy[0])) + abs(int(xy[1])), abs(int(xy[0])), abs(int(xy[1]))),
        )

        renderer = None
        used_bboxes = getattr(annotate_point, "_used_bboxes", None)
        if used_bboxes is None:
            used_bboxes = []
            setattr(annotate_point, "_used_bboxes", used_bboxes)

        for idx, (dx, dy) in enumerate(offset_candidates):
            if dx > 0:
                ha = "left"
            elif dx < 0:
                ha = "right"
            else:
                ha = "center"
            if dy > 0:
                va = "bottom"
            elif dy < 0:
                va = "top"
            else:
                va = "center"

            arrowprops = None
            # When we have to move a label away from its first-choice offset, add a subtle
            # connector line so it's still clear which point it belongs to.
            if idx != 0:
                arrowprops = {
                    "arrowstyle": "-",
                    "lw": 0.8,
                    "color": color,
                    "alpha": 0.65,
                    "shrinkA": 2,
                    "shrinkB": 2,
                }
            txt = ax.annotate(
                text,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                ha=ha,
                va=va,
                color=color,
                arrowprops=arrowprops,
            )
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bbox = txt.get_window_extent(renderer=renderer)
                if any(bbox.overlaps(b) for b in used_bboxes):
                    txt.remove()
                    continue
                used_bboxes.append(bbox)
                return
            except Exception:
                # If we can't compute extents (rare backend issues), keep the first label.
                return

    # Baseline (ELIC → MVSplat): one curve, colored markers per ELIC λ.
    baseline_curve_pts = sorted(groups.get("v1", []), key=lambda d: float(d["bpp"]))
    if baseline_curve_pts:
        xs = [float(p["bpp"]) for p in baseline_curve_pts]
        ys = [float(p["y"]) for p in baseline_curve_pts]
        ax.plot(xs, ys, linestyle="-", color="0.35", zorder=1)
        for p in baseline_curve_pts:
            lmbda = p.get("lmbda")
            lkey = _format_lambda(float(lmbda)) if lmbda is not None else None
            color = lambda_color.get(lkey, "0.35")
            ax.plot(
                [float(p["bpp"])],
                [float(p["y"])],
                marker="o",
                linestyle="None",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.6,
                zorder=3,
            )

    # E2E fine-tuned: triangles, colored by ELIC λ; point labels show only λ_rd.
    e2e_pts: list[dict[str, Any]] = []
    for group, pts in groups.items():
        if group == baseline_key:
            continue
        if "e2e" in group:
            e2e_pts.extend(pts)

    e2e_by_lambda: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in e2e_pts:
        lmbda = p.get("lmbda")
        if lmbda is None:
            continue
        e2e_by_lambda[_format_lambda(float(lmbda))].append(p)

    offset_cycle = [(10, 12), (10, -16), (-10, 12), (-10, -16)]
    for li, lkey in enumerate(lambda_keys_sorted):
        pts = e2e_by_lambda.get(lkey, [])
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda d: float(d["bpp"]))
        xs = [float(p["bpp"]) for p in pts_sorted]
        ys = [float(p["y"]) for p in pts_sorted]
        color = lambda_color.get(lkey, "0.35")
        ax.plot(
            xs,
            ys,
            marker="^",
            linestyle="--",
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            zorder=2,
        )
        if label_points:
            for pi, (x, y, p) in enumerate(zip(xs, ys, pts_sorted)):
                rd = p.get("rd")
                if rd is None:
                    continue
                label = f"λ_rd={_format_lambda(float(rd))}"
                base_offset = offset_cycle[(li + pi) % len(offset_cycle)]
                annotate_point(x=x, y=y, text=label, color=color, base_offset=base_offset)

    if baseline_pts and zoom_xs:
        baseline_y = sum(float(p["y"]) for p in baseline_pts) / float(len(baseline_pts))
        if zoom_xlim is None:
            x_min, x_max = min(zoom_xs), max(zoom_xs)
        else:
            x_min, x_max = zoom_xlim
        ax.plot(
            [x_min, x_max],
            [baseline_y, baseline_y],
            linestyle=":",
            linewidth=2.2,
            color="0.25",
            label="Uncompressed context (24 bpp)",
        )
        txt = ax.annotate(
            "uncompressed (24 bpp)",
            xy=(x_max, baseline_y),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="center",
            color="0.25",
        )
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.set_xlabel("Rate (bpp)")
    ylabel = {
        "psnr": "PSNR (dB) ↑",
        "ssim": "SSIM ↑",
        "lpips": "LPIPS ↓",
    }.get(metric, metric)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(x=0.06, y=0.12)
    if zoom_xlim is not None:
        ax.set_xlim(*zoom_xlim)
    try:
        from matplotlib.lines import Line2D
    except Exception:
        Line2D = None  # type: ignore[assignment]

    if Line2D is not None:
        method_handles: list[Any] = []
        if baseline_curve_pts:
            method_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="-",
                    color="0.35",
                    marker="o",
                    markerfacecolor="0.35",
                    markeredgecolor="0.35",
                    label="Baseline (ELIC → MVSplat)",
                )
            )
        if e2e_pts:
            method_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color="0.35",
                    marker="^",
                    markerfacecolor="white",
                    markeredgecolor="0.35",
                    label="E2E fine-tuned",
                )
            )
        if baseline_pts and zoom_xs:
            method_handles.append(Line2D([0], [0], linestyle=":", color="0.25", label="Uncompressed (24 bpp)"))
        if method_handles:
            method_legend = ax.legend(handles=method_handles, loc="upper left", frameon=False, title="Method")
            ax.add_artist(method_legend)

        lambda_handles: list[Any] = []
        for lkey in lambda_keys_sorted:
            color = lambda_color[lkey]
            lambda_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker="o",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    label=f"λ={lkey}",
                )
            )
        if lambda_handles:
            ax.legend(handles=lambda_handles, loc="lower right", frameon=False, title="ELIC λ")
    if note:
        fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Plot fair RD curves from CSV metrics")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[repo_root / "outputs" / "v1_baseline" / "results" / "fair_rd.csv"],
        help="One or more CSV files to read (rows are concatenated).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "v1_baseline" / "results" / "plots",
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
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(8.6, 5.8),
        metavar=("W", "H"),
        help="Figure size in inches (width height).",
    )
    parser.add_argument(
        "--label-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate E2E points with λ_rd (baseline uses colors for λ).",
    )
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
            label_points=bool(args.label_points),
            figsize=(float(args.figsize[0]), float(args.figsize[1])),
        )
        _plot_metric(
            rows=rows,
            metric=metric,
            out_path=args.outdir / f"{stem}.png",
            note=args.note,
            title=args.title,
            label_points=bool(args.label_points),
            figsize=(float(args.figsize[0]), float(args.figsize[1])),
        )
        print("Wrote:", args.outdir / f"{stem}.pdf")
        print("Wrote:", args.outdir / f"{stem}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
