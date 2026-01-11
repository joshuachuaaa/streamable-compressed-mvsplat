#!/usr/bin/env python3
"""Export real ELIC bitstreams + run fair MVSplat eval for a V1-E2E run.

This is a thin orchestration wrapper around:
  1) `experiments/v1_baseline/compress.py`   (real entropy-coded bytes + recon PNGs)
  2) `experiments/v1_baseline/eval_fair_mvsplat.py` (fair fixed-index evaluation)

The goal is to make the end-to-end pipeline reproducible and "one-command" per
RD point, without duplicating the baseline implementations.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_lambda(value: float) -> str:
    return ("%.3f" % value).rstrip("0").rstrip(".")


def main() -> int:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser(description="V1-E2E: export bitstreams + evaluate fairly (RE10K)")
    parser.add_argument("--tag", required=True, help="Row label to write into the results CSV.")
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        required=True,
        help="Lambda value for this RD point (e.g., 0.032).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output directory containing mvsplat_finetuned.ckpt and ELIC_*.pth.tar.",
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
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for compression and evaluation.",
    )
    parser.add_argument(
        "--compressed-output-root",
        type=Path,
        default=repo_root / "outputs" / "v1_e2e" / "compressed",
        help="Where to write exported recon PNGs + manifests (gitignored).",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=repo_root / "outputs" / "v1_e2e" / "results" / "fair_rd.csv",
        help="Results CSV to write/append.",
    )
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a row to --results-csv (disable to overwrite).",
    )
    parser.add_argument(
        "--save-bitstreams",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save .bin bitstreams for archival/debug (not required for bpp).",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)

    lmbda_str = _format_lambda(args.lmbda)
    mvsplat_ckpt = args.run_dir / "mvsplat_finetuned.ckpt"
    if not mvsplat_ckpt.exists():
        raise FileNotFoundError(mvsplat_ckpt)

    # 1) Export compressed reconstructions for the fixed evaluation index (true entropy-coded bytes).
    export_root = args.compressed_output_root / args.tag
    compress_cmd = [
        sys.executable,
        str(repo_root / "experiments" / "v1_baseline" / "compress.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--index-path",
        str(args.index_path),
        "--lambdas",
        str(args.lmbda),
        "--elic-checkpoints",
        str(args.run_dir),
        "--output-root",
        str(export_root),
        "--device",
        str(args.device),
        "--skip-existing",
    ]
    if args.save_bitstreams:
        compress_cmd.append("--save-bitstreams")

    print("\n[1/2] Export ELIC bitstreams + recon PNGs")
    print(" ".join(compress_cmd))
    subprocess.run(compress_cmd, check=True)

    compressed_root = export_root / f"lambda_{lmbda_str}"
    if not compressed_root.exists():
        raise FileNotFoundError(compressed_root)

    # 2) Run fair evaluation against MVSplat using decoded contexts (and manifest bpp).
    eval_cmd = [
        sys.executable,
        str(repo_root / "experiments" / "v1_baseline" / "eval_fair_mvsplat.py"),
        "--tag",
        args.tag,
        "--dataset-root",
        str(args.dataset_root),
        "--index-path",
        str(args.index_path),
        "--mvsplat-ckpt",
        str(mvsplat_ckpt),
        "--compressed-root",
        str(compressed_root),
        "--device",
        str(args.device),
        "--output",
        str(args.results_csv),
    ]
    if args.append:
        eval_cmd.append("--append")

    print("\n[2/2] Fair MVSplat evaluation")
    print(" ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

