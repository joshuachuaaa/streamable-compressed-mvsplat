#!/usr/bin/env python3
"""Evaluate conference baselines on the fixed RE10K evaluation index.

Baselines covered:
  1) Vanilla MVSplat (no compression).
  2) Vanilla ELIC -> MVSplat (precomputed decoded context images + manifest bpp).

This is an orchestration wrapper around `experiments/eval/eval_fair_mvsplat.py`
to standardize output locations and avoid "where did my CSV go?" confusion.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_lambda(value: float) -> str:
    return ("%.3f" % value).rstrip("0").rstrip(".")


def _load_elic_model(*, ckpt_path: Path, device: str, entropy_coder: str) -> object:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "third_party" / "ELiC-ReImplemetation"))

    import compressai
    import torch
    from compressai.zoo import load_state_dict

    from Network import TestModel  # type: ignore

    compressai.set_entropy_coder(entropy_coder)
    state_dict = load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = TestModel().from_state_dict(state_dict).eval().to(device)
    try:
        model.update(force=True)
    except Exception:
        try:
            model.update()
        except Exception:
            pass
    return model


def _materialize_reconstructions(
    *,
    lambda_dir: Path,
    device: str,
    entropy_coder: str,
    limit_images: int | None,
) -> None:
    """Decode `bitstreams/<scene>/<frame>.bin` into `recon/<scene>/<frame>.png` if missing."""
    manifest_path = lambda_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    # Early-out: if any recon PNG exists in the canonical location, assume materialized.
    recon_root = lambda_dir / "recon"
    if recon_root.exists():
        try:
            next(recon_root.rglob("*.png"))
            return
        except StopIteration:
            pass

    # Load checkpoint path from the manifest (written by the exporter/compressor).
    ckpt_path: Path | None = None
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ckpt = row.get("ckpt")
            if ckpt:
                ckpt_path = Path(ckpt)
                break
    if ckpt_path is None:
        raise RuntimeError(f"manifest.csv missing 'ckpt' column/values: {manifest_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ELIC checkpoint not found: {ckpt_path}")

    model = _load_elic_model(ckpt_path=ckpt_path, device=device, entropy_coder=entropy_coder)

    import torch

    try:
        import torchvision
    except Exception as exc:
        raise RuntimeError("torchvision is required to write recon PNGs from bitstreams.") from exc

    decoded = 0
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit_images is not None and decoded >= int(limit_images):
                break

            scene = row.get("scene")
            frame = row.get("frame")
            if scene is None or frame is None:
                continue

            # Do not trust `recon_png` paths inside the manifest: users may rename/move outputs folders.
            # Always decode into the canonical location relative to `lambda_dir`.
            recon_path = recon_root / str(scene) / f"{int(frame):0>6}.png"
            if recon_path.exists():
                continue

            bin_path = lambda_dir / "bitstreams" / str(scene) / f"{int(frame):0>6}.bin"
            if not bin_path.exists():
                raise FileNotFoundError(bin_path)

            payload = torch.load(bin_path, map_location="cpu", weights_only=False)
            strings = payload.get("strings")
            shape = payload.get("shape")
            if strings is None or shape is None:
                raise RuntimeError(f"Malformed bitstream payload: {bin_path}")

            out_dec = model.decompress(strings, shape)
            x_hat = out_dec["x_hat"].detach().clamp(0, 1).cpu()

            recon_path.parent.mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(x_hat, recon_path, nrow=1)
            decoded += 1


def main() -> int:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser(description="Baselines: vanilla MVSplat + vanilla ELIC->MVSplat (RE10K fixed eval)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
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
        "--mvsplat-ckpt",
        type=Path,
        default=repo_root / "checkpoints" / "vanilla" / "MVSplat" / "re10k.ckpt",
        help="Pretrained MVSplat checkpoint.",
    )
    parser.add_argument(
        "--compressed-base",
        type=Path,
        default=repo_root / "outputs" / "v1_baseline" / "compressed",
        help="Directory containing lambda_<λ>/ subfolders (each with recon/ + manifest.csv).",
    )
    parser.add_argument(
        "--materialize-recon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If recon PNGs are missing, decode them from saved bitstreams before evaluation.",
    )
    parser.add_argument(
        "--entropy-coder",
        type=str,
        default="ans",
        help="CompressAI entropy coder (must match what was used to create the saved bitstreams).",
    )
    parser.add_argument(
        "--materialize-limit-images",
        type=int,
        default=None,
        help="Debug: only decode the first N context images per lambda when materializing recon PNGs.",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.004, 0.008, 0.016, 0.032, 0.15, 0.45],
        help="ELIC lambda points to evaluate.",
    )
    parser.add_argument("--tag-vanilla", type=str, default="vanilla")
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="v1_lambda_",
        help="Prefix for compressed baseline tags (suffix is the lambda value).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=repo_root / "outputs" / "v1_baseline" / "results" / "fair_rd.csv",
        help="Where to write the concatenated metrics CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite --out-csv (if false, only appends).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Debug: stop after N scenes.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader workers for evaluation.",
    )
    parser.add_argument(
        "--persistent-workers",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override dataloader persistence for evaluation.",
    )
    args = parser.parse_args()

    eval_script = repo_root / "experiments" / "eval" / "eval_fair_mvsplat.py"
    if not eval_script.exists():
        raise FileNotFoundError(eval_script)

    # Backward-compatible defaults: older runs used outputs/baseline_ELIC_crop256/.
    if args.compressed_base == (repo_root / "outputs" / "v1_baseline" / "compressed"):
        legacy = repo_root / "outputs" / "baseline_ELIC_crop256" / "compressed"
        if not args.compressed_base.exists() and legacy.exists():
            print(f"[baseline] Using legacy compressed base: {legacy}")
            args.compressed_base = legacy

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.out_csv.exists():
        args.out_csv.unlink()

    base_cmd = [
        sys.executable,
        str(eval_script),
        "--dataset-root",
        str(args.dataset_root),
        "--index-path",
        str(args.index_path),
        "--mvsplat-ckpt",
        str(args.mvsplat_ckpt),
        "--device",
        str(args.device),
        "--output",
        str(args.out_csv),
    ]
    if args.max_scenes is not None:
        base_cmd.extend(["--max-scenes", str(args.max_scenes)])
    if args.num_workers is not None:
        base_cmd.extend(["--num-workers", str(args.num_workers)])
    if args.persistent_workers != "auto":
        base_cmd.extend(["--persistent-workers", args.persistent_workers])

    # 1) Vanilla point (writes header).
    cmd = base_cmd + ["--tag", args.tag_vanilla]
    print("\n[1/2] vanilla MVSplat")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # 2) Compressed baselines (append).
    print("\n[2/2] vanilla ELIC -> MVSplat")
    for lmbda in args.lambdas:
        l_str = _format_lambda(float(lmbda))
        compressed_root = args.compressed_base / f"lambda_{l_str}"
        if not compressed_root.exists():
            raise FileNotFoundError(f"Missing compressed root: {compressed_root}")

        if args.materialize_recon:
            _materialize_reconstructions(
                lambda_dir=compressed_root,
                device=str(args.device),
                entropy_coder=str(args.entropy_coder),
                limit_images=args.materialize_limit_images,
            )

        tag = f"{args.tag_prefix}{l_str}"
        cmd = base_cmd + [
            "--tag",
            tag,
            "--compressed-root",
            str(compressed_root),
            "--append",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\nWrote:", args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
