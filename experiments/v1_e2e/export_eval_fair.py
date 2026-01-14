#!/usr/bin/env python3
"""Export real ELIC bitstreams + run fair evaluation for a V1-E2E run.

Conference-grade requirements addressed:
  - Uses the fixed RE10K evaluation index (2 context → 3 target).
  - Bitrate is computed from *real entropy-coded bytes* from `ELIC.compress`.
  - Exported context reconstructions match MVSplat's actual inputs (RE10K pipeline crops to 256×256).
  - Evaluation uses the same fixed-protocol evaluator as the baselines:
      `experiments/eval/eval_fair_mvsplat.py`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_lambda(value: float) -> str:
    return ("%.3f" % value).rstrip("0").rstrip(".")


def _add_third_party_to_path(repo_root: Path) -> None:
    # MVSplat is an implicit namespace package called `src`.
    sys.path.insert(0, str(repo_root / "third_party" / "mvsplat"))
    # ELIC reimplementation uses top-level modules (e.g., `Network.py`).
    sys.path.insert(0, str(repo_root / "third_party" / "ELiC-ReImplemetation"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lambda_to_checkpoint_name(lmbda: float) -> str:
    mapping = {
        0.004: "0004",
        0.008: "0008",
        0.016: "0016",
        0.032: "0032",
        0.15: "0150",
        0.45: "0450",
    }
    for k, v in mapping.items():
        if math.isclose(float(lmbda), float(k), rel_tol=0.0, abs_tol=1e-12):
            return f"ELIC_{v}_ft_3980_Plateau.pth.tar"
    raise ValueError(f"Unsupported lambda={lmbda}; expected one of {sorted(mapping.keys())}")


def _sum_bytes(obj: Any) -> int:
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_sum_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_sum_bytes(v) for v in obj.values())
    return 0


def _load_mvsplat_cfg(
    *,
    dataset_root: Path,
    index_path: Path,
    data_loader_num_workers: int,
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
        "data_loader.test.batch_size=1",
        f"data_loader.test.num_workers={int(data_loader_num_workers)}",
        # Avoid worker persistence edge cases for IterableDataset.
        f"data_loader.test.persistent_workers={str(bool(data_loader_num_workers > 0)).lower()}",
        "test.save_image=false",
        "test.save_video=false",
        "test.compute_scores=false",
    ]

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    return cfg_dict, cfg


def _iter_eval_context_batches(cfg: Any) -> Iterable[dict[str, Any]]:
    from src.dataset.data_module import DataModule

    dm = DataModule(cfg.dataset, cfg.data_loader, step_tracker=None, global_rank=0)
    yield from dm.test_dataloader()


def _load_elic_model(checkpoint_path: Path, device: str, entropy_coder: str) -> Any:
    import torch
    import compressai
    from compressai.zoo import load_state_dict

    from Network import TestModel

    compressai.set_entropy_coder(entropy_coder)
    state_dict = load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = TestModel().from_state_dict(state_dict).eval().to(device)
    try:
        model.update(force=True)
    except Exception:
        try:
            model.update()
        except Exception:
            pass
    return model


def _pad_to_multiple(x_bchw: Any, multiple: int) -> tuple[Any, tuple[int, int, int, int]]:
    import torch.nn.functional as F

    _, _, h, w = x_bchw.shape
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
    return F.pad(x_bchw, padding, mode="constant", value=0.0), padding


def _unpad(x_bchw: Any, padding: tuple[int, int, int, int]) -> Any:
    left, right, top, bottom = padding
    if any(v != 0 for v in padding):
        return x_bchw[..., top : x_bchw.shape[-2] - bottom, left : x_bchw.shape[-1] - right]
    return x_bchw


def _read_existing_manifest(manifest_path: Path) -> set[tuple[str, int]]:
    if not manifest_path.exists():
        return set()
    processed: set[tuple[str, int]] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                processed.add((row["scene"], int(row["frame"])))
            except Exception:
                continue
    return processed


def _write_manifest_header_if_needed(manifest_path: Path) -> None:
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["scene", "frame", "lambda", "num_pixels", "num_bytes", "num_bits", "bpp", "recon_png", "ckpt"]
        )


def main() -> int:
    repo_root = _repo_root()
    _add_third_party_to_path(repo_root)

    parser = argparse.ArgumentParser(description="V1-E2E: export bitstreams + evaluate fairly (RE10K)")
    parser.add_argument("--tag", required=True, help="Row label to write into the results CSV.")
    parser.add_argument("--lambda", dest="lmbda", type=float, required=True, help="Lambda value (e.g., 0.032).")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output directory containing mvsplat_finetuned.ckpt and ELIC_*.pth.tar.",
    )
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "dataset" / "re10k")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=repo_root / "assets" / "indices" / "re10k" / "evaluation_index_re10k.json",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
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
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-bitstreams", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--entropy-coder", type=str, default="ans")
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for export (default: 0 for determinism).",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)

    lmbda_str = _format_lambda(args.lmbda)
    mvsplat_ckpt = args.run_dir / "mvsplat_finetuned.ckpt"
    if not mvsplat_ckpt.exists():
        raise FileNotFoundError(mvsplat_ckpt)

    # Export root layout matches baseline layout: lambda_<λ>/{manifest.csv,recon/,bitstreams/}.
    tag_root = args.compressed_output_root / args.tag
    export_root = tag_root / f"lambda_{lmbda_str}"
    recon_root = export_root / "recon"
    bitstream_root = export_root / "bitstreams"
    manifest_path = export_root / "manifest.csv"
    export_root.mkdir(parents=True, exist_ok=True)

    (tag_root / "_meta").mkdir(parents=True, exist_ok=True)
    (tag_root / "_meta" / "run_args.json").write_text(
        json.dumps({**vars(args), "dataset_root": str(args.dataset_root)}, indent=2, default=str),
        encoding="utf-8",
    )

    elic_ckpt_name = _lambda_to_checkpoint_name(args.lmbda)
    elic_ckpt_path = args.run_dir / elic_ckpt_name
    if not elic_ckpt_path.exists():
        matches = sorted(args.run_dir.glob("ELIC_*.pth.tar"))
        if len(matches) == 1:
            elic_ckpt_path = matches[0]
        else:
            raise FileNotFoundError(f"Missing ELIC checkpoint in {args.run_dir}: expected {elic_ckpt_name}")

    print("\n[1/2] Export ELIC bitstreams + recon PNGs (true bytes)")
    print("export_root:", export_root)
    print("elic_ckpt  :", elic_ckpt_path)

    eval_index = _load_json(args.index_path)
    expected_images = sum(len(v.get("context", [])) for v in eval_index.values() if v is not None)

    _, cfg = _load_mvsplat_cfg(
        dataset_root=args.dataset_root,
        index_path=args.index_path,
        data_loader_num_workers=int(args.num_workers),
    )

    model = _load_elic_model(elic_ckpt_path, device=str(args.device), entropy_coder=str(args.entropy_coder))
    processed = _read_existing_manifest(manifest_path)
    _write_manifest_header_if_needed(manifest_path)

    try:
        import torchvision
    except Exception as exc:
        raise RuntimeError("torchvision is required to write recon PNGs during export_eval_fair.") from exc

    images_done = 0
    with manifest_path.open("a", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=["scene", "frame", "lambda", "num_pixels", "num_bytes", "num_bits", "bpp", "recon_png", "ckpt"],
        )
        for batch in _iter_eval_context_batches(cfg):
            scene = batch["scene"][0]
            context_indices = [int(x) for x in batch["context"]["index"][0].tolist()]
            context_images = batch["context"]["image"][0]  # [2,3,H,W] already cropped to 256×256
            for view_slot, frame in enumerate(context_indices):
                key = (scene, int(frame))
                if key in processed:
                    continue

                x = context_images[view_slot].to(args.device).unsqueeze(0)  # [1,3,H,W]
                x_padded, padding = _pad_to_multiple(x, int(args.pad_multiple))
                out_enc = model.compress(x_padded)
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                x_hat = _unpad(out_dec["x_hat"], padding).detach().clamp(0, 1).cpu()

                num_pixels = int(x.shape[0] * x.shape[2] * x.shape[3])  # B*H*W
                num_bytes = _sum_bytes(out_enc.get("strings"))
                num_bits = 8 * num_bytes
                bpp = num_bits / float(num_pixels)

                recon_path = recon_root / scene / f"{int(frame):0>6}.png"
                recon_path.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(x_hat, recon_path, nrow=1)

                if args.save_bitstreams:
                    bit_path = bitstream_root / scene / f"{int(frame):0>6}.bin"
                    bit_path.parent.mkdir(parents=True, exist_ok=True)
                    import torch

                    torch.save({"strings": out_enc["strings"], "shape": out_enc["shape"]}, bit_path)

                writer.writerow(
                    {
                        "scene": scene,
                        "frame": int(frame),
                        "lambda": lmbda_str,
                        "num_pixels": num_pixels,
                        "num_bytes": num_bytes,
                        "num_bits": num_bits,
                        "bpp": bpp,
                        "recon_png": str(recon_path),
                        "ckpt": str(elic_ckpt_path),
                    }
                )
                images_done += 1
                if images_done % 200 == 0:
                    print(f"exported context images: {images_done:,}/{expected_images:,}")

    # 2) Run fair evaluation against MVSplat using decoded contexts (and manifest bpp).
    eval_cmd = [
        sys.executable,
        str(repo_root / "experiments" / "eval" / "eval_fair_mvsplat.py"),
        "--tag",
        args.tag,
        "--dataset-root",
        str(args.dataset_root),
        "--index-path",
        str(args.index_path),
        "--mvsplat-ckpt",
        str(mvsplat_ckpt),
        "--compressed-root",
        str(export_root),
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
