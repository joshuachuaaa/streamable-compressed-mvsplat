# thesis-clean

Conference-ready repository for thesis experiments on learned compression for novel view synthesis.

This codebase studies bitrate-constrained novel view synthesis built around **MVSplat** (Gaussian splatting NVS) and an **ELIC** reimplementation (learned compression):

- **V1 baseline**: ELIC-compress context RGB → decode → run MVSplat (true bitstream bpp)
- **V1 E2E**: end-to-end fine-tune ELIC + MVSplat with a rate–distortion objective (current focus)
- **V2/V3**: reserved for future variants (feature-level integration, split inference)

## Getting started

- Install: `docs/INSTALL.md`
- Data layout: `docs/DATA.md`
- Reproducing plots: `docs/REPRODUCING.md`

## Repo layout

- `experiments/`: entry points
  - evaluation + baselines: `experiments/eval/`
  - E2E: `experiments/v1_e2e/`
  - shared plotting: `experiments/plot_fair_rd.py`
- `third_party/`: vendored dependencies (MVSplat, ELIC, rasterizer)
- `checkpoints/`, `dataset/`, `outputs/`: local artifacts (ignored by git)

## Third-party provenance

See `THIRD_PARTY_VERSIONS.txt` and `docs/THIRD_PARTY.md`.

## License

This repo currently does **not** declare a license for the new/modified thesis glue code.
Third-party components under `third_party/` have their own licenses.
