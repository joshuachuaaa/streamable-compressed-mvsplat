# thesis-clean

Conference-ready repository for thesis experiments on learned compression for novel view synthesis.

This codebase studies three variants built around **MVSplat** (Gaussian splatting NVS) and an **ELIC** reimplementation (learned compression):

- **V1 (baseline)**: compress context RGB frames with ELIC bitstreams → decode → run MVSplat (true bitstream bpp)
- **V2 (E2E)**: end-to-end train ELIC (image codec) + MVSplat encoder with a differentiable rate term (planned)
- **V3 (feature compression)**: compress intermediate MVSplat feature maps with ELIC (split inference) (planned)

## Getting started

- Install: `docs/INSTALL.md`
- Data layout: `docs/DATA.md`
- Reproducing plots: `docs/REPRODUCING.md`

## Repo layout

- `experiments/`: experiment entry points (V1/V2/V3) + plotting
- `third_party/`: vendored dependencies (MVSplat, ELIC, rasterizer)
- `checkpoints/`, `dataset/`, `outputs/`: local artifacts (ignored by git)

## Third-party provenance

See `THIRD_PARTY_VERSIONS.txt` and `docs/THIRD_PARTY.md`.

## License

This repo currently does **not** declare a license for the new/modified thesis glue code.
Third-party components under `third_party/` have their own licenses.
