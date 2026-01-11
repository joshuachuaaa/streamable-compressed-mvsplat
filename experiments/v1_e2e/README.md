# V1 (End-to-End): ELIC ↔ MVSplat (Joint Fine-tuning)

## What this variant is
`v1_e2e` performs **joint optimization** of the codec + renderer:

- Context images are compressed and decoded by ELIC.
- Decoded contexts are fed into MVSplat.
- Gradients flow through the full pipeline to optimize task quality under bitrate constraints.

This tests the strongest hypothesis: *Can tight coupling of rate–distortion learning with the downstream renderer shift the RD curve?*

## Trainable / Frozen
Two common regimes:
1) **Fully joint (default):**
   - Trainable: ELIC + MVSplat
2) **Stabilized joint (often better):**
   - Stage A: fine-tune MVSplat only (`v1_renderer`)
   - Stage B: jointly fine-tune ELIC + MVSplat starting from Stage A

## Recommended objective (minimal, comparable, publishable)
- **Rate term:** `R` from ELIC entropy model
- **Novel-view distortion:**
  - `D_nvs = MSE(Î_tgt, I_tgt) + 0.05 · LPIPS(Î_tgt, I_tgt)`
- **Total:**
  - `L = D_nvs + λ · R`

### Optional regularizer (use if you observe degenerate artifacts)
- `+ β · MSE(Î_ctx, I_ctx)` with small β, ablate β.

## Protocol expectations
- **Training split:** `dataset/re10k/train`
- **View sampling:** bounded sampler for training.
- **Evaluation split/protocol:** identical fixed evaluation protocol for fair comparison:
  - `assets/indices/re10k/evaluation_index_re10k.json` (2 context → 3 target)
  - `experiments/v1_baseline/eval_fair_mvsplat.py`

## Notes on experimental design
- Prefer **one model per λ** initially (clean RD curve, minimal confounds).
- Report:
  - RD curves using *actual* bitstreams (bpp) and target-view metrics
  - Ablations: `baseline` vs `v1_renderer` vs `v1_compressor` vs `v1_e2e`

