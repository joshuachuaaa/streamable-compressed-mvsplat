# V1 (Compressor Adaptation): ELIC ↔ MVSplat (Fine-tune ELIC Only)

## What this variant is
`v1_compressor` studies **sender-side adaptation** for downstream rendering:

- **Sender side (trainable):** ELIC is fine-tuned so that decoded context images are more useful for novel view synthesis.
- **Receiver side (frozen):** MVSplat remains the pretrained renderer.

This isolates the question: *Can a task-driven codec improve NVS quality at the same bitrate without changing the renderer?*

## Trainable / Frozen
- **Frozen:** MVSplat (encoder + decoder)
- **Trainable:** ELIC (encoder + entropy model + decoder)

## Recommended objective (rate–distortion for NVS)
Use a Lagrangian objective that optimizes **task distortion** under bitrate pressure:

- **Rate term:** `R` from ELIC’s entropy model (expected bits; bpp at eval uses real bitstreams)
- **Novel-view distortion:**
  - `D_nvs = MSE(Î_tgt, I_tgt) + 0.05 · LPIPS(Î_tgt, I_tgt)`
- **Total:**
  - `L = D_nvs + λ · R`

### Conference-grade guardrail (strongly recommended)
Add a *small* context-reconstruction regularizer to prevent degenerate “cheating” solutions:

- `L = D_nvs + λ · R + β · MSE(Î_ctx, I_ctx)` (β small; ablate β)

## Protocol expectations
- **Training split:** `dataset/re10k/train`
- **View sampling:** bounded sampler (standard MVSplat training sampler).
- **Evaluation split/protocol:** same fixed evaluation index as the baseline:
  - `assets/indices/re10k/evaluation_index_re10k.json` (2 context → 3 target)
  - `experiments/v1_baseline/eval_fair_mvsplat.py`

## Notes on experimental design
- Start with **per-λ fine-tuning** (one ELIC checkpoint per RD point), initialized from the matching vanilla ELIC λ-checkpoint.
- Report both:
  - Target-view metrics (PSNR/SSIM/LPIPS)
  - Context-view metrics (PSNR/LPIPS of decoded contexts), to demonstrate non-degenerate reconstructions

