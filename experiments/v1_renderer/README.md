# V1 (Renderer Adaptation): ELIC → MVSplat (Fine-tune MVSplat Only)

## What this variant is
`v1_renderer` studies **receiver-side adaptation** under bitrate constraints:

- **Sender side (fixed):** ELIC compresses each context image independently.
- **Receiver side (trainable):** MVSplat is fine-tuned to better handle decoded ELIC artifacts.

This isolates the question: *How much of the gain comes from making the renderer robust to compression artifacts, without changing the codec?*

## Trainable / Frozen
- **Frozen:** ELIC (encoder + entropy model + decoder)
- **Trainable:** MVSplat (typically encoder + decoder; optionally only early layers)

## Recommended objective (conference-ready baseline)
Train with the standard novel-view synthesis loss used in MVSplat training:

- **Novel-view distortion**
  - `D_nvs = MSE(Î_tgt, I_tgt) + 0.05 · LPIPS(Î_tgt, I_tgt)`

Since the codec is frozen, there is **no rate term** in the training loss for this variant; bitrate is controlled by which ELIC checkpoint (λ) you choose.

## Protocol expectations
- **Training split:** `dataset/re10k/train`
- **View sampling:** use the standard MVSplat training sampler (bounded). Do *not* train on the fixed evaluation index.
- **Evaluation split:** `dataset/re10k/test`
- **Evaluation protocol:** fixed 2-context → 3-target index at `assets/indices/re10k/evaluation_index_re10k.json`
- **Evaluation script:** `experiments/v1_baseline/eval_fair_mvsplat.py` (uses the same protocol for all variants)

## Notes on experimental design
- Run **per-λ fine-tuning** first (one MVSplat checkpoint per bitrate point). This gives the cleanest RD curve and avoids confounding “variable-rate robustness” effects.
- Report:
  - RD curves (bpp vs PSNR/SSIM/LPIPS on targets)
  - Optionally, robustness across unseen scenes (standard RE10K test split)

## File/Artifact conventions (recommended)
- Checkpoints: `checkpoints/v1_renderer/<tag>/...` (gitignored)
- Results CSV: `outputs/v1_renderer/results/fair_rd.csv` (gitignored)

