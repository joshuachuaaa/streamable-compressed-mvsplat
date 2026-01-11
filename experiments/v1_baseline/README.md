# V1 baseline (ELIC → RGB → MVSplat)

This experiment implements the **bitstream baseline**:

1. Select 2 context frames per scene from a **fixed evaluation index** (`assets/indices/re10k/evaluation_index_re10k.json`).
2. Compress each context frame independently with **ELIC** at a given λ (true bitstream size).
3. Decode back to RGB and run **vanilla MVSplat** to render the target views.

The intended use is to produce a **rate–distortion curve** (bpp vs PSNR/SSIM/LPIPS).

## 1) Compress required context frames

```bash
python experiments/v1_baseline/compress.py \
  --index-path assets/indices/re10k/evaluation_index_re10k.json \
  --dataset-root dataset/re10k \
  --lambdas 0.004 0.008 0.016 0.032 0.15 0.45 \
  --skip-existing
```

If you already have precomputed ELIC outputs under `outputs/baseline_ELIC/compressed/` (legacy layout),
you can skip this step and use `--compressed-root outputs/baseline_ELIC/compressed/lambda_<λ>` during eval.

Outputs (per λ):

```text
experiments/v1_baseline/compressed/lambda_<λ>/
  manifest.csv              # per-image bpp + bookkeeping
  recon/<scene>/<frame>.png # decoded RGB used as MVSplat context input
```

## 2) Evaluate MVSplat fairly

Vanilla MVSplat (no compression):

```bash
python experiments/v1_baseline/eval_fair_mvsplat.py \
  --tag vanilla \
  --mvsplat-ckpt checkpoints/vanilla/MVSplat/re10k.ckpt \
  --output experiments/v1_baseline/results/fair_val_metrics.csv
```

Compressed V1 (one row per λ; append to same CSV):

```bash
out=experiments/v1_baseline/results/fair_val_metrics.csv
rm -f "$out"
for l in 0.004 0.008 0.016 0.032 0.15 0.45; do
  python experiments/v1_baseline/eval_fair_mvsplat.py \
    --tag "v1_lambda_${l}" \
    --mvsplat-ckpt checkpoints/vanilla/MVSplat/re10k.ckpt \
    --compressed-root "experiments/v1_baseline/compressed/lambda_${l}" \
    --output "$out" --append
done
```

## 3) Plot

```bash
bash scripts/plot_fair_rd.sh --input experiments/v1_baseline/results/fair_val_metrics.csv
```
