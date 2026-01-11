# Reproducing results

This repo currently contains the **V1 baseline** (ELIC-compressed context frames → MVSplat).
V2/V3 (end-to-end and feature-level compression) are planned but not implemented in this clean restart.

- **V1**: ELIC bitstream compression of context frames → decode → MVSplat (`experiments/v1_baseline/`)

## Fair R–D plot (V1 baseline)

Prereqs:
- Dataset at `dataset/re10k/...`
- Checkpoints under `checkpoints/` (see `docs/INSTALL.md`)

### 1) Compress only the required V1 context frames

```bash
python experiments/v1_baseline/compress.py \
  --split test \
  --index_path assets/indices/re10k/evaluation_index_re10k.json \
  --lambdas 0.004 0.008 0.016 0.032 0.15 0.45 \
  --skip_existing
```

If you're on a CPU-only machine, add `--device cpu` (expect it to be slow).

If you already have precomputed ELIC outputs in the legacy layout under `outputs/baseline_ELIC/compressed/`,
skip this step and point `--compressed-root` at `outputs/baseline_ELIC/compressed/lambda_<λ>` during eval.

### 2) Run fair evaluation (vanilla + V1)

Vanilla MVSplat (no compression):

```bash
python experiments/v1_baseline/eval_fair_mvsplat.py \
  --tag vanilla \
  --output experiments/v1_baseline/results/vanilla_fair_val_metrics.csv
```

If you're on a CPU-only machine, add `--device cpu` (expect it to be slow).

V1 for multiple lambdas (append rows to one CSV):

```bash
out=experiments/v1_baseline/results/fair_val_metrics.csv
rm -f "$out"
for l in 0.004 0.008 0.016 0.032 0.15 0.45; do
  python experiments/v1_baseline/eval_fair_mvsplat.py \
    --compressed-root "experiments/v1_baseline/compressed/lambda_${l}" \
    --tag "v1_lambda_${l}" \
    --output "$out" --append
done
```

### 3) Plot

```bash
bash scripts/plot_fair_rd.sh \
  --note "V1 BPP: bitstream (avg over 2 context views)"
```

Outputs:
- `experiments/v1_baseline/results/plots/fair_rd_psnr.pdf` (+ `.png`)
- one plot per metric when using `--all-metrics`
