# Reproducing results

This repo contains:
- **V1 baseline** (ELIC-compressed context frames → MVSplat).
- **V1 E2E** (end-to-end fine-tuning of ELIC + MVSplat with an RD objective).

- Evaluation + baselines: `experiments/eval/`
- V1 E2E: `experiments/v1_e2e/`

## Fair R–D plot (baselines)

Prereqs:
- Dataset at `dataset/re10k/...`
- Checkpoints under `checkpoints/` (see `docs/INSTALL.md`)

### 1) Ensure ELIC baseline artifacts exist

Baseline inputs are assumed to already exist under `outputs/v1_baseline/compressed/`.
This repo focuses on evaluation, not (re)generating those artifacts.

If you're on a CPU-only machine, add `--device cpu` (expect it to be slow).

If you already have precomputed ELIC outputs in the legacy layout under `outputs/baseline_ELIC/compressed/`,
you *can* point `--compressed-root` at `outputs/baseline_ELIC/compressed/lambda_<λ>` during eval, but note that
those bitrates are typically computed on the **full-resolution** inputs and are not comparable to E2E runs that
compress the **cropped** 256×256 inputs. For conference-grade RD curves, regenerate bitstreams via `compress.py`.

### 2) Run baseline evaluation (vanilla + ELIC→MVSplat)

This writes one canonical CSV:

```bash
python experiments/eval/eval_baselines.py \
  --compressed-base outputs/v1_baseline/compressed \
  --out-csv outputs/v1_baseline/results/fair_rd.csv \
  --device cuda
```

### 3) Plot

```bash
bash scripts/plot_fair_rd.sh \
  --input outputs/v1_baseline/results/fair_rd.csv \
  --note "V1 BPP: bitstream (avg over 2 context views)"
```

Outputs:
- `outputs/v1_baseline/results/plots/fair_rd_psnr.pdf` (+ `.png`)
- one plot per metric when using `--all-metrics`

---

## V1 E2E (one RD point)
Train:
```bash
python experiments/v1_e2e/train_e2e.py \
  --tag e2e \
  --lambda 0.032 \
  --rd-lambda 0.032 \
  --nvs-mse-scale 65025 \
  --device cuda \
  --max-steps 18000 \
  --batch-size 15 \
  --num-workers 8 \
  --progress rich
```

Plot training curves:
```bash
python experiments/v1_e2e/plot_curves.py \
  --run-dir checkpoints/v1_e2e/e2e_lambda_0.032_rd_1 \
  --train-mode step --smooth-window 200 \
  --out outputs/v1_e2e/results/plots/train_curves.png
```

## Plot baselines vs E2E

```bash
bash scripts/plot_baseline_vs_e2e.sh \
  --baseline outputs/v1_baseline/results/fair_rd.csv \
  --e2e outputs/v1_e2e/results/fair_rd.csv \
  --outdir outputs/plots/baseline_vs_e2e
```
