# V1 (End-to-End): ELIC ↔ MVSplat (Joint Fine-tuning)

This folder contains a **conference-grade** implementation plan for the strongest V1 variant:
**jointly fine-tuning the context-image codec (ELIC) and the NVS renderer (MVSplat)**.

The intent is not “train something that looks good”, but to produce a pipeline that is:
- **fair** (fixed evaluation protocol, no hidden tuning on test),
- **measurable** (bpp is from real bitstreams, not proxy losses),
- **defensible** (explicit design choices and sanity checks),
- **reproducible** (command-line driven, logged configs).

## Directory contents
- `experiments/v1_e2e/train_e2e.py`: joint fine-tuning (ELIC + MVSplat).
- `experiments/v1_e2e/export_eval_fair.py`: one-command export of real bitstreams + fair evaluation.
- Baseline tooling reused:
  - `experiments/v1_baseline/compress.py` (writes `manifest.csv` from true entropy-coded bytes)
  - `experiments/v1_baseline/eval_fair_mvsplat.py` (fixed-index evaluation + PSNR/SSIM/LPIPS)

---

## 1) System definition (what is transmitted, what is optimized)
### 1.1 Transmitted payload
Only **context images** are transmitted (streaming/online setting).

```
sender:   I_ctx  --ELIC.compress-->  bitstreams
channel:  bitstreams
receiver: bitstreams --ELIC.decompress--> Î_ctx  --MVSplat--> Î_tgt
```

Targets are **never** compressed/transmitted. They are used only:
- as supervision during training (train split),
- as ground-truth during evaluation (test split).

Why this is a critical (defensible) choice:
- It directly matches the deployment bottleneck (context bandwidth).
- It avoids the “are you compressing the output?” confusion.

### 1.2 What “end-to-end” means here
Gradients from the **target-view error** backpropagate through:
- MVSplat (renderer),
- the reconstructed context images,
- ELIC (codec),
so the codec is trained to produce reconstructions that are **useful for NVS**, not merely pixel-faithful.

---

## 2) Protocol (generalization + fairness)
### 2.1 Data split
- **Train:** `dataset/re10k/train`
- **Test/Eval:** `dataset/re10k/test`

No test-time tuning, no selection on test.

### 2.2 Training view sampling
We use the upstream MVSplat training sampler:
- **bounded sampler:** 2 context views → 1 target view
- Config: `third_party/mvsplat/config/dataset/view_sampler/bounded.yaml`

Why we do this (and why it’s defensible):
- It matches the pretrained MVSplat distribution (minimizes confounds).
- It avoids accidental protocol changes like “more context views during fine-tuning”.

### 2.3 Evaluation protocol
We use a fixed, canonical evaluation index:
- `assets/indices/re10k/evaluation_index_re10k.json` (2 context → 3 target)

Evaluation is run with:
- `experiments/v1_baseline/eval_fair_mvsplat.py`

Why fixed-index evaluation:
- Eliminates “sampling luck” and makes RD points comparable.
- Enables apples-to-apples comparisons across all variants.

### 2.4 Terminology: test vs evaluation
In this repo, “evaluation” refers to the **fixed-index test protocol** on the test split.
We do not run a separate “validation” protocol unless explicitly introduced later.

---

## 3) Loss design (rate–distortion for novel view synthesis)
### 3.1 Rate term `R` (differentiable proxy)
We use ELIC’s entropy-model likelihoods (CompressAI convention) as a differentiable proxy:

`R = bpp_est = Σ log p(ŷ, ẑ) / (-log(2) · #pixels)`

Key constraints:
- `R` is used for **training gradients**.
- `R` is **not** what we report as final bitrate (see §5).

Why this is defensible:
- This is the standard approach in learned image compression.
- It provides gradients without running an entropy coder inside backprop.

### 3.2 Distortion term `D_nvs` (task loss)
We define task distortion on target views:

`D_nvs = MSE(Î_tgt, I_tgt) + 0.05 · LPIPS(Î_tgt, I_tgt)`

Why this is not redundant with vanilla ELIC:
- ELIC is trained to minimize distortion **of the context images**.
- MVSplat needs context information that supports geometry/matching and multi-view consistency.
- At low bpp, “best PSNR context recon” is often not “best NVS input”.

### 3.3 Total objective used by `train_e2e.py`
We use the **ELIC/CompressAI-style** Lagrangian form:

`L = R + λ · (255^2 · D_nvs)`

Why the `255^2` scaling exists:
- The provided ELIC checkpoints were trained with `MSE * 255^2` (pixel units).
- Scaling `D_nvs` by `255^2` puts the magnitude of the distortion term in a similar regime,
  so the canonical λ values `{0.004, 0.008, 0.016, 0.032, 0.15, 0.45}` remain meaningful.

### 3.4 Optional guardrail: context reconstruction regularizer (anti-degeneracy)
If you observe unnatural context reconstructions that “help MVSplat” but look pathological,
add a small regularizer:

`+ β · (255^2 · MSE(Î_ctx, I_ctx))`

Why this is publishable:
- It makes the method robust against “codec cheating” (unreasonable signals).
- It is easy to ablate (β=0 vs β>0).
- It is standard to report context distortion alongside task distortion in task-driven compression.

---

## 4) Training implementation details (what reviewers will dig into)
### 4.1 Initialization (per RD point)
Per λ (recommended), initialize from:
- MVSplat: `checkpoints/vanilla/MVSplat/re10k.ckpt`
- ELIC: `checkpoints/vanilla/ELIC/ELIC_<λ>_ft_3980_Plateau.pth.tar`

Why this is defensible:
- Controls for “better pretraining” as a confound.
- Ensures each RD point starts from a codec already operating in that bitrate regime.

### 4.2 What is optimized
`train_e2e.py` uses:
- Main optimizer: MVSplat params + ELIC params (excluding entropy-bottleneck quantiles)
- Aux optimizer: ELIC entropy-bottleneck quantiles via `elic.aux_loss()` (CompressAI convention)

Why two optimizers:
- This matches the established CompressAI training recipe.
- It maintains a well-behaved entropy model as the analysis transform shifts.

### 4.3 Quantization path
`train_e2e.py` defaults to `--elic-noisequant` to match the ELIC reimplementation’s training behavior.
This is a standard relaxation for quantization-aware training.

### 4.4 Batch size and compute
Training supports `batch_size > 1` (MVSplat supports it in train), but ELIC-in-the-loop is memory heavy.
For reproducibility, start with `--batch-size 1` and scale only if you have headroom.

---

## 5) Evaluation and bitrate accounting (non-negotiable for conference-grade)
### 5.1 Report bitrate from *real* bitstreams
We report bpp from **actual entropy-coded bytes**:
- `ELIC.compress(...)` produces entropy-coded strings.
- `experiments/v1_baseline/compress.py` computes byte counts from these strings and writes `manifest.csv`.

Why this is critical:
- Prevents the classic failure mode: “optimized estimated rate, but actual bits drifted”.
- Makes your RD plots comparable to any compression paper that reports real bitstream sizes.

### 5.2 Use a fixed evaluation protocol
All metrics are computed on:
- `assets/indices/re10k/evaluation_index_re10k.json`
using:
- `experiments/v1_baseline/eval_fair_mvsplat.py`

---

## 6) Canonical commands (one RD point)
### 6.1 Train end-to-end (example: λ=0.032)
```bash
python experiments/v1_e2e/train_e2e.py \
  --tag e2e \
  --lambda 0.032 \
  --mvsplat-init-ckpt checkpoints/vanilla/MVSplat/re10k.ckpt \
  --elic-checkpoints checkpoints/vanilla/ELIC \
  --output-dir checkpoints/v1_e2e \
  --device cuda \
  --max-steps 10000 \
  --batch-size 1 \
  --num-workers 4
```

This writes:
`checkpoints/v1_e2e/e2e_lambda_0.032/`
containing:
- `mvsplat_finetuned.ckpt`
- `ELIC_0032_ft_3980_Plateau.pth.tar`
- `train_log.csv`
- `run_args.json`

### 6.2 Export bitstreams + evaluate fairly (same RD point)
```bash
python experiments/v1_e2e/export_eval_fair.py \
  --tag e2e_lambda_0.032 \
  --lambda 0.032 \
  --run-dir checkpoints/v1_e2e/e2e_lambda_0.032 \
  --device cuda \
  --results-csv outputs/v1_e2e/results/fair_rd.csv --append
```

### 6.3 Plot
```bash
bash scripts/plot_fair_rd.sh --input outputs/v1_e2e/results/fair_rd.csv
```

### 6.4 Sweep all λ values (recommended RD curve recipe)
```bash
csv=outputs/v1_e2e/results/fair_rd.csv
rm -f "$csv"

for l in 0.004 0.008 0.016 0.032 0.15 0.45; do
  python experiments/v1_e2e/train_e2e.py \
    --tag e2e \
    --lambda "$l" \
    --max-steps 10000 \
    --batch-size 1 \
    --device cuda

  python experiments/v1_e2e/export_eval_fair.py \
    --tag "e2e_lambda_${l}" \
    --lambda "$l" \
    --run-dir "checkpoints/v1_e2e/e2e_lambda_${l}" \
    --device cuda \
    --results-csv "$csv" --append
done
```

---

## 7) Anticipating skeptical reviews (attack → answer)
### “You trained with an estimated rate, so your bpp is not credible.”
Training uses entropy-model bpp for gradients (standard), but evaluation reports **actual bytes**
from `ELIC.compress` recorded in `manifest.csv`.

### “You might be leaking target information through the codec.”
Targets are never passed through ELIC and never transmitted. The codec only processes context frames.

### “End-to-end fine-tuning might overfit.”
We train on `train` and report on a fixed `test` protocol. To strengthen:
- run ≥3 seeds and report mean±std,
- report BD-rate/BD-PSNR,
- include qualitative results on held-out scenes.

### “Why not train on the evaluation index?”
Because that would be test leakage. Training uses the bounded sampler on the train split.

### “Why 2 context views?”
It matches the upstream MVSplat training sampler; increasing context count would be a confound.

---

## 8) Minimum publishable reporting checklist
- RD curves (bpp vs PSNR/SSIM/LPIPS) on the fixed test index
- Ablations: `baseline` vs `v1_renderer` vs `v1_compressor` vs `v1_e2e`
- Rate sanity: estimated bpp vs actual bpp at evaluation (at least one table/plot)
- (Optional but recommended) context recon quality + qualitative decoded contexts at low bpp
