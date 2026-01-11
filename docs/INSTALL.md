# Installation

This is research code and assumes a Linux + CUDA environment.

## Environment

- Python 3.10+ recommended
- CUDA-capable GPU for training/eval (V2/V3 are heavy)
- A working C++ toolchain (CUDA extensions)

## Setup (example)

Create an environment:

```bash
conda create -n thesis-clean python=3.10 -y
conda activate thesis-clean
```

Install PyTorch (pick the wheel matching your CUDA; example for CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install Python dependencies:

```bash
pip install -r third_party/mvsplat/requirements.txt
pip install compressai pytorch-msssim
pip install rich  # nicer progress bars in this repo's scripts
```

Build/install the CUDA rasterizer:

```bash
pip install -e third_party/diff-gaussian-rasterization-modified
```

(Optional) make imports work from anywhere:

```bash
export PYTHONPATH="$(pwd):$(pwd)/third_party:${PYTHONPATH}"
```

## Checkpoints

Place files under `checkpoints/` (not tracked by git):

- `checkpoints/vanilla/MVSplat/re10k.ckpt` (MVSplat pretrained on RE10K)
- `checkpoints/vanilla/ELIC/ELIC_*.pth.tar` (ELIC checkpoints; see `experiments/v1_baseline/compress.py` for λ mapping)
- (Optional, training-only) `checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth` (UniMatch/GMDepth init weights; used by MVSplat's cost-volume encoder in `mode=train`)

Where to download these depends on your setup; the upstream MVSplat README includes links for the RE10K checkpoint and UniMatch weights.
