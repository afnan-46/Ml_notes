# Lab 1 — Image Classification with Transfer Learning (CIFAR-10)

CSE475 Machine Learning course lab. Fine-tunes a pretrained MobileNetV2 on CIFAR-10 using PyTorch, with EDA, augmentation, checkpointing, multi-GPU support, and test-set evaluation — plus a full homework solutions notebook covering 10 follow-up exercises.

## Contents

| File | Description |
|---|---|
| `lab1-image-classification-multigpu.ipynb` | Original lab notebook: EDA → train/val/test split → augmentation → MobileNetV2 transfer learning → checkpointed training → test evaluation → confusion matrix → prediction inspection. |
| `lab1-homework-solutions.ipynb` | Solutions to the 10 end-of-lab exercises (pretraining baseline, frozen backbone, augmentation ablation, confusion analysis, ResNet18 comparison, checkpoint resume, best-vs-last, optimizer stripping, normalization ablation, multi-GPU scaling). |

## What the lab does

1. **Load data** — CIFAR-10 (60,000 32×32 RGB images, 10 classes), downloaded via `torchvision.datasets.CIFAR10`.
2. **EDA** — class distribution, sample images per class, pixel statistics.
3. **Split** — 10,000 train / 2,000 val / 2,000 test (subsampled for lab runtime; caps are easy to remove for a full run).
4. **Augmentation** — random horizontal flip, rotation, and color jitter applied to training data only; validation/test use deterministic resize + normalize.
5. **Model** — MobileNetV2 pretrained on ImageNet, classifier head replaced with a fresh `nn.Linear(*, 10)`; wraps in `nn.DataParallel` automatically when more than one GPU is visible.
6. **Training** — 10 epochs, Adam optimizer (`lr=1e-4`), cross-entropy loss. Saves a `best.pt` checkpoint whenever validation accuracy improves, plus periodic snapshots every 3 epochs.
7. **Evaluation** — reloads the best checkpoint from disk (mirroring a real deploy step), evaluates on the held-out test set, and reports a confusion matrix and classification report.
8. **Inspection** — visualizes individual test predictions, highlighting misclassifications.

## Homework exercises covered

1. No-pretraining baseline (`weights=None`) vs. pretrained
2. Frozen backbone (train classifier head only)
3. Augmentation ablation (drop flip/color jitter)
4. Confusion analysis (most-confused class pairs)
5. Backbone swap (MobileNetV2 vs. ResNet18)
6. Resume training from a mid-run checkpoint
7. Best checkpoint vs. last checkpoint comparison
8. Stripping optimizer state to shrink checkpoint size
9. Normalization ablation (no mean/std normalization)
10. Multi-GPU scaling (1 GPU vs. 2 GPUs epoch time)

## Requirements

- Python 3.9+
- PyTorch and TorchVision (with CUDA support recommended)
- `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm
```

## Usage

Run cells top to bottom in each notebook. The homework notebook's first cell rebuilds all shared state (data, splits, transforms, loaders, training/eval helpers) needed by every exercise, so run it before any exercise cell. Exercises 6–8 depend on checkpoint files (`best.pt`, `epoch_03.pt`, etc.) produced by Exercise 1's training run.

A GPU is strongly recommended — training at 224×224 resolution on CPU is slow. Exercise 10 (multi-GPU scaling) specifically requires 2+ visible GPUs, such as Kaggle's T4×2 runtime, to produce meaningful numbers.

## Notes

- Dataset caps (`N_TRAIN`, `N_VAL`, `N_TEST`) are set low for fast lab iteration; increase or remove them for a full CIFAR-10 run.
- Checkpoints are written to `./checkpoints/` (and per-experiment subfolders in the homework notebook) and are not included in this repo — add `checkpoints/` and `data/` to `.gitignore` before pushing.
