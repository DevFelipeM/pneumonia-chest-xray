# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pneumonia classification pipeline for chest X-ray images using transfer learning with ResNet50 (FastAI on top of PyTorch). The dataset is the Kaggle `paultimothymooney/chest-xray-pneumonia` set, downloaded at runtime via `kagglehub` (no local data is checked in).

Code comments, docstrings, and CLI help are written in Portuguese — follow that convention when editing.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline: download → train → evaluate
python main.py --epochs 15 --batch-size 32 --lr 1e-3
```

CLI flags (see `main.py:parse_args`): `--epochs` (fine-tuning epochs), `--frozen-epochs` (head-only epochs), `--batch-size`, `--lr`, `--output-dir` (default `outputs`), `--seed`.

There is no test suite, linter config, or build step — the project is run end-to-end through `main.py`.

## Architecture

`main.py` orchestrates a strictly linear 5-stage pipeline; the other modules are pure stages with no cross-dependencies between them:

1. **`data.py`** — `download_dataset()` fetches via kagglehub and resolves the `chest_xray` root (handles a doubly-nested-folder fallback). `build_dataloaders()` builds a FastAI `DataBlock`. Key design decision: the official `val/` split has only ~16 images, so `train/` and `val/` are **merged** and re-split 80/20 via `RandomSplitter`; `test/` is held out untouched for final evaluation. `get_test_files()` returns the official test set.

2. **`model.py`** — `build_learner()` creates a FastAI `vision_learner` over ImageNet-pretrained ResNet50. The positive class is auto-detected by name prefix `"pneumonia"` (`_pos_label_index`), not by index — this matters because class ordering depends on `dls.vocab`.

3. **`train.py`** — `train_model()` runs two phases: (1) frozen backbone, head-only (`fit_one_cycle`); (2) `unfreeze()` + discriminative LRs `slice(lr/100, lr/10)`. Best checkpoint tracked by `valid_loss` via `SaveModelCallback`. Returns training history as a DataFrame extracted from `learner.recorder`.

4. **`evaluate.py`** — `evaluate_on_test()` runs predictions on the held-out test set and writes all artifacts. `plot_training_curves()` is called from `main.py` (not from within evaluate).

The positive-class-by-name logic is duplicated as `_pos_label_index` (model.py) and `_pos_index` (evaluate.py) — keep both consistent if the labeling scheme changes.

## Outputs

All artifacts go to `outputs/` (configurable via `--output-dir`): `training_history.csv`, `training_curves.png`, `metrics.json`, `confusion_matrix.png`, `roc_curve.png`, `classification_report.txt`, and the exported model `pneumonia_resnet50.pkl`. Intermediate best-model checkpoints are saved by FastAI under a `models/` dir relative to the data path.
