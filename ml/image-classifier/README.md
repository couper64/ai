# Image Classifier

A Python image classifier project using PyTorch.

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/couper64/ai.git#subdirectory=ml/image-classifier"
```

This installs the default PyTorch wheels, which are suitable for CPU-only Linux and macOS, including Apple Silicon with MPS support.

For NVIDIA CUDA 13.0, install PyTorch from the dedicated CUDA 13.0 wheel index first, then install this package without reinstalling dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install --no-deps "git+https://github.com/couper64/ai.git#subdirectory=ml/image-classifier"
```

If you need a different CUDA toolkit build (CUDA 12.6, 13.2, ROCm, and so on), copy the exact command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and run it before `pip install --no-deps ...`.

## Development Setup

Create and activate the Conda environment. Conda is only used to manage the Python version:

```bash
conda env create -f environment.yml
conda activate image-classifier
```

Install the project for local development:

```bash
./script/install.sh
```

## Test

Install the test extra, then run smoke tests from the project root:

```bash
pip install -e ".[test]"
python -m pytest test
```

## Dataset Layout

Put training images in a folder where each subfolder is one class:

```text
data/
  apple/
    image-001.jpg
    image-002.jpg
  cup/
    image-001.jpg
  keyboard/
    image-001.jpg
```

Create a reproducible train/validation/test split:

```bash
image-classifier-preprocess ./data/animals10/raw-img ./data/animals10/prepared \
  --train-split 0.7 \
  --validation-split 0.15 \
  --test-split 0.15 \
  --seed 42
```

The split dataset is written as:

```text
prepared/
  train/
  validation/
  test/
```

## Training

Train from the prepared training split and validate against the prepared validation split:

```bash
image-classifier-train ./data/animals10/prepared/train \
  --validation-dir ./data/animals10/prepared/validation \
  --epochs 10 \
  --batch-size 192 \
  --image-size 224 \
  --num-workers 4
```

`best.pt` is selected by validation accuracy. For quick experiments, you can omit `--validation-dir` and use `--validation-split` to create a random validation split during training.

By default, checkpoints are saved to:

```text
output/checkpoint/best.pt
output/checkpoint/last.pt
```

## Evaluation

Evaluate the best checkpoint on a held-out test set:

```bash
image-classifier-evaluate ./data/animals10/prepared/test \
  --checkpoint output/checkpoint/best.pt \
  --batch-size 192 \
  --image-size 224 \
  --num-workers 4 \
  --metrics-output output/metrics/test.json
```

The evaluation report includes loss, accuracy, random guessing accuracy, majority-class baseline accuracy, a confusion matrix, and per-class precision, recall, and F1-score.

## Packaging

Package the selected checkpoint into an inference artifact:

```bash
image-classifier-package \
  --checkpoint output/checkpoint/best.pt \
  --metrics output/metrics/test.json \
  --output-dir output/artifacts
```

The artifact folder contains:

```text
output/artifacts/<run-id>/
  model.pt
  metadata.json
  preprocessing.json
  metrics.json
  class_to_idx.json
  idx_to_class.json
```

## Prediction

Run prediction from a packaged artifact:

```bash
image-classifier-predict ./data/animals10/prepared/test/cane/example.jpg \
  --artifact-dir output/artifacts/<run-id> \
  --top-k 3
```
