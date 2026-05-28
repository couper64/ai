# Image Classifier

A Python image classifier project using PyTorch.

## Conda Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate classifier
```

Install this project into the environment:

```bash
pip install -e . --no-deps
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

## Training

Train from a folder of class subfolders:

```bash
python -m classifier.train ./data --epochs 10 --batch-size 32 --image-size 224 --validation-split 0.2
```

The trainer randomly splits the dataset into training and validation subsets. `best.pt` is selected by validation accuracy.

By default, checkpoints are saved to:

```text
output/checkpoint/best.pt
output/checkpoint/last.pt
```

After installing the project, you can also run:

```bash
classifier-train ./data --epochs 10
```
