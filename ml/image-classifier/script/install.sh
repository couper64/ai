#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected; installing PyTorch CUDA 13.0 wheels."
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    python -m pip install -e . --no-deps
else
    echo "No NVIDIA GPU detected; installing default PyTorch wheels."
    python -m pip install -e .
fi
