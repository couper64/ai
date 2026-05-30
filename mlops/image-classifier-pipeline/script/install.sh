#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_CLASSIFIER_DIR="../../ml/image-classifier"

if [[ ! -x "${IMAGE_CLASSIFIER_DIR}/script/install.sh" ]]; then
    echo "Missing executable installer: ${IMAGE_CLASSIFIER_DIR}/script/install.sh" >&2
    exit 1
fi

echo "Installing local image-classifier package and prerequisites."
(
    cd "${IMAGE_CLASSIFIER_DIR}"
    ./script/install.sh
)

echo "Installing image-classifier-pipeline package and API dependencies."
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
