#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kimodo/bin/python}"
SEED="${1:-42}"
VALID_LENGTH="${2:-90}"
OUTPUT_DIR="${3:-${REPO_ROOT}/output/mask-visualizations}"

mkdir -p "${OUTPUT_DIR}"

run_visualization() {
    local config="$1"
    local output_name="$2"
    "${PYTHON_BIN}" "${REPO_ROOT}/visualize_mask.py" \
        --config "${REPO_ROOT}/${config}" \
        --output "${OUTPUT_DIR}/${output_name}-seed${SEED}.png" \
        --seed "${SEED}" \
        --valid-length "${VALID_LENGTH}"
}

run_visualization "configs/mjepa_patch_2d_tiny_coarse7.yaml" "patch-2d-trajectory-coarse7"

echo "Saved all mask visualizations under ${OUTPUT_DIR}"
