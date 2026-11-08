#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements-geometry.txt"
# PyTorch and NumPy are installed by environment.yml before this command.
# Pin the reference build without replacing its existing runtime dependencies.
conda install --yes --no-deps -c pytorch3d pytorch3d=0.7.8=py38_cu118_pyt240
"${PYTHON_BIN}" -c 'import pytorch3d; from pytorch3d.transforms import so3_log_map; print("PyTorch3D", pytorch3d.__version__)'
