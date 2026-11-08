#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_ROOT}/third_party/pointnet2"
"${PYTHON_BIN}" setup.py build_ext --inplace
"${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps .
