#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "${ROOT_DIR}"
export PATCHSELECT_REPO_ROOT="${ROOT_DIR}"
exec "${PYTHON_BIN}" -m patchselect.jepa.launch "$@"
