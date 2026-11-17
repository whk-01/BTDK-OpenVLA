#!/usr/bin/env bash
# CPU-only verification of the A6 offline-BTKD pipeline (no model / dataset / GPU needed).
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "conda not found; run: bash install_btkd_oneclick.sh --env-name BTKD" >&2
    exit 1
fi
conda activate "${BTKD_ENV_NAME:-BTKD}"
cd "$(dirname "$0")"

echo "== 1. unit tests =============================================="
python tests/test_core.py

echo
echo "== 2. end-to-end smoke (write cache -> read cache -> optimise) ="
python scripts/smoke_test_a6.py

echo
echo "== 3. CLI round-trip on a synthetic cache ====================="
CACHE_DIR="$(mktemp -d)/btkd_dummy_cache"
python scripts/export_teacher_logits.py --dummy --limit 256 --cache_dir "$CACHE_DIR"
python scripts/train_btkd_student.py --cache_dir "$CACHE_DIR" --dataset_name dummy --check_cache_only

echo
echo "All A6 smoke checks passed. Cache left at: $CACHE_DIR"
