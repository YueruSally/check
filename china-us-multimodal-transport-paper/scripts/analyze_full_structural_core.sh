#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_core.py \
  --results results/v21_full_structural_core_exact \
  --output analysis/v21_full_structural_core_exact

echo "Full structural exact core analysis completed."
