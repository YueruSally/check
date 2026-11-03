#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_release_robustness.py \
  --single-wave-analysis analysis/v21_full_structural_core_exact \
  --release-results results/v24_full_structural_release_robustness \
  --output analysis/v24_full_structural_release_robustness
