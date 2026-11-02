#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_baseline_calibration.py \
  --results results/v23_full_structural_baseline_calibration \
  --output analysis/v23_full_structural_baseline_calibration
