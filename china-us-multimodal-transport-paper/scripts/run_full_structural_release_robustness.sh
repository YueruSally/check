#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_baseline_calibration.py \
  --results results/v23_full_structural_baseline_calibration \
  --output analysis/v23_full_structural_baseline_calibration

python -u scripts/run_full_structural_release_robustness.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --structural-results results/v21_full_structural_core_exact \
  --output results/v24_full_structural_release_robustness \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8

echo "Full structural v2.3 analysis and v2.4 release robustness completed."
