#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_cost_sensitivity.py \
  --results results/v22_full_structural_cost_sensitivity \
  --output analysis/v22_full_structural_cost_sensitivity

python -u scripts/run_full_structural_baseline_calibration.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --structural-results results/v21_full_structural_core_exact \
  --output results/v23_full_structural_baseline_calibration \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8

echo "Full structural v2.2 analysis and v2.3 baseline calibration completed."
