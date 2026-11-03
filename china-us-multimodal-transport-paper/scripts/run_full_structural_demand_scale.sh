#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_release_robustness.py \
  --single-wave-analysis analysis/v21_full_structural_core_exact \
  --release-results results/v24_full_structural_release_robustness \
  --output analysis/v24_full_structural_release_robustness

python -u scripts/run_full_structural_demand_scale.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --structural-results results/v21_full_structural_core_exact \
  --output results/v25_full_structural_demand_scale \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8

echo "Full structural v2.4 analysis and v2.5 demand scaling completed."
