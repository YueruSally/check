#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_demand_scale.py \
  --core-analysis analysis/v21_full_structural_core_exact \
  --demand-results results/v25_full_structural_demand_scale \
  --output analysis/v25_full_structural_demand_scale

python -u scripts/run_full_structural_inland_bottleneck.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --structural-results results/v21_full_structural_core_exact \
  --output results/v26_full_structural_inland_bottleneck \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8

echo "Full structural v2.5 analysis and v2.6 inland-bottleneck interaction completed."
