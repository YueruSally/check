#!/usr/bin/env bash
set -euo pipefail

bash scripts/analyze_full_structural_core.sh

python -u scripts/run_full_structural_cost_sensitivity.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --structural-results results/v21_full_structural_core_exact \
  --output results/v22_full_structural_cost_sensitivity \
  --enumeration-cap-per-gateway 100000 \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6

echo "Full structural exact cost sensitivity completed."
