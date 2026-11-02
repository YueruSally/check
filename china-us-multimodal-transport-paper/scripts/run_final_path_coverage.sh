#!/usr/bin/env bash
set -euo pipefail

python -u scripts/run_final_path_coverage.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --paths8-results results/v17_core_exact_paths8 \
  --output results/v20_final_path_coverage \
  --enumeration-cap-per-gateway 100000 \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6

echo "Final path coverage experiment completed."
