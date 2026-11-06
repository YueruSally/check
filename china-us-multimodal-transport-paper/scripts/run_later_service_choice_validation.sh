#!/usr/bin/env bash
set -euo pipefail

python -u scripts/run_later_service_choice_validation.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --core-results results/v21_full_structural_core_exact \
  --inland-results results/v26_full_structural_inland_bottleneck \
  --regional-inland-results results/v28_regional_inland_capacity \
  --output results/v31_later_service_choice_validation \
  --enumeration-cap-per-gateway 100000 \
  --time-limit-seconds 1800 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6

echo "Later-service choice validation completed."
