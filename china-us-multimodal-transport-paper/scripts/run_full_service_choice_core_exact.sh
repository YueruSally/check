#!/usr/bin/env bash
set -euo pipefail

python -u scripts/run_full_service_choice_core_exact.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --reuse-expanded-results results/v31_later_service_choice_validation \
  --output results/v32_full_service_choice_core_exact \
  --enumeration-cap-per-gateway 100000 \
  --time-limit-seconds 1800 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6

echo "Full service-choice exact core grid completed."
