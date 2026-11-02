#!/usr/bin/env bash
set -euo pipefail

python -u scripts/run_path_set_sensitivity.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --core-results results/v14_core_exact_suite \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6 \
  --output results/v15_path_set_sensitivity

echo "v1.5 four-vs-eight path sensitivity completed." >&2
