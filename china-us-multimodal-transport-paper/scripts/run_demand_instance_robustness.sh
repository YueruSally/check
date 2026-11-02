#!/usr/bin/env bash
set -euo pipefail

python -u scripts/run_demand_instance_robustness.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6 \
  --output results/v16_demand_robustness

echo "v1.6 ten-instance demand robustness completed." >&2
