#!/usr/bin/env bash
set -euo pipefail

python -u scripts/run_regional_inland_capacity.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --output results/v28_regional_inland_capacity

python -u scripts/analyze_regional_inland_capacity.py \
  --results results/v28_regional_inland_capacity \
  --output analysis/v28_regional_inland_capacity

python -u scripts/analyze_memphis_penalty_switching.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --results results/v22_full_structural_cost_sensitivity \
  --output analysis/v29_memphis_penalty_switching

echo "Final-scope regional inland-capacity and penalty analyses completed."
