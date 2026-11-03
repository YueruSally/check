#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_demand_scale.py \
  --core-analysis analysis/v21_full_structural_core_exact \
  --demand-results results/v25_full_structural_demand_scale \
  --output analysis/v25_full_structural_demand_scale
