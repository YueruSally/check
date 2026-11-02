#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_cost_sensitivity.py \
  --results results/v22_full_structural_cost_sensitivity \
  --output analysis/v22_full_structural_cost_sensitivity
