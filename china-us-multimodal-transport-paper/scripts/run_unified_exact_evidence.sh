#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_unified_exact_evidence.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --closed-points analysis/v12_final_validation/closed_front_points.csv \
  --historical-points analysis/paper_revised/counterfactual_front_points.csv \
  --exact-root results/v12_exact_endpoint_validation \
  --output analysis/v13_unified_exact_evidence

echo "Unified exact-evidence post-processing completed." >&2
