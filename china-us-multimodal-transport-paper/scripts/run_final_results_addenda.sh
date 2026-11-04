#!/usr/bin/env bash
set -euo pipefail

python -u scripts/run_final_results_addenda.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --output analysis/v30_final_results_addenda

python -u paper/render_final_results_figures.py

echo "Final Results addenda, matched zero-penalty baselines, and corrected figures completed."
