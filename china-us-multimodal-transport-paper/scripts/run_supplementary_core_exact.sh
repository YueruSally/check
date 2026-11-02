#!/usr/bin/env bash
set -euo pipefail

python -u scripts/run_supplementary_core_exact.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --reuse-exact-root results/v12_exact_endpoint_validation \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6 \
  --output results/v14_core_exact_suite

python -u scripts/analyze_supplementary_core.py \
  --results results/v14_core_exact_suite \
  --output analysis/v14_core_exact_suite

python -u scripts/render_supplementary_core_figures.py \
  --results results/v14_core_exact_suite \
  --output paper/figures

echo "v1.4 exact core suite and value curves completed." >&2
