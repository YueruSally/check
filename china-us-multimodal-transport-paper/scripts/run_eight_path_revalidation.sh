#!/usr/bin/env bash
set -euo pipefail

echo "Stage 1/3: eight-path exact core endpoints, share bounds, and diagnostics" >&2
python -u scripts/run_supplementary_core_exact.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --max-paths-per-gateway 8 \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6 \
  --output results/v17_core_exact_paths8

python -u scripts/analyze_supplementary_core.py \
  --results results/v17_core_exact_paths8 \
  --output analysis/v17_core_exact_paths8

python -u scripts/render_supplementary_core_figures.py \
  --results results/v17_core_exact_paths8 \
  --output paper/figures

echo "Stage 2/3: eight-path ten-instance demand robustness" >&2
python -u scripts/run_demand_instance_robustness.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --max-paths-per-gateway 8 \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6 \
  --output results/v18_demand_robustness_paths8

echo "Stage 3/3: targeted binding LA/LB inland-arc capacity sensitivity" >&2
python -u scripts/run_binding_arc_capacity_sensitivity.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8 \
  --relative-tolerance 1e-9 \
  --absolute-tolerance 1e-6 \
  --output results/v19_binding_arc_capacity_sensitivity

echo "Eight-path exact revalidation and targeted capacity sensitivity completed." >&2
