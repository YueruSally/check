#!/usr/bin/env bash
set -euo pipefail

python -u scripts/analyze_full_structural_inland_bottleneck.py \
  --core-analysis analysis/v21_full_structural_core_exact \
  --inland-results results/v26_full_structural_inland_bottleneck \
  --output analysis/v26_full_structural_inland_bottleneck

python -u scripts/run_full_structural_epsilon_tradeoffs.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --core-results results/v21_full_structural_core_exact \
  --output results/v27_full_structural_epsilon_tradeoffs \
  --internal-levels 8 \
  --time-limit-seconds 900 \
  --mip-relative-gap 1e-8

python -u scripts/analyze_full_structural_epsilon_tradeoffs.py \
  --results results/v27_full_structural_epsilon_tradeoffs \
  --output analysis/v27_full_structural_epsilon_tradeoffs

echo "Full structural v2.6 analysis and v2.7 exact epsilon trade-offs completed."
