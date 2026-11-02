#!/usr/bin/env bash
set -euo pipefail

echo "Stage 1/3: activation regime and utilization mechanism" >&2
bash scripts/run_activation_regime.sh

echo "Stage 2/3: NY/NJ allowed-versus-disabled counterfactual" >&2
bash scripts/run_ny_nj_counterfactual.sh

echo "Stage 3/3: destination-asymmetry mechanism decomposition" >&2
bash scripts/run_destination_asymmetry_analysis.sh

echo "All argument-focused supplementary experiments and analyses completed." >&2
