#!/usr/bin/env bash
set -euo pipefail

echo "Stage 1/3: exact endpoints, share bounds, diagnostics, and feasibility" >&2
bash scripts/run_supplementary_core_exact.sh

echo "Stage 2/3: four-vs-eight candidate-path sensitivity" >&2
bash scripts/run_path_set_sensitivity.sh

echo "Stage 3/3: ten pre-registered demand instances" >&2
bash scripts/run_demand_instance_robustness.sh

echo "All fixed supplementary experiments completed." >&2
