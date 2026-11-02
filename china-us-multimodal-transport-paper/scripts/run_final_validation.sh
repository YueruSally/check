#!/usr/bin/env bash
set -euo pipefail

results_root="${RESULTS_ROOT:-results/v12_final_closed_fronts_30seeds}"
analysis_root="${ANALYSIS_ROOT:-analysis/v12_final_validation}"
exact_root="${EXACT_ROOT:-results/v12_exact_endpoint_validation}"
exact_time_limit="${EXACT_TIME_LIMIT_SECONDS:-600}"

mkdir -p "${analysis_root}" "${exact_root}"

echo "Final validation 1/4: targeted allowed/restricted NSGA-II reruns" >&2
OUTPUT="${results_root}" bash scripts/run_final_closed_fronts.sh

echo "Final validation 2/4: archive closure and batch-matched mechanism audit" >&2
python -u scripts/analyze_final_closed_fronts.py \
  "${results_root}" data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --output "${analysis_root}"

echo "Final validation 3/4: full retained-configuration capacity diagnostics" >&2
for case_id in D1 D2 D3; do
  python -u scripts/analyze_capacity_utilization.py \
    "${results_root}/${case_id}" data/pilot_v2 \
    --config configs/nsga2_pilot.toml \
    --output "${analysis_root}/capacity/${case_id}" \
    --all-found-configurations
done

echo "Final validation 4/4: MILP endpoint validation (20 tasks; at most 40 calls)" >&2
python -u scripts/run_exact_endpoint_validation.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --output "${exact_root}" \
  --time-limit-seconds "${exact_time_limit}"

echo "Final evidence-consistency validation completed." >&2
