#!/usr/bin/env bash
set -euo pipefail

workers="${WORKERS:-2}"
seed_start="${SEED_START:-20260906}"
runs="${RUNS:-30}"
population="${POPULATION:-48}"
generations="${GENERATIONS:-99}"
main_results="results/v09_activation_regime_30seeds"
equivalence_results="results/v09_load_pressure_equivalent_30seeds"
analysis_output="analysis/v09_activation_regime"
excluded_scenario="demand_150pct_uswc_capacity_60pct"

if [[ ! -f "${main_results}/aggregate.csv" ]]; then
  echo "Missing completed v0.9 results: ${main_results}/aggregate.csv" >&2
  exit 1
fi

mkdir -p "${equivalence_results}" "${analysis_output}"

echo "Recovery stage 1/4: feasible equal-load-pressure supplement (90 runs)" >&2
python -u -m china_us_multimodal.cli run-experiment \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --cases D1 D2 D3 \
  --scenarios demand_125pct_uswc_capacity_50pct \
  --seed-start "${seed_start}" \
  --runs "${runs}" \
  --workers "${workers}" \
  --population "${population}" \
  --generations "${generations}" \
  --output "${equivalence_results}"

echo "Recovery stage 2/4: rebuild v0.9 analyses without the structurally infeasible case" >&2
python scripts/analyze_scenarios.py \
  "${main_results}" \
  --output "${analysis_output}" \
  --exclude-scenarios "${excluded_scenario}"
python scripts/analyze_paired_statistics.py \
  "${main_results}" \
  --output "${analysis_output}" \
  --exclude-scenarios "${excluded_scenario}"
python scripts/analyze_activation_regime.py \
  "${main_results}" data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --equivalence-results "${equivalence_results}" \
  --output "${analysis_output}"
python scripts/analyze_capacity_utilization.py \
  "${main_results}" data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --output "${analysis_output}"
python scripts/analyze_capacity_utilization.py \
  "${equivalence_results}" data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --output "${analysis_output}/equivalence_capacity"

echo "Recovery stage 3/4: NY/NJ counterfactual" >&2
bash scripts/run_ny_nj_counterfactual.sh

echo "Recovery stage 4/4: destination-asymmetry decomposition" >&2
bash scripts/run_destination_asymmetry_analysis.sh

echo "Recovered v0.9 analyses and completed v1.0-v1.1 experiments." >&2
