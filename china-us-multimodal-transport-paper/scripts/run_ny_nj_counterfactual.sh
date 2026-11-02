#!/usr/bin/env bash
set -euo pipefail

workers="${WORKERS:-2}"
seed_start="${SEED_START:-20260906}"
runs="${RUNS:-30}"
population="${POPULATION:-48}"
generations="${GENERATIONS:-99}"
output_root="${OUTPUT_ROOT:-results/v10_ny_nj_counterfactual_30seeds}"
analysis_output="${ANALYSIS_OUTPUT:-analysis/v10_ny_nj_counterfactual}"

mkdir -p "${output_root}" "${analysis_output}"

run_case() {
  local case_id="$1"
  local disabled_delay_scenario="$2"
  local output="${output_root}/${case_id}"
  mkdir -p "${output}"

  echo "Starting v1.0 NY/NJ counterfactual for ${case_id}: ${runs} paired seeds" >&2
  python -u -m china_us_multimodal.cli run-experiment \
    data/pilot_v2 \
    --config configs/nsga2_pilot.toml \
    --cases "${case_id}" \
    --scenarios \
      "${disabled_delay_scenario}" \
      uswc_compound_30pct_delay_168h_no_ny_nj \
    --seed-start "${seed_start}" \
    --runs "${runs}" \
    --workers "${workers}" \
    --population "${population}" \
    --generations "${generations}" \
    --output "${output}"
}

run_case D1 uswc_departure_delay_288h_no_ny_nj
run_case D2 uswc_departure_delay_264h_no_ny_nj
run_case D3 uswc_departure_delay_240h_no_ny_nj

python scripts/analyze_ny_nj_value.py \
  "${output_root}" \
  --v07 results/v07_activation_scenarios_30seeds \
  --v08 results/v08_focused_activation_thresholds_30seeds \
  --output "${analysis_output}"

echo "v1.0 NY/NJ counterfactual suite and paired value analysis completed." >&2
