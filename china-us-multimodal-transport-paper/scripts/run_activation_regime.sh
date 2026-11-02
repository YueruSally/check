#!/usr/bin/env bash
set -euo pipefail

workers="${WORKERS:-2}"
seed_start="${SEED_START:-20260906}"
runs="${RUNS:-30}"
population="${POPULATION:-48}"
generations="${GENERATIONS:-99}"
output="${OUTPUT:-results/v09_activation_regime_30seeds}"
analysis_output="${ANALYSIS_OUTPUT:-analysis/v09_activation_regime}"

scenarios=(
  pilot_baseline
  uswc_port_capacity_60pct
  uswc_port_capacity_40pct
  uswc_port_capacity_35pct
  uswc_capacity_60pct_delay_168h
  uswc_compound_40pct_delay_168h
  uswc_compound_35pct_delay_168h
  uswc_capacity_60pct_delay_288h
  uswc_capacity_40pct_delay_288h
  uswc_capacity_35pct_delay_288h
  demand_125pct_uswc_capacity_50pct
)

mkdir -p "${output}" "${analysis_output}"

echo "Starting v0.9 activation-regime suite: ${#scenarios[@]} scenarios, 3 cases, ${runs} paired seeds" >&2
python -u -m china_us_multimodal.cli run-experiment \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --cases D1 D2 D3 \
  --scenarios "${scenarios[@]}" \
  --seed-start "${seed_start}" \
  --runs "${runs}" \
  --workers "${workers}" \
  --population "${population}" \
  --generations "${generations}" \
  --output "${output}"

python scripts/analyze_scenarios.py "${output}" --output "${analysis_output}"
python scripts/analyze_paired_statistics.py "${output}" --output "${analysis_output}"
python scripts/analyze_activation_regime.py \
  "${output}" data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --output "${analysis_output}"
python scripts/analyze_capacity_utilization.py \
  "${output}" data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --output "${analysis_output}"

echo "v0.9 activation-regime suite and analyses completed." >&2
