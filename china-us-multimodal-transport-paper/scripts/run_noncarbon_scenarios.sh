#!/usr/bin/env bash
set -euo pipefail

workers="${WORKERS:-2}"
seed_start="${SEED_START:-20260906}"
runs="${RUNS:-30}"
population="${POPULATION:-48}"
generations="${GENERATIONS:-99}"
output="${OUTPUT:-results/v05_noncarbon_scenarios_30seeds}"

scenarios=(
  pilot_baseline
  uswc_port_capacity_125pct
  uswc_port_capacity_75pct
  uswc_port_capacity_50pct
  uswc_departure_delay_24h
  uswc_departure_delay_72h
  all_departure_delay_48h
  uswc_compound_moderate
  uswc_compound_severe
)

mkdir -p "${output}"

echo "Starting v0.5 non-carbon scenario suite: ${#scenarios[@]} scenarios, 3 cases, ${runs} seeds" >&2
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

echo "v0.5 non-carbon scenario suite completed." >&2
