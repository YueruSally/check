#!/usr/bin/env bash
set -euo pipefail

workers="${WORKERS:-2}"
seed_start="${SEED_START:-20260906}"
runs="${RUNS:-30}"
population="${POPULATION:-48}"
generations="${GENERATIONS:-99}"
output="${OUTPUT:-results/v07_activation_scenarios_30seeds}"

scenarios=(
  pilot_baseline
  uswc_port_capacity_675pct
  uswc_departure_delay_240h
  uswc_departure_delay_336h
  uswc_compound_30pct_delay_168h
  uswc_compound_40pct_delay_168h
)

mkdir -p "${output}"

echo "Starting v0.7 non-carbon activation suite: ${#scenarios[@]} scenarios, 3 cases, ${runs} paired seeds" >&2
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

echo "v0.7 non-carbon activation suite completed." >&2
