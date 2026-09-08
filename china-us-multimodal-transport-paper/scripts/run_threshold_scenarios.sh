#!/usr/bin/env bash
set -euo pipefail

workers="${WORKERS:-2}"
seed_start="${SEED_START:-20260906}"
runs="${RUNS:-30}"
population="${POPULATION:-48}"
generations="${GENERATIONS:-99}"
output="${OUTPUT:-results/v06_threshold_scenarios_30seeds}"

scenarios=(
  pilot_baseline
  uswc_port_capacity_70pct
  uswc_port_capacity_65pct
  uswc_port_capacity_60pct
  la_lb_port_capacity_50pct
  seattle_tacoma_port_capacity_50pct
  la_lb_departure_delay_72h
  seattle_tacoma_departure_delay_72h
  uswc_departure_delay_96h
  uswc_departure_delay_120h
  uswc_departure_delay_168h
  uswc_compound_50pct_delay_120h
)

mkdir -p "${output}"

echo "Starting v0.6 non-carbon threshold suite: ${#scenarios[@]} scenarios, 3 cases, ${runs} seeds" >&2
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

echo "v0.6 non-carbon threshold suite completed." >&2
