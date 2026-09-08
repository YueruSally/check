#!/usr/bin/env bash
set -euo pipefail

workers="${WORKERS:-2}"
seed_start="${SEED_START:-20260906}"
runs="${RUNS:-30}"
population="${POPULATION:-48}"
generations="${GENERATIONS:-99}"
output="${OUTPUT:-results/v08_focused_activation_thresholds_30seeds}"

scenarios=(
  pilot_baseline
  uswc_departure_delay_264h
  uswc_departure_delay_288h
  uswc_departure_delay_312h
  uswc_compound_35pct_delay_168h
)

mkdir -p "${output}"

echo "Starting v0.8 focused non-carbon threshold suite: ${#scenarios[@]} scenarios, D1/D2, ${runs} paired seeds" >&2
python -u -m china_us_multimodal.cli run-experiment \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --cases D1 D2 \
  --scenarios "${scenarios[@]}" \
  --seed-start "${seed_start}" \
  --runs "${runs}" \
  --workers "${workers}" \
  --population "${population}" \
  --generations "${generations}" \
  --output "${output}"

echo "v0.8 focused non-carbon threshold suite completed." >&2
