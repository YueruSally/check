#!/usr/bin/env bash
set -euo pipefail

workers="${WORKERS:-2}"
seed_start="${SEED_START:-20260906}"
runs="${RUNS:-30}"
population="${POPULATION:-48}"
generations="${GENERATIONS:-99}"
root="${OUTPUT:-results/v12_final_closed_fronts_30seeds}"

mkdir -p "${root}/D1" "${root}/D2" "${root}/D3"

echo "Stage 1/3: Chicago 288 h allowed/restricted (60 runs)" >&2
python -u -m china_us_multimodal.cli run-experiment \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --cases D1 \
  --scenarios \
    uswc_departure_delay_288h \
    uswc_departure_delay_288h_no_ny_nj \
  --seed-start "${seed_start}" \
  --runs "${runs}" \
  --workers "${workers}" \
  --population "${population}" \
  --generations "${generations}" \
  --output "${root}/D1"

echo "Stage 2/3: Memphis 264/288 h allowed/restricted (120 runs)" >&2
python -u -m china_us_multimodal.cli run-experiment \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --cases D2 \
  --scenarios \
    uswc_departure_delay_264h \
    uswc_departure_delay_264h_no_ny_nj \
    uswc_departure_delay_288h \
    uswc_departure_delay_288h_no_ny_nj \
  --seed-start "${seed_start}" \
  --runs "${runs}" \
  --workers "${workers}" \
  --population "${population}" \
  --generations "${generations}" \
  --output "${root}/D2"

echo "Stage 3/3: Columbus 240 h allowed/restricted (60 runs)" >&2
python -u -m china_us_multimodal.cli run-experiment \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --cases D3 \
  --scenarios \
    uswc_departure_delay_240h \
    uswc_departure_delay_240h_no_ny_nj \
  --seed-start "${seed_start}" \
  --runs "${runs}" \
  --workers "${workers}" \
  --population "${population}" \
  --generations "${generations}" \
  --output "${root}/D3"

echo "Completed 240 final closed-front NSGA-II runs." >&2
