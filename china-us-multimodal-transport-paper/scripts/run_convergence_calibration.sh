#!/usr/bin/env bash
set -euo pipefail

workers="${WORKERS:-2}"
seed_start="${SEED_START:-20260906}"
runs="${RUNS:-10}"

mkdir -p results/convergence_calibration

run_shape() {
  local population="$1"
  local generations="$2"
  local label="p${population}_g${generations}"
  echo "Starting ${label}: 4800 evaluations/run, ${runs} seeds/case" >&2
  python -u -m china_us_multimodal.cli run-experiment \
    data/pilot_v2 \
    --config configs/nsga2_pilot.toml \
    --cases D1 D2 D3 \
    --scenarios pilot_baseline \
    --seed-start "${seed_start}" \
    --runs "${runs}" \
    --workers "${workers}" \
    --population "${population}" \
    --generations "${generations}" \
    --output "results/convergence_calibration/${label}"
}

run_shape 48 99
run_shape 96 49

echo "Convergence calibration completed." >&2
