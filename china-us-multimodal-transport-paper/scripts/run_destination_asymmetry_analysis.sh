#!/usr/bin/env bash
set -euo pipefail

output="${OUTPUT:-analysis/v11_destination_asymmetry}"
mkdir -p "${output}"

python scripts/analyze_destination_asymmetry.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --v06 results/v06_threshold_scenarios_30seeds \
  --v07 results/v07_activation_scenarios_30seeds \
  --v08 results/v08_focused_activation_thresholds_30seeds \
  --output "${output}"

echo "Destination-asymmetry mechanism analysis completed." >&2
