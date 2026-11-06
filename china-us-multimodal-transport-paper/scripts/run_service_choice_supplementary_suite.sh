#!/usr/bin/env bash
set -euo pipefail

CORE_RESULTS="results/v32_full_service_choice_core_exact"
CORE_ANALYSIS="analysis/v32_full_service_choice_core_exact"
RESULT_ROOT="results/v33_service_choice_supplementary"
ANALYSIS_ROOT="analysis/v33_service_choice_supplementary"
COMMON_ARGS=(
  data/pilot_v2
  --config configs/nsga2_pilot.toml
  --max-extra-service-departures 2
  --time-limit-seconds 900
  --mip-relative-gap 1e-8
  --relative-tolerance 1e-9
  --absolute-tolerance 1e-6
)

mkdir -p "$RESULT_ROOT" "$ANALYSIS_ROOT"

python -u scripts/analyze_full_structural_core.py \
  --results "$CORE_RESULTS" \
  --output "$CORE_ANALYSIS" \
  --reference-results results/v21_full_structural_core_exact

echo "Stage 1/7: cost-parameter sensitivity with selectable services"
python -u scripts/run_full_structural_cost_sensitivity.py \
  "${COMMON_ARGS[@]}" \
  --structural-results "$CORE_RESULTS" \
  --output "$RESULT_ROOT/cost_sensitivity"
python -u scripts/analyze_full_structural_cost_sensitivity.py \
  --results "$RESULT_ROOT/cost_sensitivity" \
  --output "$ANALYSIS_ROOT/cost_sensitivity"
python -u scripts/analyze_memphis_penalty_switching.py \
  data/pilot_v2 \
  --config configs/nsga2_pilot.toml \
  --results "$RESULT_ROOT/cost_sensitivity" \
  --output "$ANALYSIS_ROOT/memphis_penalty_switching"

echo "Stage 2/7: baseline parameter sensitivity with selectable services"
python -u scripts/run_full_structural_baseline_calibration.py \
  "${COMMON_ARGS[@]}" \
  --structural-results "$CORE_RESULTS" \
  --output "$RESULT_ROOT/baseline_sensitivity"
python -u scripts/analyze_full_structural_baseline_calibration.py \
  --results "$RESULT_ROOT/baseline_sensitivity" \
  --output "$ANALYSIS_ROOT/baseline_sensitivity"

echo "Stage 3/7: release-pattern robustness with selectable services"
python -u scripts/run_full_structural_release_robustness.py \
  "${COMMON_ARGS[@]}" \
  --structural-results "$CORE_RESULTS" \
  --output "$RESULT_ROOT/release_robustness"
python -u scripts/analyze_full_structural_release_robustness.py \
  --single-wave-analysis "$CORE_ANALYSIS" \
  --release-results "$RESULT_ROOT/release_robustness" \
  --output "$ANALYSIS_ROOT/release_robustness"

echo "Stage 4/7: demand-scale robustness with selectable services"
python -u scripts/run_full_structural_demand_scale.py \
  "${COMMON_ARGS[@]}" \
  --structural-results "$CORE_RESULTS" \
  --output "$RESULT_ROOT/demand_scale"
python -u scripts/analyze_full_structural_demand_scale.py \
  --core-analysis "$CORE_ANALYSIS" \
  --demand-results "$RESULT_ROOT/demand_scale" \
  --output "$ANALYSIS_ROOT/demand_scale"

echo "Stage 5/7: local and regional inland-capacity tests"
python -u scripts/run_full_structural_inland_bottleneck.py \
  "${COMMON_ARGS[@]}" \
  --structural-results "$CORE_RESULTS" \
  --output "$RESULT_ROOT/local_inland_bottleneck"
python -u scripts/analyze_full_structural_inland_bottleneck.py \
  --core-analysis "$CORE_ANALYSIS" \
  --inland-results "$RESULT_ROOT/local_inland_bottleneck" \
  --output "$ANALYSIS_ROOT/local_inland_bottleneck"
python -u scripts/run_regional_inland_capacity.py \
  "${COMMON_ARGS[@]}" \
  --structural-results "$CORE_RESULTS" \
  --output "$RESULT_ROOT/regional_inland_capacity"
python -u scripts/analyze_regional_inland_capacity.py \
  --results "$RESULT_ROOT/regional_inland_capacity" \
  --output "$ANALYSIS_ROOT/regional_inland_capacity"

echo "Stage 6/7: epsilon-constrained trade-off solutions"
python -u scripts/run_full_structural_epsilon_tradeoffs.py \
  "${COMMON_ARGS[@]}" \
  --core-results "$CORE_RESULTS" \
  --output "$RESULT_ROOT/epsilon_tradeoffs" \
  --internal-levels 8
python -u scripts/analyze_full_structural_epsilon_tradeoffs.py \
  --results "$RESULT_ROOT/epsilon_tradeoffs" \
  --output "$ANALYSIS_ROOT/epsilon_tradeoffs"

echo "Stage 7/7: final addenda under the selectable-service model"
python -u scripts/run_final_results_addenda.py \
  "${COMMON_ARGS[@]}" \
  --structural-results "$CORE_RESULTS" \
  --cost-sensitivity-results "$RESULT_ROOT/cost_sensitivity" \
  --inland-results "$RESULT_ROOT/local_inland_bottleneck" \
  --output "$ANALYSIS_ROOT/final_results_addenda"

echo "All selectable-service supplementary experiments completed."
