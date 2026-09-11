# Project Memory and Cross-Window Handoff

## 1. Purpose and operating rules

This is the canonical handoff document for the China-US multimodal freight paper. Every new writing or experiment task should read this file first.

The current Codex task is reserved for experiments: experiment design, GitHub code and configuration, remote-server commands, completion checks, result interpretation and upload checking. A separate task will be used for manuscript writing.

Important rules:

- Never run the project, tests or optimization programs on the user's local computer.
- Computational work is run by the user on the remote Linux server through VS Code, normally with nohup.
- Codex may read and edit the GitHub repository directly.
- Do not add carbon prices, emissions constraints, carbon objectives or any other carbon variables.
- New tasks must recover context from this document and the repository rather than relying on chat memory.

## 2. Repository and environment

- Repository: YueruSally/check
- Branch: main
- Project folder: china-us-multimodal-transport-paper/
- GitHub: https://github.com/YueruSally/check/tree/main/china-us-multimodal-transport-paper
- Remote-server checkout: ~/check/china-us-multimodal-transport-paper
- Remote Python environment: chinaus311

## 3. Canonical research positioning

The paper is not a simple comparison of China-US routes and is not an attempt to force more cargo through New York/New Jersey.

The research problem is:

> In a China-US multimodal freight network with fixed schedules, port capacity and shipment-batch constraints, how do West Coast disruptions alter the cost-time trade-off, and under what conditions does New York/New Jersey emerge endogenously as a meaningful alternative gateway?

NY/NJ share is not an optimization objective. It is an emergent network-reconfiguration outcome produced by the two-objective optimization model and its operational constraints.

For result interpretation, a meaningful alternative gateway is defined as an NY/NJ share of at least 10% of total network freight in the cross-seed non-dominated union. This differs from the model constraint requiring every active path for an individual shipment batch to carry at least 10% of that shipment's demand.

## 4. Research questions

Main question: How does a scheduled China-US multimodal freight network reallocate cargo under West Coast port-capacity reductions and sailing delays, and when does NY/NJ become an operationally meaningful alternative gateway?

RQ1 — Baseline structure: How is freight allocated among Los Angeles/Long Beach, Seattle/Tacoma and NY/NJ without disruption?

RQ2 — Disruption mechanism: Do capacity reductions, sailing delays and compound capacity-delay disruptions produce different gateway-reallocation responses?

RQ3 — Activation thresholds: At what tested disruption intervals does NY/NJ reach at least 10% of total gateway share?

RQ4 — Consequences and stability: What cost and delivery-time penalties accompany activation, how do they differ by inland destination, and are the effects stable across 30 NSGA-II seeds?

Current threshold findings:

- D1, Chicago: isolated-delay threshold 264–288 hours.
- D2, Memphis: isolated-delay threshold 240–264 hours.
- D3, Columbus: isolated-delay threshold 168–240 hours.
- D1 with a 168-hour delay: compound capacity threshold between 30% and 35% remaining West Coast capacity.

## 5. Objective functions

The model minimizes two objectives.

Objective 1: total operational cost.

min f1(x) = transport cost + transfer cost + timetable waiting/holding cost + processing cost + shipment-specific tardiness cost.

A compact form is:

\min f_1(x)=\sum_{s\in S}\sum_{p\in P_s}q_{sp}c_{sp}+\sum_{s\in S}\pi_s\max(0,T_s-d_s)

Here q_sp is shipment-batch quantity assigned to path p; c_sp includes transport, transfer, waiting and processing cost; T_s is realized delivery time; d_s is due time; and pi_s is the batch-specific tardiness rate.

Objective 2: quantity-weighted mean delivery time.

\min f_2(x)=\frac{\sum_{s\in S}Q_sT_s}{\sum_{s\in S}Q_s}

NY/NJ share, gateway activation, makespan and resilience are reported indicators, not optimization objectives. Carbon emissions are neither an objective nor a variable in this paper.

## 6. Main constraints

- Shipment-demand conservation.
- Path continuity and flow conservation.
- Fixed departure schedules and timetable feasibility.
- Service capacity.
- Daily arc and node/port capacity.
- Transfer feasibility and minimum processing time.
- Maximum three active paths per shipment.
- Every active path carries at least 10% of the relevant shipment demand.
- Maximum eight legs per path.
- Maximum four mode changes.
- Scenario-adjusted demand and capacity consistency.

Canonical configuration: configs/nsga2_pilot.toml

The minimum active-path share is min_path_share = 0.10.

## 7. Data and implementation map

Source and solver-ready data:

- data/source/ChinaUS_Gateway_Model_Ready_v2.xlsx — canonical source workbook.
- data/pilot_v2/ — solver-ready FEU network tables.
- data/pilot_v2/conversion_report.json — source hash, TEU-to-FEU conversion and integrity checks.

The workbook uses TEU and the solver uses FEU. Quantities and capacities are divided by two. Per-TEU cost and tardiness rates are multiplied by two.

Implementation:

- src/china_us_multimodal/ — model, evaluator, path generation, NSGA-II and CLI.
- configs/nsga2_pilot.toml — model, constraints, algorithm and scenarios.
- scripts/build_pilot_from_workbook.py — builds solver-ready data.
- scripts/analyze_scenarios.py — produces scenario effects, gateway shares, NY/NJ activation and SVG figures.
- scripts/analyze_paired_statistics.py — produces same-seed paired statistics.

Experiment launchers:

- scripts/run_noncarbon_scenarios.sh — v0.5.
- scripts/run_threshold_scenarios.sh — v0.6.
- scripts/run_activation_scenarios.sh — v0.7.
- scripts/run_focused_activation_thresholds.sh — v0.8.

## 8. Experiment chronology and storage

Calibration and baseline:

- results/baseline_30seeds/ — early 30-seed baseline; not the final 4,800-evaluation paper setting.
- results/convergence_calibration/ — equal-budget 48x99 versus 96x49 calibration.
- results/nsga2_final_30seeds/ — final 30-seed NSGA-II baseline.
- results/v04_baseline_30seeds/ — baseline with the operational 10% path-share rule.

For paper comparisons, use the version-matched baseline inside each scenario suite rather than an unrelated baseline folder.

v0.5:

- Results: results/v05_noncarbon_scenarios_30seeds/
- Nine scenarios x three cases x 30 seeds = 810 runs.
- Capacity, delay and compound disruptions; no carbon mechanism.

v0.6:

- Results: results/v06_threshold_scenarios_30seeds/
- Analysis: analysis/v06_threshold_scenarios/
- Twelve scenarios x three cases x 30 seeds = 1,080 runs.
- Capacity 70%, 65%, 60%; gateway-specific 50% capacity; gateway-specific 72-hour delays; regional 96, 120, 168-hour delays; and 50% capacity plus 120-hour delay.

v0.7:

- Results: results/v07_activation_scenarios_30seeds/
- Analysis: analysis/v07_activation_scenarios/
- Six scenarios x three cases x 30 seeds = 540 runs.
- Tests 67.5% capacity, 240/336-hour delays and 30%/40% capacity plus 168-hour delay.
- Includes paired statistics.

v0.8:

- Results: results/v08_focused_activation_thresholds_30seeds/
- Analysis: analysis/v08_focused_activation_thresholds/
- Five scenarios x two cases x 30 seeds = 300 runs: 240 new scenario runs and 60 same-seed baseline controls.
- Tests D1/D2 delays of 264, 288 and 312 hours and 35% capacity plus 168-hour delay.
- Includes paired statistics.

## 9. Meaning of stored result files

Each results/<suite>/ folder can contain:

- manifest.json — cases, scenarios, seeds, solver settings and execution metadata.
- runs.csv — one row per independent algorithm run.
- fronts.csv — final non-dominated solutions from individual runs.
- combined_pareto.csv — cross-seed non-dominated union by case and scenario.
- representative_solutions.csv — minimum-cost, balanced and minimum-delivery-time solutions.
- representative_allocations.csv — batch/path allocations for representative solutions.
- aggregate.csv — aggregated run-level algorithm-performance statistics.

Each analysis/<suite>/ folder can contain:

- scenario_effects.csv — median cost and delivery-time changes from the version-matched baseline.
- gateway_shares.csv — gateway shares for representative solutions.
- ny_activation.csv — NY/NJ use rate and minimum, mean and maximum share across the cross-seed non-dominated union.
- paired_statistics.csv — paired inference for cost and weighted mean delivery time.
- paired_differences.csv — each paired difference by case, scenario, metric and seed.
- paired_statistics_manifest.json — statistical-method metadata.
- scenario_cost_change.svg, scenario_time_change.svg and balanced_gateway_shares.svg — figures.

## 10. How to extract results

Run analysis only on the remote Linux server. Example for v0.8:

    cd ~/check/china-us-multimodal-transport-paper
    conda activate chinaus311
    python scripts/analyze_scenarios.py results/v08_focused_activation_thresholds_30seeds --output analysis/v08_focused_activation_thresholds
    python scripts/analyze_paired_statistics.py results/v08_focused_activation_thresholds_30seeds --output analysis/v08_focused_activation_thresholds

Replace the directory names to regenerate v0.6 or v0.7.

For manuscript extraction:

1. Read manifest.json to confirm cases, scenarios, seeds and solver settings.
2. Read scenario_effects.csv for cost and delivery-time changes.
3. Read ny_activation.csv for gateway activation and threshold claims.
4. Read gateway_shares.csv for representative allocations.
5. Read paired_statistics.csv for confidence intervals, sign tests, Holm-adjusted p-values and paired Cohen dz.
6. Use paired_differences.csv for run-level diagnostics.
7. Trace specific solutions through representative_solutions.csv and representative_allocations.csv.

Do not compare hypervolume values normalized separately in different case-scenario groups as direct scenario-effect sizes.

## 11. Same-seed paired statistics

All scenarios and their version-matched baseline use the same seed sequence, normally 20260906–20260935.

For seed r:

paired cost difference(r) = scenario cost(r) - baseline cost(r)

paired time difference(r) = scenario weighted mean delivery time(r) - baseline weighted mean delivery time(r)

This controls part of the NSGA-II stochastic variation because each scenario is compared with the baseline run using the same seed.

Reported statistics include mean and median paired differences, percentage differences, deterministic 95% bootstrap intervals, exact two-sided sign tests, Holm-adjusted p-values and paired Cohen dz.

These tests apply to total cost and quantity-weighted mean delivery time. They are not a paired significance test of NY/NJ share. NY/NJ share is a descriptive result from the cross-seed non-dominated union.

## 12. Confirmed findings

Baseline:

- NY/NJ share is 0% for D1, D2 and D3.
- Baseline union sizes: D1 84, D2 68 and D3 71.

Capacity-only shocks:

- West Coast capacity at 70%, 67.5%, 65% or 60% does not activate NY/NJ.
- Gateway-specific 50% capacity at LA/LB or Seattle/Tacoma does not activate NY/NJ.
- At 60% regional capacity, median delivery-time increases are 7.22 hours for D1, 11.11 for D2 and 11.31 for D3.
- Median cost changes at 60% remain between -0.34% and 0.33%.
- Interpretation: moderate capacity loss is absorbed mainly within the West Coast system.

Isolated sailing delays:

- 96 and 120 hours: no NY/NJ use.
- 168 hours: absent in D1 and below 1.2% in D2/D3.
- 240 hours: D1 mean share 0.75%; D2 1.24%; D3 26.14%.
- 264 hours: D1 mean 6.59%, range 4.22%–8.06%; D2 mean 19.28%, range 12.47%–26.04%.
- 288 hours: D1 mean 39.00%, range 32.75%–50.56%; D2 mean 32.36%, range 31.72%–32.79%.
- 312 hours: D1 mean 67.14%; D2 mean 43.64%.
- 336 hours: D1 mean 71.39%; D2 mean 67.75%; D3 mean 83.62%.

Meaningful isolated-delay intervals:

- D1: 264–288 hours.
- D2: 240–264 hours.
- D3: 168–240 hours.

Compound disruption with 168-hour delay:

| Remaining West Coast capacity | D1 mean NY/NJ share | D2 mean | D3 mean |
| --- | ---: | ---: | ---: |
| 40% | 6.19% | 12.59% | 35.59% |
| 35% | 5.61% | 10.83% | not rerun |
| 30% | 17.90% | 17.32% | 45.11% |

D1 crosses the criterion between 30% and 35% remaining capacity. D2 at 35% ranges from 6.88% to 16.97%; the 30%/35%/40% means are not strictly monotonic. Describe D2 as a transition region, not an exact deterministic threshold.

Cost and delivery consequences at the first stable isolated-delay point:

- D1 at 288 hours: median cost +210.69%; median weighted delivery time +264.58 hours.
- D2 at 264 hours: median cost +186.80%; median weighted delivery time +248.26 hours.
- D3 at 240 hours: median cost +156.94%; median weighted delivery time +222.32 hours.

NY/NJ is a resilience pathway that preserves feasible routing choices under severe disruption. It does not restore baseline cost or service performance.

For v0.8 severe scenarios, all 30 paired cost and time differences are positive. Holm-adjusted p-values are approximately 2.98 x 10^-8 and deterministic 95% bootstrap intervals exclude zero.

## 13. Interpretation boundaries

- Do not say that the model optimizes NY/NJ share.
- Do not say that NY/NJ improves cost or delivery time.
- Say that NY/NJ emerges endogenously as an alternative under severe disruption.
- Do not confuse the per-shipment 10% active-path constraint with the network-level 10% meaningful-gateway criterion.
- Do not claim an exact threshold where only an interval was tested.
- Do not describe D2's compound capacity response as monotonic.
- Do not claim paired significance for gateway share; current paired inference covers cost and time.
- Do not present NSGA-II as the main algorithmic novelty unless a genuinely new operator, encoding, repair or dominance mechanism is added.
- Do not introduce carbon content.

## 14. Paper narrative

Normal operation -> substitution within West Coast gateways -> moderate capacity loss absorbed internally -> ordinary delays do not activate NY/NJ -> destination-specific delay threshold crossed -> discontinuous NY/NJ increase -> cross-coast gateway structure -> severe cost and delivery-time penalties remain.

Resilience means retaining feasible alternative routing and gateway options under disruption, not returning to baseline cost or delivery time.

## 15. Manuscript structure

1. Introduction: gateway dependence, disruption exposure, research questions and contributions.
2. Literature Review: multimodal routing; timetable and capacity; port disruption and resilience; multi-objective optimization; research gap.
3. Problem Description and Data: origins, US gateways, inland destinations, modes, batches, services, capacities, transfers and TEU-to-FEU conversion.
4. Mathematical Model: sets, parameters, variables, two objectives and constraints.
5. Solution Method: gateway-aware Martins path generation, constrained NSGA-II allocation, feasibility-first dominance, 10% path-share repair and 30-seed protocol.
6. Experimental Design: baseline, capacity, delay, compound, threshold refinement and paired inference.
7. Results: paper/04_results.md and paper/RESULTS_SUMMARY_CN.md.
8. Discussion: mechanisms, destination heterogeneity, resilience interpretation and managerial implications.
9. Conclusion: findings, contributions, limitations and future research.

## 16. Contribution claims

1. A schedule-driven shipment-batch multimodal model combining fixed services, waiting, processing, transfers, service and node capacity, limited splitting and minimum executable shares.
2. Endogenous alternative-gateway activation: NY/NJ is neither forced nor optimized directly.
3. Identification of destination-specific nonlinear activation intervals rather than one universal threshold.
4. Separation of capacity-loss, delay and compound-disruption mechanisms.
5. A same-seed multi-run statistical design that distinguishes stable cost/time effects from one NSGA-II realization.

The strongest contribution is the endogenous, destination-dependent activation threshold. Do not frame the paper primarily as a novel NSGA-II paper.

## 17. Working titles

Preferred:

> When Does an Alternative Gateway Become Viable? A Scheduled Multimodal Freight Routing Study of China-US Port Disruptions

Alternative:

> Bi-objective Scheduled Multimodal Freight Routing under Port Capacity and Sailing Delay Disruptions: Evidence from China-US Gateway Networks

## 18. Existing paper and protocol files

- paper/04_results.md — consolidated English results chapter.
- paper/RESULTS_SUMMARY_CN.md — Chinese interpretation and reporting cautions.
- docs/v07_activation_protocol.md — v0.7 protocol.
- docs/v08_focused_threshold_protocol.md — v0.8 protocol.
- docs/experiment_protocol.md — broader protocol.
- README.md — project overview and commands.

## 19. Prompt for the new manuscript-writing task

> This task is for manuscript writing only. Do not run experiments locally or remotely and do not change experimental results. First read china-us-multimodal-transport-paper/docs/PROJECT_MEMORY_AND_HANDOFF.md, then read paper/04_results.md, paper/RESULTS_SUMMARY_CN.md, the v0.7/v0.8 analysis tables and the model source files needed for the section being written. Treat the handoff file as the canonical research framing. The paper has two objectives: total operational cost and quantity-weighted mean delivery time. NY/NJ activation is an endogenous result, not an objective. Keep the paper completely free of carbon variables. Preserve interval claims, statistical limitations and the distinction between the per-shipment 10% path constraint and the network-level 10% meaningful-gateway criterion.

## 20. Use of this experiment task

Use this task for narrow new scenarios, GitHub code/configuration changes, remote-server nohup commands, completion checks, uploaded-result interpretation and experimental diagnostics. Do not use this task as the main manuscript-writing workspace.
