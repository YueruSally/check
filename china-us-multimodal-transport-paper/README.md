# China–US Scheduled Multimodal Freight Routing

This project studies scheduled multi-batch freight routing between China and
the United States. The pilot compares West Coast landbridge routes with East
Coast all-water routes for Chicago, Memphis and Columbus.

## Solver-ready v2 network

`data/pilot_v2` is generated from
`data/source/ChinaUS_Gateway_Model_Ready_v2.xlsx` and contains:

- 9 nodes and 28 directed arcs;
- 17 scheduled services and 496 departures over an eight-week horizon;
- 32 transfer rules;
- three independent 40-batch OD cases (120 records in one table).

The workbook uses TEU while the solver uses FEU. The conversion is explicit:
quantities and capacities are divided by two; per-TEU costs and tardiness rates
are multiplied by two. `conversion_report.json` records the source hash and
all integrity checks.

## Optimization flow

1. Validate workbook cross-references and generate FEU CSV tables.
2. Validate node, arc, service, transfer and shipment constraints.
3. Run Martins separately through each US gateway and merge unique paths.
4. Allocate each shipment across at most three paths with constrained NSGA-II.
5. Evaluate every solution with the same timetable, capacity, cost and
   delivery-time evaluator.

Gateway-aware path generation is intentional: a static global dominance filter
could discard the East Coast all-water option before schedule, capacity and
scenario effects are evaluated.

## Commands

```bash
python scripts/build_pilot_from_workbook.py \
  data/source/ChinaUS_Gateway_Model_Ready_v2.xlsx data/pilot_v2

python -m china_us_multimodal.cli validate-network \
  data/pilot_v2 --config configs/nsga2_pilot.toml

python -m china_us_multimodal.cli run-nsga2 \
  data/pilot_v2 --config configs/nsga2_pilot.toml \
  --case D1 --output results/D1

python -m china_us_multimodal.cli run-experiment \
  data/pilot_v2 --config configs/nsga2_pilot.toml \
  --cases D1 D2 D3 --scenarios pilot_baseline \
  --seed-start 20260906 --runs 30 --workers 2 \
  --population 48 --generations 99 \
  --output results/v04_baseline_30seeds
```

Use D1 for Chicago, D2 for Memphis and D3 for Columbus. Scenario overrides use
`--scenario`.

## Model coverage

The shared evaluator includes arc and transfer cost, timetable waiting and
processing cost, shipment-specific tardiness penalties, service capacity,
daily arc capacity, daily node capacity, flow conservation, a 10% minimum share
for every active path, split limits, leg limits and mode-change limits.
NSGA-II uses feasibility-first constrained dominance and deterministic seeds.

## Reproducibility

The two optimization objectives are total USD cost and quantity-weighted mean
delivery time. Makespan remains a reported service-level indicator.

The committed smoke outputs only prove that the full pipeline runs. They are
not paper findings. Follow `docs/experiment_protocol.md` for multi-seed paper
experiments and equal evaluation budgets across NSGA-II, SPEA2 and MOEA/D.
`run-experiment` writes per-run metrics, every final objective-space front,
the cross-seed non-dominated union, representative route allocations, all
distinct non-dominated configurations retained by the observational search
archive, seed-level cost/time endpoints, and an experiment manifest. The
observational archive is not used for selection. Each completed run is printed
immediately so nohup logs show live progress. Hypervolume and spacing are
calculated after normalizing every run in a case/scenario group against the
same pooled ideal and nadir points.

## Convergence calibration

The selected paper configuration uses 48 individuals and 99 generations,
giving 4,800 evaluations per run. Calibration artifacts compare it with
96 individuals and 49 generations under the same evaluation budget.

## v0.4 operational split rule

Every active route for a shipment must carry at least 10% of that shipment's
scenario-adjusted demand. NSGA-II prunes smaller genome shares and renormalizes
the remaining paths; the shared evaluator independently enforces the same rule
for future comparator algorithms.

## v0.5 non-carbon scenario suite

The v0.5 suite adds regional daily port-capacity multipliers and fixed ocean
departure delays. It contains no carbon cost, emissions cap or carbon
objective. The nine-scenario matrix covers:

- the baseline control;
- West Coast capacity at 125%, 75% and 50%;
- West Coast departure delays of 24 and 72 hours;
- a 48-hour delay on both West and East Coast ocean services;
- moderate and severe West Coast compound disruptions.

Run all 810 NSGA-II experiments (nine scenarios × three cases × 30 seeds) in
the background:

```bash
mkdir -p logs

nohup nice -n 10 bash scripts/run_noncarbon_scenarios.sh \
  > logs/v05_noncarbon_scenarios.log 2>&1 &

echo $! > logs/v05_noncarbon_scenarios.pid
```

The default is two workers, 48 individuals, 99 generations and 30 seeds. The
previous baseline runtime suggests approximately 1.5–2 hours on two workers,
but shared-server load may change this estimate. Use
`tail -f logs/v05_noncarbon_scenarios.log` to monitor progress.

## v0.6 threshold and gateway-specific suite

v0.6 refines the West Coast capacity threshold at 70%, 65% and 60%,
separates LA/LB from Seattle/Tacoma capacity and 72-hour delay shocks, extends
regional delays to 96, 120 and 168 hours, and tests a 50%-capacity plus
120-hour compound disruption. It contains no carbon mechanism. The baseline
is included in the same run with the same seed sequence.

Run 1,080 experiments (12 scenarios × three cases × 30 seeds):

```bash
mkdir -p logs
nohup nice -n 10 bash scripts/run_threshold_scenarios.sh \
  > logs/v06_threshold_scenarios.log 2>&1 &
echo $! > logs/v06_threshold_scenarios.pid
```

With two workers, allow roughly 2–3 hours depending on shared-server load.
After completion, create the comparison tables and paper-ready SVG figures:

```bash
python scripts/analyze_scenarios.py \
  results/v06_threshold_scenarios_30seeds \
  --output analysis/v06_threshold_scenarios
```

The analysis reports raw median changes relative to the version-matched
baseline. Hypervolume values normalized within different case/scenario groups
should not be interpreted as directly comparable effect sizes.

## v0.7 NY/NJ activation experiment

v0.7 adds a 67.5% West Coast capacity threshold, 240-hour and 336-hour
West Coast departure delays, and 30%/40% capacity plus 168-hour compound
disruptions. The six-scenario design includes a same-seed baseline and contains
no carbon mechanism. It runs 540 experiments.

```bash
mkdir -p logs
nohup nice -n 10 bash scripts/run_activation_scenarios.sh \
  > logs/v07_activation_scenarios.log 2>&1 &
echo $! > logs/v07_activation_scenarios.pid
```

After the run, generate the route and paired statistical analyses using the
commands in `docs/v07_activation_protocol.md`. The paired analysis matches
each scenario to the baseline by case and seed and reports exact sign tests,
Holm-adjusted p-values, paired Cohen's dz and deterministic 95% bootstrap
confidence intervals.


## v0.8 focused activation thresholds

v0.8 narrows the NY/NJ activation boundary for D1 and D2. It tests West
Coast departure delays of 264, 288 and 312 hours plus a 35%-capacity,
168-hour compound disruption. The same-seed baseline is included, giving 300
runs in total: 240 new scenario runs and 60 baseline controls. The experiment
contains no carbon mechanism.

```bash
mkdir -p logs
nohup nice -n 10 bash scripts/run_focused_activation_thresholds.sh \
  > logs/v08_focused_activation_thresholds.log 2>&1 &
echo $! > logs/v08_focused_activation_thresholds.pid
```

After completion, use the scenario and paired-statistics analysis commands in
`docs/v08_focused_threshold_protocol.md`. The decision rule treats NY/NJ as a
meaningful alternative gateway when its cross-seed non-dominated-union share
reaches at least 10%.

## v0.9–v1.1 argument-focused supplementary experiments

The next supplementary stage is limited to three analyses that strengthen the
paper's mechanism and managerial claims:

1. a 3 × 3 remaining-capacity by sailing-delay activation regime, including
   actual utilization diagnostics and a nominal capacity-days comparison;
2. same-seed, same-disruption counterfactuals with NY/NJ allowed versus
   disabled, which quantify the value of retaining the alternative gateway;
3. a destination-level cost and time decomposition immediately below and
   above the observed activation intervals.

Run all three stages sequentially with one `nohup` job:

```bash
mkdir -p logs
nohup nice -n 10 bash scripts/run_argument_experiments.sh \
  > logs/v09_v11_argument_experiments.log 2>&1 &
echo $! > logs/v09_v11_argument_experiments.pid
```

The full protocol, runtime estimate, monitoring commands, output paths and
result-upload commands are in `docs/v09_v11_argument_experiments.md`.

After the completed v0.9–v1.1 result folders have been pulled, build the
archive-closed counterfactual tables and all publication figures with:

```bash
bash scripts/build_revised_paper_outputs.sh
```

This command performs post-processing only; it does not run NSGA-II. It writes
the manuscript tables to `analysis/paper_revised/` and refreshes PNG, SVG and
PDF files in `paper/figures/`. The canonical experimental-design and results
texts are `paper/06_experimental_design.md` and `paper/07_results.md`, with
Chinese working versions in the corresponding `_CN.md` files.

## v1.2 final evidence-consistency validation

The final validation stage does not expand the broad scenario grid. It reruns
four targeted allowed/restricted delay comparisons (240 runs), retains all
distinct configurations found at equal non-dominated objective coordinates,
performs a seed-paired shipment-matched time decomposition with an identity
check, and validates selected cost/time endpoints with a path-allocation MILP.
The MILP includes service-departure, daily-arc, and node-day capacity and is
independently checked by the shared evaluator before the batch proceeds.

Run it only on the server after installing the optional exact dependency:

```bash
python -m pip install -e '.[analysis,exact]'
mkdir -p logs
nohup nice -n 10 bash scripts/run_final_validation.sh \
  > logs/v12_final_validation.log 2>&1 &
echo $! > logs/v12_final_validation.pid
```

Definitions, scope limits, output files, runtime bounds, and completion rules
are documented in `docs/v12_final_validation_protocol.md`.
