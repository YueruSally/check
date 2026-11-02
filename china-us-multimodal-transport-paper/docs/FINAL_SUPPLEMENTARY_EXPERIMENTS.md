# Final supplementary experiments (non-carbon)

This document freezes the final validation programme. It must not be expanded
into another broad scenario grid. The objectives are to validate the main
claims, explain the source of the NY/NJ counterfactual value, and test whether
the conclusions survive alternative constructed demand instances.

## Stage 1: exact core suite

`scripts/run_supplementary_core_exact.sh` runs the three destinations under:

- the baseline and 0, 24, 72, 96, 120, 168, 240, 264, 288, 312, and 336 h
  common West Coast sailing delays;
- 60%, 40%, and 35% remaining West Coast port capacity at zero added delay;
- 30% remaining capacity plus 168 h delay.

Every condition compares allowed and restricted NY/NJ networks at the two
lexicographic endpoints. The script reuses compatible v1.2 exact endpoints,
stores full batch--path--service allocations, solves the minimum and maximum
NY/NJ share within each endpoint-equivalent objective region, decomposes cost,
records all capacity accounts, and performs the three 1.5-demand/60%-capacity
feasibility checks.

The checkpoint directory is deliberately retained. Re-running the same command
after interruption skips completed endpoints and bounded-share tasks.

Run on the Linux server from the project root:

```bash
mkdir -p logs
nohup nice -n 10 bash scripts/run_supplementary_core_exact.sh \
  > logs/v14_core_exact_suite.log 2>&1 &
echo $! > logs/v14_core_exact_suite.pid
```

Monitor with:

```bash
ps -p "$(cat logs/v14_core_exact_suite.pid)" -o pid,etime,%cpu,%mem,cmd
tail -n 30 logs/v14_core_exact_suite.log
```

The conservative expected duration is 10--30 minutes. Actual endpoint solves
may be substantially faster; path construction and detailed output generation
dominate this stage. A single solver call is capped at 900 seconds, so difficult
conditions are reported as indeterminate rather than silently labelled
infeasible.

## Interpretation rules fixed before viewing the results

- Cost and time value curves refer to different lexicographic endpoints and do
  not represent benefits attained simultaneously by one solution.
- Equal endpoint share bounds establish only the uniqueness of the aggregate
  NY/NJ share within the numerical objective region, not allocation uniqueness.
- Failure to use NY/NJ at either endpoint does not establish non-use over the
  complete Pareto front.
- The 1.5-demand feasibility result has three possible states: feasible solution
  found, proven infeasible, or indeterminate within the time limit.
- No carbon variable or carbon-related objective is introduced.

Stages 2 and 3 contain the fixed 4-vs-8 path test and the ten pre-registered
demand instances. A targeted sensitivity to explain repeated 727.5/427.5 FEU
amounts is selected only after the Stage-1 resource and shipment diagnostic is
read; capacity, timetable, due-date, and penalty scans must not all be run.

## Stages 2 and 3

The path-set check is restricted to the conditions listed in the study design.
It asserts programmatically that every four-path candidate occurs in the
corresponding eight-path library. Run it after Stage 1:

```bash
nohup nice -n 10 bash scripts/run_path_set_sensitivity.sh \
  > logs/v15_path_set_sensitivity.log 2>&1 &
echo $! > logs/v15_path_set_sensitivity.pid
```

The ten demand instances use seeds 20261101--20261110. Batch quantities are
perturbed and rescaled to the original total; batch count, release times,
destination, and tardiness rates remain fixed. Due allowances receive centred
jitter from -48 to +48 h. The generator, numerical tolerances, material-effect
thresholds, and mutually exclusive classification order are written to the
result manifest before any interpretation. Run with:

```bash
nohup nice -n 10 bash scripts/run_demand_instance_robustness.sh \
  > logs/v16_demand_robustness.log 2>&1 &
echo $! > logs/v16_demand_robustness.pid
```

Conservative expected durations are 10--30 minutes for Stage 2 and 30--90
minutes for Stage 3. To run all three fixed stages in one resumable background
job, use:

```bash
mkdir -p logs
nohup nice -n 10 bash scripts/run_fixed_supplementary_experiments.sh \
  > logs/final_supplementary_experiments.log 2>&1 &
echo $! > logs/final_supplementary_experiments.pid
```

The combined conservative estimate is 1--3 hours. Checkpoints make the same
command safe to rerun. The targeted driver sensitivity is intentionally not in
the combined script: it is selected only after the v1.4 diagnostic identifies
capacity, timetable connection, or due-date/penalty structure as the dominant
driver.

## Evidence-triggered eight-path revalidation

The completed Stage-2 check added four Seattle/Tacoma paths per shipment and
changed gateway use at several core endpoints. The four-path library therefore
cannot be treated as the final evidentiary model. This is a failed robustness
check, not a reason to add a broader scenario grid. The corrective action is to
repeat the already frozen exact core conditions and the same ten demand
instances with eight paths per gateway:

```bash
mkdir -p logs
nohup nice -n 10 bash scripts/run_eight_path_revalidation.sh \
  > logs/eight_path_revalidation.log 2>&1 &
echo $! > logs/eight_path_revalidation.pid
```

Monitor with:

```bash
ps -p "$(cat logs/eight_path_revalidation.pid)" \
  -o pid,etime,%cpu,%mem,cmd
tail -n 40 logs/eight_path_revalidation.log
```

The observed solver times in v1.4--v1.6 imply an expected duration of roughly
5--15 minutes on the Linux server; allow up to 30 minutes for path construction,
analysis, and figure rendering. The job writes the corrected core evidence to
`results/v17_core_exact_paths8`, the matched analysis to
`analysis/v17_core_exact_paths8`, and the ten-instance results to
`results/v18_demand_robustness_paths8`.

The repeated 727.5 and 427.5 FEU quantities in the four-path run coincide with
binding West Coast resources, not binding NY/NJ resources. A separate capacity
sensitivity is deferred until the eight-path revalidation establishes which
repeated quantities remain in the corrected candidate library. No timetable,
due-date, penalty, or NY/NJ-capacity scan should be run in parallel.
