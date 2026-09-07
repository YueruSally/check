# Pilot experiment protocol

## Experimental unit

Run each destination independently:

- D1: Xi'an–Chicago, 40 batches;
- D2: Xi'an–Memphis, 40 batches;
- D3: Xi'an–Columbus, 40 batches.

All algorithms must use the same FEU data, gateway-aware Martins path library,
shared timetable/capacity evaluator and evaluation budget.

## Objectives and constraints

Minimize total USD cost and quantity-weighted mean delivery time in hours. Total cost includes arc cost,
transfer cost, schedule-wait holding cost, transfer-processing time cost and
shipment-specific tardiness penalties. Timetable, service capacity, daily arc
capacity, daily node capacity, flow conservation, maximum path count, maximum
leg count and maximum mode-change count are enforced.

Quantity-weighted mean delivery time equals the sum of each route allocation's
FEU quantity multiplied by its release-to-arrival lead time, divided by total
allocated FEU.

NSGA-II uses feasibility-first constrained dominance. Feasible individuals
dominate infeasible individuals; among infeasible individuals, lower aggregate
constraint violation is preferred.

## Candidate-path policy

Run Martins separately for Los Angeles/Long Beach, Seattle/Tacoma and New
York/New Jersey, then merge unique paths. This prevents a globally dominated
all-water gateway path from disappearing before schedule, capacity and scenario
effects are evaluated.

## Reproducibility

For paper experiments, use at least 30 independent seeds per case and scenario.
Record the source workbook SHA-256, configuration file, seed, population size,
generations, wall-clock time and full Pareto front. Keep one-factor ablations
and comparator algorithms on the same total number of solution evaluations.

The committed smoke run only verifies that the pipeline is executable. It is
not evidence for a research conclusion.

## Multi-seed output contract

The batch runner fixes the seed sequence before optimization and writes five
compact artifacts:

- `runs.csv`: feasibility, extrema, normalized hypervolume, spacing and time
  for every independent run;
- `fronts.csv`: every final feasible non-dominated objective point;
- `aggregate.csv`: median, quartiles and success rate across seeds;
- `combined_pareto.csv`: the non-dominated union over all seeds;
- `representative_solutions.csv`: roles and objective values for the
  cross-seed union front;
- `representative_allocations.csv`: shipment quantities and arc sequences for
  those representative solutions;
- `manifest.json`: source/config hashes, evaluation budget and seed list.

Within each case/scenario group, all runs share pooled ideal and nadir values
and the normalized hypervolume reference point (1.1, 1.1). For algorithm
comparisons, recompute indicators using bounds pooled across every compared
algorithm; do not compare separately normalized hypervolumes.


## Convergence calibration

Use a small fixed-seed calibration before scenario and algorithm comparisons.
Compare population/generation shapes at the same evaluation budget:

- 48 individuals × (99 + 1) populations = 4,800 evaluations;
- 96 individuals × (49 + 1) populations = 4,800 evaluations.

Run ten common seeds for D1-D3, select the shape using feasible rate,
cross-seed union front, shared-bound hypervolume and objective stability, then
rerun the selected configuration with 30 seeds. The calibration results are
parameter-selection evidence, not scenario findings.
