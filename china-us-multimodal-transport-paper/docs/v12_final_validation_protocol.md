# v1.2 final evidence-consistency validation

This is the final validation suite before rewriting the manuscript. It adds no
new broad scenario grid, carbon variable, algorithm comparison, or extra random
seeds. Its purpose is to make the existing activation, mechanism, and gateway-
value evidence internally consistent.

## What changed before rerunning

The experiment writer now preserves two separate data products:

1. unique non-dominated objective points for hypervolume, spacing, and front
   plots; and
2. all distinct route-allocation configurations *found* at those retained
   objective coordinates for gateway shares and mechanism analysis.

Exact duplicate configurations may be deduplicated across seeds, with all
source seeds retained. This is not a claim that the heuristic enumerates every
possible allocation. A separate observational archive records feasible
non-dominated configurations throughout all generations; it is never used in
tournaments or environmental selection and therefore does not alter the search
budget or selection pressure. Seed-level minimum-cost and minimum-time endpoint
allocations are also stored separately.

## Scope

The targeted NSGA-II rerun contains 240 independent runs:

| Case | Allowed/restricted conditions | Runs |
|---|---|---:|
| D1 Chicago | 288 h West Coast delay | 60 |
| D2 Memphis | 264 h and 288 h West Coast delay | 120 |
| D3 Columbus | 240 h West Coast delay | 60 |

Each run uses 48 individuals, 99 generations after initialization, and thus
4,800 objective evaluations. The seed range is 20260906--20260935. The actual
paper budget is therefore 48 × (99 + 1) = 4,800 evaluations per run; the former
250 × 200 / 50,250 statement must not appear in the revised manuscript.

The new closure is limited to these four allowed/restricted delay comparisons.
It does not turn every cell in the historical capacity-delay map into a closed
counterfactual comparison. The already observed Chicago 30%-capacity + 168 h
compound condition remains a transition under its existing closed archive and
is included in the exact endpoint checks, not in another broad NSGA-II grid.

## Activation definitions

For each condition, allowed and restricted archives are pooled and screened
again for non-dominance. All distinct found configurations attached to retained
objective coordinates are then used to calculate NY/NJ shares.

- `first_solution_activation = 1`: at least one retained found configuration
  has NY/NJ share at or above 10%;
- `front_wide_activation = 1`: every retained found configuration has NY/NJ
  share at or above 10%;
- `transition`: the first condition is met but the second is not.

"Front-wide" always means the retained best-found approximate archive, not the
unknown exact Pareto set.

## Batch-matched time decomposition

The minimum-time endpoint is paired by seed. If the restricted endpoint is
better than the independently found allowed endpoint, archive closure selects
the restricted solution because it is feasible in the allowed problem. For
each shipment, its restricted-solution mean lead time becomes the matching
benchmark. The program verifies, for every paired seed,

\[
\Delta T_{\mathrm{NY/NJ}}+\Delta T_{\mathrm{West}}+
\Delta T_{\mathrm{other}}=T^{A}-T^{R}.
\]

The sign is **allowed minus restricted**, so a negative value is an improvement.
The decomposition is shipment-matched; it is not unique tracing of each unit of
split freight. Service-departure quantity changes are output separately. A
claim about release of a specific scheduled service is allowed only when that
file supports it.

## MILP endpoint validation

Ten case-condition-access models are checked at two lexicographic endpoints,
giving 20 endpoint tasks and at most 40 solver calls. The path-allocation MILP
uses the same candidate library and includes:

- shipment demand conservation;
- at most three active paths per shipment;
- the 10% minimum share on every active path;
- fixed timetable, transfer, and scenario-delay accounting;
- service-departure capacity;
- daily arc capacity;
- node-day capacity; and
- disabled-gateway restrictions through the scenario-specific path library.

The first endpoint is a preflight. Its decoded allocation must match the shared
evaluator's cost, quantity-weighted time, feasibility, and three capacity
accounts before the remaining tasks run. Stage 2 is attempted only if stage 1
terminates optimally within the requested MIP gap. It constrains the primary
objective using an explicit absolute-plus-relative tolerance. A time-limited
incumbent is reported with its bound and gap but is not labelled a proven
lexicographic endpoint.

## Server command

Install the optional exact-solver dependency in the existing isolated
environment, then run the suite in the background:

```bash
cd ~/check/china-us-multimodal-transport-paper
git pull --ff-only origin main
conda activate chinaus311
python -m pip install -e '.[analysis,exact]'

mkdir -p logs
nohup nice -n 10 bash scripts/run_final_validation.sh \
  > logs/v12_final_validation.log 2>&1 &
echo $! > logs/v12_final_validation.pid
```

Monitor it with:

```bash
process_id="$(cat logs/v12_final_validation.pid)"
ps -p "$process_id" -o pid,etime,%cpu,%mem,cmd
tail -n 40 logs/v12_final_validation.log
```

Based on the earlier approximately 15-second NSGA-II run time, the 240 reruns
should take about 35--60 minutes with two workers; the observational archive may
add modest overhead. Post-processing should take a few minutes. The MILP stage
is data-dependent; a practical overall estimate is 1--3 hours. With the default
600-second cap on each of at most 40
solver calls, the conservative worst-case bound is about 7 hours plus the
NSGA-II stage. A solver timeout is a reportable result and does not trigger more
experiments.

## Outputs

- `results/v12_final_closed_fronts_30seeds/{D1,D2,D3}/`
  - point fronts and legacy representative files;
  - `found_configuration_solutions.csv` and
    `found_configuration_allocations.csv`;
  - `run_endpoint_solutions.csv` and `run_endpoint_allocations.csv`.
- `analysis/v12_final_validation/`
  - `closed_front_classification.csv`;
  - `closed_front_points.csv`;
  - `batch_matched_time_decomposition.csv` and its summary;
  - `service_departure_quantity_changes.csv`;
  - capacity diagnostics for all retained found configurations.
- `results/v12_exact_endpoint_validation/`
  - endpoint summaries, solver statuses/bounds/gaps, allocations, capacity
    usage, allowed-versus-restricted endpoint values, and the best 30-seed
    NSGA-II primary-objective gap against each proven MILP endpoint.

## Completion rule

The suite is complete when outputs are internally consistent and limitations
are reported. It is not necessary for Memphis to become front-wide active, for
the compound case to lose its zero-share solutions, for cost benefits to be
positive and large, or for the shipment-matched decomposition to support the
old capacity-release explanation. No additional scenario search is authorized
to make the findings look cleaner.
