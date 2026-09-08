# v0.7 activation experiment protocol

## Research purpose

The v0.7 experiment estimates the West Coast capacity threshold more precisely
and identifies the disruption level at which NY/NJ becomes a meaningful
alternative gateway. It retains the two objectives of total transport cost and
quantity-weighted mean delivery time. It contains no carbon variable, carbon
price, emissions objective or emissions constraint.

## Scenario matrix

| Scenario | West Coast capacity | West Coast departure delay |
| --- | ---: | ---: |
| `pilot_baseline` | 100% | 0 h |
| `uswc_port_capacity_675pct` | 67.5% | 0 h |
| `uswc_departure_delay_240h` | 100% | 240 h |
| `uswc_departure_delay_336h` | 100% | 336 h |
| `uswc_compound_30pct_delay_168h` | 30% | 168 h |
| `uswc_compound_40pct_delay_168h` | 40% | 168 h |

Each scenario is run for D1, D2 and D3 using the same 30 seeds. Population size
is 48 and generations are 99, giving 4,800 evaluations per independent run.
The complete design therefore contains 540 runs.

## Constraints retained

Timetables, sailing and daily node capacities, transfer rules, split limits,
leg and mode-change limits remain active. Every active path must carry at least
10% of its shipment's scenario-adjusted demand.

## Paired statistical analysis

Scenario outcomes are paired to the baseline by `case_id` and `seed`.
The analysis refuses to continue if a scenario has a missing or extra feasible
seed. It evaluates the per-run minimum cost and per-run minimum
quantity-weighted delivery time.

For each case, scenario and metric, the analysis reports:

- mean and median within-seed differences and percentage differences;
- a two-sided exact sign test, omitting exact ties;
- Holm-adjusted p-values across all reported comparisons;
- paired Cohen's dz;
- deterministic 95% percentile-bootstrap confidence intervals for the mean
  and median paired differences using 10,000 resamples;
- the complete seed-level paired differences.

The sign test is distribution-free but tests direction rather than difference
magnitude. Effect sizes and confidence intervals should therefore be reported
alongside p-values.

## Execution

Run the suite from the project root:

```bash
mkdir -p logs
nohup nice -n 10 bash scripts/run_activation_scenarios.sh \
  > logs/v07_activation_scenarios.log 2>&1 &
echo $! > logs/v07_activation_scenarios.pid
```

After completion, create route-allocation summaries and paired statistics:

```bash
python scripts/analyze_scenarios.py \
  results/v07_activation_scenarios_30seeds \
  --output analysis/v07_activation_scenarios

python scripts/analyze_paired_statistics.py \
  results/v07_activation_scenarios_30seeds \
  --output analysis/v07_activation_scenarios
```

The first command creates scenario effects, gateway shares, NY/NJ activation
statistics and SVG figures. The second creates `paired_statistics.csv`,
`paired_differences.csv` and `paired_statistics_manifest.json`.
