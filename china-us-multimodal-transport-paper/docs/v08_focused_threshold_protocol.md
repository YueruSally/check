# v0.8 focused NY/NJ activation-threshold protocol

## Purpose

Locate the disruption threshold at which NY/NJ becomes a meaningful alternative gateway for D1 and D2. A meaningful alternative is defined as an NY/NJ gateway share of at least 10% in the cross-seed non-dominated union. The model retains the separate constraint that every active shipment path carries at least 10% of that shipment demand.

This experiment contains no carbon objective, carbon cost, emissions cap or carbon variable.

## Scenarios

| Scenario | West Coast capacity | West Coast departure delay |
| --- | ---: | ---: |
| `pilot_baseline` | 100% | 0 h |
| `uswc_departure_delay_264h` | 100% | 264 h |
| `uswc_departure_delay_288h` | 100% | 288 h |
| `uswc_departure_delay_312h` | 100% | 312 h |
| `uswc_compound_35pct_delay_168h` | 35% | 168 h |

Only D1 and D2 are run because v0.7 already showed meaningful activation for D3 at 240 hours. Thirty identical seeds are used for every scenario. Four new scenarios produce 240 new optimization runs; including the same-seed baseline control gives 300 runs in the self-contained v0.8 result set.

## Fixed solver settings

- Population: 48
- Generations: 99
- Evaluations per run: 4,800
- Seeds: 20260906 through 20260935
- Workers: 2 by default
- Maximum paths per shipment: 3
- Minimum share per active path: 10%

## Run and analyse

```bash
bash scripts/run_focused_activation_thresholds.sh

python scripts/analyze_scenarios.py \
  results/v08_focused_activation_thresholds_30seeds \
  --output analysis/v08_focused_activation_thresholds

python scripts/analyze_paired_statistics.py \
  results/v08_focused_activation_thresholds_30seeds \
  --output analysis/v08_focused_activation_thresholds
```

The statistical analysis pairs scenario and baseline runs by case and seed. Report the mean and median paired differences, deterministic 95% bootstrap confidence intervals, exact sign-test p-values, Holm-adjusted p-values and paired Cohen dz.

## Decision rule

For each case, identify the first tested delay at which the NY/NJ mean gateway share is at least 10% and activation is stable across the non-dominated union. If adjacent tested points straddle 10%, report the activation interval instead of claiming an exact threshold.
