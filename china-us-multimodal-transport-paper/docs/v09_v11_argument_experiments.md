# v0.9–v1.1 Argument-Focused Supplementary Experiments

These experiments contain no carbon variables. They are deliberately limited
to analyses that advance the paper's causal and managerial argument.

## Research logic

### v0.9: nominal capacity-days × sailing delay

The first suite tests a 3 × 3 grid:

- remaining combined US West Coast port capacity: 60%, 40%, 35%;
- common West Coast ocean departure delay: 0, 168, 288 h.

It also compares two scenarios with the same nominal demand/capacity pressure:
40% capacity with baseline demand versus 50% capacity with demand multiplied by
1.25. This checks whether comparable demand relative to daily capacity, rather
than the capacity percentage in isolation, reproduces gateway allocation. The
ratio has units of capacity-days and is not a dimensionless utilization rate.
The analysis reconstructs actual daily node,
arc and service utilization for representative solutions.

The historical primary output is `activation_regime.csv`. The revised paper
analysis uses `analysis/paper_revised/activation_classification.csv`, where
archives are classified as no meaningful activation, transition, or front-wide
activation using all pooled-front solutions, including zero-share solutions.

### v1.0: value of access to NY/NJ

For each destination, the model compares NY/NJ allowed and NY/NJ disabled under
the same disruption and the same 30 seeds. The allowed runs are reused from
v0.7/v0.8; only the 180 disabled-gateway runs are new:

- D1: 288 h West Coast delay;
- D2: 264 h West Coast delay;
- D3: 240 h West Coast delay;
- all three cases: 30% remaining West Coast capacity plus 168 h delay.

Positive paired differences mean that disabling NY/NJ worsens the objective,
so the difference is the loss avoided by retaining access to the alternative
gateway. The analysis also reports whether NY/NJ access changes the number of
feasible runs.

The manuscript-facing post-processing closes each allowed archive with its
matching restricted archive before objective comparison. This enforces the
known feasible-set inclusion relation and prevents independent heuristic
search error from being misinterpreted as a negative value of gateway access.

### v1.1: destination asymmetry mechanism

This is an analysis of already completed v0.6–v0.8 outputs; it does not rerun
NSGA-II. For balanced solutions immediately below and above each destination's
activation interval, it decomposes gateway-level cost and time into:

- China inland transport;
- ocean transport;
- US inland transport;
- waiting and holding;
- transfer and processing;
- tardiness.

This table is intended to explain why Columbus, Memphis and Chicago activate
NY/NJ at different delay levels.

## One-command background run

Run from the project folder after pulling the latest `main` branch:

```bash
mkdir -p logs

nohup nice -n 10 bash scripts/run_argument_experiments.sh \
  > logs/v09_v11_argument_experiments.log 2>&1 &

echo $! > logs/v09_v11_argument_experiments.pid
```

The default configuration is two workers, 48 individuals, 99 generations and
30 paired seeds. Do not launch a second copy while this process is running.

If the earlier v0.9 run completed all 990 optimizations but stopped because the
obsolete `demand_150pct_uswc_capacity_60pct` case was infeasible, do not rerun
v0.9. Pull the current branch and use the recovery entry point:

```bash
nohup nice -n 10 bash scripts/resume_argument_experiments_after_v09.sh \
  > logs/v09_v11_recovery.log 2>&1 &
echo $! > logs/v09_v11_recovery.pid
```

This runs only 90 replacement load-pressure experiments, rebuilds the v0.9
analysis without the structurally infeasible case, and continues v1.0–v1.1.

## Expected runtime

Existing v0.6–v0.8 records average 12.84–13.93 seconds per independent run on
the same server configuration.

- v0.9: 990 runs, about 1 h 52 min of ideal two-worker wall time; allow
  **2–2.75 hours** under shared-server load and analysis overhead.
- v1.0: 180 new runs; allow **18–35 minutes** because the disabled-gateway
  search space is smaller but server load varies.
- v1.1: analysis only, normally **under 2 minutes**.
- complete sequential command: allow approximately **2.3–3.4 hours**.

## Monitoring

```bash
project=~/check/china-us-multimodal-transport-paper
cd "$project"

process_id="$(cat logs/v09_v11_argument_experiments.pid)"
ps -p "$process_id" -o pid,etime,%cpu,%mem,cmd
tail -n 30 logs/v09_v11_argument_experiments.log
```

During v0.9 the log counter advances to `[990/990]`. During v1.0 it restarts
for each destination and advances to `[60/60]` three times. Completion is
confirmed only when the final log line is:

```text
All argument-focused supplementary experiments and analyses completed.
```

## Output locations

- `results/v09_activation_regime_30seeds/`
- `analysis/v09_activation_regime/`
- `results/v10_ny_nj_counterfactual_30seeds/D1/`
- `results/v10_ny_nj_counterfactual_30seeds/D2/`
- `results/v10_ny_nj_counterfactual_30seeds/D3/`
- `analysis/v10_ny_nj_counterfactual/`
- `analysis/v11_destination_asymmetry/`

The most important new paper tables are:

- `analysis/v09_activation_regime/activation_regime.csv`;
- `analysis/v09_activation_regime/capacity_utilization_summary.csv`;
- `analysis/v10_ny_nj_counterfactual/ny_nj_value_summary.csv`;
- `analysis/v11_destination_asymmetry/gateway_mechanism_components.csv`.

After the v0.9–v1.1 outputs have been uploaded, regenerate the corrected paper
tables and figures with:

```bash
bash scripts/build_revised_paper_outputs.sh
```

This is post-processing only and normally completes in under one minute. It
writes `analysis/paper_revised/` and refreshes `paper/figures/`.

## Uploading completed results

The repository ignores generated result folders by default, so add only these
new result and analysis directories explicitly:

```bash
git add -f -- \
  results/v09_activation_regime_30seeds \
  results/v10_ny_nj_counterfactual_30seeds

git add -- \
  analysis/v09_activation_regime \
  analysis/v10_ny_nj_counterfactual \
  analysis/v11_destination_asymmetry

git commit -m "Add argument-focused gateway activation experiments"
git push origin main
```

Do not add `logs/` or `src/china_us_multimodal.egg-info/`.

## Deliberately excluded

The following are not part of this supplementary run because their main role
would be defensive rather than advancing the paper's argument:

- exact MILP versus NSGA-II benchmarking;
- candidate-path count sensitivity;
- broad one-at-a-time price, penalty and headway sweeps;
- additional random-seed counts;
- dynamic recovery and rollover modelling.

The 5%/10%/20% classification is now included as post-processing of the
existing pooled archives. It does not add optimization runs and is used only
to check whether the destination ordering depends on the reporting criterion.
