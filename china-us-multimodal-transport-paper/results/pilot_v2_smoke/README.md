# Pilot v2 NSGA-II smoke run

This deterministic run verifies data loading, gateway-aware Martins path
generation, constrained NSGA-II, timetable simulation and result export for all
three destination cases.

Parameters: population 48, generations 40, seed 20260906, pilot baseline
scenario. Every final population contained 48 feasible individuals.

| Case | Destination | Feasible Pareto points | Lowest observed cost (USD) | Lowest observed makespan (h) |
|---|---|---:|---:|---:|
| D1 | Chicago | 1 | 26,913,791.00 | 748.8 |
| D2 | Memphis | 1 | 25,903,610.20 | 760.8 |
| D3 | Columbus | 2 | 27,538,620.92 | 772.8 |

These are pipeline-validation outputs, not paper results. The paper experiment
must use the multi-seed protocol in `docs/experiment_protocol.md`.
