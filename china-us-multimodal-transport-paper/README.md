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
are multiplied by two. `conversion_report.json` records the source hash and all
integrity checks.

## Optimization flow

1. Validate workbook cross-references and generate FEU CSV tables.
2. Validate node, arc, service, transfer and shipment constraints.
3. Run Martins separately through each US gateway and merge unique paths.
4. Allocate each shipment across at most three paths with constrained NSGA-II.
5. Evaluate every solution with the same timetable, capacity, cost and makespan
   evaluator.

Gateway-aware path generation is intentional: a static global dominance filter
could discard the East Coast all-water option before Panama, schedule and
capacity scenarios are evaluated.

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
  --seed-start 20260906 --runs 30 --workers 4 \
  --output results/baseline_30seeds
```

Use D1 for Chicago, D2 for Memphis and D3 for Columbus. Scenario overrides use
`--scenario`, for example `--scenario panama_restricted`.

## Model coverage

The shared evaluator includes arc and transfer cost, timetable waiting and
processing cost, shipment-specific tardiness penalties, service capacity,
daily arc capacity, daily node capacity, flow conservation, split limits, leg
limits and mode-change limits. NSGA-II uses feasibility-first constrained
dominance and deterministic seeds.

## Reproducibility

The committed smoke outputs only prove that the full pipeline runs. They are
not paper findings. Follow `docs/experiment_protocol.md` for multi-seed paper
experiments and equal evaluation budgets across NSGA-II, SPEA2 and MOEA/D.
`run-experiment` writes per-run metrics, every final objective-space front,
the cross-seed non-dominated union and an experiment manifest. Hypervolume and
spacing are calculated after normalizing every run in a case/scenario group
against the same pooled ideal and nadir points.
