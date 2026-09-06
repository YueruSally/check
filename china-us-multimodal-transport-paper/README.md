# China–US Scheduled Multimodal Freight Routing

This project studies scheduled multi-batch freight routing between China and the United States. The pilot compares West Coast landbridge routes with East Coast all-water routes.

## Research design

- Objectives: minimize total cost and makespan.
- Candidate paths: Martins bi-objective label-setting algorithm.
- Allocation optimization: improved NSGA-II is the planned primary algorithm; SPEA2 and MOEA/D will be retained as benchmarks.
- Main scenarios: Panama service restrictions and the US West Coast–East Coast ocean freight-rate differential.
- Preferred inland destinations: Chicago, Memphis and Columbus, subject to completion of solver-ready network data.

## Current implementation

The first model layer is now in place:

- typed nodes, arcs, recurring services, shipments and transfer rules;
- strict schema and network validation;
- China-to-US stage-sequence validation;
- recurring timetable and cutoff handling;
- service/voyage, daily arc and daily node capacity checks;
- split-allocation and path-complexity constraints;
- soft or hard due-date treatment;
- one shared cost–makespan evaluator for every algorithm;
- Martins candidate-path generation;
- configurable Panama, ocean-rate, port-capacity and demand scenarios.

No optimizer result should be generated until the CSV network tables pass validation. The supplied Excel workbook is a calibration/evidence workbook, not a complete node–arc–service instance.

## Repository structure

```text
configs/                   model constraints and scenario values
data/source/               original evidence workbook
data/templates/            solver-ready CSV schemas
src/china_us_multimodal/   shared model, validation and evaluation code
tests/                     unit tests for timetable and evaluation logic
```

## Quantity and cost convention

The pilot uses FEU throughout: one FEU is one 40-foot-equivalent container. Source values expressed in TEU must be converted explicitly before loading. Ocean rates from the workbook are already USD per 40-foot container and therefore map directly to USD/FEU.

## Install and validate

```bash
python -m pip install -e ".[dev]"
china-us-model validate-evidence data/source/ChinaUS_Gateway_Data.xlsx
china-us-model validate-network data/templates --config configs/baseline.toml
pytest
```

The template network intentionally contains headers only. `validate-network` must fail until the missing network data is populated; this prevents missing costs, times or capacities from silently becoming zero.

## Constraint policy

The baseline enables timetable and capacity constraints and allows at most three paths per shipment. Due dates are soft by default, but tardiness remains explicitly reported. Each ablation must change exactly one model mechanism while keeping algorithms, operators, seeds and evaluation budgets fixed.

The Panama multiplier is applied only to services on arcs marked `via_panama=true`. It represents effective service availability, not the physical TEU capacity of the canal. The 0.8 case is a hypothetical stress test.

## Next implementation step

Populate the five CSV tables with a complete pilot network, then verify hand-calculated routes before connecting improved NSGA-II, SPEA2 and MOEA/D to the shared evaluator.
