# v1.3 unified exact-evidence post-processing

This step does not run NSGA-II or solve another MILP. It combines the retained
heuristic configurations with the already-proven v1.2 MILP endpoints, removes
objective records dominated by the unified evidence, and decomposes the exact
minimum-time counterfactual by shipment and scheduled service departure.

Run from the project directory in the existing `chinaus311` environment:

```bash
bash scripts/run_unified_exact_evidence.sh
```

Expected runtime is normally below one minute and should remain below three
minutes on the Linux grid. `nohup` is unnecessary because no optimization is
performed. The command writes only to:

```text
analysis/v13_unified_exact_evidence/
```

The key outputs are:

- `unified_nondominated_points.csv`: current-known objective points after
  pooling heuristic configurations and exact endpoints;
- `displaced_evidence_records.csv`: records removed by the unified dominance
  screen, including whether an exact endpoint displaced them;
- `unified_classification.csv`: NY/NJ share ranges and deliberately qualified
  current-known classifications;
- `exact_batch_matched_time_decomposition.csv`: shipment-matched decomposition
  of each proven minimum-time counterfactual;
- `exact_shipment_time_changes.csv`: batch-level time and path changes;
- `exact_service_departure_quantity_changes.csv`: changes in fixed service
  departure use; and
- `manifest.json`: tolerances, evidence scope, and interpretation limits.

For objective-space classification, coincident proven minimum-cost and
minimum-time endpoints establish a singleton Pareto objective point within the
recorded tolerance. They do not establish that the corresponding allocation
configuration is unique. Likewise, the unified set is described as the
*current-known non-dominated set* unless a complete exact interior front has
been enumerated.

After checking the output, add only this analysis directory:

```bash
git add -- analysis/v13_unified_exact_evidence
git commit -m "Add unified exact endpoint evidence analysis"
git push origin main
```
