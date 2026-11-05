# Figure sources and reproducibility

The figures in this directory are generated from stored experimental outputs. The rendering workflow does not execute the routing model or NSGA-II.

After pulling the final validation results, run from the project directory:

```bash
python paper/render_figures.py
```

The renderer does not run an optimization model. It writes 300 dpi PNG, SVG,
and PDF versions from the stored result and analysis CSV files.

| Figure | Main source files | Interpretation |
| --- | --- | --- |
| 1 | baseline/compound archives plus v1.2 targeted `combined_pareto.csv` | Discrete cost–time alternatives under baseline and severe disruption |
| 2 | historical activation tables plus `analysis/v12_final_validation/closed_front_classification.csv` | Full retained-configuration NYNJ share ranges; ranges are not confidence intervals |
| 3 | `analysis/paper_revised/activation_classification.csv` | Exploratory allowed-search capacity–delay regime map |
| 4 | `analysis/v12_final_validation/closed_front_points.csv` and v1.2 restricted fronts | Configuration-preserving closed approximate fronts |
| 5 | `results/v12_exact_endpoint_validation/exact_counterfactual_value.csv` | Proven lexicographic endpoint differences |
| 6 | `analysis/v12_final_validation/batch_matched_time_decomposition_summary.csv` | Seed-paired shipment-matched time contributions |
| S1 | historical activation tables plus the v1.2 Chicago correction | Compound-scenario ranges, with unclosed cells treated as exploratory |

The figure captions and inferential limits are stated in `paper/07_results.md` and `paper/07_results_CN.md`.
