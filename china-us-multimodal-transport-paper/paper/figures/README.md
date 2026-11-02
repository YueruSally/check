# Figure sources and reproducibility

The figures in this directory are generated from stored experimental outputs. The rendering workflow does not execute the routing model or NSGA-II.

Run from the project directory:

```bash
bash scripts/build_revised_paper_outputs.sh
```

The command first builds manuscript-facing CSV tables in `analysis/paper_revised/` and then writes 300 dpi PNG, SVG, and PDF versions of each figure.

| Figure | Main source files | Interpretation |
| --- | --- | --- |
| 1 | v0.7/v0.8 `combined_pareto.csv` | Discrete cost–time alternatives under baseline and severe disruption |
| 2 | v0.6–v0.8 `ny_activation.csv` and `gateway_shares.csv` | Balanced solution plus full pooled-front NY/NJ share range; ranges are not confidence intervals |
| 3 | `analysis/paper_revised/activation_classification.csv` | Categorical capacity–delay regime based on the full pooled-front range |
| 4 | `analysis/paper_revised/counterfactual_front_points.csv` | Common comparison of archive-closed allowed and restricted fronts |
| 5 | `analysis/paper_revised/counterfactual_value_summary.csv` | Paired archive-closed endpoint benefits and bootstrap intervals |
| 6 | `analysis/paper_revised/destination_mechanism_summary.csv` | Descriptive gateway-component differences in balanced solutions |
| S1 | v0.7/v0.8 activation and gateway-share tables | Full pooled-front ranges under 168-hour compound disruption |

The figure captions and inferential limits are stated in `paper/07_results.md` and `paper/07_results_CN.md`.
