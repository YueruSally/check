# 7. Results

## 7.1 Computational outcomes and baseline gateway structure

Every case--scenario combination retained for the main analysis produced feasible solutions in all 30 NSGA-II runs. The pooled baseline archives contained 84 solutions for Chicago, 68 for Memphis, and 71 for Columbus; none routed freight through New York/New Jersey (NY/NJ). Baseline cost--time alternatives were therefore supported entirely by Los Angeles/Long Beach and Seattle/Tacoma.

The baseline archives spanned USD 24.84--26.77 million and 615.64--660.47 h for Chicago, USD 25.04--26.64 million and 614.22--644.22 h for Memphis, and USD 26.58--28.61 million and 628.06--661.82 h for Columbus. These are ranges across retained search solutions, not statistical intervals. A diagnostic combining 1.5 times demand with 60% remaining West Coast capacity produced no feasible population for any destination and was excluded from the activation analysis; the reported activation experiments vary disruption conditions at the original demand.

Figure 1 contrasts each baseline search archive with the current-known non-dominated points after the heuristic evidence was pooled with proven mixed-integer linear programming (MILP) endpoints. The disruption conditions are 288 h for Chicago and Memphis and 240 h for Columbus. The exact-evidence points are discrete; no continuous frontier is inferred between distinct endpoints.

![Baseline and disruption objective values](figures/fig01_cost_time_fronts.png)

Figure 1. Baseline NSGA-II archives and current-known non-dominated objective points under the selected delay disruptions. The targeted disruption points incorporate the proven MILP endpoints and exclude historical heuristic points subsequently shown to be dominated.

## 7.2 West Coast reallocation accommodated capacity-only reductions

Capacity-only disruptions did not generate operationally material NY/NJ use. NY/NJ remained absent when regional West Coast capacity was reduced to 70%, 67.5%, 65%, 60%, 40%, or 35% of baseline, and when either West Coast gateway was individually reduced to 50%.

This result cannot be attributed to all tested capacity constraints being slack. Table 2 reports the maximum West Coast node-day utilization in one balanced retained solution per scenario. At 60%, 40%, and 35% remaining capacity, at least one node-day reached 97.89%--99.99% utilization while the NY/NJ share remained zero. Thus, under the tested demand and service structure, the solutions accommodated capacity reductions through West Coast allocation and timing changes. The diagnostic does not imply that every West Coast resource or every non-dominated solution was saturated.

Table 2. Maximum West Coast node-day utilization in balanced capacity-only solutions.

| Destination | Baseline (%) | 60% capacity (%) | 40% capacity (%) | 35% capacity (%) | NY/NJ share (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chicago | 57.68 | 97.89 | 99.86 | 99.31 | 0 |
| Memphis | 70.21 | 99.99 | 99.40 | 99.79 | 0 |
| Columbus | 66.86 | 99.79 | 99.90 | 99.91 | 0 |

The capacity-days comparison provides supporting rather than causal evidence. Baseline demand at 40% capacity and 1.25 times demand at 50% capacity both required 0.928 nominal capacity-days. Their balanced Los Angeles/Long Beach shares differed by less than 0.10 percentage points for every destination and neither condition used NY/NJ. Because other service capacities were not scaled proportionally, this comparison is not used to claim that demand divided by port capacity is the sole driver.

## 7.3 Exact endpoints changed the inferred activation structure

Regional delays of 96 and 120 h produced no NY/NJ use in the exploratory archives. At 168 h, Chicago still used none; Memphis reached only 0.28% in one retained solution, and Columbus reached 0.37%--1.14% in five solutions. All were below the 10% material-use criterion. The first tested conditions containing a material current-known configuration were therefore bracketed by 264--288 h for Chicago, 240--264 h for Memphis, and 168--240 h for Columbus (Figure 2). These brackets are conditional on the tested delay grid and are not estimates of universal switching times.

![Destination-specific delay activation](figures/fig02_delay_activation.png)

Figure 2. NY/NJ shares under regional West Coast sailing delays. At the four validated delay conditions, vertical segments report the range among the current-known non-dominated configurations after exact endpoints were added; other delay points retain their exploratory allowed-search scope. Segments are configuration ranges, not confidence intervals. The dashed line is the 10% material-use criterion.

Integrating the exact endpoints materially changed the structure inferred from the heuristic archives. For every validated condition, all historical heuristic points were dominated by at least one proven exact endpoint (Table 3). Chicago at 288 h and the Chicago compound condition each retained two distinct exact endpoint coordinates, but the exact interior between them was not enumerated. Memphis at 264 h also retained two coordinates: the minimum-cost coordinate contained only zero-NY/NJ configurations, whereas the minimum-time coordinate used 32.66%. The condition is therefore a preference-dependent transition.

For Memphis at 288 h and Columbus at 240 h, the proven minimum-cost and minimum-time solutions attained the same objective coordinates within the recorded tolerances. Their candidate-path MILP Pareto fronts consequently degenerated to one objective point. This does not prove allocation uniqueness: different configurations can attain the same objectives, and the exhaustive set of objective-equivalent configurations was not enumerated. Both exact solutions found at each singleton point used 32.66% NY/NJ.

Table 3. Unified current-known non-dominated evidence after exact endpoints were added.

| Destination | Disruption | Displaced heuristic points | Current-known objective points | NY/NJ share range among retained found configurations (%) | Exact objective-space result |
| --- | --- | ---: | ---: | ---: | --- |
| Chicago | 288 h delay | 24 | 2 | 32.66--64.89 | Distinct exact endpoints; interior not enumerated |
| Memphis | 264 h delay | 26 | 2 | 0--32.66 | Preference-dependent transition |
| Memphis | 288 h delay | 12 | 1 | 32.66 | Proven singleton objective front within tolerance |
| Columbus | 240 h delay | 11 | 1 | 32.66 | Proven singleton objective front within tolerance |
| Chicago | 30% capacity + 168 h | 5 | 2 | 19.19 | Distinct exact endpoints; interior not enumerated |

Figure 3 makes the evidence correction explicit. Hollow grey points are historical heuristic records that are no longer non-dominated after exact solutions are admitted. The coloured and black markers are proven exact endpoints. In particular, the five historical Chicago compound points--including the two zero-NY/NJ points previously used to classify the condition as a transition--were all dominated and therefore cannot support that interpretation. The two retained exact compound endpoints both used 19.19% NY/NJ, although universal use over an unenumerated exact interior is not claimed.

![Unified heuristic and exact endpoint evidence](figures/fig04_counterfactual_fronts.png)

Figure 3. Historical heuristic points and proven lexicographic MILP endpoints. Hollow grey circles are superseded heuristic points; coloured circles and diamonds are the exact NY/NJ-allowed minimum-cost and minimum-time endpoints; black crosses are exact NY/NJ-restricted endpoints. Overlapping allowed endpoint markers identify singleton objective fronts for Memphis at 288 h and Columbus at 240 h. No line between endpoints is interpreted as an enumerated exact frontier.

The unified evidence therefore does not support the former claim of a complete destination ordering in front-wide activation. It supports a narrower result: the first material configuration appeared at different tested delays across destinations, and Memphis at 264 h exhibits a proven cost--time preference split. Statements about all Pareto-optimal configurations are avoided unless the exact objective front itself is proven to be a singleton, and even then configuration uniqueness is not inferred.

## 7.4 Exact counterfactual endpoints quantify the value of NY/NJ access

All allowed and NY/NJ-restricted counterfactual models were feasible, so NY/NJ was not required to preserve feasibility in the tested conditions. Both lexicographic MILP stages were proven for all 20 endpoint tasks. Every decoded allocation matched the independent evaluator and the service-departure, arc-day, and node-day capacity accounts within numerical tolerance.

Table 4 reports restricted minus allowed outcomes. Positive values indicate an avoided cost or time loss from retaining NY/NJ access; a negative cost value means that the faster NY/NJ-allowed solution incurred additional cost. Cost and time differences in the same row refer to one lexicographically selected allocation, whereas differences across endpoint rows need not be jointly attainable by another solution.

Table 4. Exact value of NY/NJ access at lexicographic endpoints of the candidate-path MILP.

| Destination | Disruption | Endpoint | NY/NJ share (%) | Restricted minus allowed cost (USD million) | Restricted minus allowed time (h) |
| --- | --- | --- | ---: | ---: | ---: |
| Chicago | 288 h delay | Minimum cost, then time | 64.89 | +1.732 | +5.030 |
| Chicago | 288 h delay | Minimum time, then cost | 32.66 | +1.497 | +9.994 |
| Chicago | 30% capacity + 168 h | Minimum cost, then time | 19.19 | +0.309 | +3.723 |
| Chicago | 30% capacity + 168 h | Minimum time, then cost | 19.19 | +0.309 | +3.723 |
| Memphis | 264 h delay | Minimum cost, then time | 0 | 0.000 | 0.000 |
| Memphis | 264 h delay | Minimum time, then cost | 32.66 | -0.797 | +2.939 |
| Memphis | 288 h delay | Minimum cost, then time | 32.66 | +1.407 | +13.913 |
| Memphis | 288 h delay | Minimum time, then cost | 32.66 | +1.407 | +13.913 |
| Columbus | 240 h delay | Minimum cost, then time | 32.66 | +0.832 | +3.070 |
| Columbus | 240 h delay | Minimum time, then cost | 32.66 | +0.832 | +3.070 |

Memphis at 264 h provides the clearest preference-dependent case. The exact minimum-cost solution matched the restricted objective coordinates and used no NY/NJ; allocation uniqueness at those coordinates was not assumed. The exact minimum-time solution assigned 32.66% to NY/NJ, reduced mean delivery time by 2.939 h, and increased cost by USD 0.797 million. NY/NJ therefore offered a paid time-improvement option rather than a required or cost-minimizing route.

At Memphis 288 h and Columbus 240 h, the coincident exact endpoints improved both objectives relative to their restricted counterparts: USD 1.407 million and 13.913 h for Memphis, and USD 0.832 million and 3.070 h for Columbus. Chicago at 288 h retained a genuine exact cost--time conflict, but NY/NJ access improved both corresponding counterfactual endpoints. The compound Chicago condition also retained two objective points; both used 19.19% NY/NJ and produced the same endpoint-specific avoided losses of USD 0.309 million and 3.723 h.

The exact solutions also reveal the limits of the heuristic archives. Across the validated access models, the best 30-seed NSGA-II primary-objective gaps were 0.516%--4.338% for minimum cost and below 0.258% for minimum time. More importantly, every targeted historical heuristic point was displaced in the unified dominance screen. The heuristic results are therefore retained as exploratory search evidence, not as the final description of the targeted Pareto structures.

## 7.5 Exact shipment-matched accounting identifies the time-improvement mechanism

The exact minimum-time allocation for each allowed case was compared with its exact restricted counterpart using the same shipment batches. Table 5 reports allowed minus restricted mean delivery time, so negative values denote improvement. The NY/NJ component compares each quantity assigned to NY/NJ in the allowed solution with that shipment's restricted-solution mean; the West Coast component is defined analogously. The two components summed to the system change with absolute identity error below \(1.8\times10^{-13}\) h, and the residual “other” component was zero.

Table 5. Exact shipment-matched decomposition of the minimum-time counterfactual, hours.

| Destination | Disruption | System change | NY/NJ-assigned component | West Coast-assigned component | Share of time improvement attributed to NY/NJ-assigned freight (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Chicago | 288 h delay | -9.994 | -9.730 | -0.264 | 97.36 |
| Memphis | 264 h delay | -2.939 | -2.582 | -0.357 | 87.84 |
| Memphis | 288 h delay | -13.913 | -13.501 | -0.412 | 97.04 |
| Columbus | 240 h delay | -3.070 | -2.727 | -0.343 | 88.83 |
| Chicago | 30% capacity + 168 h | -3.723 | -3.569 | -0.154 | 95.85 |

![Exact shipment-matched time decomposition](figures/fig06_destination_mechanism.png)

Figure 4. Exact shipment-matched contributions to the minimum-time difference. Bars show NY/NJ- and West Coast-assigned components, and diamonds show their system total. Negative values favour allowing NY/NJ. These are deterministic differences between proven candidate-path MILP endpoints, not confidence intervals for real-world effects.

Contrary to the earlier heuristic interpretation, the exact decomposition did not show a slower NY/NJ component being offset by large gains for West Coast freight. The NY/NJ-assigned component was favourable in all five comparisons and accounted for 87.84%--97.36% of the total time improvement. West Coast-assigned freight contributed the remaining 2.64%--12.16%. Thus, within the exact minimum-time solutions, most of the benefit arose directly because the shipments reassigned to the alternative gateway achieved lower delivery times than the same shipments' average outcome when NY/NJ was unavailable.

The service-departure audit is consistent with this allocation change. In each delay-only case, 727.5 FEU moved from the scheduled Shanghai--Los Angeles/Long Beach ocean departure to the Shanghai--NY/NJ departure; in the Chicago compound case, 427.5 FEU moved from Shanghai--Seattle/Tacoma to Shanghai--NY/NJ. The number of shipment batches whose path set changed was 27 for Chicago at 288 h, 31 for Memphis at 264 h, 28 for Memphis at 288 h, 28 for Columbus at 240 h, and 15 for the Chicago compound condition. These records establish path and fixed-departure reallocation. They do not establish a unique freight-unit lineage for split batches and are not described as endogenous congestion relief because the model contains no port-queue process.

## 7.6 Consolidated findings and evidence boundaries

Three findings remain after the exact endpoints are integrated into the same evidence set as the heuristic configurations. First, capacity reduction and sailing delay generated asymmetric gateway responses under the tested network: capacity-only reductions were accommodated through West Coast reallocation even near binding node-day limits, whereas sufficiently long delays made cross-coast substitution competitive. Second, the value of NY/NJ access depended on both destination and objective preference. Memphis at 264 h exhibited a proven paid time-improvement option, while Memphis at 288 h and Columbus at 240 h had a common solution point minimizing both objectives within tolerance. Third, exact shipment-matched accounting attributed most of the minimum-time benefit to the freight reassigned to NY/NJ itself, with smaller accompanying improvements among West Coast-assigned freight.

These results are conditional on the constructed case-study network, the generated candidate paths, and the tested discrete disruption conditions. The delay brackets are not universal empirical switching thresholds. Thirty seeds quantify stochastic-search variation rather than input uncertainty. Exactness is established for the candidate-path MILP endpoints, not for paths absent from the generated set. Where the two exact endpoints differ, the exact interior was not enumerated; where they coincide, the objective-space front is a singleton within tolerance but allocation uniqueness remains unproven. Accordingly, the manuscript reports current-known configuration ranges and endpoint-specific value rather than claiming universal front-wide gateway adoption.

## Source tables

- `analysis/v13_unified_exact_evidence/unified_nondominated_points.csv`
- `analysis/v13_unified_exact_evidence/displaced_evidence_records.csv`
- `analysis/v13_unified_exact_evidence/unified_classification.csv`
- `analysis/v13_unified_exact_evidence/exact_batch_matched_time_decomposition.csv`
- `analysis/v13_unified_exact_evidence/exact_shipment_time_changes.csv`
- `analysis/v13_unified_exact_evidence/exact_service_departure_quantity_changes.csv`
- `analysis/paper_revised/capacity_saturation_summary.csv`
- `results/v12_exact_endpoint_validation/exact_endpoint_summary.csv`
- `results/v12_exact_endpoint_validation/exact_counterfactual_value.csv`
- `results/v12_exact_endpoint_validation/heuristic_endpoint_quality.csv`
