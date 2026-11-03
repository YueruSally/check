# 7. Results

## 7.1 Structural path coverage and exact-solution verification

The final analysis uses the complete set of simple paths satisfying the stated limits on path length, mode changes, and node revisits. Structural enumeration found eight feasible paths for every origin--destination--gateway combination. The Martins library retained all eight Seattle/Tacoma paths but only four Los Angeles/Long Beach paths and four New York/New Jersey (NY/NJ) paths because its static cost--time dominance test removed resource- and schedule-distinct alternatives. Consequently, the full allowed model contained 960 shipment--path variables and the NY/NJ-restricted model contained 640. Replacing the Martins library with the structural library materially changed 74 of the 180 core endpoint solutions. All findings below therefore use the structurally complete library; results based on the truncated libraries are not used as substantive evidence.

The core design contained 45 destination--condition combinations. For each combination, the NY/NJ-allowed and NY/NJ-restricted models were solved at the minimum-cost-then-time and minimum-time-then-cost lexicographic endpoints. All 180 endpoint models were evaluator-consistent and proven optimal. All 360 solver calls terminated as optimal within tolerance, with a maximum relative gap of \(9.98\times10^{-9}\). Minimum and maximum NY/NJ shares were additionally proven for 88 distinct endpoint-equivalent regions. The gateway-use conclusions therefore do not depend on stochastic NSGA-II convergence. Exactness is conditional on the stated path-structure restrictions and the constructed network.

Table 2 establishes the undisturbed reference state. NY/NJ was absent from every endpoint-equivalent region: both the proven lower and upper share bounds were zero. The reference network was therefore West Coast-supported, although the two endpoints reveal a sizeable cost--time trade-off.

Table 2. Exact baseline endpoints under the complete structural path library.

| Destination | Endpoint | Total cost (USD million) | Weighted mean delivery time (h) | NY/NJ share (%) |
| --- | --- | ---: | ---: | ---: |
| Chicago | Minimum cost, then time | 22.863 | 668.351 | 0 |
| Chicago | Minimum time, then cost | 29.886 | 602.296 | 0 |
| Memphis | Minimum cost, then time | 22.430 | 687.534 | 0 |
| Memphis | Minimum time, then cost | 29.390 | 600.282 | 0 |
| Columbus | Minimum cost, then time | 23.969 | 700.544 | 0 |
| Columbus | Minimum time, then cost | 31.558 | 609.177 | 0 |

## 7.2 Gateway-node capacity reductions did not activate NY/NJ

Reducing the common West Coast gateway-node capacity to 60%, 40%, or 35% of baseline did not introduce NY/NJ at either exact endpoint for any destination. The compound condition combining 30% remaining gateway-node capacity with 168 h of West Coast sailing delay also produced a zero NY/NJ share at every endpoint. Allowing NY/NJ consequently produced no material cost or time improvement in these conditions.

This result was robust to proportional demand changes. At 35% gateway-node capacity, all endpoint-equivalent NY/NJ share bounds remained zero when total demand was set to 80%, 100%, or 120% of baseline. This was not evidence that every West Coast resource was slack: in the 80% and 120% demand experiments, maximum daily-arc utilization reached 100% for every destination, and maximum node-day utilization ranged from 95.24% to approximately 100%. The appropriate interpretation is therefore that the tested gateway-node reductions were accommodated through the remaining West Coast path and timetable choices. It is not a general claim that all capacity disruptions are ineffective; Section 7.6 shows that destination-specific inland capacity changes can alter gateway selection when combined with delay.

The baseline calibration checks also preserved the reference conclusion. Reducing NY/NJ ocean cost by as much as 30% or reducing Panama-route transit time by 48 h did not activate NY/NJ at either endpoint. A 96 h transit-time reduction introduced NY/NJ only in the Columbus minimum-cost endpoint, at 7.16%, below the pre-specified 10% material-use criterion. The corresponding cost benefit was USD 0.027 million and weighted mean time increased by 1.547 h. Hence, the zero-share baseline is robust to the tested one-factor changes, but remains a constructed timetable- and delivery-sensitive setting rather than a universal market calibration.

## 7.3 Sailing delay generated destination- and objective-specific gateway value

Figure 1 reports the exact value of retaining NY/NJ access over the common West Coast delay grid. Panel (a) compares the minimum-cost objective, and Panel (b) compares the minimum-time objective. The two panels contain different lexicographic solutions; their benefits must not be added or interpreted as jointly attained by one allocation.

![Exact NY/NJ value curves](figures/fig07_final_gateway_value_curves.png)

Figure 1. Exact value of NY/NJ access under additional West Coast sailing delay. Positive values are the restricted objective minus the allowed objective. Cost and time panels report separate lexicographic endpoints.

Chicago and Memphis used no NY/NJ through 264 h at either endpoint. At 288 h, both destinations used NY/NJ materially at both endpoints. Columbus responded earlier and differently by objective: at 240 h the minimum-cost endpoint used 32.66% NY/NJ, whereas the minimum-time endpoint used none; at 264 h both endpoints used NY/NJ. Thus, on the tested grid, the first material use lay in \((264,288]\) h for Chicago and Memphis, in \((168,240]\) h for the Columbus minimum-cost endpoint, and in \((240,264]\) h for the Columbus minimum-time endpoint. These intervals are experimental brackets, not continuous real-world switching-time estimates.

Table 3 reports the decisive grid points. Positive differences are losses avoided by allowing NY/NJ. A time difference in a minimum-cost row describes the time of that cost-optimal allocation; it is not the global minimum-time benefit, and vice versa.

Table 3. Exact endpoint value at decisive delay conditions.

| Destination | Delay (h) | Endpoint | NY/NJ share (%) | Cost avoided (USD million) | Time avoided (h) |
| --- | ---: | --- | ---: | ---: | ---: |
| Chicago | 264 | Minimum cost, then time | 0 | 0.000 | 0.000 |
| Chicago | 264 | Minimum time, then cost | 0 | 0.000 | 0.000 |
| Chicago | 288 | Minimum cost, then time | 64.89 | 1.732 | 5.030 |
| Chicago | 288 | Minimum time, then cost | 32.66 | 2.260 | 3.266 |
| Memphis | 264 | Minimum cost, then time | 0 | 0.000 | 0.000 |
| Memphis | 264 | Minimum time, then cost | 0 | 0.000 | 0.000 |
| Memphis | 288 | Minimum cost, then time | 32.66 | 1.370 | 11.311 |
| Memphis | 288 | Minimum time, then cost | 32.66 | 1.966 | 2.613 |
| Columbus | 168 | Minimum cost, then time | 0 | 0.000 | 0.000 |
| Columbus | 168 | Minimum time, then cost | 0 | 0.000 | 0.000 |
| Columbus | 240 | Minimum cost, then time | 32.66 | 0.827 | 2.764 |
| Columbus | 240 | Minimum time, then cost | 0 | 0.000 | 0.000 |
| Columbus | 264 | Minimum cost, then time | 56.48 | 2.960 | -1.579 |
| Columbus | 264 | Minimum time, then cost | 32.66 | 2.944 | 0.555 |

The share-bound models show that these were not arbitrary configurations selected from wide endpoint-equivalent ranges. At Chicago 288 h, the minimum-cost region fixed the NY/NJ share at 64.89%, while the minimum-time region fixed it at approximately 32.66%. At Memphis 288 h, the two endpoints coincided in objective space and fixed the aggregate NY/NJ share at approximately 32.66%; this does not imply uniqueness of the shipment-level allocation. At Columbus 240 h, the minimum-cost region fixed the share at 32.66% and the minimum-time region fixed it at zero. The exact evidence therefore supports destination- and objective-specific use rather than a single gateway-activation ordering.

## 7.4 Cost savings were not solely a tardiness-penalty artifact

The minimum-cost benefit arose from different component combinations across destinations (Table 4). For Chicago at 288 h, the USD 1.732 million saving included reductions of USD 0.564 million in transport cost, USD 0.694 million in waiting and holding cost, and USD 0.474 million in tardiness cost. For Memphis, NY/NJ increased transport cost by USD 1.109 million but reduced waiting and holding cost by USD 0.350 million and tardiness cost by USD 2.129 million, producing a net USD 1.370 million saving. For Columbus at 240 h, the USD 0.827 million saving consisted mainly of USD 0.525 million in avoided tardiness and USD 0.284 million in waiting and holding cost.

Table 4. Exact minimum-cost saving decomposition (restricted minus allowed, USD million).

| Destination and delay | Transport | Waiting and holding | Tardiness | Total |
| --- | ---: | ---: | ---: | ---: |
| Chicago, 288 h | +0.564 | +0.694 | +0.474 | +1.732 |
| Memphis, 288 h | -1.109 | +0.350 | +2.129 | +1.370 |
| Columbus, 240 h | +0.018 | +0.284 | +0.525 | +0.827 |

Setting the tardiness penalty to zero did not eliminate the minimum-cost use of NY/NJ. The proven shares were 89.79% for Chicago, 10.21% for Memphis, and 100% for Columbus, with respective cost savings of USD 1.113, 0.113, and 2.408 million. NY/NJ value was therefore not generated solely by the tardiness term. The response was not monotonic: Memphis did not use NY/NJ at 25% of the baseline penalty, but used 22.60% at 50% and 32.66% at the baseline penalty. This reflects discrete changes in path, service, and minimum-share choices rather than a smooth penalty threshold. Changing the holding-cost rate from zero to twice baseline did not change the qualitative use state in any of the three focal conditions, although it changed the exact share and value.

## 7.5 Release timing and demand intensity changed the location and strength of gateway use

Dispersing shipment releases materially changed the first tested delay at which NY/NJ reached 10% (Figure 2 and Table 5). Chicago minimum-cost use remained at 288 h in every multi-wave instance, but minimum-time use moved from 288 h in the single-wave case to 336 h in every multi-wave instance. Memphis generally moved from 288 h to 336 h; only one of the five daily-release instances reached 10% at the 288 h minimum-cost endpoint. Columbus was most release-sensitive: its minimum-cost first-use point ranged from 168 to 264 h, while its minimum-time first-use point moved from 264 h in the single-wave case to 336 h in every multi-wave instance.

![Activation under alternative release patterns](figures/fig08_release_robustness.png)

Figure 2. First tested delay at which all endpoint-equivalent solutions used at least 10% NY/NJ under single-wave and dispersed-release designs. Points represent constructed instances, not occurrence probabilities.

Table 5. First tested material-use delays under alternative release designs.

| Destination | Endpoint | Single wave | Two weekly waves | Four weekly waves | Daily over four weeks |
| --- | --- | ---: | ---: | ---: | ---: |
| Chicago | Minimum cost | 288 | 288 (5/5) | 288 (5/5) | 288 (5/5) |
| Chicago | Minimum time | 288 | 336 (5/5) | 336 (5/5) | 336 (5/5) |
| Memphis | Minimum cost | 288 | 336 (5/5) | 336 (5/5) | 288 (1/5); 336 (4/5) |
| Memphis | Minimum time | 288 | 336 (5/5) | 336 (5/5) | 336 (5/5) |
| Columbus | Minimum cost | 240 | 264 (5/5) | 264 (5/5) | 168 (1/5); 240 (4/5) |
| Columbus | Minimum time | 264 | 336 (5/5) | 336 (5/5) | 336 (5/5) |

The multi-wave results also reject the interpretation of 32.66% as a universal capacity ceiling: exact NY/NJ shares ranged across several intermediate values and reached 100% in some dispersed-release solutions. Most highlighted multi-wave solutions did not bind service, daily-arc, or node-day capacity, so release-time alignment and connection opportunities, rather than capacity overflow alone, affected gateway choice.

Proportional demand scaling preserved the main endpoint asymmetries. At Chicago 288 h, the minimum-cost NY/NJ share was 64.89%, 64.89%, and 61.48% at 80%, 100%, and 120% demand, while the minimum-time share rose from 15.83% to 32.66% and 37.41%. Memphis 288 h used 15.83%, 32.66%, and 37.41% at both endpoints. Columbus 240 h used 15.83%, 32.66%, and 37.41% at the minimum-cost endpoint but zero at the minimum-time endpoint. Thus, demand intensity changed the amount and value of diversion without erasing destination and objective dependence. These are controlled robustness cases, not empirical demand distributions.

## 7.6 Inland capacity and sailing delay interacted

The earlier repeated 727.5 FEU diversion equalled total demand minus the 1,500 FEU/day capacity of a destination-specific Los Angeles/Long Beach inland arc. The full interaction experiment varied that diagnosed arc to 50%, 100%, and 150% of baseline while also varying delay. Figure 3 and Table 6 show that inland capacity affected both the level and the onset of NY/NJ use, but its effect depended on destination, delay, and objective.

![Delay--inland-capacity interaction](figures/fig09_inland_capacity_interaction.png)

Figure 3. Exact NY/NJ shares under joint changes in West Coast sailing delay and the diagnosed Los Angeles/Long Beach inland daily-arc capacity. Solid and dashed lines denote the minimum-cost and minimum-time endpoints.

Table 6. Exact NY/NJ shares under the inland-capacity interaction (%; minimum cost / minimum time).

| Destination | Delay (h) | 50% inland capacity | 100% inland capacity | 150% inland capacity |
| --- | ---: | ---: | ---: | ---: |
| Memphis | 264 | 0 / 0 | 0 / 0 | 0 / 0 |
| Memphis | 276 | 21.44 / 0 | 0 / 0 | 0 / 0 |
| Memphis | 288 | 66.33 / 66.33 | 32.66 / 32.66 | 0 / 0 |
| Columbus | 216 | 17.98 / 0 | 10.06 / 0 | 0 / 0 |
| Columbus | 240 | 66.33 / 0 | 32.66 / 0 | 7.25 / 0 |
| Columbus | 264 | 66.33 / 66.33 | 56.48 / 32.66 | 56.48 / 0 |

For Memphis, tightening the inland arc to 50% made NY/NJ cost-effective at 276 h, but did not make it time-optimal. At 288 h, the same capacity level produced 66.33% NY/NJ at both endpoints and avoided USD 3.653 million at the minimum-cost endpoint and USD 4.568 million at the minimum-time endpoint. Raising capacity to 150% removed NY/NJ completely. Columbus displayed a different pattern. Its minimum-cost endpoint used NY/NJ as early as 216 h at 50% and 100% capacity, while the minimum-time endpoint remained at zero until the more severe 264 h condition. At 240 h, increasing inland capacity reduced minimum-cost NY/NJ use from 66.33% to 32.66% and then 7.25%; at 264 h, the minimum-cost allocation remained at 56.48% even at 150% capacity, but the minimum-time allocation returned to zero. Hence, gateway-node capacity, inland-arc capacity, and sailing delay cannot be represented by one generic “capacity disruption” effect.

## 7.7 Exact intermediate solutions revealed how preference changed gateway value

Eight internal time limits were solved exactly between the two allowed endpoints for Chicago at 288 h and Columbus at 240 h. At each limit, cost was minimized and time was used as a lexicographic tie-breaker; the same limit was imposed on the NY/NJ-restricted model. These are exact epsilon-constrained trade-off solutions, not a claim that the entire continuous Pareto set was enumerated.

![Exact epsilon-constrained trade-off solutions](figures/fig10_exact_epsilon_tradeoffs.png)

Figure 4. Exact endpoint and epsilon-constrained solutions. Color denotes the proven NY/NJ share in the allowed solution; crosses denote feasible restricted solutions at the same internal time limit.

For Chicago, the allowed NY/NJ share increased from 32.66% at the minimum-time endpoint to 64.89% at the minimum-cost endpoint. Across the eight internal limits it increased monotonically from 36.24% to 61.31%, and the minimum and maximum share bounds differed by less than \(1.3\times10^{-7}\). The restricted model was proven infeasible at the first five limits, from 887.581 to 889.787 h. It became feasible at the final three limits; even then, NY/NJ access reduced minimum cost by USD 2.456, 2.375, and 2.289 million at the same respective time limits. NY/NJ therefore expanded the attainable time region as well as lowering cost within the overlapping region.

For Columbus, the restricted model remained feasible at all eight internal limits. The allowed NY/NJ share rose from 3.63% to 29.03%, with the 10% criterion crossed between the second and third limits, whose caps were 850.796 and 851.605 h. Cost avoided at the same time cap rose from USD 0.110 million to USD 0.774 million. Together with the zero-share minimum-time endpoint and the 32.66% minimum-cost endpoint, these points show that Columbus gateway use varied systematically with the decision-maker's tolerated delivery time rather than switching only at a single front-wide threshold.

## 7.8 Consolidated findings and evidential limits

Four findings remain after structural path enumeration, exact counterfactual optimization, parameter checks, and release and demand perturbations. First, West Coast gateway-node capacity reductions were absorbed without NY/NJ in the tested cases, whereas long sailing delays and delay--inland-capacity interactions created alternative-gateway value. Second, gateway use was asymmetric across destinations and objectives: Columbus used NY/NJ for cost reduction before doing so for minimum time, while Chicago and Memphis required the longer tested delay under the single-wave demand. Third, NY/NJ value was not limited to making an otherwise infeasible allocation feasible. It reduced exact endpoint objectives, expanded Chicago's attainable time region, and reduced cost at common time limits. Fourth, activation shares were shaped by timetable alignment, release dispersion, demand intensity, and destination-specific inland capacity; the repeated single-wave share was therefore a feature of a particular binding configuration rather than a universal diversion ratio.

These results remain conditional on the constructed case-study network, the stated path-structure restrictions, the discrete delay grid, and the specified objective components. The release and demand variants are controlled robustness instances and are not estimates of real-world probabilities. A proven aggregate gateway-share range does not imply uniqueness of the complete shipment allocation. The epsilon experiment provides exact internal trade-off solutions but does not claim exhaustive enumeration of every Pareto-optimal configuration. Finally, the model contains no carbon variable; all reported effects concern operational cost, delivery time, capacity, timetable, and gateway allocation.

## Source tables

- `results/v21_full_structural_core_exact/path_coverage_summary.csv`
- `results/v21_full_structural_core_exact/full_exact_endpoint_summary.csv`
- `results/v21_full_structural_core_exact/full_endpoint_share_ranges.csv`
- `analysis/v21_full_structural_core_exact/delay_gateway_value.csv`
- `analysis/v21_full_structural_core_exact/capacity_compound_value.csv`
- `analysis/v21_full_structural_core_exact/disruption_loss_recovery.csv`
- `analysis/v22_full_structural_cost_sensitivity/cost_sensitivity_value.csv`
- `analysis/v22_full_structural_cost_sensitivity/counterfactual_cost_components.csv`
- `analysis/v23_full_structural_baseline_calibration/baseline_calibration_summary.csv`
- `analysis/v24_full_structural_release_robustness/activation_by_release_instance.csv`
- `analysis/v24_full_structural_release_robustness/release_value_distributions.csv`
- `analysis/v25_full_structural_demand_scale/demand_scale_comparison.csv`
- `analysis/v26_full_structural_inland_bottleneck/inland_capacity_delay_interaction.csv`
- `analysis/v27_full_structural_epsilon_tradeoffs/epsilon_tradeoff_comparison.csv`
