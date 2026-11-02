# 7. Results

## 7.1 Computational outcomes and baseline gateway structure

Every case--scenario combination retained for the main analysis produced feasible solutions in all 30 NSGA-II runs. The pooled baseline archives contained 84 solutions for Chicago, 68 for Memphis, and 71 for Columbus; none routed freight through New York/New Jersey (NY/NJ). Baseline cost--time trade-offs were therefore supported entirely by Los Angeles/Long Beach and Seattle/Tacoma.

The baseline archives spanned USD 24.84--26.77 million and 615.64--660.47 h for Chicago, USD 25.04--26.64 million and 614.22--644.22 h for Memphis, and USD 26.58--28.61 million and 628.06--661.82 h for Columbus. These are ranges across retained non-dominated alternatives, not sampling intervals. A separate diagnostic that combined 1.5 times demand with 60% remaining West Coast capacity produced no feasible population for any destination and was not used in the activation analysis; the reported activation suite varies disruption conditions at the original demand.

Figure 1 compares the baseline archive with a closure-validated delay condition and the 30% capacity plus 168 h compound condition. Both disruption types shifted the found alternatives towards higher cost and longer delivery time. The points are discrete best-found solutions under the stated budget; no continuous frontier is inferred between them.

![Cost and delivery time trade-offs](figures/fig01_cost_time_fronts.png)

Figure 1. Cross-seed non-dominated cost and quantity-weighted mean delivery-time solutions. The delay condition is 288 h for Chicago and Memphis and 240 h for Columbus. Baseline and compound points are historical allowed-search archives; delay points are from the targeted final rerun. Common axes permit comparison across destinations, and insets enlarge the delay-only archives.

## 7.2 West Coast reallocation accommodated capacity reductions

Capacity-only disruptions did not generate operationally material NY/NJ use. NY/NJ remained absent when regional West Coast capacity was reduced to 70%, 67.5%, 65%, 60%, 40%, or 35% of baseline, and when either West Coast gateway was individually reduced to 50%.

This finding cannot be attributed to all tested capacity constraints being slack. Table 2 reports the maximum West Coast node-day utilization in one balanced retained solution per scenario. At 60%, 40%, and 35% remaining capacity, at least one node-day reached 97.89%--99.99% utilization, while the NY/NJ share remained zero. The evidence shows that the tested solutions accommodated these reductions through West Coast allocation and timing changes. It does not imply that the whole West Coast system, every resource type, or every non-dominated solution was saturated.

Table 2. Maximum West Coast node-day utilization in balanced capacity-only solutions.

| Destination | Baseline (%) | 60% capacity (%) | 40% capacity (%) | 35% capacity (%) | NY/NJ share (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chicago | 57.68 | 97.89 | 99.86 | 99.31 | 0 |
| Memphis | 70.21 | 99.99 | 99.40 | 99.79 | 0 |
| Columbus | 66.86 | 99.79 | 99.90 | 99.91 | 0 |

The capacity-days diagnostic gives supporting, but not causal, evidence. Baseline demand at 40% capacity and 1.25 times demand at 50% capacity both required 0.928 nominal capacity-days. Their balanced Los Angeles/Long Beach shares differed by 0.09 percentage points for Chicago, 0.09 for Memphis, and 0.04 for Columbus; neither condition used NY/NJ. Other service capacities were not scaled proportionally, so this comparison is not used to claim that demand divided by daily port capacity is the sole driver.

## 7.3 Delay activation depended on destination and cost--time preference

Regional delays of 96 and 120 h produced no NY/NJ use. At 168 h, Chicago still used none; Memphis reached only 0.28% in one retained solution, and Columbus reached 0.37%--1.14% in five solutions. These shares remained below the 10% operational criterion.

The final closure changes the interpretation of the critical delays. It preserves distinct route configurations at equal objective coordinates and pools NY/NJ-allowed and restricted configurations before non-dominance filtering. Table 3 reports both first-solution and front-wide activation. Chicago moved from 4.22%--8.06% at 264 h to 32.75%--50.56% at 288 h; its first material use and front-wide activation therefore coincided at 288 h. Columbus similarly moved from 0%--1.14% at 168 h to 24.17%--28.13% at 240 h.

Memphis behaved differently. At 264 h, the closed archive contained 26 objective points but 27 distinct configurations because one objective point had two allocation variants. Eight configurations used no NY/NJ, while others used up to 26.04%. The condition is therefore a transition, not front-wide activation. At 288 h, all 12 retained configurations used 31.72%--32.79%, establishing front-wide activation on the tested grid.

Table 3. Configuration-preserving activation classification at the critical delays.

| Destination | Delay | Objective points | Distinct configurations | Zero-share configurations | NY/NJ share range (%) | Classification |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Chicago | 288 h | 24 | 24 | 0 | 32.75--50.56 | Front-wide activation |
| Memphis | 264 h | 26 | 27 | 8 | 0--26.04 | Transition |
| Memphis | 288 h | 12 | 12 | 0 | 31.72--32.79 | Front-wide activation |
| Columbus | 240 h | 11 | 11 | 0 | 24.17--28.13 | Front-wide activation |

![Destination-specific delay activation](figures/fig02_delay_activation.png)

Figure 2. NY/NJ shares under regional West Coast sailing delays. Dots show balanced solutions and vertical segments show complete retained-configuration ranges, including zero-share configurations. The 264 and 288 h Memphis ranges, 288 h Chicago range, and 240 h Columbus range use configuration-preserving closed archives; other points retain their original allowed-search scope. Vertical segments are solution ranges, not confidence intervals. The dashed line is the 10% reporting criterion.

The two activation concepts now yield different orderings. The first retained configuration reaching 10% appeared at 240 h for Columbus, 264 h for Memphis, and 288 h for Chicago. Front-wide activation appeared at 240 h for Columbus and at 288 h for both Chicago and Memphis. The evidence therefore supports destination- and preference-dependent transition regions, not the earlier claim of a simple Columbus--Memphis--Chicago ordering for the whole front. The large changes between adjacent tested delays are not proof of a mathematical discontinuity because intermediate delays were not sampled.

## 7.4 Capacity--delay interaction was evident, but most grid cells remain exploratory

Figure 3 retains the broad allowed-search grid as a screening result. With no added delay, capacity reduction alone did not activate NY/NJ. At 168 h, the allowed-search archives suggested earlier substitution for Columbus and some Memphis conditions than for Chicago; at 288 h, all displayed allowed-search cells contained material NY/NJ use. Because most cells were not closed with matching restricted configurations, blue cells in this map mean front-wide within the allowed-search archive, not final closure-validated front-wide activation.

![Capacity-delay activation regimes](figures/fig03_activation_regime.png)

Figure 3. Exploratory capacity--delay regime map from the NY/NJ-allowed searches. Grey denotes no configuration reaching 10%, orange denotes a range crossing 10%, and blue denotes all retained allowed-search configurations exceeding 10%. Cell labels show the balanced-solution NY/NJ share. Only separately reported targeted conditions have matching restricted-archive closure.

The corrected Chicago compound result illustrates why this distinction matters. Under 30% remaining West Coast capacity and 168 h delay, the existing closed archive contained five objective points, two with zero NY/NJ share, and an overall range of 0%--18.17%. The condition is a transition, not front-wide activation. The exact cost- and time-optimal endpoints nevertheless both assigned 19.19% to NY/NJ and improved on their restricted counterparts (Section 7.5). Capacity scarcity can therefore make NY/NJ valuable for particular preferences without making its use universal across the retained trade-off set. Supplementary Figure S1 reports the historical compound ranges with this Chicago correction; unclosed cells remain exploratory.

## 7.5 Exact endpoints quantify value without implying universal cost savings

All allowed and restricted counterfactual models were feasible, so NY/NJ was not required to preserve feasibility in the tested conditions. Figure 4 shows the configuration-preserving closed heuristic fronts for Chicago at 288 h, Memphis at 264 h, and Columbus at 240 h. The Memphis panel retains zero-share restricted configurations on the non-dominated union, visually confirming a preference-dependent transition.

![Closed NY/NJ counterfactual fronts](figures/fig04_counterfactual_fronts.png)

Figure 4. Configuration-preserving closed approximate fronts. Grey crosses are NY/NJ-restricted objective points. Coloured circles are retained points in the closed allowed union; colour gives the range midpoint of NY/NJ share at that objective coordinate. Where equal-objective configurations differ in share, the full range is retained in the underlying classification table.

Table 4 replaces heuristic endpoint means with proven lexicographic MILP endpoints. Both solver stages were proven for all 20 tasks. Every decoded solution matched the independent evaluator and the three resource accounts within numerical tolerance. Positive cost and time differences mean that the restricted endpoint was worse; a negative cost difference means that the time-optimal allowed solution bought time at additional cost.

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

![Exact endpoint effects of NY/NJ access](figures/fig05_counterfactual_benefits.png)

Figure 5. Exact endpoint differences, restricted minus allowed. Circles denote minimum-cost endpoints and diamonds minimum-time endpoints. Positive values favour NY/NJ access. The Memphis 264 h minimum-time endpoint saves 2.939 h but increases cost by USD 0.797 million, while its minimum-cost endpoint does not use NY/NJ.

The endpoint evidence separates gateway share from economic value. At 264 h, Memphis is a genuine transition: cost minimization selects zero NY/NJ, whereas time minimization selects 32.66% and accepts a higher cost. At 288 h, access improves both endpoints by USD 1.407 million and 13.913 h. Chicago at 288 h shows the largest exact cost benefit (USD 1.732 million at the cost endpoint), while Columbus at 240 h gains USD 0.832 million and 3.070 h. High NY/NJ share therefore does not by itself imply cost saving; value depends on destination, disruption, and decision preference.

The exact comparison also calibrates the heuristic evidence. Across the ten validated access models, the best 30-seed NSGA-II primary-objective gap was 0.516%--4.338% for minimum cost and below 0.258% for minimum time. In particular, small heuristic cost benefits are not treated as endpoint truth. NSGA-II remains useful for mapping trade-off structure, while the exact MILP anchors the selected endpoints.

## 7.6 Shipment-matched accounting supports a reallocation mechanism, with destination differences

Table 5 decomposes the paired minimum-time change over 30 seeds. Values are allowed minus restricted, so negative values are improvements. The accounting identity held for every seed with maximum absolute error below (7.1\times10^{-13}) h, and the unassigned “other” component was zero.

Table 5. Seed-paired shipment-matched delivery-time decomposition; mean [95% bootstrap interval], hours.

| Destination | Delay | Total change | NY/NJ-assigned component | West Coast-assigned component |
| --- | ---: | ---: | ---: | ---: |
| Chicago | 288 h | -8.83 [-9.13, -8.53] | -0.23 [-0.55, 0.07] | -8.60 [-9.02, -8.17] |
| Memphis | 264 h | -1.52 [-1.68, -1.35] | +6.07 [5.54, 6.59] | -7.59 [-8.19, -6.99] |
| Memphis | 288 h | -13.19 [-13.47, -12.90] | +0.94 [0.35, 1.50] | -14.13 [-14.74, -13.50] |
| Columbus | 240 h | -2.18 [-2.30, -2.07] | +6.78 [6.37, 7.20] | -8.97 [-9.44, -8.51] |

![Shipment-matched time decomposition](figures/fig06_destination_mechanism.png)

Figure 6. Mean shipment-matched contributions to the minimum-time change. Bars show NY/NJ and West Coast components, and diamonds show their system total. Error bars are 95% paired-bootstrap intervals. Negative values favour allowing NY/NJ. These are search-run summaries under fixed inputs, not confidence intervals for real-world effects.

For Memphis at 264 h and Columbus at 240 h, freight assigned to NY/NJ increased the matched time account in all 30 seeds, by 6.07 and 6.78 h on average. This was more than offset by improvements of 7.59 and 8.97 h among freight remaining on West Coast routes. The corresponding mean service changes shifted 402.66 and 520.18 FEU onto the Shanghai--NY/NJ departure and 402.44 and 518.93 FEU away from Shanghai--Los Angeles/Long Beach. The results are therefore consistent with system-level reallocation: a slower component for some freight accompanies better connections for the remaining West Coast-assigned freight.

Chicago at 288 h was different. Its NY/NJ component averaged -0.23 h and the interval included zero, whereas the West Coast component improved in all 30 seeds and accounted for -8.60 of the -8.83 h total. Memphis at 288 h showed an even larger West Coast component (-14.13 h) and only a small adverse NY/NJ component (+0.94 h). Thus the data do not support one universal “slower NY/NJ route” mechanism. They support destination- and disruption-specific reallocation of fixed path and departure capacity. Because the model does not contain endogenous port queues, these changes are not described as congestion relief.

## 7.7 Sensitivity to the operational share criterion

Reclassifying the same closed archives at 5%, 10%, and 20% changes no front-wide point among the three destinations (Table 6). The result does not validate the business meaning of 10%; it shows only that the reported front-wide classification is not an artefact of that single cutoff.

Table 6. First closure-validated front-wide point under alternative share criteria.

| Destination | 5% criterion | 10% criterion | 20% criterion |
| --- | ---: | ---: | ---: |
| Chicago | 288 h | 288 h | 288 h |
| Memphis | 288 h | 288 h | 288 h |
| Columbus | 240 h | 240 h | 240 h |

Memphis still reaches first-solution activation at 264 h under the 10% criterion, because some configurations use as much as 26.04%, but its zero-share configurations prevent front-wide classification at every tested criterion. This distinction is substantively important: it separates the onset of a viable option from adoption across the entire retained cost--time trade-off.

## 7.8 Principal findings and inferential limits

Four findings survive the final consistency checks. First, capacity reduction and schedule delay produce asymmetric responses: the tested capacity-only reductions were accommodated within West Coast allocations, while long delay made cross-coast substitution competitive. Second, gateway activation is preference dependent. Columbus reaches front-wide activation at 240 h, Memphis enters transition at 264 h, and Chicago and Memphis are front-wide at 288 h. Third, exact counterfactual value differs by objective: NY/NJ can improve both endpoints, but at Memphis 264 h it provides time value only by accepting extra cost. Fourth, shipment-matched accounting attributes the system time gains primarily to reconfiguration of freight remaining on West Coast paths, with the NY/NJ component varying by destination.

These findings apply to the constructed case-study network, generated candidate paths, and tested discrete conditions. They are not universal empirical switching times. The 30 seeds quantify stochastic-search variation, not input uncertainty; “front-wide” describes retained found configurations, not a probability of real-world gateway choice; and exact endpoint claims are conditional on the candidate-path MILP. Within those boundaries, NY/NJ is best interpreted as a performance-diversification option whose adoption and value depend jointly on disruption severity, inland destination, and the decision maker's position on the cost--time trade-off.

## Source tables

- `analysis/v12_final_validation/closed_front_classification.csv`
- `analysis/v12_final_validation/closed_front_points.csv`
- `analysis/v12_final_validation/historical_closed_front_corrections.csv`
- `analysis/v12_final_validation/batch_matched_time_decomposition_summary.csv`
- `analysis/v12_final_validation/service_departure_quantity_changes.csv`
- `results/v12_exact_endpoint_validation/exact_endpoint_summary.csv`
- `results/v12_exact_endpoint_validation/exact_counterfactual_value.csv`
- `results/v12_exact_endpoint_validation/heuristic_endpoint_quality.csv`
- `results/v12_exact_endpoint_validation/exact_endpoint_capacity_usage.csv`
