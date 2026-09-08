# 4. Results

## 4.1 Computational performance and baseline structure

All reported NSGA-II experiments completed successfully. Each case-scenario combination used 30 independent runs with an identical seed sequence, a population of 48 and 99 generations, corresponding to 4,800 objective evaluations per run. Feasible solutions satisfied timetable, service-capacity, daily node-capacity, flow-conservation and transfer constraints. Each shipment used no more than three paths, and every active path carried at least 10% of that shipment demand.

The baseline cross-seed non-dominated unions contained 84 solutions for D1 (Chicago), 68 for D2 (Memphis) and 71 for D3 (Columbus). None used the New York/New Jersey gateway. Baseline freight was therefore assigned entirely through Los Angeles/Long Beach and Seattle/Tacoma.

## 4.2 Capacity-only disruptions

Capacity reductions alone did not activate New York/New Jersey. Regional West Coast capacity multipliers of 70%, 67.5%, 65% and 60% all produced a zero New York/New Jersey share in every case. A 50% reduction applied separately to Los Angeles/Long Beach or Seattle/Tacoma also failed to activate the East Coast gateway.

The operational response to capacity loss was primarily reallocation within the West Coast system. At 60% regional capacity, the median delivery-time increase relative to baseline was 7.22 hours for D1, 11.11 hours for D2 and 11.31 hours for D3, while the median cost change remained between -0.34% and 0.33%. The 67.5% capacity experiment similarly produced no material cost change. These results show that spare capacity and substitution between the two West Coast gateways absorb moderate capacity shocks before an all-water East Coast route becomes competitive.

## 4.3 Departure-delay activation thresholds

Short and medium West Coast departure delays changed cost and delivery performance but did not create a meaningful New York/New Jersey alternative. Delays of 96 and 120 hours produced no East Coast use. At 168 hours, New York/New Jersey appeared only marginally in D2 and D3, with shares below 1.2%, and remained absent in D1.

The targeted v0.7 and v0.8 experiments reveal a discontinuous activation response. For D1, a 264-hour delay produced a mean New York/New Jersey share of 6.59% (range 4.22%-8.06%), whereas a 288-hour delay increased the mean share to 39.00% (range 32.75%-50.56%). The meaningful activation threshold for D1 therefore lies in the tested interval from 264 to 288 hours.

D2 activated earlier. Its mean share was only 1.24% at 240 hours but rose to 19.28% at 264 hours, with every solution in the non-dominated union allocating at least 12.47% through New York/New Jersey. The D2 threshold lies between 240 and 264 hours. D3 was the most delay-sensitive case: a 240-hour delay already produced a mean East Coast share of 26.14%, with a range of 24.17%-28.13%.

At 336 hours, New York/New Jersey became a dominant gateway in all cases. Mean shares reached 71.39% for D1, 67.75% for D2 and 83.62% for D3. This confirms that prolonged sailing disruption changes the preferred gateway structure rather than merely shifting freight between West Coast ports.

| Case | Last tested point below 10% | First tested point stably above 10% | Activation interval |
| --- | ---: | ---: | ---: |
| D1 (Chicago) | 264 h: 6.59% mean | 288 h: 39.00% mean | 264-288 h |
| D2 (Memphis) | 240 h: 1.24% mean | 264 h: 19.28% mean | 240-264 h |
| D3 (Columbus) | Not isolated below 240 h | 240 h: 26.14% mean | At or below 240 h, above 168 h |

## 4.4 Compound capacity-and-delay disruptions

Compound disruption activated New York/New Jersey at lower delays, but the response depended on destination. With West Coast capacity at 30% and departure delay at 168 hours, mean New York/New Jersey shares were 17.90% for D1, 17.32% for D2 and 45.11% for D3. All three cases therefore met the 10% meaningful-alternative criterion.

At 40% capacity and 168 hours of delay, D1 remained below the criterion with a mean share of 6.19%, while D2 and D3 reached 12.59% and 35.59%. The additional 35% experiment placed D1 at 5.61%. Its compound activation boundary consequently lies between 30% and 35% remaining West Coast capacity at a 168-hour delay.

For D2, the mean share at 35% capacity was 10.83%, but individual non-dominated solutions ranged from 6.88% to 16.97%. The tested 30%, 35% and 40% results were not strictly monotonic. The D2 compound result should therefore be described as a transition region rather than an exact deterministic threshold. This variation reflects changes in the composition of the cross-seed non-dominated union and warrants caution when interpreting a single mean gateway share.

## 4.5 Cost and delivery-time consequences

New York/New Jersey activation did not eliminate the performance loss created by severe disruption. At the first stable delay activation point, the D1 median cost was 210.69% above baseline and median weighted delivery time increased by 264.58 hours. For D2 at 264 hours, the corresponding changes were 186.80% and 248.26 hours. For D3 at 240 hours, median cost increased by 156.94% and median weighted delivery time by 222.32 hours.

The compound cases also imposed large penalties. The D1 30%-capacity and 168-hour scenario increased median cost by 138.21% and delivery time by 217.20 hours. For D2 at 40% capacity and 168 hours, the increases were 124.90% and 202.80 hours. New York/New Jersey should thus be interpreted as a resilience pathway that preserves feasible routing options under severe disruption, not as a mechanism that restores baseline cost or service performance.

## 4.6 Paired statistical evidence

Paired inference compared each disruption run with the baseline run having the same case and random seed. The v0.7 and v0.8 severe delay and compound scenarios showed the same directional increase in cost and weighted delivery time for all 30 seed pairs. Exact two-sided sign tests remained significant after Holm correction. In v0.8, the adjusted p-value was 2.98 x 10^-8 for every reported cost and delivery-time comparison, and deterministic 95% bootstrap confidence intervals excluded zero.

Paired Cohen dz values were extremely large because between-seed variation in the paired differences was small relative to the disruption effect. These values confirm numerical stability across seeds but should not be interpreted using conventional small, medium and large behavioural-science labels. Statistical significance applies to the cost and time response; gateway activation itself is reported descriptively from the cross-seed non-dominated union.

## 4.7 Main finding

The experiments identify two distinct resilience regimes. Moderate capacity loss and delays up to 168 hours are absorbed mainly within the West Coast gateway system. New York/New Jersey becomes operationally meaningful only when West Coast sailing delays become prolonged or when a 168-hour delay is combined with an extreme regional capacity contraction. The threshold is destination-dependent: Memphis and Columbus activate the East Coast route earlier than Chicago under isolated delay, while Chicago requires remaining West Coast capacity to fall below approximately 35% under the tested compound disruption.

These findings are based exclusively on operational cost, timetable, capacity and delivery-time variables. No carbon price, emissions constraint or carbon objective is included.

## Source tables

- [v0.6 scenario effects](../analysis/v06_threshold_scenarios/scenario_effects.csv)
- [v0.6 New York/New Jersey activation](../analysis/v06_threshold_scenarios/ny_activation.csv)
- [v0.7 scenario effects](../analysis/v07_activation_scenarios/scenario_effects.csv)
- [v0.7 New York/New Jersey activation](../analysis/v07_activation_scenarios/ny_activation.csv)
- [v0.7 paired statistics](../analysis/v07_activation_scenarios/paired_statistics.csv)
- [v0.8 scenario effects](../analysis/v08_focused_activation_thresholds/scenario_effects.csv)
- [v0.8 New York/New Jersey activation](../analysis/v08_focused_activation_thresholds/ny_activation.csv)
- [v0.8 paired statistics](../analysis/v08_focused_activation_thresholds/paired_statistics.csv)
