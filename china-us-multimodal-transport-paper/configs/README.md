# Constraint and scenario rules

The baseline is a bi-objective pilot: total cost and makespan. All quantities are FEU (one 40-foot-equivalent container). Convert TEU evidence before loading it.

Rules for experiments:

1. Use `pilot_baseline` as the control.
2. Change one mechanism at a time for an ablation.
3. Never alter algorithm operators, population size or evaluation budget in a constraint ablation.
4. Keep timetable and capacity enabled for all policy scenarios.
5. Panama multipliers apply only to services whose ocean arc has `via_panama=true`.
6. The Panama multiplier represents effective service availability, not canal TEU capacity.
7. The `panama_stress` case is hypothetical and must be labelled as a stress test.
8. Soft due dates do not affect feasibility. If tardiness is priced, document the coefficient before running experiments.

The freight-gap scenarios keep the Shanghai–LA baseline at USD 5,861.75/FEU and move the Shanghai–NY price to produce the observed premium endpoints of USD 940 and USD 2,515.
