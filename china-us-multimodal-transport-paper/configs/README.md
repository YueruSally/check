# Constraint and scenario rules

The model minimizes total USD cost and quantity-weighted mean delivery time.
Makespan remains a reported service-level indicator. All quantities and
capacities use FEU.

## Common experimental rules

1. Use `pilot_baseline` as the control.
2. Keep timetable, capacity and the 10% minimum active-path share enabled.
3. Use the same NSGA-II operators, seed sequence and 4,800-evaluation budget for
   every case and scenario.
4. Change one mechanism at a time in capacity and delay sensitivity runs.
5. Treat compound scenarios separately from one-factor sensitivity results.
6. Soft due dates do not affect feasibility. If tardiness is priced, document
   the coefficient before running experiments.
7. The v0.5 scenario suite contains no carbon-price, carbon-cap or emissions
   objective.

## Regional port-capacity mechanism

`port_capacity_multiplier` scales every Chinese and US port.
`uswc_port_capacity_multiplier` additionally scales LA/LB and
Seattle/Tacoma, while `usec_port_capacity_multiplier` additionally scales
NY/NJ. Multipliers are applied to daily node capacity; values above one
represent capacity expansion and values below one represent disruption.

The v0.5 matrix uses West Coast capacity at 125%, 75% and 50% of baseline.

## Voyage-delay mechanism

`uswc_ocean_departure_delay_h` applies to non-Panama ocean arcs serving
LA/LB and Seattle/Tacoma. `usec_ocean_departure_delay_h` applies to Panama
all-water arcs serving NY/NJ. Delay is added after the published sailing
departure. It therefore increases port waiting time, holding cost, arrival
time, weighted mean delivery time and any tardiness.

The v0.5 matrix includes West Coast delays of 24 and 72 hours, a 48-hour delay
on all ocean services, and moderate/severe West Coast compound disruptions.

Historical Panama, freight-gap and demand scenarios remain in the configuration
for reproducibility but are not part of `scripts/run_noncarbon_scenarios.sh`.
