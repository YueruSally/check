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

## v0.6 threshold and gateway-specific mechanisms

The v0.6 experiment retains the 10% minimum active-path share and contains no
carbon variable, objective, price or emissions constraint.

`la_lb_port_capacity_multiplier` and
`seattle_tacoma_port_capacity_multiplier` scale the two West Coast gateway
capacities separately. They multiply the common
`port_capacity_multiplier` and `uswc_port_capacity_multiplier`, so regional
and gateway-specific disruptions can be combined without changing the base
data. The capacity-threshold runs refine the interval between the non-binding
75% scenario and the binding 50% scenario at 70%, 65% and 60%.

`la_lb_ocean_departure_delay_h` and
`seattle_tacoma_ocean_departure_delay_h` add gateway-specific delays on top
of any common West Coast delay. The matrix separates the two gateways at 72
hours, extends the regional delay to 96, 120 and 168 hours, and includes one
50%-capacity plus 120-hour compound scenario. The baseline is rerun with the
same seeds to provide a paired control for this version.
