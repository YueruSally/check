# Source workbook mapping

The source workbook is an evidence and calibration workbook, not yet a solver-ready network table.

## Inputs already mapped

| Workbook area | Model field | Status |
|---|---|---|
| Baseline_Inputs ocean costs | Ocean arc `cost_usd_per_feu` | Ready |
| Baseline_Inputs LA–Chicago rail time | Rail arc `transit_time_h` | Ready |
| Baseline_Inputs terminal handling | Transfer `cost_usd_per_feu` | Ready |
| Panama scenario multipliers | Scenario `panama_availability_multiplier` | Ready |
| Ocean_Freight premium range | Low/high USEC freight scenarios | Ready |
| Gateways_Destinations | Node shortlist | Ready for screening |

## Required before the first optimization run

- Complete node identifiers for Chinese origins, ports, US ports, hubs and destinations.
- Directed arc distance, transit time and standardized USD/FEU cost.
- Named ocean and rail services with recurring departures, cutoff and FEU capacity.
- China inland rail tariffs and timetables.
- US inland costs for Memphis and Columbus, not only LA–Chicago.
- Transfer time and cost for every mode change used by a path.
- Shipment quantities, releases and due dates.
- Explicit classification of each USEC ocean service as Panama-exposed or not.

Missing values must remain blank and fail validation. They must never be silently replaced with zero.
