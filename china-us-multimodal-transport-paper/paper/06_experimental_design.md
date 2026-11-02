# 6. Experimental design

## 6.1 Research logic and computational settings

The experiments identify when New York/New Jersey (NY/NJ) becomes an operationally material alternative in a scheduled China--US multimodal freight network. Three cases represent the inland destinations Chicago (D1), Memphis (D2), and Columbus (D3). Freight can enter the United States through Los Angeles/Long Beach, Seattle/Tacoma, or NY/NJ. The model contains no carbon price, emission constraint, or emission objective.

The two objectives minimize total operational cost and quantity-weighted mean delivery time. Cost includes transport, transfer, waiting, processing, and shipment-specific tardiness. Every scenario enforces demand conservation, fixed timetables, transfer compatibility, service-departure capacity, daily arc capacity, and node-day capacity. A shipment may use at most three paths, and every active path must carry at least 10% of that shipment's demand. This path-level execution constraint is distinct from the network-level gateway criterion in Section 6.3.

Each case--scenario combination is solved by 30 independent NSGA-II runs with the common seed sequence 20260906--20260935. A run uses 48 individuals and 99 generations after initialization, hence (48(99+1)=4{,}800) objective evaluations. Crossover and mutation probabilities are 0.90 and 0.12. Paired comparisons use the same seeds and evaluation budget. These searches provide best-found approximate fronts; they are not labelled exact Pareto fronts.

## 6.2 Experiment architecture

The experiments are organized by inferential purpose rather than software version (Table 1).

Table 1. Experimental architecture and evidential role.

| Stage | Conditions | Purpose | Primary evidence |
| --- | --- | --- | --- |
| Baseline | Original capacities and schedules | Establish normal gateway allocation and cost--time trade-offs | Cross-seed non-dominated archive |
| Capacity mechanism | Regional West Coast capacity at 70%, 67.5%, 65%, 60%, 40%, and 35%; either West Coast gateway at 50% | Test whether capacity reductions are accommodated through West Coast reallocation | Gateway shares and reconstructed utilization |
| Delay activation | Regional delays of 96, 120, 168, 240, 264, 288, 312, and 336 h | Locate destination-specific gateway-use regimes | Configuration-preserving closed archives at the critical conditions |
| Capacity--delay interaction | Remaining capacity of 60%, 40%, or 35% crossed with delays of 0, 168, or 288 h; additional 30% compound condition | Screen how capacity modifies delay-driven substitution | Allowed-search regime map plus a closed Chicago compound comparison |
| NY/NJ counterfactual | NY/NJ allowed versus prohibited under four critical delays and one compound disruption | Separate gateway value from disruption loss | Closed fronts and exact lexicographic endpoints |
| Mechanism audit | Seed-paired minimum-time solutions at four critical delays | Attribute system time changes to shipment-matched NY/NJ and West Coast components | Paired decomposition and service-departure flow changes |

Capacity percentages denote capacity remaining relative to baseline. Delays are added to affected West Coast ocean departures and propagated to arrivals and inland connections. The quantity obtained by dividing demand (FEU) by combined daily West Coast capacity (FEU/day) is reported only as *nominal capacity-days required*. Because it has units of days, it is not treated as a utilization ratio; binding is assessed from the reconstructed resource accounts.

## 6.3 Configuration-preserving archives and activation definitions

For each run, objective-point records used for hypervolume and plots are separated from route-allocation configurations used for gateway shares. All distinct configurations *found* at retained objective coordinates are preserved; exact duplicates may be removed. A separate observational archive records feasible non-dominated configurations during the search but does not enter tournament selection or environmental replacement. It therefore does not change the 4,800-evaluation search budget.

At the four critical delay conditions, the allowed and matching NY/NJ-restricted configurations are pooled and screened again for non-dominance. The restricted set is nested in the allowed feasible set, (X^R\subseteq X^A); adding restricted solutions before filtering closes an omission that can arise from independent heuristic searches. All distinct found configurations attached to retained objective coordinates remain in the share calculation. This is a configuration-preserving closure of the solutions found, not a claim to enumerate every feasible configuration or the unknown exact Pareto set.

For a retained configuration (x), the network share routed through NY/NJ is

\[
g(x)=\frac{Q_{\mathrm{NY/NJ}}(x)}{Q_{\mathrm{total}}}.
\]

The primary operational-materiality criterion is (	au=0.10). Two events are distinguished:

\[
\text{first-solution activation if }\max_{x\in F_{d,s}}g(x)\geq\tau,
\]

\[
\text{front-wide activation if }\min_{x\in F_{d,s}}g(x)\geq\tau.
\]

If the first condition holds but the second does not, the condition is a *transition*. Thus first-solution activation shows that NY/NJ is competitive for at least one found cost--time preference, whereas front-wide activation means that every retained found configuration uses it materially. “Front-wide” always refers to the retained best-found approximate archive. Because disruption levels are discrete, boundaries are interval-censored and no interpolation is made between tested conditions. The stored archives are also reclassified at 5% and 20% without rerunning the optimizer.

## 6.4 Counterfactual comparison and exact endpoint validation

Comparing a disrupted case with the undisrupted baseline cannot isolate NY/NJ value. The counterfactual instead compares the same disruption with NY/NJ allowed and prohibited. Closed heuristic fronts describe the shape of the found trade-off set, but primary endpoint values are obtained from a path-allocation mixed-integer linear program using the same candidate-path library and all original allocation constraints.

For each access condition, two lexicographic endpoints are solved:

1. minimize cost, then minimize time while retaining the proven cost optimum within an explicit absolute-plus-relative tolerance; and
2. minimize time, then minimize cost under the analogous time-optimality tolerance.

The validation covers ten case--condition--access models at two endpoints, giving 20 endpoint tasks and at most 40 solver calls. The MILP contains shipment conservation, at most three active paths, the 10% active-path minimum, timetable and transfer accounting, service-departure capacity, daily arc capacity, node-day capacity, and scenario-specific gateway restrictions. An endpoint is called *proven* only when both solver stages terminate optimally within the recorded tolerance. Decoded allocations are independently re-evaluated for feasibility, both objectives, and all three capacity accounts. Exactness therefore applies to the candidate-path MILP, not to routes outside the generated candidate library.

For an endpoint (e), the value of access is reported as

\[
\Delta C_e=C^R_e-C^A_e,
\qquad
\Delta T_e=T^R_e-T^A_e.
\]

Positive values favour allowing NY/NJ. Because the cost-optimal and time-optimal rows represent different decision preferences, their cost and time differences are not combined into a single benefit score.

## 6.5 Seed-paired mechanism audit

Mechanism evidence uses the minimum-time configuration from each allowed--restricted seed pair. If the independently found restricted endpoint dominates the allowed endpoint, closure selects the restricted configuration because it is also feasible in the allowed problem. Each shipment's mean lead time in the restricted configuration becomes its matching benchmark. The quantity-weighted system change is decomposed as

\[
\Delta T_{\mathrm{NY/NJ}}+
\Delta T_{\mathrm{West}}+
\Delta T_{\mathrm{other}}
=T^A-T^R.
\]

The sign convention is allowed minus restricted, so negative values denote improvement. The identity is checked for every seed. This is a shipment-matched decomposition, not unique tracing of each unit of split freight. Service-departure quantity changes are reported separately; a capacity-release interpretation is made only where both the component accounting and service changes support it.

For each component, the analysis reports the mean, median, a 95% percentile bootstrap interval from 10,000 paired resamples, and the numbers of negative, positive, and tied seeds. These intervals measure stochastic-search variation under fixed inputs, not real-world uncertainty in demand, prices, schedules, or disruption duration.

## 6.6 Scope of the capacity--delay map

The broad capacity--delay grid was generated by the NY/NJ-allowed search and is retained as a screening map. Only the four targeted delay conditions were rerun with configuration-preserving allowed/restricted closure. The historical Chicago condition with 30% remaining capacity and a 168 h delay has an existing closed objective archive sufficient to reject front-wide activation, but it was not rerun as a configuration-complete condition. Cells without a matching restricted closure are therefore labelled exploratory rather than final front-wide evidence.
