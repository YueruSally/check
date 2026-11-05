# 6. Experimental design and analysis

## 6.1 Exact solution framework

The final analysis is based on a path-allocation mixed-integer linear program rather than on the earlier exploratory evolutionary searches. All simple structural paths satisfying the model limits were enumerated: a path may contain at most eight transport legs and four mode changes, and it may not revisit a node. This produced eight feasible structural paths for every origin–destination–gateway combination. A structural path fixes the sequence of nodes and modes; for each scheduled arc, the evaluator assigns the earliest departure whose cutoff can still be met after the shipment arrives and completes the required transfer processing. The model does not allow a shipment to wait deliberately for a later departure on the same structural path.

Every model enforces shipment conservation, at most three active paths per shipment, a minimum allocation of 10% of the shipment quantity on every active path, service-departure capacity, daily arc capacity, node-day capacity, path continuity, timetable compatibility, and transfer processing time. The daily capacity period is zero based: period (j) represents the 24 h interval ([24j,24(j+1))), measured from the model time origin. A shipment's delivery time is its final arrival time minus its release time. The second objective is the quantity-weighted mean of these shipment delivery times.

For every case and condition, two access designs were solved: one with NYNJ in the gateway choice set and one without NYNJ. Two lexicographic endpoints were obtained for each design:

1. minimise total cost, then minimise weighted mean delivery time while retaining the proven cost optimum within the specified absolute-plus-relative tolerance; and
2. minimise weighted mean delivery time, then minimise cost while retaining the proven time optimum within the corresponding tolerance.

An endpoint is reported as proven only when both stages terminate optimally within the recorded tolerance. Each decoded allocation is then recomputed by an independent evaluator. The term *evaluator-consistent* means that the independently recomputed feasibility status, objective values, shipment balances, and all service, arc-day, and node-day capacity accounts agree with the solver output within the stated numerical tolerances. At every endpoint with NYNJ, two additional models minimise and maximise the NYNJ share while retaining both endpoint objectives within their tolerances. The resulting interval identifies all proven aggregate shares in that endpoint-equivalent objective region; equality of the bounds establishes a unique aggregate gateway share, not a unique shipment-level allocation.

## 6.2 Case structure and reference design

Chicago, Memphis, and Columbus are analysed as three separate destination cases. They do not compete simultaneously for one shared capacity pool. The capacity inputs are therefore interpreted as capacity available to the modelled focal flow in each case, not as total observed facility utilisation or as an empirically observed contract quota. Each case contains 40 shipment batches and 2,227.5 FEU. The single-wave reference releases all batches at time zero and represents a concentrated-release case. It is retained as the controlled reference for the core delay, capacity, and counterfactual comparisons; the multi-wave experiments test how far its results depend on the temporal distribution of demand.

The total-cost objective contains transport, transfer and processing, waiting and holding, and shipment-specific tardiness costs. Tardiness is measured against the shipment due time. The batch-specific penalty rates in the case-study workbook range from USD 60 to USD 200 per FEU-hour. They are constructed inputs for a time-sensitive cargo case and are not calibrated to an observed shipper contract. This modelling choice is material: a high tardiness price favours West Coast routes when they retain a delivery-time advantage, whereas a cost-focused shipper facing a lower tardiness price can use more East Coast capacity. The zero-penalty experiment is therefore treated as a specification sensitivity analysis rather than as an alternative empirical calibration.

## 6.3 Core experimental conditions

The final experiments are organised by their evidential role.

| Experiment | Conditions | Primary question |
|---|---|---|
| Undisturbed reference | Original timetable and capacities | How do the exact cost and time endpoints allocate freight? |
| Gateway-node capacity | 60%, 40%, and 35% of baseline; 30% plus 168 h delay | Can West Coast reallocation absorb gateway-node reductions? |
| Sailing delay | Tested single-wave grid plus focused observations at Columbus 216 h and Memphis 276 h | At which tested delays does NYNJ become materially used? |
| Counterfactual access | With NYNJ versus without NYNJ under the same disruption | What cost or time loss does NYNJ avoid? |
| Cost sensitivity | Tardiness penalty from 0 to 200% of baseline; holding cost from 0 to 200% | How does the objective specification change gateway use and value? |
| Release design | Single wave, two weekly waves, four weekly waves, and daily release over four weeks | How does the temporal distribution of demand affect gateway use? |
| Demand scale | 80%, 100%, and 120% of reference demand | Does freight volume change the endpoint response? |
| Inland capacity | Regional West Coast inland capacity at 50% and 30%; focused local-arc levels at 50%, 100%, and 150% | Does capacity location change the gateway response? |
| Delivery-time preference | Eight internal time limits for Chicago 288 h and Columbus 240 h | How does gateway use change between the two exact endpoints? |

When release times are reassigned in the multi-wave experiments, each shipment due time is shifted by the same amount, so its release-to-due allowance is preserved. Five predetermined assignments distribute batches across each multi-wave pattern while keeping shipment quantities and total demand unchanged. The same West Coast delay applies throughout each constructed instance.

## 6.4 Gateway use and criterion sensitivity

For allocation (x), the NYNJ share is

\[
g(x)=\frac{Q_{\mathrm{NYNJ}}(x)}{Q_{\mathrm{total}}}.
\]

The primary material-use criterion is 10%. Five and 20% are reported as criterion sensitivities. For each endpoint, classification uses the proven minimum share in the endpoint-equivalent objective region. A destination's first material-use condition is the first tested single-wave delay at which the proven minimum share at either endpoint reaches 10%. Because delays are evaluated at discrete levels, the reported brackets describe adjacent tested conditions rather than continuously estimated switching times.

The release-design comparison uses the common grid \(\{0,168,240,264,288,336\}\) h. The focused Columbus observation at 216 h and Memphis observation at 276 h refine the single-wave evidence but are not inserted into multi-wave designs that were not solved at those levels.

## 6.5 Counterfactual value, disruption loss, and recovery

Let (f_{m,B}) be the undisturbed reference value for metric (m), (f_{m,R}(\delta)) the disrupted value without NYNJ, and (f_{m,A}(\delta)) the disrupted value with NYNJ. The avoided loss from access to NYNJ is

\[
G_m(\delta)=f_{m,R}(\delta)-f_{m,A}(\delta).
\]

The disrupted loss without NYNJ is (L_m(\delta)=f_{m,R}(\delta)-f_{m,B}). When this denominator is positive and exceeds the numerical tolerance, the recovered fraction is

\[
R_m(\delta)=\frac{G_m(\delta)}{L_m(\delta)}.
\]

Cost recovery is evaluated at the minimum-cost endpoint and time recovery at the minimum-time endpoint. These are different allocations and are not combined into one benefit measure.

Under the zero-penalty specification, the undisturbed solution with NYNJ may already outperform the one without it. The analysis therefore reports incremental gateway value instead of a recovery ratio. With

\[
V(\delta)=C_R(\delta)-C_A(\delta), \qquad
V(0)=C_R(0)-C_A(0),
\]

the disruption-induced increment is

\[
\Delta V(\delta)=V(\delta)-V(0).
\]

All differences and ratios are computed from unrounded objective values.

## 6.6 Exact intermediate trade-off solutions

For Chicago at 288 h and Columbus at 240 h, eight internal delivery-time limits were placed between the exact endpoints with NYNJ. At each limit, cost was minimised and time was used as the lexicographic tie-breaker. The identical limit was imposed on the model without NYNJ. Additional share-minimisation and share-maximisation models established the NYNJ share range at each exact intermediate solution. These points are exact solutions to the stated epsilon-constrained problems; they are not presented as an enumeration of a continuous Pareto frontier.

## 6.7 Evidential scope

The principal claims use the complete structural path set, proven lexicographic endpoints, proven endpoint share bounds, and exact epsilon-constrained solutions. Historical NSGA-II results are retained only as exploratory development records and do not determine the final endpoint, gateway-share, or activation conclusions. Exactness remains conditional on the modelled choice set and the earliest-reachable-service rule. The three destination cases are independent, the case-study parameters are constructed from the workbook and cited secondary sources, and the network contains NYNJ as its only East Coast alternative. These scope conditions are considered when interpreting the numerical delay brackets and capacity responses.
