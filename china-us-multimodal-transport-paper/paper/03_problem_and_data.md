# 3. Problem setting and case-study data

## 3.1 Network and decision setting

The case study represents scheduled container transport from Chinese origins to three United States inland destinations: Chicago, Memphis, and Columbus. United States entry is possible through Los Angeles and Long Beach (LALB), Seattle and Tacoma (SETA), and New York and New Jersey (NYNJ). LALB and SETA form the West Coast alternatives; NYNJ is the East Coast alternative. Ocean, rail, and road services are connected by transfer and processing activities at the corresponding nodes.

The three inland destinations are analysed separately. Each destination case contains 40 shipment batches and a total demand of 2,227.5 FEU. The optimisation selects shipment paths, service connections, and split quantities subject to timetable and capacity constraints. It does not represent simultaneous competition among the three destinations for one common facility-capacity pool.

## 3.2 Data construction and units

The model-ready dataset was derived from `ChinaUS_Gateway_Model_Ready_v2.xlsx` and is stored in FEU. Source quantities and capacities recorded in TEU were divided by two, while per-TEU costs and tardiness rates were multiplied by two when converted to FEU. The service calendar spans eight weeks and records recurring ocean, rail, and road departures together with travel time, cutoff time, transfer requirements, departure capacity, daily arc capacity, and node-day capacity.

The data combine workbook records, publicly available route and service information cited in the data appendix, and constructed case-study parameters where directly comparable observations were unavailable. They should therefore be interpreted as a reproducible constructed network rather than as a calibrated representation of one carrier's realised operations or market shares.

## 3.3 Time, service, and capacity definitions

All times are measured in hours from a common model origin. A shipment's delivery time is its final arrival time minus its release time. Tardiness is the positive difference between final arrival and the shipment due time. In release-robustness experiments, release and due times are shifted together, preserving the shipment's delivery allowance.

Daily capacity accounting uses zero-based 24 h periods. Model period (j) is the interval ([24j,24(j+1))). For each scheduled arc, a shipment uses the earliest departure whose cutoff remains feasible after arrival and required processing at the departure node. The model does not allow an active choice to wait for a later departure on the same structural path. Capacity conclusions therefore describe freight reallocation under this earliest-reachable-service rule.

Because destinations are solved separately, gateway-node, daily-arc, and service capacities represent capacity available to the focal modelled flow in each destination case. They are not estimates of total facility throughput shared by all market demand, and they are not described as observed contractual allocations.

## 3.4 Cost specification and cargo profile

Total operational cost includes transport, transfer and processing, waiting and holding, and tardiness. Shipment-specific tardiness penalties in the workbook range from USD 60 to USD 200 per FEU-hour. These are constructed case-study inputs intended to represent a time-sensitive cargo profile; they are not calibrated from an observed shipper contract. The reference scenario should consequently be interpreted as a time-sensitive decision setting. A shipper placing less monetary weight on tardiness may select more East Coast capacity, as examined in the penalty sensitivity analysis.

This distinction is important for interpretation. Gateway share, monetary value, and time recovery are separate outcomes. The minimum-time result is independent of the tardiness penalty because it optimises delivery time first, whereas the minimum-cost gateway configuration depends on how tardiness is priced.

## 3.5 Reference and disruption conditions

The single-wave reference releases all 40 batches at time zero and is used as a concentrated-release comparison. Disruptions are introduced as reductions in West Coast gateway-node or inland capacity, additions to West Coast sailing time, or combinations of the two. Alternative release designs retain the same shipment quantities and total demand but distribute batches across two weekly waves, four weekly waves, or daily release over four weeks. These designs test the portability of the concentrated-release findings rather than replacing the reference with an assumed empirical arrival process.
