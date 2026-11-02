# 6. Experimental design

## 6.1 Research logic and computational settings

The experiments were designed to identify when an East Coast gateway becomes operationally relevant in a scheduled China–US multimodal freight network and to distinguish this question from the separate question of whether disruption worsens network performance. Three cases represented the inland destinations Chicago (D1), Memphis (D2), and Columbus (D3). Freight could enter the United States through Los Angeles/Long Beach, Seattle/Tacoma, or New York/New Jersey (NY/NJ). No carbon price, emission constraint, or emission objective was included.

The model minimized total operational cost and quantity-weighted mean delivery time. Total cost comprised transport, transfer, waiting, processing, and shipment-specific tardiness costs. Demand conservation, fixed service schedules, service and node capacities, and transfer requirements applied in every scenario. Each shipment could use at most three paths, and an active path had to carry at least 10% of that shipment's demand. This path-level execution constraint is distinct from the network-level NY/NJ activation criterion defined in Section 6.3.

Each case–scenario combination was solved by 30 independent NSGA-II runs using the common seed sequence 20260906–20260935. The population size was 48 and the algorithm ran for 99 generations after the initial population, giving 4,800 objective evaluations per run. Crossover and mutation probabilities were 0.90 and 0.12. The same evaluation budget and seed sequence were retained across paired scenario comparisons. The reported fronts are best-found approximations under this computational budget; they are not claimed to be exact Pareto fronts.

## 6.2 Experiment architecture

The experiments were organized by inferential purpose rather than by software version. Table 1 summarizes the resulting architecture.

Table 1. Experimental architecture and analytical role.

| Stage | Conditions | Purpose | Primary evidence |
| --- | --- | --- | --- |
| Baseline structure | Original capacities and schedules | Establish normal gateway allocation and cost–time trade-offs | Cross-seed non-dominated archive |
| Capacity mechanism | Regional West Coast capacity at 70%, 67.5%, 65%, 60%, 40%, and 35%; individual West Coast gateway at 50% | Determine whether capacity scarcity is absorbed within the West Coast system | Gateway shares and reconstructed node-day utilization |
| Delay activation | Regional delays of 96, 120, 168, 240, 264, 288, 312, and 336 h | Identify destination-specific activation intervals | Full pooled-front NY/NJ share ranges |
| Capacity–delay interaction | Remaining West Coast capacity of 60%, 40%, or 35% crossed with delays of 0, 168, or 288 h; additional 30% compound case | Distinguish no activation, transition, and front-wide activation regimes | Categorical two-dimensional regime map |
| Capacity-days comparison | Baseline demand at 40% capacity versus 1.25 times demand at 50% capacity | Examine whether comparable demand relative to daily capacity reproduces gateway allocation | Balanced allocations and utilization diagnostics |
| NY/NJ counterfactual | NY/NJ allowed versus prohibited under the first delay activation point and the 30% + 168 h compound disruption | Quantify feasibility and performance value attributable to access to NY/NJ | Archive-closed paired endpoints and common-scale hypervolume |
| Destination mechanism | Balanced solutions immediately below and above each delay activation interval | Explain destination asymmetry | Gateway-level cost and time decomposition |

Capacity percentages denote capacity remaining relative to the baseline. A 30% value therefore represents a 70% reduction. Delays were added to affected West Coast ocean departures and propagated through subsequent arrival and inland-connection times.

The quantity obtained by dividing scenario demand (FEU) by combined daily West Coast capacity (FEU/day) is reported as *nominal capacity-days required*. It has units of days and is not interpreted as a dimensionless utilization ratio. Capacity binding is assessed instead from reconstructed node-day, arc-day, and service-departure utilization.

## 6.3 Pareto archive and gateway activation

For each case and scenario, the feasible non-dominated solutions from the 30 runs were pooled and filtered again for dominance. The resulting cross-seed archive describes the cost–time alternatives found under the common search budget. The number of points in an archive is not treated as a statistical sample size, and solutions are not assigned probabilities.

For solution \(x\), the network share routed through NY/NJ is

\[
g(x)=\frac{Q_{\mathrm{NY/NJ}}(x)}{Q_{\mathrm{total}}}.
\]

A share of 10% was used as the primary operational-materiality criterion, \(\tau=0.10\). The archive was classified as follows:

\[
\text{No meaningful activation if }
\max_{x\in F_{d,s}}g(x)<\tau,
\]

\[
\text{Transition regime if }
\min_{x\in F_{d,s}}g(x)<\tau\leq
\max_{x\in F_{d,s}}g(x),
\]

\[
\text{Front-wide activation if }
\min_{x\in F_{d,s}}g(x)\geq\tau.
\]

All archive solutions, including zero-share solutions, enter the minimum and maximum. The balanced solution is shown as an interpretable reference point but does not determine the activation classification. The earlier term *stable activation* is therefore replaced by *front-wide operational activation*. This term refers to the best-found pooled front, not to a real-world activation probability.

Because disruption levels were tested on a discrete grid, the analysis identifies interval-censored activation boundaries. If \(h^-\) is the last tested delay below front-wide activation and \(h^+\) is the first tested delay with front-wide activation, the result is reported as \((h^-,h^+]\). No response is interpolated between tested conditions.

As a supplementary classification check, the same stored archives are reclassified at 5%, 10%, and 20%. This does not rerun the optimization. Its purpose is to determine whether the ordering of destination responses depends on the reporting threshold.

## 6.4 Counterfactual value of NY/NJ access

The value of NY/NJ cannot be inferred by comparing a disrupted scenario with the undisrupted baseline because that comparison combines the disruption loss and the gateway response. The counterfactual analysis instead compares the same disruption with NY/NJ allowed and prohibited.

The restricted feasible set is nested within the allowed feasible set:

\[
X^{R}\subseteq X^{A}.
\]

Independent heuristic searches may nevertheless return an allowed archive that misses a point found by the restricted search. Before comparing objectives, the allowed archive is therefore completed with the restricted archive:

\[
\widetilde F^{A}_{r}=ND\left(F^{A}_{r}\cup F^{R}_{r}\right),
\]

for each paired seed \(r\). Endpoint benefits are then

\[
\Delta C_r=min_{x\in F^R_r}C(x)-
\min_{x\in \widetilde F^A_r}C(x)\geq0,
\]

\[
\Delta T_r=min_{x\in F^R_r}T(x)-
\min_{x\in \widetilde F^A_r}T(x)\geq0.
\]

This archive-closure step enforces the known feasible-set relationship; it is not an ex-post deletion of unfavorable observations. The raw differences are retained in the analysis file for auditability.

The comparison also reports normalized hypervolume. Allowed and restricted fronts are normalized using a common ideal and nadir obtained from their pooled objective points, with reference point \((1.1,1.1)\). Hypervolumes stored by the original independent experiment batches are not compared because their normalization bounds differ.

## 6.5 Statistical summaries and scope of inference

Scenario–baseline comparisons and allowed–restricted comparisons use the same random seed on both sides of each pair. For an unconstrained paired difference, the analysis reports the mean, median, a 95% percentile bootstrap interval based on 10,000 paired resamples, and the number of positive and negative pairs. Exact sign tests with Holm correction are retained for scenario–baseline comparisons.

Archive-closed counterfactual benefits are structurally non-negative. Sign tests are not used for these benefits because archive closure converts missed allowed points into ties by construction. Instead, the analysis reports the mean and median benefit, bootstrap intervals, and the number of runs with a strict improvement. Cost benefits are additionally expressed as a percentage of the corresponding restricted value.

These intervals quantify variation across stochastic optimization runs under fixed model inputs. They do not represent uncertainty in actual freight demand, prices, service schedules, or disruption duration. Gateway shares are descriptive properties of the pooled approximate front and are not subjected to inferential tests.

## 6.6 Destination-mechanism decomposition

For each destination, a balanced pooled-front solution was selected immediately below and above the delay activation interval. Gateway-specific cost was decomposed into China inland transport, ocean transport, US inland transport, holding, transfer, processing, and tardiness components. Lead time was decomposed into China inland transit, ocean transit, US inland transit, waiting, and transfer time.

The decomposition is used to assess whether the longer all-water service to NY/NJ is offset by reduced West Coast waiting and a different US inland connection. It is descriptive evidence from representative solutions, not an average causal effect over all solutions.
