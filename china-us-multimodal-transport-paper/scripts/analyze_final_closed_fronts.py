#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from analyze_paired_statistics import bootstrap_interval, stable_seed
from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import capacity_usage_rows, route_component_rows
from china_us_multimodal.domain import RouteAllocation
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.metrics import nondominated_points


COMPARISONS = (
    (
        "D1",
        "delay_288h",
        "uswc_departure_delay_288h",
        "uswc_departure_delay_288h_no_ny_nj",
    ),
    (
        "D2",
        "delay_264h",
        "uswc_departure_delay_264h",
        "uswc_departure_delay_264h_no_ny_nj",
    ),
    (
        "D2",
        "delay_288h",
        "uswc_departure_delay_288h",
        "uswc_departure_delay_288h_no_ny_nj",
    ),
    (
        "D3",
        "delay_240h",
        "uswc_departure_delay_240h",
        "uswc_departure_delay_240h_no_ny_nj",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def point(row: dict[str, str]) -> tuple[float, float]:
    return (
        float(row["total_cost_usd"]),
        float(row["quantity_weighted_mean_delivery_time_h"]),
    )


def gateway(data, arc_ids: tuple[str, ...]) -> str:
    for arc_id in arc_ids:
        arc = data.arcs[arc_id]
        for node_id in (arc.origin, arc.destination):
            if data.nodes[node_id].kind.value == "us_port":
                return node_id
    return "unclassified"


def load_allocations(
    rows: list[dict[str, str]], id_field: str
) -> dict[str, tuple[RouteAllocation, ...]]:
    indexed: dict[str, list[RouteAllocation]] = {}
    for row in rows:
        indexed.setdefault(row[id_field], []).append(
            RouteAllocation(
                shipment_id=row["shipment_id"],
                quantity_feu=float(row["quantity_feu"]),
                arc_ids=tuple(row["arc_ids"].split("|")),
            )
        )
    return {key: tuple(value) for key, value in indexed.items()}


def gateway_share(data, allocations: tuple[RouteAllocation, ...], target: str) -> float:
    total = sum(item.quantity_feu for item in allocations)
    selected = sum(
        item.quantity_feu
        for item in allocations
        if gateway(data, item.arc_ids) == target
    )
    return selected / total if total else 0.0


def endpoint_index(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    result: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        if "min_delivery_time" not in row["roles"].split("|"):
            continue
        result[(row["scenario"], int(row["seed"]))] = row
    return result


def weighted_mean_lead(components: list[dict[str, object]]) -> float:
    total = sum(float(row["quantity_feu"]) for row in components)
    return (
        sum(
            float(row["quantity_feu"]) * float(row["lead_time_h"])
            for row in components
        )
        / total
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Close the four targeted allowed/restricted archives, classify NY/NJ "
            "activation using all distinct found configurations, and perform the "
            "seed-paired batch-matched time decomposition."
        )
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-threshold", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument(
        "--historical-closed-front",
        type=Path,
        default=Path("analysis/paper_revised/counterfactual_front_points.csv"),
        help=(
            "Existing closure used only to carry the known Chicago compound "
            "transition correction into the final audit."
        ),
    )
    args = parser.parse_args()
    if not 0 < args.activation_threshold < 1:
        parser.error("--activation-threshold must be in (0, 1)")

    args.output.mkdir(parents=True, exist_ok=True)
    full_data = load_model_data(args.data)
    closed_point_rows: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    service_change_rows: list[dict[str, object]] = []

    for case_id, comparison, allowed_scenario, restricted_scenario in COMPARISONS:
        result_dir = args.results / case_id
        data = select_case(full_data, case_id)
        fronts = read_csv(result_dir / "combined_pareto.csv")
        front_by_scenario = {
            scenario: [point(row) for row in fronts if row["scenario"] == scenario]
            for scenario in (allowed_scenario, restricted_scenario)
        }
        closed_front = nondominated_points(
            front_by_scenario[allowed_scenario]
            + front_by_scenario[restricted_scenario]
        )

        configuration_solutions = read_csv(
            result_dir / "found_configuration_solutions.csv"
        )
        configuration_allocations = load_allocations(
            read_csv(result_dir / "found_configuration_allocations.csv"),
            "configuration_id",
        )
        configuration_by_point: dict[
            tuple[str, tuple[float, float]], list[dict[str, str]]
        ] = {}
        for row in configuration_solutions:
            configuration_by_point.setdefault(
                (row["scenario"], point(row)), []
            ).append(row)

        all_shares: list[float] = []
        zero_configurations = 0
        total_configurations = 0
        points_with_equal_objective_variants = 0
        for point_index, objective_point in enumerate(closed_front, start=1):
            shares: list[float] = []
            sources: list[str] = []
            configuration_ids: list[str] = []
            for source, scenario in (
                ("allowed", allowed_scenario),
                ("restricted", restricted_scenario),
            ):
                for row in configuration_by_point.get((scenario, objective_point), []):
                    configuration_id = row["configuration_id"]
                    allocations = configuration_allocations[configuration_id]
                    shares.append(gateway_share(data, allocations, "ny_nj"))
                    sources.append(source)
                    configuration_ids.append(configuration_id)
            if not shares:
                raise RuntimeError(
                    f"No saved configuration for closed point {case_id}/{comparison}/"
                    f"{objective_point}."
                )
            if len(shares) > 1:
                points_with_equal_objective_variants += 1
            all_shares.extend(shares)
            zero_configurations += sum(value <= 1e-12 for value in shares)
            total_configurations += len(shares)
            closed_point_rows.append(
                {
                    "case_id": case_id,
                    "comparison": comparison,
                    "allowed_scenario": allowed_scenario,
                    "restricted_scenario": restricted_scenario,
                    "point_id": point_index,
                    "total_cost_usd": objective_point[0],
                    "quantity_weighted_mean_delivery_time_h": objective_point[1],
                    "configuration_count": len(shares),
                    "source_archives": "|".join(sorted(set(sources))),
                    "ny_nj_share_min": min(shares),
                    "ny_nj_share_max": max(shares),
                    "configurations_below_threshold": sum(
                        value < args.activation_threshold - 1e-12 for value in shares
                    ),
                    "configurations_at_or_above_threshold": sum(
                        value >= args.activation_threshold - 1e-12 for value in shares
                    ),
                    "configuration_ids": "|".join(configuration_ids),
                }
            )
        if not all_shares:
            classification = "no_retained_solution"
            first_solution_activation = 0
            front_wide_activation = 0
            minimum_share = maximum_share = ""
        else:
            minimum_share = min(all_shares)
            maximum_share = max(all_shares)
            first_solution_activation = int(
                maximum_share >= args.activation_threshold - 1e-12
            )
            front_wide_activation = int(
                minimum_share >= args.activation_threshold - 1e-12
            )
            if front_wide_activation:
                classification = "front_wide_activation"
            elif first_solution_activation:
                classification = "transition"
            else:
                classification = "inactive"
        classification_rows.append(
            {
                "case_id": case_id,
                "comparison": comparison,
                "allowed_scenario": allowed_scenario,
                "restricted_scenario": restricted_scenario,
                "closed_objective_points": len(closed_front),
                "distinct_found_configurations": total_configurations,
                "zero_ny_nj_configurations": zero_configurations,
                "points_with_equal_objective_configuration_variants": (
                    points_with_equal_objective_variants
                ),
                "activation_threshold": args.activation_threshold,
                "ny_nj_share_min": minimum_share,
                "ny_nj_share_max": maximum_share,
                "first_solution_activation": first_solution_activation,
                "front_wide_activation": front_wide_activation,
                "classification": classification,
                "evidence_scope": (
                    "closed non-dominated union of the allowed/restricted "
                    "configurations found in the targeted 30-seed rerun"
                ),
            }
        )

        endpoint_solutions = read_csv(result_dir / "run_endpoint_solutions.csv")
        endpoints = endpoint_index(endpoint_solutions)
        endpoint_allocations = load_allocations(
            read_csv(result_dir / "run_endpoint_allocations.csv"),
            "endpoint_solution_id",
        )
        allowed_config = load_config(args.config, allowed_scenario)
        restricted_config = load_config(args.config, restricted_scenario)
        allowed_seeds = {
            seed for scenario, seed in endpoints if scenario == allowed_scenario
        }
        restricted_seeds = {
            seed for scenario, seed in endpoints if scenario == restricted_scenario
        }
        for seed in sorted(allowed_seeds & restricted_seeds):
            allowed_row = endpoints[(allowed_scenario, seed)]
            restricted_row = endpoints[(restricted_scenario, seed)]
            allowed_key = (
                float(allowed_row["quantity_weighted_mean_delivery_time_h"]),
                float(allowed_row["total_cost_usd"]),
            )
            restricted_key = (
                float(restricted_row["quantity_weighted_mean_delivery_time_h"]),
                float(restricted_row["total_cost_usd"]),
            )
            closure_selected_restricted = restricted_key < allowed_key
            if closure_selected_restricted:
                selected_allowed_row = restricted_row
                selected_allowed_allocations = endpoint_allocations[
                    restricted_row["endpoint_solution_id"]
                ]
                selected_allowed_config = allowed_config
            else:
                selected_allowed_row = allowed_row
                selected_allowed_allocations = endpoint_allocations[
                    allowed_row["endpoint_solution_id"]
                ]
                selected_allowed_config = allowed_config
            restricted_allocations = endpoint_allocations[
                restricted_row["endpoint_solution_id"]
            ]
            allowed_components = route_component_rows(
                data, list(selected_allowed_allocations), selected_allowed_config
            )
            restricted_components = route_component_rows(
                data, list(restricted_allocations), restricted_config
            )
            restricted_by_shipment: dict[str, list[dict[str, object]]] = {}
            for row in restricted_components:
                restricted_by_shipment.setdefault(str(row["shipment_id"]), []).append(row)
            restricted_mean_by_shipment = {
                shipment_id: weighted_mean_lead(rows)
                for shipment_id, rows in restricted_by_shipment.items()
            }
            total_quantity = sum(float(row["quantity_feu"]) for row in allowed_components)
            contributions = {"ny_nj": 0.0, "west_coast": 0.0, "other": 0.0}
            for row in allowed_components:
                shipment_id = str(row["shipment_id"])
                route_gateway = str(row["gateway"])
                if route_gateway == "ny_nj":
                    group = "ny_nj"
                elif route_gateway in {"la_lb", "seattle_tacoma"}:
                    group = "west_coast"
                else:
                    group = "other"
                contributions[group] += (
                    float(row["quantity_feu"])
                    * (
                        float(row["lead_time_h"])
                        - restricted_mean_by_shipment[shipment_id]
                    )
                    / total_quantity
                )
            allowed_mean = weighted_mean_lead(allowed_components)
            restricted_mean = weighted_mean_lead(restricted_components)
            observed_difference = allowed_mean - restricted_mean
            contribution_sum = sum(contributions.values())
            identity_error = contribution_sum - observed_difference
            if abs(identity_error) > 1e-7:
                raise RuntimeError(
                    f"Time decomposition identity failed for {case_id}/{comparison}/"
                    f"{seed}: {identity_error}."
                )
            reported_allowed_mean = float(
                selected_allowed_row["quantity_weighted_mean_delivery_time_h"]
            )
            reported_restricted_mean = float(
                restricted_row["quantity_weighted_mean_delivery_time_h"]
            )
            endpoint_accounting_error = max(
                abs(reported_allowed_mean - allowed_mean),
                abs(reported_restricted_mean - restricted_mean),
            )
            if endpoint_accounting_error > 1e-7:
                raise RuntimeError(
                    f"Endpoint time accounting failed for {case_id}/{comparison}/{seed}: "
                    f"{endpoint_accounting_error}."
                )
            decomposition_rows.append(
                {
                    "case_id": case_id,
                    "comparison": comparison,
                    "seed": seed,
                    "allowed_scenario": allowed_scenario,
                    "restricted_scenario": restricted_scenario,
                    "raw_allowed_endpoint_solution_id": allowed_row[
                        "endpoint_solution_id"
                    ],
                    "restricted_endpoint_solution_id": restricted_row[
                        "endpoint_solution_id"
                    ],
                    "closed_allowed_endpoint_source": (
                        "restricted_archive"
                        if closure_selected_restricted
                        else "allowed_archive"
                    ),
                    "closed_allowed_endpoint_solution_id": selected_allowed_row[
                        "endpoint_solution_id"
                    ],
                    "ny_nj_share_closed_allowed": gateway_share(
                        data, selected_allowed_allocations, "ny_nj"
                    ),
                    "allowed_mean_time_h": allowed_mean,
                    "restricted_mean_time_h": restricted_mean,
                    "allowed_minus_restricted_time_h": observed_difference,
                    "ny_nj_contribution_h": contributions["ny_nj"],
                    "west_coast_contribution_h": contributions["west_coast"],
                    "other_contribution_h": contributions["other"],
                    "contribution_sum_h": contribution_sum,
                    "identity_error_h": identity_error,
                    "endpoint_accounting_error_h": endpoint_accounting_error,
                    "sign_convention": "allowed_minus_restricted; negative is improvement",
                }
            )

            allowed_usage = {
                (str(row["resource_id"]), str(row["period"])): row
                for row in capacity_usage_rows(
                    data, list(selected_allowed_allocations), selected_allowed_config
                )
                if row["resource_type"] == "service_departure"
            }
            restricted_usage = {
                (str(row["resource_id"]), str(row["period"])): row
                for row in capacity_usage_rows(
                    data, list(restricted_allocations), restricted_config
                )
                if row["resource_type"] == "service_departure"
            }
            for resource_key in sorted(set(allowed_usage) | set(restricted_usage)):
                allowed_used = float(
                    allowed_usage.get(resource_key, {}).get("used_feu", 0.0)
                )
                restricted_used = float(
                    restricted_usage.get(resource_key, {}).get("used_feu", 0.0)
                )
                difference = allowed_used - restricted_used
                if abs(difference) <= 1e-9:
                    continue
                row = allowed_usage.get(resource_key) or restricted_usage[resource_key]
                service_change_rows.append(
                    {
                        "case_id": case_id,
                        "comparison": comparison,
                        "seed": seed,
                        "service_id": resource_key[0],
                        "scheduled_departure_h": resource_key[1],
                        "origin": row["origin"],
                        "destination": row["destination"],
                        "allowed_used_feu": allowed_used,
                        "restricted_used_feu": restricted_used,
                        "allowed_minus_restricted_used_feu": difference,
                    }
                )

    summary_rows: list[dict[str, object]] = []
    component_fields = (
        "allowed_minus_restricted_time_h",
        "ny_nj_contribution_h",
        "west_coast_contribution_h",
        "other_contribution_h",
    )
    for case_id, comparison, _, _ in COMPARISONS:
        group = [
            row
            for row in decomposition_rows
            if row["case_id"] == case_id and row["comparison"] == comparison
        ]
        for field in component_fields:
            values = [float(row[field]) for row in group]
            if values:
                low, high = bootstrap_interval(
                    values,
                    statistics.mean,
                    args.bootstrap_repetitions,
                    stable_seed(case_id, comparison, field),
                )
                mean = statistics.mean(values)
                median = statistics.median(values)
            else:
                low = high = mean = median = ""
            summary_rows.append(
                {
                    "case_id": case_id,
                    "comparison": comparison,
                    "component": field,
                    "paired_seeds": len(values),
                    "mean_h": mean,
                    "mean_bootstrap_95_low_h": low,
                    "mean_bootstrap_95_high_h": high,
                    "median_h": median,
                    "negative_seed_count": sum(value < -1e-9 for value in values),
                    "positive_seed_count": sum(value > 1e-9 for value in values),
                    "tie_seed_count": sum(abs(value) <= 1e-9 for value in values),
                    "sign_convention": "allowed_minus_restricted; negative is improvement",
                }
            )

    write_csv(
        args.output / "closed_front_points.csv",
        closed_point_rows,
        [
            "case_id", "comparison", "allowed_scenario", "restricted_scenario",
            "point_id", "total_cost_usd",
            "quantity_weighted_mean_delivery_time_h", "configuration_count",
            "source_archives", "ny_nj_share_min", "ny_nj_share_max",
            "configurations_below_threshold",
            "configurations_at_or_above_threshold", "configuration_ids",
        ],
    )
    write_csv(
        args.output / "closed_front_classification.csv",
        classification_rows,
        [
            "case_id", "comparison", "allowed_scenario", "restricted_scenario",
            "closed_objective_points", "distinct_found_configurations",
            "zero_ny_nj_configurations",
            "points_with_equal_objective_configuration_variants",
            "activation_threshold", "ny_nj_share_min", "ny_nj_share_max",
            "first_solution_activation", "front_wide_activation", "classification",
            "evidence_scope",
        ],
    )
    write_csv(
        args.output / "batch_matched_time_decomposition.csv",
        decomposition_rows,
        [
            "case_id", "comparison", "seed", "allowed_scenario",
            "restricted_scenario", "raw_allowed_endpoint_solution_id",
            "restricted_endpoint_solution_id", "closed_allowed_endpoint_source",
            "closed_allowed_endpoint_solution_id", "ny_nj_share_closed_allowed",
            "allowed_mean_time_h", "restricted_mean_time_h",
            "allowed_minus_restricted_time_h", "ny_nj_contribution_h",
            "west_coast_contribution_h", "other_contribution_h",
            "contribution_sum_h", "identity_error_h", "endpoint_accounting_error_h",
            "sign_convention",
        ],
    )
    write_csv(
        args.output / "batch_matched_time_decomposition_summary.csv",
        summary_rows,
        [
            "case_id", "comparison", "component", "paired_seeds", "mean_h",
            "mean_bootstrap_95_low_h", "mean_bootstrap_95_high_h", "median_h",
            "negative_seed_count", "positive_seed_count", "tie_seed_count",
            "sign_convention",
        ],
    )
    write_csv(
        args.output / "service_departure_quantity_changes.csv",
        service_change_rows,
        [
            "case_id", "comparison", "seed", "service_id",
            "scheduled_departure_h", "origin", "destination", "allowed_used_feu",
            "restricted_used_feu", "allowed_minus_restricted_used_feu",
        ],
    )
    historical_corrections: list[dict[str, object]] = []
    if args.historical_closed_front.exists():
        rows = [
            row
            for row in read_csv(args.historical_closed_front)
            if row.get("case_id") == "D1"
            and row.get("comparison") == "extreme_compound"
            and row.get("condition") == "archive_closed_allowed"
        ]
        if rows:
            shares = [float(row["ny_nj_share_pct"]) / 100.0 for row in rows]
            first = int(max(shares) >= args.activation_threshold - 1e-12)
            front_wide = int(min(shares) >= args.activation_threshold - 1e-12)
            historical_corrections.append(
                {
                    "case_id": "D1",
                    "comparison": "compound_30pct_delay_168h",
                    "retained_objective_points": len(rows),
                    "zero_ny_nj_objective_points": sum(
                        value <= 1e-12 for value in shares
                    ),
                    "ny_nj_share_min": min(shares),
                    "ny_nj_share_max": max(shares),
                    "first_solution_activation": first,
                    "front_wide_activation": front_wide,
                    "classification": (
                        "front_wide_activation"
                        if front_wide
                        else "transition"
                        if first
                        else "inactive"
                    ),
                    "evidence_scope": (
                        "existing archive-closed objective points; sufficient to "
                        "reject front-wide activation but not a configuration-complete "
                        "rerun"
                    ),
                }
            )
    write_csv(
        args.output / "historical_closed_front_corrections.csv",
        historical_corrections,
        [
            "case_id", "comparison", "retained_objective_points",
            "zero_ny_nj_objective_points", "ny_nj_share_min", "ny_nj_share_max",
            "first_solution_activation", "front_wide_activation", "classification",
            "evidence_scope",
        ],
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_threshold": args.activation_threshold,
        "comparisons": [
            {
                "case_id": case_id,
                "comparison": comparison,
                "allowed_scenario": allowed,
                "restricted_scenario": restricted,
            }
            for case_id, comparison, allowed, restricted in COMPARISONS
        ],
        "archive_rule": (
            "pool allowed and restricted objective points, retain the non-dominated "
            "union, then retain every distinct configuration found at those coordinates"
        ),
        "activation_definitions": {
            "first_solution_activation": (
                "at least one retained found configuration has NY/NJ share at or "
                "above the threshold"
            ),
            "front_wide_activation": (
                "every retained found configuration has NY/NJ share at or above "
                "the threshold"
            ),
        },
        "decomposition_identity": (
            "NY/NJ contribution + West Coast contribution + other contribution "
            "= allowed minus restricted quantity-weighted mean time"
        ),
        "decomposition_limit": (
            "shipment-matched change relative to each shipment's restricted mean; "
            "it is not unique tracking of every split freight unit"
        ),
        "scope_limit": (
            "Only the four targeted delay comparisons are closed by the new rerun. "
            "No claim is made that every cell in the historical capacity-delay map "
            "has a restricted-archive closure."
        ),
        "historical_compound_correction": (
            "The existing Chicago 30%-capacity + 168 h closed archive contains "
            "zero-share points and is carried as a transition correction. It is "
            "not relabelled as a new configuration-complete rerun."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Final closed-front analysis written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
