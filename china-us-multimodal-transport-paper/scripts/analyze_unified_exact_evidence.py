#!/usr/bin/env python3
"""Unify heuristic archives with proven MILP endpoints.

This is a post-processing analysis only.  It does not invoke NSGA-II or the
MILP solver.  It performs two tasks needed by the Results section:

1. pool the retained heuristic configurations and the proven exact endpoints,
   remove objective records dominated by the unified evidence, and report
   NY/NJ activation only for the *current-known* non-dominated set; and
2. decompose the exact minimum-time counterfactual by shipment, gateway group,
   and scheduled service departure.

The script deliberately distinguishes a proven singleton objective front from
configuration uniqueness.  Coincident exact cost/time endpoints prove that the
objective-space front collapses (within the recorded solver tolerances), but do
not prove that every allocation attaining that point has the same gateway mix.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import capacity_usage_rows, route_component_rows
from china_us_multimodal.domain import RouteAllocation
from china_us_multimodal.io import load_model_data, select_case


COMPARISONS = (
    ("D1", "delay_288h"),
    ("D2", "delay_264h"),
    ("D2", "delay_288h"),
    ("D3", "delay_240h"),
    ("D1", "compound_30pct_delay_168h"),
)


@dataclass(frozen=True)
class EvidenceRecord:
    case_id: str
    comparison: str
    record_id: str
    evidence_source: str
    gateway_access: str
    endpoint: str
    cost: float
    time: float
    configuration_count: int
    below_threshold_count: int
    at_or_above_threshold_count: int
    share_min: float
    share_max: float
    configuration_ids: str
    exact_signature: tuple[tuple[str, float, tuple[str, ...]], ...] | None = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_allocations(
    rows: Iterable[dict[str, str]], id_field: str
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


def allocation_signature(
    allocations: tuple[RouteAllocation, ...],
) -> tuple[tuple[str, float, tuple[str, ...]], ...]:
    return tuple(
        sorted(
            (
                allocation.shipment_id,
                round(allocation.quantity_feu, 9),
                allocation.arc_ids,
            )
            for allocation in allocations
        )
    )


def equivalent(
    left: EvidenceRecord,
    right: EvidenceRecord,
    cost_tolerance: float,
    time_tolerance: float,
) -> bool:
    return (
        abs(left.cost - right.cost) <= cost_tolerance
        and abs(left.time - right.time) <= time_tolerance
    )


def dominates(
    left: EvidenceRecord,
    right: EvidenceRecord,
    cost_tolerance: float,
    time_tolerance: float,
) -> bool:
    no_worse = (
        left.cost <= right.cost + cost_tolerance
        and left.time <= right.time + time_tolerance
    )
    materially_better = (
        left.cost < right.cost - cost_tolerance
        or left.time < right.time - time_tolerance
    )
    return no_worse and materially_better


def union_find_clusters(
    records: list[EvidenceRecord],
    cost_tolerance: float,
    time_tolerance: float,
) -> list[list[EvidenceRecord]]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            if equivalent(
                left,
                records[right_index],
                cost_tolerance,
                time_tolerance,
            ):
                union(left_index, right_index)

    grouped: dict[int, list[EvidenceRecord]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    return list(grouped.values())


def weighted_mean_lead(rows: list[dict[str, object]]) -> float:
    total = sum(float(row["quantity_feu"]) for row in rows)
    return (
        sum(float(row["quantity_feu"]) * float(row["lead_time_h"]) for row in rows)
        / total
    )


def gateway_group(gateway: str) -> str:
    if gateway == "ny_nj":
        return "ny_nj"
    if gateway in {"la_lb", "seattle_tacoma"}:
        return "west_coast"
    return "other"


def load_heuristic_records(
    closed_points_path: Path,
    historical_points_path: Path,
    activation_threshold: float,
) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for row in read_csv(closed_points_path):
        case_id = row["case_id"]
        comparison = row["comparison"]
        if (case_id, comparison) not in COMPARISONS:
            continue
        records.append(
            EvidenceRecord(
                case_id=case_id,
                comparison=comparison,
                record_id=f"heuristic_{case_id}_{comparison}_{row['point_id']}",
                evidence_source="targeted_heuristic_closed_archive",
                gateway_access=row["source_archives"],
                endpoint="",
                cost=float(row["total_cost_usd"]),
                time=float(row["quantity_weighted_mean_delivery_time_h"]),
                configuration_count=int(row["configuration_count"]),
                below_threshold_count=int(row["configurations_below_threshold"]),
                at_or_above_threshold_count=int(
                    row["configurations_at_or_above_threshold"]
                ),
                share_min=float(row["ny_nj_share_min"]),
                share_max=float(row["ny_nj_share_max"]),
                configuration_ids=row["configuration_ids"],
            )
        )

    for row in read_csv(historical_points_path):
        if not (
            row.get("case_id") == "D1"
            and row.get("comparison") == "extreme_compound"
            and row.get("condition") == "archive_closed_allowed"
        ):
            continue
        share = float(row["ny_nj_share_pct"]) / 100.0
        below = int(share < activation_threshold - 1e-12)
        records.append(
            EvidenceRecord(
                case_id="D1",
                comparison="compound_30pct_delay_168h",
                record_id=f"historical_compound_{row['point_id']}",
                evidence_source="historical_heuristic_closed_archive",
                gateway_access=row.get("point_source", ""),
                endpoint="",
                cost=float(row["total_cost_usd"]),
                time=float(row["quantity_weighted_mean_delivery_time_h"]),
                configuration_count=1,
                below_threshold_count=below,
                at_or_above_threshold_count=1 - below,
                share_min=share,
                share_max=share,
                configuration_ids=f"historical_compound_{row['point_id']}",
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge retained heuristic configurations with proven MILP endpoints, "
            "re-screen the current-known non-dominated set, and decompose the "
            "exact minimum-time counterfactual without re-optimizing."
        )
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--closed-points",
        type=Path,
        default=Path("analysis/v12_final_validation/closed_front_points.csv"),
    )
    parser.add_argument(
        "--historical-points",
        type=Path,
        default=Path("analysis/paper_revised/counterfactual_front_points.csv"),
    )
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=Path("results/v12_exact_endpoint_validation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v13_unified_exact_evidence"),
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.10,
        help=(
            "Must remain 0.10 because the archived heuristic configuration counts "
            "were generated for the manuscript's 10%% material-use definition."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.activation_threshold < 1.0:
        parser.error("--activation-threshold must be in (0, 1)")
    if abs(args.activation_threshold - 0.10) > 1e-12:
        parser.error(
            "This unification must use the archived 10% material-use definition."
        )

    exact_summary = read_csv(args.exact_root / "exact_endpoint_summary.csv")
    exact_allocations = load_allocations(
        read_csv(args.exact_root / "exact_endpoint_allocations.csv"),
        "validation_id",
    )
    exact_rows = {
        row["validation_id"]: row
        for row in exact_summary
        if int(row["lexicographic_proven"]) == 1
        and int(row["evaluator_consistent"]) == 1
        and int(row["feasible"]) == 1
    }
    if len(exact_rows) != len(exact_summary):
        raise RuntimeError(
            "Every exact endpoint must be proven, feasible, and evaluator-consistent "
            "before it enters the unified evidence set."
        )

    cost_tolerance = max(
        float(row["primary_tolerance"])
        for row in exact_summary
        if row["primary_objective"] == "cost"
    )
    time_tolerance = max(
        float(row["primary_tolerance"])
        for row in exact_summary
        if row["primary_objective"] == "time"
    )

    records = load_heuristic_records(
        args.closed_points,
        args.historical_points,
        args.activation_threshold,
    )
    for row in exact_summary:
        share = float(row["ny_nj_share"])
        below = int(share < args.activation_threshold - 1e-12)
        records.append(
            EvidenceRecord(
                case_id=row["case_id"],
                comparison=row["comparison"],
                record_id=row["validation_id"],
                evidence_source="proven_exact_endpoint",
                gateway_access=row["gateway_access"],
                endpoint=row["endpoint"],
                cost=float(row["total_cost_usd"]),
                time=float(row["quantity_weighted_mean_delivery_time_h"]),
                configuration_count=1,
                below_threshold_count=below,
                at_or_above_threshold_count=1 - below,
                share_min=share,
                share_max=share,
                configuration_ids=row["validation_id"],
                exact_signature=allocation_signature(
                    exact_allocations[row["validation_id"]]
                ),
            )
        )

    retained_rows: list[dict[str, object]] = []
    displaced_rows: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []
    for case_id, comparison in COMPARISONS:
        group = [
            record
            for record in records
            if record.case_id == case_id and record.comparison == comparison
        ]
        if not group:
            raise RuntimeError(f"No evidence records for {case_id}/{comparison}.")
        retained = [
            record
            for record in group
            if not any(
                dominates(other, record, cost_tolerance, time_tolerance)
                for other in group
                if other.record_id != record.record_id
            )
        ]
        displaced = [record for record in group if record not in retained]
        for record in displaced:
            dominators = [
                other
                for other in group
                if other.record_id != record.record_id
                and dominates(other, record, cost_tolerance, time_tolerance)
            ]
            displaced_rows.append(
                {
                    "case_id": case_id,
                    "comparison": comparison,
                    "record_id": record.record_id,
                    "evidence_source": record.evidence_source,
                    "total_cost_usd": record.cost,
                    "quantity_weighted_mean_delivery_time_h": record.time,
                    "dominated_by_record_ids": "|".join(
                        sorted(item.record_id for item in dominators)
                    ),
                    "dominated_by_exact_endpoint": int(
                        any(
                            item.evidence_source == "proven_exact_endpoint"
                            for item in dominators
                        )
                    ),
                }
            )

        clusters = union_find_clusters(retained, cost_tolerance, time_tolerance)
        all_share_min: list[float] = []
        all_share_max: list[float] = []
        below_count = 0
        above_count = 0
        unique_configuration_count = 0
        for point_index, cluster in enumerate(
            sorted(clusters, key=lambda rows: (min(r.cost for r in rows), min(r.time for r in rows))),
            start=1,
        ):
            # Exact cost/time lexicographic runs may recover the same allocation.
            # Count that exact configuration once while retaining every heuristic
            # configuration already represented by its archived count.
            exact_seen: set[tuple[tuple[str, float, tuple[str, ...]], ...]] = set()
            cluster_configuration_count = 0
            cluster_below = 0
            cluster_above = 0
            share_min_values: list[float] = []
            share_max_values: list[float] = []
            for record in cluster:
                if record.exact_signature is not None:
                    if record.exact_signature in exact_seen:
                        continue
                    exact_seen.add(record.exact_signature)
                cluster_configuration_count += record.configuration_count
                cluster_below += record.below_threshold_count
                cluster_above += record.at_or_above_threshold_count
                share_min_values.append(record.share_min)
                share_max_values.append(record.share_max)
            unique_configuration_count += cluster_configuration_count
            below_count += cluster_below
            above_count += cluster_above
            all_share_min.extend(share_min_values)
            all_share_max.extend(share_max_values)
            representative = min(cluster, key=lambda row: (row.cost, row.time))
            retained_rows.append(
                {
                    "case_id": case_id,
                    "comparison": comparison,
                    "current_known_point_id": point_index,
                    "display_total_cost_usd": representative.cost,
                    "display_quantity_weighted_mean_delivery_time_h": representative.time,
                    "cost_min_usd": min(row.cost for row in cluster),
                    "cost_max_usd": max(row.cost for row in cluster),
                    "time_min_h": min(row.time for row in cluster),
                    "time_max_h": max(row.time for row in cluster),
                    "objective_equivalent_record_count": len(cluster),
                    "distinct_found_configuration_count": cluster_configuration_count,
                    "evidence_sources": "|".join(
                        sorted({row.evidence_source for row in cluster})
                    ),
                    "gateway_access_sources": "|".join(
                        sorted({row.gateway_access for row in cluster})
                    ),
                    "record_ids": "|".join(sorted(row.record_id for row in cluster)),
                    "ny_nj_share_min": min(share_min_values),
                    "ny_nj_share_max": max(share_max_values),
                    "configurations_below_threshold": cluster_below,
                    "configurations_at_or_above_threshold": cluster_above,
                    "contains_proven_exact_endpoint": int(
                        any(row.evidence_source == "proven_exact_endpoint" for row in cluster)
                    ),
                }
            )

        allowed_exact = [
            record
            for record in group
            if record.evidence_source == "proven_exact_endpoint"
            and record.gateway_access == "allowed"
        ]
        allowed_cost_endpoint = next(
            record for record in allowed_exact if record.endpoint == "min_cost_then_time"
        )
        allowed_time_endpoint = next(
            record for record in allowed_exact if record.endpoint == "min_time_then_cost"
        )
        singleton_objective_front = equivalent(
            allowed_cost_endpoint,
            allowed_time_endpoint,
            cost_tolerance,
            time_tolerance,
        )
        minimum_share = min(all_share_min)
        maximum_share = max(all_share_max)
        first_activation = int(maximum_share >= args.activation_threshold - 1e-12)
        all_current_known = int(minimum_share >= args.activation_threshold - 1e-12)
        if all_current_known:
            classification = "current_known_all_at_or_above_threshold"
        elif first_activation:
            classification = "current_known_transition"
        else:
            classification = "current_known_inactive"
        classification_rows.append(
            {
                "case_id": case_id,
                "comparison": comparison,
                "pooled_evidence_records": len(group),
                "current_known_objective_points": len(clusters),
                "current_known_distinct_found_configurations": unique_configuration_count,
                "displaced_evidence_records": len(displaced),
                "displaced_heuristic_records": sum(
                    row.evidence_source != "proven_exact_endpoint" for row in displaced
                ),
                "activation_threshold": args.activation_threshold,
                "ny_nj_share_min": minimum_share,
                "ny_nj_share_max": maximum_share,
                "configurations_below_threshold": below_count,
                "configurations_at_or_above_threshold": above_count,
                "first_current_known_activation": first_activation,
                "all_current_known_at_or_above_threshold": all_current_known,
                "current_known_classification": classification,
                "exact_allowed_endpoint_relation": (
                    "proven_singleton_objective_front_within_tolerance"
                    if singleton_objective_front
                    else "proven_endpoints_distinct_interior_not_enumerated"
                ),
                "configuration_uniqueness": "not_proven",
                "universal_configuration_activation": "not_claimed",
                "evidence_scope": (
                    "non-dominated union of retained heuristic configurations and "
                    "proven exact endpoints; not a complete exact interior front"
                ),
            }
        )

    full_data = load_model_data(args.data)
    decomposition_rows: list[dict[str, object]] = []
    shipment_rows: list[dict[str, object]] = []
    service_rows: list[dict[str, object]] = []
    for case_id, comparison in COMPARISONS:
        allowed_id = f"{case_id}_{comparison}_allowed_min_time_then_cost"
        restricted_id = f"{case_id}_{comparison}_restricted_min_time_then_cost"
        allowed_row = exact_rows[allowed_id]
        restricted_row = exact_rows[restricted_id]
        allowed_allocations = exact_allocations[allowed_id]
        restricted_allocations = exact_allocations[restricted_id]
        data = select_case(full_data, case_id)
        allowed_config = load_config(args.config, allowed_row["scenario"])
        restricted_config = load_config(args.config, restricted_row["scenario"])
        allowed_components = route_component_rows(
            data, list(allowed_allocations), allowed_config
        )
        restricted_components = route_component_rows(
            data, list(restricted_allocations), restricted_config
        )
        allowed_by_shipment: dict[str, list[dict[str, object]]] = {}
        restricted_by_shipment: dict[str, list[dict[str, object]]] = {}
        for row in allowed_components:
            allowed_by_shipment.setdefault(str(row["shipment_id"]), []).append(row)
        for row in restricted_components:
            restricted_by_shipment.setdefault(str(row["shipment_id"]), []).append(row)
        if set(allowed_by_shipment) != set(restricted_by_shipment):
            raise RuntimeError(f"Shipment mismatch for {case_id}/{comparison}.")
        restricted_mean_by_shipment = {
            shipment_id: weighted_mean_lead(rows)
            for shipment_id, rows in restricted_by_shipment.items()
        }
        total_quantity = sum(float(row["quantity_feu"]) for row in allowed_components)
        contributions = {"ny_nj": 0.0, "west_coast": 0.0, "other": 0.0}
        for shipment_id in sorted(allowed_by_shipment):
            allowed_group = allowed_by_shipment[shipment_id]
            restricted_group = restricted_by_shipment[shipment_id]
            allowed_quantity = sum(float(row["quantity_feu"]) for row in allowed_group)
            restricted_quantity = sum(
                float(row["quantity_feu"]) for row in restricted_group
            )
            if abs(allowed_quantity - restricted_quantity) > 1e-8:
                raise RuntimeError(
                    f"Demand mismatch for {case_id}/{comparison}/{shipment_id}."
                )
            allowed_mean = weighted_mean_lead(allowed_group)
            restricted_mean = restricted_mean_by_shipment[shipment_id]
            shipment_rows.append(
                {
                    "case_id": case_id,
                    "comparison": comparison,
                    "shipment_id": shipment_id,
                    "quantity_feu": allowed_quantity,
                    "allowed_mean_time_h": allowed_mean,
                    "restricted_mean_time_h": restricted_mean,
                    "allowed_minus_restricted_mean_time_h": allowed_mean
                    - restricted_mean,
                    "allowed_ny_nj_quantity_feu": sum(
                        float(row["quantity_feu"])
                        for row in allowed_group
                        if str(row["gateway"]) == "ny_nj"
                    ),
                    "allowed_west_coast_quantity_feu": sum(
                        float(row["quantity_feu"])
                        for row in allowed_group
                        if str(row["gateway"]) in {"la_lb", "seattle_tacoma"}
                    ),
                    "allowed_route_count": len(allowed_group),
                    "restricted_route_count": len(restricted_group),
                    "allowed_arc_paths": "|".join(
                        sorted(
                            ">".join(allocation.arc_ids)
                            for allocation in allowed_allocations
                            if allocation.shipment_id == shipment_id
                        )
                    ),
                    "restricted_arc_paths": "|".join(
                        sorted(
                            ">".join(allocation.arc_ids)
                            for allocation in restricted_allocations
                            if allocation.shipment_id == shipment_id
                        )
                    ),
                    "path_configuration_changed": int(
                        {
                            allocation.arc_ids
                            for allocation in allowed_allocations
                            if allocation.shipment_id == shipment_id
                        }
                        != {
                            allocation.arc_ids
                            for allocation in restricted_allocations
                            if allocation.shipment_id == shipment_id
                        }
                    ),
                }
            )
            for row in allowed_group:
                group_name = gateway_group(str(row["gateway"]))
                contributions[group_name] += (
                    float(row["quantity_feu"])
                    * (float(row["lead_time_h"]) - restricted_mean)
                    / total_quantity
                )
        allowed_mean = weighted_mean_lead(allowed_components)
        restricted_mean = weighted_mean_lead(restricted_components)
        observed_difference = allowed_mean - restricted_mean
        contribution_sum = sum(contributions.values())
        identity_error = contribution_sum - observed_difference
        if abs(identity_error) > 1e-7:
            raise RuntimeError(
                f"Exact decomposition identity failed for {case_id}/{comparison}: "
                f"{identity_error}."
            )
        endpoint_accounting_error = max(
            abs(float(allowed_row["quantity_weighted_mean_delivery_time_h"]) - allowed_mean),
            abs(
                float(restricted_row["quantity_weighted_mean_delivery_time_h"])
                - restricted_mean
            ),
        )
        if endpoint_accounting_error > 1e-7:
            raise RuntimeError(
                f"Exact endpoint accounting failed for {case_id}/{comparison}: "
                f"{endpoint_accounting_error}."
            )
        decomposition_rows.append(
            {
                "case_id": case_id,
                "comparison": comparison,
                "allowed_validation_id": allowed_id,
                "restricted_validation_id": restricted_id,
                "allowed_ny_nj_share": float(allowed_row["ny_nj_share"]),
                "allowed_mean_time_h": allowed_mean,
                "restricted_mean_time_h": restricted_mean,
                "allowed_minus_restricted_time_h": observed_difference,
                "time_avoided_by_allowing_ny_nj_h": -observed_difference,
                "ny_nj_contribution_h": contributions["ny_nj"],
                "west_coast_contribution_h": contributions["west_coast"],
                "other_contribution_h": contributions["other"],
                "contribution_sum_h": contribution_sum,
                "identity_error_h": identity_error,
                "endpoint_accounting_error_h": endpoint_accounting_error,
                "sign_convention": "allowed_minus_restricted; negative is improvement",
                "interpretation_limit": (
                    "shipment-matched decomposition relative to each shipment's "
                    "restricted mean; not unique tracking of every split freight unit"
                ),
            }
        )

        allowed_usage = {
            (str(row["resource_id"]), str(row["period"])): row
            for row in capacity_usage_rows(
                data, list(allowed_allocations), allowed_config
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
            source = allowed_usage.get(resource_key) or restricted_usage[resource_key]
            service_rows.append(
                {
                    "case_id": case_id,
                    "comparison": comparison,
                    "service_id": resource_key[0],
                    "scheduled_departure_h": resource_key[1],
                    "origin": source["origin"],
                    "destination": source["destination"],
                    "allowed_used_feu": allowed_used,
                    "restricted_used_feu": restricted_used,
                    "allowed_minus_restricted_used_feu": difference,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output / "unified_nondominated_points.csv",
        retained_rows,
        [
            "case_id", "comparison", "current_known_point_id",
            "display_total_cost_usd",
            "display_quantity_weighted_mean_delivery_time_h", "cost_min_usd",
            "cost_max_usd", "time_min_h", "time_max_h",
            "objective_equivalent_record_count",
            "distinct_found_configuration_count", "evidence_sources",
            "gateway_access_sources", "record_ids", "ny_nj_share_min",
            "ny_nj_share_max", "configurations_below_threshold",
            "configurations_at_or_above_threshold",
            "contains_proven_exact_endpoint",
        ],
    )
    write_csv(
        args.output / "displaced_evidence_records.csv",
        displaced_rows,
        [
            "case_id", "comparison", "record_id", "evidence_source",
            "total_cost_usd", "quantity_weighted_mean_delivery_time_h",
            "dominated_by_record_ids", "dominated_by_exact_endpoint",
        ],
    )
    write_csv(
        args.output / "unified_classification.csv",
        classification_rows,
        [
            "case_id", "comparison", "pooled_evidence_records",
            "current_known_objective_points",
            "current_known_distinct_found_configurations",
            "displaced_evidence_records", "displaced_heuristic_records",
            "activation_threshold", "ny_nj_share_min", "ny_nj_share_max",
            "configurations_below_threshold",
            "configurations_at_or_above_threshold",
            "first_current_known_activation",
            "all_current_known_at_or_above_threshold",
            "current_known_classification", "exact_allowed_endpoint_relation",
            "configuration_uniqueness", "universal_configuration_activation",
            "evidence_scope",
        ],
    )
    write_csv(
        args.output / "exact_batch_matched_time_decomposition.csv",
        decomposition_rows,
        [
            "case_id", "comparison", "allowed_validation_id",
            "restricted_validation_id", "allowed_ny_nj_share",
            "allowed_mean_time_h", "restricted_mean_time_h",
            "allowed_minus_restricted_time_h",
            "time_avoided_by_allowing_ny_nj_h", "ny_nj_contribution_h",
            "west_coast_contribution_h", "other_contribution_h",
            "contribution_sum_h", "identity_error_h",
            "endpoint_accounting_error_h", "sign_convention",
            "interpretation_limit",
        ],
    )
    write_csv(
        args.output / "exact_shipment_time_changes.csv",
        shipment_rows,
        [
            "case_id", "comparison", "shipment_id", "quantity_feu",
            "allowed_mean_time_h", "restricted_mean_time_h",
            "allowed_minus_restricted_mean_time_h",
            "allowed_ny_nj_quantity_feu", "allowed_west_coast_quantity_feu",
            "allowed_route_count", "restricted_route_count", "allowed_arc_paths",
            "restricted_arc_paths", "path_configuration_changed",
        ],
    )
    write_csv(
        args.output / "exact_service_departure_quantity_changes.csv",
        service_rows,
        [
            "case_id", "comparison", "service_id", "scheduled_departure_h",
            "origin", "destination", "allowed_used_feu",
            "restricted_used_feu", "allowed_minus_restricted_used_feu",
        ],
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "postprocessing_only_no_optimization",
        "comparisons": [
            {"case_id": case_id, "comparison": comparison}
            for case_id, comparison in COMPARISONS
        ],
        "objective_equivalence_tolerances": {
            "cost_absolute_usd": cost_tolerance,
            "time_absolute_h": time_tolerance,
            "source": "maximum recorded primary tolerance among proven exact endpoints",
        },
        "dominance_rule": (
            "two-objective minimization with the recorded exact-endpoint absolute "
            "tolerances; equivalent endpoint allocations are counted once"
        ),
        "classification_scope": (
            "current-known non-dominated union only; no universal full-front "
            "configuration claim is made"
        ),
        "singleton_front_rule": (
            "coincident proven minimum-cost and minimum-time allowed endpoints imply "
            "a singleton objective-space front within tolerance, but not a unique "
            "allocation configuration"
        ),
        "decomposition_identity": (
            "NY/NJ contribution + West Coast contribution + other contribution = "
            "allowed minus restricted quantity-weighted mean delivery time"
        ),
        "decomposition_scope": (
            "proven exact minimum-time endpoints; shipment-matched, not unique "
            "freight-unit lineage"
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Unified exact-evidence analysis written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
