#!/usr/bin/env python3
"""Post-process exact core solutions without adding optimization scenarios."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def weighted_shipment_lead(rows: list[dict[str, str]]) -> float:
    total = sum(float(row["quantity_feu"]) for row in rows)
    return sum(
        float(row["quantity_feu"]) * float(row["lead_time_h"]) for row in rows
    ) / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=Path("results/v14_core_exact_suite")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/v14_core_exact_suite")
    )
    args = parser.parse_args()
    summary = read_csv(args.results / "exact_endpoint_summary.csv")
    allocations = read_csv(args.results / "exact_endpoint_allocations.csv")
    components = read_csv(args.results / "exact_route_components.csv")
    cost_components = read_csv(args.results / "exact_cost_component_summary.csv")
    service_legs = read_csv(args.results / "exact_endpoint_service_legs.csv")
    summary_by_id = {row["validation_id"]: row for row in summary}
    allocation_by_id: dict[str, list[dict[str, str]]] = {}
    component_by_id: dict[str, list[dict[str, str]]] = {}
    legs_by_id: dict[str, list[dict[str, str]]] = {}
    for row in allocations:
        allocation_by_id.setdefault(row["validation_id"], []).append(row)
    for row in components:
        component_by_id.setdefault(row["validation_id"], []).append(row)
    for row in service_legs:
        legs_by_id.setdefault(row["validation_id"], []).append(row)

    index = {
        (row["case_id"], row["condition_id"], row["gateway_access"], row["endpoint"]): row
        for row in summary
    }
    cost_index = {row["validation_id"]: row for row in cost_components}
    cost_difference_rows: list[dict[str, object]] = []
    condition_keys = sorted({(row["case_id"], row["condition_id"]) for row in summary})
    for case_id, condition_id in condition_keys:
        for endpoint in ("min_cost_then_time", "min_time_then_cost"):
            allowed = index[(case_id, condition_id, "allowed", endpoint)]
            restricted = index[(case_id, condition_id, "restricted", endpoint)]
            allowed_cost = cost_index[allowed["validation_id"]]
            restricted_cost = cost_index[restricted["validation_id"]]
            row: dict[str, object] = {
                "case_id": case_id,
                "condition_id": condition_id,
                "endpoint": endpoint,
            }
            component_fields = (
                "transport_cost_usd",
                "transfer_cost_usd",
                "waiting_holding_cost_usd",
                "processing_cost_usd",
                "tardiness_cost_usd",
            )
            for field in component_fields:
                row[f"avoided_{field}"] = float(restricted_cost[field]) - float(
                    allowed_cost[field]
                )
            row["total_cost_avoided_usd"] = float(
                restricted["total_cost_usd"]
            ) - float(allowed["total_cost_usd"])
            row["dominant_absolute_component"] = max(
                component_fields,
                key=lambda field: abs(float(row[f"avoided_{field}"])),
            )
            cost_difference_rows.append(row)
    write_csv(
        args.output / "exact_counterfactual_cost_decomposition.csv",
        cost_difference_rows,
        list(cost_difference_rows[0].keys()),
    )

    shipment_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for case_id, condition_id in condition_keys:
        for endpoint in ("min_cost_then_time", "min_time_then_cost"):
            allowed = index[(case_id, condition_id, "allowed", endpoint)]
            restricted = index[(case_id, condition_id, "restricted", endpoint)]
            allowed_components = component_by_id[allowed["validation_id"]]
            restricted_components = component_by_id[restricted["validation_id"]]
            shipment_ids = sorted(
                {row["shipment_id"] for row in allowed_components}
                | {row["shipment_id"] for row in restricted_components}
            )
            total_demand = sum(
                float(row["quantity_feu"])
                for row in allocation_by_id[allowed["validation_id"]]
            )
            group_contributions: dict[str, float] = {}
            reconstructed = 0.0
            for shipment_id in shipment_ids:
                allowed_rows = [
                    row for row in allowed_components if row["shipment_id"] == shipment_id
                ]
                restricted_rows = [
                    row
                    for row in restricted_components
                    if row["shipment_id"] == shipment_id
                ]
                quantity = sum(float(row["quantity_feu"]) for row in allowed_rows)
                allowed_lead = weighted_shipment_lead(allowed_rows)
                restricted_lead = weighted_shipment_lead(restricted_rows)
                contribution = quantity * (allowed_lead - restricted_lead) / total_demand
                ny_quantity = sum(
                    float(row["quantity_feu"])
                    for row in allowed_rows
                    if row["gateway"] == "ny_nj"
                )
                group = (
                    "shipments_using_ny_nj_in_allowed_solution"
                    if ny_quantity > 1e-9
                    else "shipments_remaining_on_west_coast"
                )
                reconstructed += contribution
                group_contributions[group] = group_contributions.get(group, 0.0) + contribution
                shipment_rows.append(
                    {
                        "case_id": case_id,
                        "condition_id": condition_id,
                        "endpoint": endpoint,
                        "shipment_id": shipment_id,
                        "shipment_quantity_feu": quantity,
                        "allowed_ny_nj_quantity_feu": ny_quantity,
                        "allowed_mean_lead_time_h": allowed_lead,
                        "restricted_mean_lead_time_h": restricted_lead,
                        "allowed_minus_restricted_mean_lead_h": allowed_lead
                        - restricted_lead,
                        "contribution_to_system_mean_change_h": contribution,
                        "shipment_group": group,
                    }
                )
            observed = float(allowed["quantity_weighted_mean_delivery_time_h"]) - float(
                restricted["quantity_weighted_mean_delivery_time_h"]
            )
            for group in (
                "shipments_using_ny_nj_in_allowed_solution",
                "shipments_remaining_on_west_coast",
            ):
                group_rows.append(
                    {
                        "case_id": case_id,
                        "condition_id": condition_id,
                        "endpoint": endpoint,
                        "shipment_group": group,
                        "contribution_to_allowed_minus_restricted_time_h": (
                            group_contributions.get(group, 0.0)
                        ),
                        "reconstructed_total_change_h": reconstructed,
                        "observed_total_change_h": observed,
                        "identity_abs_error_h": abs(reconstructed - observed),
                        "identity_pass": int(
                            math.isclose(reconstructed, observed, rel_tol=0.0, abs_tol=1e-7)
                        ),
                        "interpretation_limit": (
                            "batch-matched contribution, not unique tracking of every split FEU"
                        ),
                    }
                )
    write_csv(
        args.output / "exact_batch_matched_time_by_shipment.csv",
        shipment_rows,
        list(shipment_rows[0].keys()),
    )
    write_csv(
        args.output / "exact_batch_matched_time_decomposition.csv",
        group_rows,
        list(group_rows[0].keys()),
    )

    compound_rows = [
        row
        for row in summary
        if row["case_id"] == "D1"
        and row["condition_id"] == "compound_30pct_delay_168h"
    ]
    span_rows: list[dict[str, object]] = []
    for access in ("allowed", "restricted"):
        min_cost = next(
            row
            for row in compound_rows
            if row["gateway_access"] == access
            and row["endpoint"] == "min_cost_then_time"
        )
        min_time = next(
            row
            for row in compound_rows
            if row["gateway_access"] == access
            and row["endpoint"] == "min_time_then_cost"
        )
        span_rows.append(
            {
                "gateway_access": access,
                "cost_span_min_time_minus_min_cost_usd": float(
                    min_time["total_cost_usd"]
                )
                - float(min_cost["total_cost_usd"]),
                "time_span_min_cost_minus_min_time_h": float(
                    min_cost["quantity_weighted_mean_delivery_time_h"]
                )
                - float(min_time["quantity_weighted_mean_delivery_time_h"]),
                "min_cost_ny_nj_share": min_cost["ny_nj_share"],
                "min_time_ny_nj_share": min_time["ny_nj_share"],
                "claim_limit": (
                    "equal spans alone do not establish that the trade-off is independent of NY/NJ"
                ),
            }
        )
    write_csv(
        args.output / "chicago_compound_endpoint_spans.csv",
        span_rows,
        list(span_rows[0].keys()),
    )

    config_change_rows: list[dict[str, object]] = []
    for access in ("allowed", "restricted"):
        identifiers = {
            endpoint: next(
                row["validation_id"]
                for row in compound_rows
                if row["gateway_access"] == access and row["endpoint"] == endpoint
            )
            for endpoint in ("min_cost_then_time", "min_time_then_cost")
        }
        by_signature: dict[tuple[str, str, str], dict[str, float]] = {}
        for endpoint, validation_id in identifiers.items():
            for row in allocation_by_id[validation_id]:
                signature = (row["shipment_id"], row["gateway"], row["arc_ids"])
                by_signature.setdefault(signature, {})[endpoint] = float(row["quantity_feu"])
        for (shipment_id, gateway, arc_ids), quantities in sorted(by_signature.items()):
            min_cost_quantity = quantities.get("min_cost_then_time", 0.0)
            min_time_quantity = quantities.get("min_time_then_cost", 0.0)
            if math.isclose(min_cost_quantity, min_time_quantity, abs_tol=1e-8):
                continue
            config_change_rows.append(
                {
                    "gateway_access": access,
                    "shipment_id": shipment_id,
                    "gateway": gateway,
                    "arc_ids": arc_ids,
                    "min_cost_endpoint_quantity_feu": min_cost_quantity,
                    "min_time_endpoint_quantity_feu": min_time_quantity,
                    "change_min_time_minus_min_cost_feu": min_time_quantity
                    - min_cost_quantity,
                }
            )
    write_csv(
        args.output / "chicago_compound_allocation_changes.csv",
        config_change_rows,
        [
            "gateway_access", "shipment_id", "gateway", "arc_ids",
            "min_cost_endpoint_quantity_feu", "min_time_endpoint_quantity_feu",
            "change_min_time_minus_min_cost_feu",
        ],
    )

    service_change_rows: list[dict[str, object]] = []
    for access in ("allowed", "restricted"):
        identifiers = {
            endpoint: next(
                row["validation_id"]
                for row in compound_rows
                if row["gateway_access"] == access and row["endpoint"] == endpoint
            )
            for endpoint in ("min_cost_then_time", "min_time_then_cost")
        }
        service_signatures: dict[str, dict[str, set[str]]] = {}
        for endpoint, validation_id in identifiers.items():
            for row in legs_by_id[validation_id]:
                if not row["service_id"]:
                    continue
                signature = f"{row['service_id']}@{row['published_departure_h']}"
                service_signatures.setdefault(row["shipment_id"], {}).setdefault(
                    endpoint, set()
                ).add(signature)
        for shipment_id, endpoint_sets in sorted(service_signatures.items()):
            cost_services = endpoint_sets.get("min_cost_then_time", set())
            time_services = endpoint_sets.get("min_time_then_cost", set())
            if cost_services == time_services:
                continue
            service_change_rows.append(
                {
                    "gateway_access": access,
                    "shipment_id": shipment_id,
                    "min_cost_endpoint_services": "|".join(sorted(cost_services)),
                    "min_time_endpoint_services": "|".join(sorted(time_services)),
                    "same_service_set": 0,
                }
            )
    write_csv(
        args.output / "chicago_compound_service_changes.csv",
        service_change_rows,
        [
            "gateway_access", "shipment_id", "min_cost_endpoint_services",
            "min_time_endpoint_services", "same_service_set",
        ],
    )
    print(f"Supplementary core analysis written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
