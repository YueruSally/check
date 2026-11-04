#!/usr/bin/env python3
"""Generate the final Results addenda and matched zero-penalty baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import route_component_rows
from china_us_multimodal.domain import RouteAllocation
from china_us_multimodal.exact_endpoint import solve_lexicographic_endpoint
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import ExactCondition, condition_config
from run_final_path_coverage import build_full_model, enumerate_case_paths
from run_full_structural_cost_sensitivity import (
    SensitivitySpec,
    scaled_case_data,
    sensitivity_config,
)
from run_supplementary_core_exact import endpoint_payload, load_json, write_json_atomic


CASES = ("D1", "D2", "D3")
COMMON_DELAY_GRID = (0.0, 168.0, 240.0, 264.0, 288.0, 336.0)
CRITERIA = (0.05, 0.10, 0.20)
FOCAL_DELAYS = {"D1": 288.0, "D2": 288.0, "D3": 240.0}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def zero_penalty_spec(case_id: str) -> SensitivitySpec:
    return SensitivitySpec(
        condition=ExactCondition(
            case_id=case_id,
            condition_id="delay_0h_penalty_000pct",
            family="matched_zero_penalty_baseline",
            uswc_delay_h=0.0,
        ),
        dimension="tardiness_penalty",
        multiplier=0.0,
        base_delay_h=0.0,
    )


def allocations_by_validation(path: Path) -> dict[str, list[RouteAllocation]]:
    result: dict[str, list[RouteAllocation]] = defaultdict(list)
    for row in read_csv(path):
        result[row["validation_id"]].append(
            RouteAllocation(
                shipment_id=row["shipment_id"],
                quantity_feu=float(row["quantity_feu"]),
                arc_ids=tuple(row["arc_ids"].split("|")),
            )
        )
    return result


def component_totals(data, allocations, config) -> dict[str, float]:
    totals = {
        "transport_cost_usd": 0.0,
        "holding_cost_usd": 0.0,
        "transfer_cost_usd": 0.0,
        "processing_cost_usd": 0.0,
        "tardiness_cost_usd": 0.0,
    }
    for row in route_component_rows(data, allocations, config):
        totals["transport_cost_usd"] += sum(
            float(row[field])
            for field in (
                "china_inland_transport_cost_usd",
                "ocean_transport_cost_usd",
                "us_inland_transport_cost_usd",
            )
        )
        for field in (
            "holding_cost_usd",
            "transfer_cost_usd",
            "processing_cost_usd",
            "tardiness_cost_usd",
        ):
            totals[field] += float(row[field])
    totals["total_cost_usd"] = sum(totals.values())
    return totals


def solve_matched_zero_penalty_baselines(args, data, base_config):
    structural_manifest = load_json(args.structural_results / "manifest.json")
    if not structural_manifest.get("structural_enumeration_complete"):
        raise RuntimeError("The structural path source is not exhaustive.")

    base_cases = {case_id: select_case(data, case_id) for case_id in CASES}
    enumerations = {
        case_id: enumerate_case_paths(
            case_data, base_config, args.enumeration_cap_per_gateway
        )
        for case_id, case_data in base_cases.items()
    }
    if any(
        not enumeration.complete
        for by_gateway in enumerations.values()
        for enumeration in by_gateway.values()
    ):
        raise RuntimeError("Structural enumeration cap reached.")

    checkpoint_root = args.output / "checkpoints" / "zero_penalty_baselines"
    total = len(CASES) * 2
    completed = 0
    for case_id in CASES:
        spec = zero_penalty_spec(case_id)
        case_data = scaled_case_data(base_cases[case_id], spec)
        for access in ("allowed", "restricted"):
            task_id = (
                f"{case_id}_{spec.condition.condition_id}_{access}_"
                "min_cost_then_time"
            )
            checkpoint = checkpoint_root / f"{task_id}.json"
            if checkpoint.exists():
                existing = load_json(checkpoint)
                if int(existing.get("lexicographic_proven", 0)) == 1:
                    completed += 1
                    print(f"[{completed}/{total}] resume {task_id}", flush=True)
                    continue
            config = sensitivity_config(base_config, spec, access)
            model, max_paths = build_full_model(
                case_data, config, enumerations[case_id]
            )
            result = solve_lexicographic_endpoint(
                model,
                primary_objective="cost",
                time_limit_seconds=args.time_limit_seconds,
                mip_relative_gap=args.mip_relative_gap,
                primary_relative_tolerance=args.relative_tolerance,
                primary_absolute_tolerance=args.absolute_tolerance,
            )
            payload = endpoint_payload(
                condition=spec.condition,
                gateway_access=access,
                endpoint="min_cost_then_time",
                model=model,
                result=result,
                max_paths_per_gateway=int(max_paths),
            )
            payload.update(
                {
                    "validation_id": task_id,
                    "library_variant": "structural_full",
                    "sensitivity_dimension": "tardiness_penalty",
                    "sensitivity_multiplier": 0.0,
                }
            )
            write_json_atomic(checkpoint, payload)
            completed += 1
            print(
                f"[{completed}/{total}] {task_id} "
                f"proven={payload['lexicographic_proven']}",
                flush=True,
            )

    payloads = [load_json(path) for path in sorted(checkpoint_root.glob("*.json"))]
    if len(payloads) != total or not all(
        int(row.get("lexicographic_proven", 0)) == 1 for row in payloads
    ):
        raise RuntimeError("Matched zero-penalty baseline endpoints are incomplete.")
    return payloads, base_cases


def criterion_rows(activation_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_id in CASES:
        for endpoint in ("min_cost_then_time", "min_time_then_cost"):
            candidates = [
                row
                for row in activation_rows
                if row["case_id"] == case_id
                and row["endpoint"] == endpoint
                and row["condition_id"].startswith("delay_")
                and float(row["delay_h"]) in COMMON_DELAY_GRID
            ]
            for criterion in CRITERIA:
                qualifying = sorted(
                    float(row["delay_h"])
                    for row in candidates
                    if float(row["ny_nj_min_share"]) >= criterion - 1e-9
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "endpoint": endpoint,
                        "criterion_share": criterion,
                        "first_common_grid_delay_h": (
                            qualifying[0] if qualifying else ""
                        ),
                        "common_delay_grid_h": "|".join(
                            f"{value:g}" for value in COMMON_DELAY_GRID
                        ),
                    }
                )
    return rows


def absolute_recovery_rows(
    recovery_rows: list[dict[str, str]],
    v26_endpoints: list[dict[str, str]],
) -> list[dict[str, object]]:
    targets = (
        ("D1", "delay_288h", 288.0),
        ("D2", "delay_288h", 288.0),
    )
    result: list[dict[str, object]] = []
    for case_id, condition_id, delay_h in targets:
        for endpoint, metric in (
            ("min_cost_then_time", "cost"),
            ("min_time_then_cost", "time"),
        ):
            row = next(
                item
                for item in recovery_rows
                if item["case_id"] == case_id
                and item["condition_id"] == condition_id
                and item["endpoint"] == endpoint
                and item["metric"] == metric
            )
            result.append(dict(row))

    endpoint_index = {
        (row["gateway_access"], row["endpoint"]): row
        for row in v26_endpoints
        if row["case_id"] == "D3"
        and row["condition_id"]
        == "delay_216h_inland_us-d-03_capacity_100pct"
    }
    baseline_cost = 23968537.06396954
    baseline_time = 609.1774410774411
    for endpoint, metric, baseline in (
        ("min_cost_then_time", "cost", baseline_cost),
        ("min_time_then_cost", "time", baseline_time),
    ):
        allowed = endpoint_index[("allowed", endpoint)]
        restricted = endpoint_index[("restricted", endpoint)]
        allowed_value = float(
            allowed[
                "total_cost_usd"
                if metric == "cost"
                else "quantity_weighted_mean_delivery_time_h"
            ]
        )
        restricted_value = float(
            restricted[
                "total_cost_usd"
                if metric == "cost"
                else "quantity_weighted_mean_delivery_time_h"
            ]
        )
        loss = restricted_value - baseline
        avoided = restricted_value - allowed_value
        result.append(
            {
                "case_id": "D3",
                "condition_id": "delay_216h",
                "delay_h": 216.0,
                "endpoint": endpoint,
                "metric": metric,
                "baseline_value": baseline,
                "disrupted_restricted_value": restricted_value,
                "disrupted_allowed_value": allowed_value,
                "disruption_loss": loss,
                "ny_nj_avoided_loss": avoided,
                "recovery_ratio": avoided / loss if loss > 1e-6 else "",
                "ratio_reportable": int(loss > 1e-6),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--structural-results",
        type=Path,
        default=Path("results/v21_full_structural_core_exact"),
    )
    parser.add_argument(
        "--cost-sensitivity-results",
        type=Path,
        default=Path("results/v22_full_structural_cost_sensitivity"),
    )
    parser.add_argument(
        "--inland-results",
        type=Path,
        default=Path("results/v26_full_structural_inland_bottleneck"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/v30_final_results_addenda")
    )
    parser.add_argument("--enumeration-cap-per-gateway", type=int, default=100_000)
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    zero_payloads, base_cases = solve_matched_zero_penalty_baselines(
        args, data, base_config
    )
    zero_index = {
        (str(row["case_id"]), str(row["gateway_access"])): row
        for row in zero_payloads
    }

    zero_summary = [
        {
            key: value
            for key, value in row.items()
            if key not in {"solver_calls", "allocations"}
        }
        for row in zero_payloads
    ]
    write_csv(args.output / "matched_zero_penalty_baselines.csv", zero_summary)

    sensitivity_endpoints = read_csv(
        args.cost_sensitivity_results / "exact_endpoint_summary.csv"
    )
    sensitivity_index = {row["validation_id"]: row for row in sensitivity_endpoints}
    zero_recovery_rows: list[dict[str, object]] = []
    for case_id in CASES:
        delay_h = FOCAL_DELAYS[case_id]
        prefix = f"{case_id}_delay_{int(delay_h)}h_penalty_000pct"
        allowed = sensitivity_index[f"{prefix}_allowed_min_cost_then_time"]
        restricted = sensitivity_index[f"{prefix}_restricted_min_cost_then_time"]
        baseline_allowed = zero_index[(case_id, "allowed")]
        baseline_restricted = zero_index[(case_id, "restricted")]
        baseline = float(baseline_allowed["total_cost_usd"])
        disrupted_allowed = float(allowed["total_cost_usd"])
        disrupted_restricted = float(restricted["total_cost_usd"])
        loss = disrupted_restricted - baseline
        avoided = disrupted_restricted - disrupted_allowed
        zero_recovery_rows.append(
            {
                "case_id": case_id,
                "delay_h": delay_h,
                "endpoint": "min_cost_then_time",
                "penalty_multiplier": 0.0,
                "baseline_reference_access": "allowed",
                "baseline_allowed_cost_usd": baseline,
                "baseline_restricted_cost_usd": float(
                    baseline_restricted["total_cost_usd"]
                ),
                "baseline_allowed_ny_nj_share": baseline_allowed["ny_nj_share"],
                "disrupted_restricted_cost_usd": disrupted_restricted,
                "disrupted_allowed_cost_usd": disrupted_allowed,
                "disruption_loss_usd": loss,
                "ny_nj_avoided_loss_usd": avoided,
                "recovery_ratio": avoided / loss if loss > 1e-6 else "",
                "ratio_reportable": int(loss > 1e-6),
            }
        )
    write_csv(args.output / "zero_penalty_loss_recovery.csv", zero_recovery_rows)

    activation_rows = read_csv(
        Path("analysis/v21_full_structural_core_exact/endpoint_activation_status.csv")
    )
    write_csv(
        args.output / "activation_criterion_common_grid.csv",
        criterion_rows(activation_rows),
    )

    supplemental_ranges = read_csv(
        args.inland_results / "endpoint_ny_nj_share_ranges.csv"
    )
    supplemental_rows = [
        {
            **row,
            "meets_5pct": int(float(row["ny_nj_min_share"]) >= 0.05 - 1e-9),
            "meets_10pct": int(float(row["ny_nj_min_share"]) >= 0.10 - 1e-9),
            "meets_20pct": int(float(row["ny_nj_min_share"]) >= 0.20 - 1e-9),
        }
        for row in supplemental_ranges
        if row["case_id"] == "D3"
        and row["condition_id"]
        == "delay_216h_inland_us-d-03_capacity_100pct"
    ]
    write_csv(
        args.output / "activation_criterion_supplemental_216h.csv",
        supplemental_rows,
    )

    recovery_rows = read_csv(
        Path("analysis/v21_full_structural_core_exact/disruption_loss_recovery.csv")
    )
    v26_endpoints = read_csv(args.inland_results / "exact_endpoint_summary.csv")
    write_csv(
        args.output / "first_material_use_absolute_recovery.csv",
        absolute_recovery_rows(recovery_rows, v26_endpoints),
    )

    core_allocations = allocations_by_validation(
        args.structural_results / "full_exact_endpoint_allocations.csv"
    )
    component_rows: list[dict[str, object]] = []
    for case_id in CASES:
        case_data = base_cases[case_id]
        delay_h = FOCAL_DELAYS[case_id]
        base_condition = ExactCondition(
            case_id=case_id,
            condition_id="delay_0h",
            family="baseline",
            uswc_delay_h=0.0,
        )
        disrupted_condition = ExactCondition(
            case_id=case_id,
            condition_id=f"delay_{int(delay_h)}h",
            family="regional_delay",
            uswc_delay_h=delay_h,
        )
        baseline_id = (
            f"{case_id}_delay_0h_restricted_min_cost_then_time_structural_full"
        )
        disrupted_id = (
            f"{case_id}_delay_{int(delay_h)}h_restricted_"
            "min_cost_then_time_structural_full"
        )
        baseline_components = component_totals(
            case_data,
            core_allocations[baseline_id],
            condition_config(base_config, base_condition, gateway_access="restricted"),
        )
        disrupted_components = component_totals(
            case_data,
            core_allocations[disrupted_id],
            condition_config(
                base_config, disrupted_condition, gateway_access="restricted"
            ),
        )
        row: dict[str, object] = {
            "case_id": case_id,
            "delay_h": delay_h,
            "baseline_validation_id": baseline_id,
            "disrupted_restricted_validation_id": disrupted_id,
        }
        for field in baseline_components:
            row[f"baseline_{field}"] = baseline_components[field]
            row[f"disrupted_restricted_{field}"] = disrupted_components[field]
            row[f"disruption_loss_{field}"] = (
                disrupted_components[field] - baseline_components[field]
            )
        component_rows.append(row)
    write_csv(
        args.output / "restricted_disruption_loss_cost_components.csv",
        component_rows,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "structural_source": str(args.structural_results),
                "cost_sensitivity_source": str(args.cost_sensitivity_results),
                "inland_source": str(args.inland_results),
                "matched_zero_penalty_endpoint_tasks": len(zero_payloads),
                "common_delay_grid_h": list(COMMON_DELAY_GRID),
                "activation_criteria": list(CRITERIA),
                "zero_penalty_recovery_definition": (
                    "(disrupted restricted cost - disrupted allowed cost) / "
                    "(disrupted restricted cost - undisturbed allowed cost), "
                    "with the tardiness penalty fixed at zero throughout"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Final Results addenda written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
