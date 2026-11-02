#!/usr/bin/env python3
"""Exact multi-wave release robustness on the exhaustive path library."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import random

from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import capacity_usage_rows
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import ExactCondition, condition_config
from run_final_path_coverage import (
    build_full_model,
    enumerate_case_paths,
    share_bound_complete,
    solve_share_bound_with_numerical_retry,
    zero_share_lower_bound_certificate,
)
from run_supplementary_core_exact import (
    ENDPOINTS,
    allocations_from_payload,
    endpoint_payload,
    endpoint_region_bounds,
    endpoints_coincident,
    load_json,
    share_bound_payload,
    write_csv,
    write_json_atomic,
)


CASES = ("D1", "D2", "D3")
DELAYS_H = (0.0, 168.0, 240.0, 264.0, 288.0, 336.0)
INSTANCE_SEEDS = tuple(range(20261121, 20261126))
CASE_SEED_OFFSETS = {"D1": 101, "D2": 202, "D3": 303}


@dataclass(frozen=True)
class ReleaseInstance:
    instance_id: str
    pattern: str
    seed: int
    wave_offsets_h: tuple[float, ...]


@dataclass(frozen=True)
class ReleaseSpec:
    condition: ExactCondition
    instance: ReleaseInstance


def release_instances() -> tuple[ReleaseInstance, ...]:
    rows: list[ReleaseInstance] = []
    pattern_offsets = {
        "two_weekly_waves": (0.0, 168.0),
        "four_weekly_waves": (0.0, 168.0, 336.0, 504.0),
        "daily_over_four_weeks": tuple(24.0 * day for day in range(28)),
    }
    for pattern, offsets in pattern_offsets.items():
        for index, seed in enumerate(INSTANCE_SEEDS, start=1):
            rows.append(
                ReleaseInstance(
                    instance_id=f"{pattern}_i{index:02d}",
                    pattern=pattern,
                    seed=seed,
                    wave_offsets_h=offsets,
                )
            )
    return tuple(rows)


def specs() -> tuple[ReleaseSpec, ...]:
    rows: list[ReleaseSpec] = []
    for instance in release_instances():
        for case_id in CASES:
            for delay_h in DELAYS_H:
                delay_label = int(round(delay_h))
                rows.append(
                    ReleaseSpec(
                        condition=ExactCondition(
                            case_id=case_id,
                            condition_id=(
                                f"{instance.instance_id}_delay_{delay_label}h"
                            ),
                            family="release_pattern_robustness",
                            uswc_delay_h=delay_h,
                        ),
                        instance=instance,
                    )
                )
    return tuple(rows)


def released_case_data(base_case, case_id: str, instance: ReleaseInstance):
    rng = random.Random(instance.seed + CASE_SEED_OFFSETS[case_id])
    shipment_ids = sorted(base_case.shipments)
    rng.shuffle(shipment_ids)
    wave_loads = [0.0 for _ in instance.wave_offsets_h]
    assignments: dict[str, int] = {}
    for shipment_id in shipment_ids:
        minimum = min(wave_loads)
        candidates = [
            index
            for index, value in enumerate(wave_loads)
            if abs(value - minimum) <= 1e-9
        ]
        wave = rng.choice(candidates)
        assignments[shipment_id] = wave
        wave_loads[wave] += base_case.shipments[shipment_id].quantity_feu

    shipments = {}
    manifest_rows: list[dict[str, object]] = []
    for shipment_id, shipment in base_case.shipments.items():
        wave = assignments[shipment_id]
        shift_h = instance.wave_offsets_h[wave]
        due_h = None if shipment.due_h is None else shipment.due_h + shift_h
        shipments[shipment_id] = replace(
            shipment,
            release_h=shipment.release_h + shift_h,
            due_h=due_h,
        )
        manifest_rows.append(
            {
                "release_instance_id": instance.instance_id,
                "release_pattern": instance.pattern,
                "generator_seed": instance.seed,
                "case_id": case_id,
                "shipment_id": shipment_id,
                "quantity_feu": shipment.quantity_feu,
                "wave_index": wave,
                "release_shift_h": shift_h,
                "base_release_h": shipment.release_h,
                "generated_release_h": shipment.release_h + shift_h,
                "base_due_h": shipment.due_h,
                "generated_due_h": due_h,
                "delivery_allowance_preserved": int(
                    shipment.due_h is None
                    or abs(
                        (due_h - (shipment.release_h + shift_h))
                        - (shipment.due_h - shipment.release_h)
                    )
                    <= 1e-9
                ),
            }
        )
    generated = replace(base_case, shipments=shipments)
    if abs(
        sum(row.quantity_feu for row in generated.shipments.values())
        - sum(row.quantity_feu for row in base_case.shipments.values())
    ) > 1e-8:
        raise RuntimeError("Release transformation changed total demand.")
    return generated, manifest_rows


def validation_id(spec: ReleaseSpec, access: str, endpoint: str) -> str:
    return f"{spec.condition.case_id}_{spec.condition.condition_id}_{access}_{endpoint}"


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
        "--output",
        type=Path,
        default=Path("results/v24_full_structural_release_robustness"),
    )
    parser.add_argument("--enumeration-cap-per-gateway", type=int, default=100_000)
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    structural_manifest = load_json(args.structural_results / "manifest.json")
    if not structural_manifest.get("structural_enumeration_complete"):
        raise RuntimeError("The structural path source is not exhaustive.")

    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    experiment_specs = specs()
    base_cases = {case_id: select_case(data, case_id) for case_id in CASES}
    enumerations = {
        case_id: enumerate_case_paths(
            base_cases[case_id], base_config, args.enumeration_cap_per_gateway
        )
        for case_id in CASES
    }
    if any(
        not enumeration.complete
        for by_gateway in enumerations.values()
        for enumeration in by_gateway.values()
    ):
        raise RuntimeError("Structural enumeration cap reached.")

    case_cache = {}
    assignment_rows: list[dict[str, object]] = []
    for instance in release_instances():
        for case_id in CASES:
            generated, rows = released_case_data(
                base_cases[case_id], case_id, instance
            )
            case_cache[(case_id, instance.instance_id)] = generated
            assignment_rows.extend(rows)
    write_csv(
        args.output / "release_assignments.csv",
        assignment_rows,
        list(assignment_rows[0]),
    )

    checkpoint_root = args.output / "checkpoints"
    total = len(experiment_specs) * 2 * len(ENDPOINTS)
    completed = 0
    for spec in experiment_specs:
        case_id = spec.condition.case_id
        case_data = case_cache[(case_id, spec.instance.instance_id)]
        for access in ("allowed", "restricted"):
            config = condition_config(base_config, spec.condition, gateway_access=access)
            model = None
            max_paths = None
            for endpoint, primary in ENDPOINTS:
                task_id = validation_id(spec, access, endpoint)
                checkpoint = checkpoint_root / "endpoints" / f"{task_id}.json"
                if checkpoint.exists():
                    existing = load_json(checkpoint)
                    if int(existing.get("lexicographic_proven", 0)) == 1:
                        completed += 1
                        print(f"[{completed}/{total}] resume {task_id}", flush=True)
                        continue
                if model is None:
                    model, max_paths = build_full_model(
                        case_data, config, enumerations[case_id]
                    )
                from china_us_multimodal.exact_endpoint import solve_lexicographic_endpoint

                result = solve_lexicographic_endpoint(
                    model,
                    primary_objective=primary,
                    time_limit_seconds=args.time_limit_seconds,
                    mip_relative_gap=args.mip_relative_gap,
                    primary_relative_tolerance=args.relative_tolerance,
                    primary_absolute_tolerance=args.absolute_tolerance,
                )
                payload = endpoint_payload(
                    condition=spec.condition,
                    gateway_access=access,
                    endpoint=endpoint,
                    model=model,
                    result=result,
                    max_paths_per_gateway=int(max_paths),
                )
                payload.update(
                    {
                        "validation_id": task_id,
                        "library_variant": "structural_full",
                        "release_instance_id": spec.instance.instance_id,
                        "release_pattern": spec.instance.pattern,
                        "release_generator_seed": spec.instance.seed,
                    }
                )
                write_json_atomic(checkpoint, payload)
                completed += 1
                print(
                    f"[{completed}/{total}] {task_id} "
                    f"proven={payload['lexicographic_proven']}",
                    flush=True,
                )

    endpoint_payloads = [
        load_json(path)
        for path in sorted((checkpoint_root / "endpoints").glob("*.json"))
    ]
    endpoint_index = {
        (
            row["case_id"],
            row["condition_id"],
            row["gateway_access"],
            row["endpoint"],
        ): row
        for row in endpoint_payloads
    }
    spec_index = {
        (spec.condition.case_id, spec.condition.condition_id): spec
        for spec in experiment_specs
    }

    planned = []
    for spec in experiment_specs:
        endpoints = {
            endpoint: endpoint_index[
                (spec.condition.case_id, spec.condition.condition_id, "allowed", endpoint)
            ]
            for endpoint, _ in ENDPOINTS
        }
        if not all(int(row["lexicographic_proven"]) == 1 for row in endpoints.values()):
            continue
        if endpoints_coincident(
            endpoints["min_cost_then_time"],
            endpoints["min_time_then_cost"],
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        ):
            planned.append((spec, "coincident_endpoints", endpoints["min_cost_then_time"]))
        else:
            planned.extend((spec, endpoint, row) for endpoint, row in endpoints.items())

    share_total = len(planned) * 2
    share_completed = 0
    for spec, region, endpoint in planned:
        case_id = spec.condition.case_id
        case_data = case_cache[(case_id, spec.instance.instance_id)]
        config = condition_config(base_config, spec.condition, gateway_access="allowed")
        model = None
        cost_bound, time_bound = endpoint_region_bounds(
            endpoint,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        for direction in ("min", "max"):
            bound_id = (
                f"{case_id}_{spec.condition.condition_id}_{region}_{direction}_"
                "ny_nj_share_structural_full"
            )
            checkpoint = checkpoint_root / "share_bounds" / f"{bound_id}.json"
            if checkpoint.exists():
                existing = load_json(checkpoint)
                if share_bound_complete(existing):
                    share_completed += 1
                    print(
                        f"[share {share_completed}/{share_total}] resume {bound_id}",
                        flush=True,
                    )
                    continue
            if direction == "min" and float(endpoint.get("ny_nj_share", 1.0)) <= 1e-12:
                payload = zero_share_lower_bound_certificate(
                    bound_id=bound_id,
                    condition=spec.condition,
                    endpoint_region=region,
                    cost_upper_bound=cost_bound,
                    time_upper_bound=time_bound,
                    endpoint=endpoint,
                )
            else:
                if model is None:
                    model, _ = build_full_model(
                        case_data, config, enumerations[case_id]
                    )
                result, applied_cost, applied_time, attempts, retry_mode = (
                    solve_share_bound_with_numerical_retry(
                        model=model,
                        direction=direction,
                        cost_bound=cost_bound,
                        time_bound=time_bound,
                        time_limit_seconds=args.time_limit_seconds,
                        mip_relative_gap=args.mip_relative_gap,
                    )
                )
                payload = share_bound_payload(
                    bound_id=bound_id,
                    condition=spec.condition,
                    endpoint_region=region,
                    direction=direction,
                    cost_upper_bound=applied_cost,
                    time_upper_bound=applied_time,
                    result=result,
                    data=case_data,
                )
                payload.update(
                    {
                        "strict_cost_upper_bound_usd": cost_bound,
                        "strict_time_upper_bound_h": time_bound,
                        "share_bound_retry_mode": retry_mode,
                        "solver_attempts": attempts,
                    }
                )
            payload.update(
                {
                    "library_variant": "structural_full",
                    "release_instance_id": spec.instance.instance_id,
                    "release_pattern": spec.instance.pattern,
                    "release_generator_seed": spec.instance.seed,
                }
            )
            write_json_atomic(checkpoint, payload)
            share_completed += 1
            print(
                f"[share {share_completed}/{share_total}] {bound_id} "
                f"proven={payload['proven_optimal']} share={payload['ny_nj_share']}",
                flush=True,
            )

    share_payloads = [
        load_json(path)
        for path in sorted((checkpoint_root / "share_bounds").glob("*.json"))
    ]
    unresolved = [row for row in share_payloads if not share_bound_complete(row)]
    if unresolved:
        raise RuntimeError(
            "Unresolved share bounds: "
            + ", ".join(str(row["bound_id"]) for row in unresolved)
        )

    endpoint_rows = [
        {key: value for key, value in row.items() if key not in {"solver_calls", "allocations"}}
        for row in endpoint_payloads
    ]
    write_csv(args.output / "exact_endpoint_summary.csv", endpoint_rows, list(endpoint_rows[0]))

    allocation_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    for payload in endpoint_payloads:
        validation = str(payload["validation_id"])
        spec = spec_index[(str(payload["case_id"]), str(payload["condition_id"]))]
        case_data = case_cache[(spec.condition.case_id, spec.instance.instance_id)]
        config = condition_config(
            base_config,
            spec.condition,
            gateway_access=str(payload["gateway_access"]),
        )
        allocations = list(allocations_from_payload(payload))
        for call in payload["solver_calls"]:
            solver_rows.append({"validation_id": validation, **call})
        for allocation_id, allocation in enumerate(payload["allocations"], start=1):
            allocation_rows.append(
                {
                    "validation_id": validation,
                    "allocation_id": allocation_id,
                    "shipment_id": allocation["shipment_id"],
                    "quantity_feu": allocation["quantity_feu"],
                    "gateway": allocation["gateway"],
                    "arc_ids": "|".join(allocation["arc_ids"]),
                }
            )
        for row in capacity_usage_rows(case_data, allocations, config):
            capacity_rows.append({"validation_id": validation, **row})
    write_csv(args.output / "exact_endpoint_allocations.csv", allocation_rows, list(allocation_rows[0]))
    write_csv(args.output / "exact_endpoint_solver_calls.csv", solver_rows, list(solver_rows[0]))
    write_csv(
        args.output / "exact_endpoint_capacity_usage.csv",
        capacity_rows,
        list(capacity_rows[0]),
    )
    binding_rows = [row for row in capacity_rows if int(row["binding_95pct"]) == 1]
    write_csv(
        args.output / "binding_resource_diagnostic.csv",
        binding_rows,
        list(capacity_rows[0]),
    )

    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in share_payloads:
        key = (str(row["case_id"]), str(row["condition_id"]), str(row["endpoint_region"]))
        groups.setdefault(key, {})[str(row["direction"])] = row
    range_rows = []
    for (case_id, condition_id, region), directions in sorted(groups.items()):
        minimum, maximum = directions["min"], directions["max"]
        spec = spec_index[(case_id, condition_id)]
        range_rows.append(
            {
                "case_id": case_id,
                "condition_id": condition_id,
                "release_instance_id": spec.instance.instance_id,
                "release_pattern": spec.instance.pattern,
                "release_generator_seed": spec.instance.seed,
                "delay_h": spec.condition.uswc_delay_h,
                "endpoint_region": region,
                "ny_nj_min_share": minimum["ny_nj_share"],
                "ny_nj_max_share": maximum["ny_nj_share"],
                "min_proven": minimum["proven_optimal"],
                "max_proven": maximum["proven_optimal"],
            }
        )
    write_csv(args.output / "endpoint_ny_nj_share_ranges.csv", range_rows, list(range_rows[0]))

    value_rows = []
    for spec in experiment_specs:
        for endpoint, _ in ENDPOINTS:
            allowed = endpoint_index[
                (spec.condition.case_id, spec.condition.condition_id, "allowed", endpoint)
            ]
            restricted = endpoint_index[
                (spec.condition.case_id, spec.condition.condition_id, "restricted", endpoint)
            ]
            value_rows.append(
                {
                    "case_id": spec.condition.case_id,
                    "condition_id": spec.condition.condition_id,
                    "release_instance_id": spec.instance.instance_id,
                    "release_pattern": spec.instance.pattern,
                    "release_generator_seed": spec.instance.seed,
                    "delay_h": spec.condition.uswc_delay_h,
                    "endpoint": endpoint,
                    "allowed_ny_nj_share": allowed["ny_nj_share"],
                    "cost_avoided_by_allowing_ny_nj_usd": (
                        float(restricted["total_cost_usd"])
                        - float(allowed["total_cost_usd"])
                    ),
                    "time_avoided_by_allowing_ny_nj_h": (
                        float(restricted["quantity_weighted_mean_delivery_time_h"])
                        - float(allowed["quantity_weighted_mean_delivery_time_h"])
                    ),
                }
            )
    write_csv(args.output / "release_pattern_gateway_value.csv", value_rows, list(value_rows[0]))

    write_json_atomic(
        args.output / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "library_variant": "structural_full",
            "structural_source": str(args.structural_results),
            "single_wave_exact_source_reused_in_analysis": str(args.structural_results),
            "single_wave_duplicate_runs_suppressed": True,
            "release_instances": [asdict(item) for item in release_instances()],
            "delay_grid_h": list(DELAYS_H),
            "delivery_allowance_preserved": True,
            "total_demand_and_batch_count_preserved": True,
            "assignment_rule": "seeded FEU-balanced greedy assignment to fixed waves",
            "endpoint_tasks": len(endpoint_payloads),
            "share_bound_tasks": len(share_payloads),
            "interpretation_limits": [
                "The five instances per multi-wave pattern are constructed release assignments, not occurrence probabilities.",
                "The single-wave exact results are reused from v2.1 rather than redundantly re-optimized.",
                "Tested delays identify grid transitions rather than continuous real-world thresholds.",
            ],
        },
    )
    print(f"Full structural release robustness written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
