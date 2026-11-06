#!/usr/bin/env python3
"""Rebuild the exact core grid with two later departures selectable per leg."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import core_conditions
from run_final_path_coverage import (
    MARTINS_LIMITS,
    enumerate_case_paths,
    finalize,
    run_full_endpoints,
    run_full_share_bounds,
    structural_variant_suffix,
    write_coverage_audit,
)
from run_supplementary_core_exact import load_json, write_json_atomic


MAX_EXTRA_SERVICE_DEPARTURES = 2


def seed_v31_endpoints(source: Path, target: Path, conditions) -> int:
    """Reuse compatible proven expanded-service endpoints from v3.1."""
    valid_conditions = {(item.case_id, item.condition_id) for item in conditions}
    source_root = source / "checkpoints" / "endpoints"
    target_root = target / "checkpoints" / "endpoints"
    target_root.mkdir(parents=True, exist_ok=True)
    variant = structural_variant_suffix(MAX_EXTRA_SERVICE_DEPARTURES)
    copied = 0
    for source_path in sorted(source_root.glob("*_earliest_plus_two.json")):
        payload = load_json(source_path)
        key = (str(payload.get("case_id")), str(payload.get("condition_id")))
        if key not in valid_conditions:
            continue
        if int(payload.get("lexicographic_proven", 0)) != 1:
            continue
        if int(payload.get("evaluator_consistent", 0)) != 1:
            continue
        validation_id = (
            f"{payload['case_id']}_{payload['condition_id']}_"
            f"{payload['gateway_access']}_{payload['endpoint']}_{variant}"
        )
        target_path = target_root / f"{validation_id}.json"
        if target_path.exists():
            continue
        payload.update(
            {
                "validation_id": validation_id,
                "library_variant": variant,
                "service_choice_rule": "earliest_plus_2",
                "max_extra_service_departures": MAX_EXTRA_SERVICE_DEPARTURES,
                "reused_from": str(source_path),
            }
        )
        payload.pop("regional_inland_capacity_multiplier", None)
        payload.pop("source_validation_id", None)
        write_json_atomic(target_path, payload)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--reuse-expanded-results",
        type=Path,
        default=Path("results/v31_later_service_choice_validation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v32_full_service_choice_core_exact"),
    )
    parser.add_argument("--enumeration-cap-per-gateway", type=int, default=100_000)
    parser.add_argument("--time-limit-seconds", type=float, default=1800.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    args.paths8_results = None

    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    conditions = core_conditions()
    case_enumerations = {
        case_id: enumerate_case_paths(
            select_case(data, case_id),
            base_config,
            args.enumeration_cap_per_gateway,
        )
        for case_id in ("D1", "D2", "D3")
    }
    audit_rows = write_coverage_audit(
        args,
        data,
        base_config,
        case_enumerations,
    )
    incomplete = [
        row
        for row in audit_rows
        if int(row["structural_enumeration_complete"]) != 1
    ]
    if incomplete:
        raise RuntimeError("Structural enumeration cap reached.")

    reused = seed_v31_endpoints(
        args.reuse_expanded_results,
        args.output,
        conditions,
    )
    run_full_endpoints(
        args,
        data,
        base_config,
        conditions,
        case_enumerations,
        max_extra_service_departures=MAX_EXTRA_SERVICE_DEPARTURES,
    )
    run_full_share_bounds(
        args,
        data,
        base_config,
        conditions,
        case_enumerations,
        max_extra_service_departures=MAX_EXTRA_SERVICE_DEPARTURES,
    )
    endpoints, shares = finalize(args, data, conditions)
    write_json_atomic(
        args.output / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "full_service_choice_core_exact",
            "conditions": [asdict(item) for item in conditions],
            "structural_enumeration_complete": True,
            "enumeration_cap_per_gateway": args.enumeration_cap_per_gateway,
            "martins_limits_audited": list(MARTINS_LIMITS),
            "max_extra_service_departures": MAX_EXTRA_SERVICE_DEPARTURES,
            "service_choice_rule": (
                "earliest feasible departure plus the next two feasible "
                "departure events on every scheduled leg"
            ),
            "reused_proven_v31_endpoints": reused,
            "endpoint_tasks": len(endpoints),
            "share_regions": len(shares),
            "interpretation_limit": (
                "Exactness is conditional on the complete registered structural "
                "path set and the three-event service-choice set."
            ),
        },
    )
    print(
        f"Full service-choice exact core grid written to {args.output}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
