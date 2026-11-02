#!/usr/bin/env python3
"""Run the complete exact core grid with the exhaustive structural library.

The v2.0 coverage experiment established that Martins-8 omits resource-distinct
paths that materially improve exact endpoints.  This script therefore rebuilds
the main delay/capacity/counterfactual evidence on the exhaustive structural
library.  Proven v2.0 endpoints are copied as immutable checkpoints; share
bounds are recomputed uniformly with the current numerical safeguards.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil

from china_us_multimodal.config import load_config
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import core_conditions
from run_final_path_coverage import (
    MARTINS_LIMITS,
    enumerate_case_paths,
    finalize,
    run_full_endpoints,
    run_full_share_bounds,
    write_coverage_audit,
)
from run_supplementary_core_exact import load_json, write_json_atomic


def seed_proven_endpoints(source: Path, target: Path) -> int:
    """Copy proven structural-full endpoint checkpoints without overwriting."""
    source_root = source / "checkpoints" / "endpoints"
    target_root = target / "checkpoints" / "endpoints"
    copied = 0
    if not source_root.exists():
        return copied
    target_root.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_root.glob("*.json")):
        payload = load_json(source_path)
        if int(payload.get("lexicographic_proven", 0)) != 1:
            continue
        if payload.get("library_variant") != "structural_full":
            continue
        target_path = target_root / source_path.name
        if target_path.exists():
            continue
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact core grid using all structurally feasible paths."
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--reuse-full-results",
        type=Path,
        default=Path("results/v20_final_path_coverage"),
    )
    parser.add_argument(
        "--paths8-results",
        type=Path,
        default=Path("results/v17_core_exact_paths8"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v21_full_structural_core_exact"),
    )
    parser.add_argument("--enumeration-cap-per-gateway", type=int, default=100_000)
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    required = (
        args.paths8_results / "exact_endpoint_summary.csv",
        args.paths8_results / "exact_endpoint_ny_nj_share_ranges.csv",
        args.reuse_full_results / "manifest.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required exact results are missing: " + ", ".join(missing))

    reused_endpoints = seed_proven_endpoints(args.reuse_full_results, args.output)
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
    audit_rows = write_coverage_audit(args, data, base_config, case_enumerations)
    incomplete = [
        row
        for row in audit_rows
        if int(row["structural_enumeration_complete"]) != 1
    ]
    if incomplete:
        write_json_atomic(
            args.output / "manifest.json",
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "structural_enumeration_complete": False,
                "enumeration_cap_per_gateway": args.enumeration_cap_per_gateway,
                "incomplete_od_gateway_rows": incomplete,
                "exact_endpoint_experiment_run": False,
            },
        )
        raise RuntimeError(
            "Structural enumeration cap reached; exact core grid was not started."
        )

    run_full_endpoints(args, data, base_config, conditions, case_enumerations)
    run_full_share_bounds(args, data, base_config, conditions, case_enumerations)
    endpoints, shares = finalize(args, data, conditions)
    write_json_atomic(
        args.output / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "structural_enumeration_complete": True,
            "enumeration_cap_per_gateway": args.enumeration_cap_per_gateway,
            "conditions": [asdict(item) for item in conditions],
            "martins_limits_audited": list(MARTINS_LIMITS),
            "reused_proven_v20_endpoints": reused_endpoints,
            "full_endpoint_tasks": len(endpoints),
            "full_share_regions": len(shares),
            "paths8_source": str(args.paths8_results),
            "structural_full_source": str(args.reuse_full_results),
            "interpretation": (
                "All simple paths satisfying the stated leg, mode-change, and "
                "no-node-revisit restrictions are included without static "
                "cost--time dominance pruning."
            ),
        },
    )
    print(f"Full structural exact core grid written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
