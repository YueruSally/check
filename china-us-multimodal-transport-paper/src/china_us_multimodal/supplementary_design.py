"""Pre-registered non-carbon designs for the final supplementary experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import ModelConfig


CASES = ("D1", "D2", "D3")
DELAY_GRID_H = (0, 24, 72, 96, 120, 168, 240, 264, 288, 312, 336)
CAPACITY_GRID = (0.60, 0.40, 0.35)


@dataclass(frozen=True)
class ExactCondition:
    case_id: str
    condition_id: str
    family: str
    uswc_capacity_multiplier: float = 1.0
    uswc_delay_h: float = 0.0


def core_conditions() -> tuple[ExactCondition, ...]:
    rows: list[ExactCondition] = []
    for case_id in CASES:
        rows.extend(
            ExactCondition(
                case_id=case_id,
                condition_id=f"delay_{delay_h}h",
                family="baseline" if delay_h == 0 else "regional_delay",
                uswc_delay_h=float(delay_h),
            )
            for delay_h in DELAY_GRID_H
        )
        rows.extend(
            ExactCondition(
                case_id=case_id,
                condition_id=f"capacity_{int(round(capacity * 100))}pct",
                family="capacity_reduction",
                uswc_capacity_multiplier=capacity,
            )
            for capacity in CAPACITY_GRID
        )
        rows.append(
            ExactCondition(
                case_id=case_id,
                condition_id="compound_30pct_delay_168h",
                family="compound_disruption",
                uswc_capacity_multiplier=0.30,
                uswc_delay_h=168.0,
            )
        )
    return tuple(rows)


def path_sensitivity_conditions() -> tuple[ExactCondition, ...]:
    requested = (
        ("D1", 0, 1.0, "delay_0h", "baseline"),
        ("D2", 0, 1.0, "delay_0h", "baseline"),
        ("D3", 0, 1.0, "delay_0h", "baseline"),
        ("D1", 264, 1.0, "delay_264h", "regional_delay"),
        ("D1", 288, 1.0, "delay_288h", "regional_delay"),
        ("D2", 240, 1.0, "delay_240h", "regional_delay"),
        ("D2", 264, 1.0, "delay_264h", "regional_delay"),
        ("D2", 288, 1.0, "delay_288h", "regional_delay"),
        ("D3", 168, 1.0, "delay_168h", "regional_delay"),
        ("D3", 240, 1.0, "delay_240h", "regional_delay"),
        ("D1", 0, 0.35, "capacity_35pct", "capacity_reduction"),
        ("D2", 0, 0.35, "capacity_35pct", "capacity_reduction"),
        ("D3", 0, 0.35, "capacity_35pct", "capacity_reduction"),
        (
            "D1",
            168,
            0.30,
            "compound_30pct_delay_168h",
            "compound_disruption",
        ),
        (
            "D2",
            168,
            0.30,
            "compound_30pct_delay_168h",
            "compound_disruption",
        ),
        (
            "D3",
            168,
            0.30,
            "compound_30pct_delay_168h",
            "compound_disruption",
        ),
    )
    return tuple(
        ExactCondition(
            case_id=case_id,
            condition_id=condition_id,
            family=family,
            uswc_capacity_multiplier=capacity,
            uswc_delay_h=float(delay_h),
        )
        for case_id, delay_h, capacity, condition_id, family in requested
    )


def demand_robustness_conditions() -> tuple[ExactCondition, ...]:
    requested = (
        ("D1", 264, 1.0, "delay_264h", "regional_delay"),
        ("D1", 288, 1.0, "delay_288h", "regional_delay"),
        (
            "D1",
            168,
            0.30,
            "compound_30pct_delay_168h",
            "compound_disruption",
        ),
        ("D2", 240, 1.0, "delay_240h", "regional_delay"),
        ("D2", 264, 1.0, "delay_264h", "regional_delay"),
        ("D2", 288, 1.0, "delay_288h", "regional_delay"),
        ("D3", 168, 1.0, "delay_168h", "regional_delay"),
        ("D3", 240, 1.0, "delay_240h", "regional_delay"),
    )
    return tuple(
        ExactCondition(
            case_id=case_id,
            condition_id=condition_id,
            family=family,
            uswc_capacity_multiplier=capacity,
            uswc_delay_h=float(delay_h),
        )
        for case_id, delay_h, capacity, condition_id, family in requested
    )


def condition_config(
    base: ModelConfig,
    condition: ExactCondition,
    *,
    gateway_access: str,
    demand_multiplier: float = 1.0,
) -> ModelConfig:
    if gateway_access not in {"allowed", "restricted"}:
        raise ValueError("gateway_access must be 'allowed' or 'restricted'.")
    disabled = () if gateway_access == "allowed" else ("ny_nj",)
    scenario = replace(
        base.scenario,
        name=(
            f"supp_{condition.condition_id}"
            + ("_no_ny_nj" if gateway_access == "restricted" else "")
        ),
        panama_availability_multiplier=1.0,
        uswc_ocean_cost_multiplier=1.0,
        usec_ocean_cost_multiplier=1.0,
        port_capacity_multiplier=1.0,
        uswc_port_capacity_multiplier=condition.uswc_capacity_multiplier,
        usec_port_capacity_multiplier=1.0,
        la_lb_port_capacity_multiplier=1.0,
        seattle_tacoma_port_capacity_multiplier=1.0,
        uswc_ocean_departure_delay_h=condition.uswc_delay_h,
        usec_ocean_departure_delay_h=0.0,
        la_lb_ocean_departure_delay_h=0.0,
        seattle_tacoma_ocean_departure_delay_h=0.0,
        demand_multiplier=demand_multiplier,
        disabled_gateways=disabled,
    )
    return replace(base, scenario=scenario)
