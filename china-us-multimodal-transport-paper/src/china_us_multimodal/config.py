from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ConstraintConfig:
    enforce_timetable: bool = True
    enforce_capacity: bool = True
    hard_due_time: bool = False
    allow_split: bool = True
    max_paths_per_shipment: int = 3
    min_path_share: float = 0.10
    max_legs_per_path: int = 8
    max_mode_changes: int = 4
    allocation_tolerance_feu: float = 1e-6


@dataclass(frozen=True)
class PenaltyConfig:
    tardiness_usd_per_feu_h: float = 0.0
    infeasible_objective: float = 1e15


@dataclass(frozen=True)
class OperationalCostConfig:
    holding_usd_per_feu_h: float = 0.0
    processing_usd_per_feu_h: float = 0.0


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    panama_availability_multiplier: float = 1.0
    uswc_ocean_cost_multiplier: float = 1.0
    usec_ocean_cost_multiplier: float = 1.0
    port_capacity_multiplier: float = 1.0
    uswc_port_capacity_multiplier: float = 1.0
    usec_port_capacity_multiplier: float = 1.0
    uswc_ocean_departure_delay_h: float = 0.0
    usec_ocean_departure_delay_h: float = 0.0
    demand_multiplier: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    quantity_unit: str
    objectives: tuple[str, ...]
    constraints: ConstraintConfig
    penalties: PenaltyConfig
    scenario: ScenarioConfig
    operational_costs: OperationalCostConfig = OperationalCostConfig()


def load_config(path: str | Path, scenario_name: str | None = None) -> ModelConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)

    model = raw.get("model", {})
    quantity_unit = model.get("quantity_unit", "FEU")
    objectives = tuple(model.get("objectives", []))
    if objectives != (
        "total_cost_usd",
        "quantity_weighted_mean_delivery_time_h",
    ):
        raise ValueError(
            "The pilot must use exactly two objectives: total_cost_usd and "
            "quantity_weighted_mean_delivery_time_h."
        )

    constraints = ConstraintConfig(**raw.get("constraints", {}))
    penalties = PenaltyConfig(**raw.get("penalties", {}))
    operational_costs = OperationalCostConfig(**raw.get("operational_costs", {}))
    scenarios = raw.get("scenarios", {})
    selected = scenario_name or model.get("active_scenario")
    if not selected or selected not in scenarios:
        raise ValueError(f"Unknown or missing active scenario: {selected!r}.")
    scenario = ScenarioConfig(name=selected, **scenarios[selected])

    if quantity_unit != "FEU":
        raise ValueError("All pilot quantities and capacities must be converted to FEU.")
    if not 0 < scenario.panama_availability_multiplier <= 1:
        raise ValueError("Panama availability multiplier must be in (0, 1].")
    for field_name, multiplier in (
        ("port_capacity_multiplier", scenario.port_capacity_multiplier),
        ("uswc_port_capacity_multiplier", scenario.uswc_port_capacity_multiplier),
        ("usec_port_capacity_multiplier", scenario.usec_port_capacity_multiplier),
        ("demand_multiplier", scenario.demand_multiplier),
    ):
        if multiplier <= 0:
            raise ValueError(f"{field_name} must be positive.")
    for field_name, delay_h in (
        ("uswc_ocean_departure_delay_h", scenario.uswc_ocean_departure_delay_h),
        ("usec_ocean_departure_delay_h", scenario.usec_ocean_departure_delay_h),
    ):
        if delay_h < 0:
            raise ValueError(f"{field_name} must be non-negative.")
    if constraints.max_paths_per_shipment < 1:
        raise ValueError("max_paths_per_shipment must be positive.")
    if not 0 < constraints.min_path_share <= 1:
        raise ValueError("min_path_share must be in (0, 1].")
    if operational_costs.holding_usd_per_feu_h < 0:
        raise ValueError("Holding cost must be non-negative.")
    if operational_costs.processing_usd_per_feu_h < 0:
        raise ValueError("Processing cost must be non-negative.")

    return ModelConfig(
        quantity_unit,
        objectives,
        constraints,
        penalties,
        scenario,
        operational_costs,
    )
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ConstraintConfig:
    enforce_timetable: bool = True
    enforce_capacity: bool = True
    hard_due_time: bool = False
    allow_split: bool = True
    max_paths_per_shipment: int = 3
    min_path_share: float = 0.10
    max_legs_per_path: int = 8
    max_mode_changes: int = 4
    allocation_tolerance_feu: float = 1e-6


@dataclass(frozen=True)
class PenaltyConfig:
    tardiness_usd_per_feu_h: float = 0.0
    infeasible_objective: float = 1e15


@dataclass(frozen=True)
class OperationalCostConfig:
    holding_usd_per_feu_h: float = 0.0
    processing_usd_per_feu_h: float = 0.0


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    panama_availability_multiplier: float = 1.0
    uswc_ocean_cost_multiplier: float = 1.0
    usec_ocean_cost_multiplier: float = 1.0
    port_capacity_multiplier: float = 1.0
    demand_multiplier: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    quantity_unit: str
    objectives: tuple[str, ...]
    constraints: ConstraintConfig
    penalties: PenaltyConfig
    scenario: ScenarioConfig
    operational_costs: OperationalCostConfig = OperationalCostConfig()


def load_config(path: str | Path, scenario_name: str | None = None) -> ModelConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)

    model = raw.get("model", {})
    quantity_unit = model.get("quantity_unit", "FEU")
    objectives = tuple(model.get("objectives", []))
    if objectives != (
        "total_cost_usd",
        "quantity_weighted_mean_delivery_time_h",
    ):
        raise ValueError(
            "The pilot must use exactly two objectives: total_cost_usd and "
            "quantity_weighted_mean_delivery_time_h."
        )

    constraints = ConstraintConfig(**raw.get("constraints", {}))
    penalties = PenaltyConfig(**raw.get("penalties", {}))
    operational_costs = OperationalCostConfig(**raw.get("operational_costs", {}))
    scenarios = raw.get("scenarios", {})
    selected = scenario_name or model.get("active_scenario")
    if not selected or selected not in scenarios:
        raise ValueError(f"Unknown or missing active scenario: {selected!r}.")
    scenario = ScenarioConfig(name=selected, **scenarios[selected])

    if quantity_unit != "FEU":
        raise ValueError("All pilot quantities and capacities must be converted to FEU.")
    if not 0 < scenario.panama_availability_multiplier <= 1:
        raise ValueError("Panama availability multiplier must be in (0, 1].")
    if constraints.max_paths_per_shipment < 1:
        raise ValueError("max_paths_per_shipment must be positive.")
    if not 0 < constraints.min_path_share <= 1:
        raise ValueError("min_path_share must be in (0, 1].")
    if operational_costs.holding_usd_per_feu_h < 0:
        raise ValueError("Holding cost must be non-negative.")
    if operational_costs.processing_usd_per_feu_h < 0:
        raise ValueError("Processing cost must be non-negative.")

    return ModelConfig(
        quantity_unit,
        objectives,
        constraints,
        penalties,
        scenario,
        operational_costs,
    )
