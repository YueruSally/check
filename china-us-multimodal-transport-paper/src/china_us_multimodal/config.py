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
    max_legs_per_path: int = 8
    max_mode_changes: int = 4
    allocation_tolerance_feu: float = 1e-6


@dataclass(frozen=True)
class PenaltyConfig:
    tardiness_usd_per_feu_h: float = 0.0
    infeasible_objective: float = 1e15


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


def load_config(path: str | Path, scenario_name: str | None = None) -> ModelConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)

    model = raw.get("model", {})
    quantity_unit = model.get("quantity_unit", "FEU")
    objectives = tuple(model.get("objectives", []))
    if objectives != ("total_cost_usd", "makespan_h"):
        raise ValueError(
            "The pilot must use exactly two objectives: total_cost_usd and makespan_h."
        )

    constraints = ConstraintConfig(**raw.get("constraints", {}))
    penalties = PenaltyConfig(**raw.get("penalties", {}))
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

    return ModelConfig(quantity_unit, objectives, constraints, penalties, scenario)
