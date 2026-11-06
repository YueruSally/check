from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
import time
from typing import Iterable

from .config import ModelConfig
from .diagnostics import capacity_usage_rows, route_component_rows
from .domain import ModelData, RouteAllocation
from .evaluator import EvaluationResult, evaluate_solution
from .path_library import build_gateway_aware_library


@dataclass(frozen=True)
class PathProfile:
    shipment_id: str
    path_index: int
    arc_ids: tuple[str, ...]
    service_departure_ranks: tuple[int, ...]
    unit_cost_usd: float
    lead_time_h: float
    resource_coefficients: tuple[tuple[tuple[str, str, str], float], ...]


@dataclass(frozen=True)
class SolverCall:
    stage: int
    objective: str
    status_code: int
    status: str
    success: bool
    objective_value: float | None
    dual_bound: float | None
    relative_gap: float | None
    node_count: int | None
    elapsed_seconds: float | None
    message: str


@dataclass(frozen=True)
class ExactEndpointResult:
    primary_objective: str
    secondary_objective: str
    stage_calls: tuple[SolverCall, ...]
    allocations: tuple[RouteAllocation, ...]
    evaluation: EvaluationResult | None
    primary_optimum: float | None
    primary_tolerance: float | None
    stage_one_proven: bool
    lexicographic_proven: bool
    evaluator_consistent: bool
    evaluator_cost_abs_error: float | None
    evaluator_time_abs_error: float | None
    resource_max_abs_error_feu: float | None


@dataclass(frozen=True)
class BoundedMilpResult:
    """One exact MILP call under optional linear upper bounds.

    This result type is used for endpoint-equivalent NY/NJ share bounds and
    feasibility checks.  ``proven_optimal`` and ``proven_infeasible`` are kept
    separate so that a time-limit result is never reported as infeasible.
    """

    objective: str
    maximize: bool
    solver_call: SolverCall
    allocations: tuple[RouteAllocation, ...]
    evaluation: EvaluationResult | None
    objective_value: float | None
    proven_optimal: bool
    proven_infeasible: bool
    evaluator_consistent: bool
    evaluator_cost_abs_error: float | None
    evaluator_time_abs_error: float | None
    resource_max_abs_error_feu: float | None


@dataclass(frozen=True)
class ExactEndpointModel:
    data: ModelData
    config: ModelConfig
    profiles: tuple[PathProfile, ...]
    demand_by_shipment: tuple[tuple[str, float], ...]
    resource_capacities: tuple[tuple[tuple[str, str, str], float], ...]
    max_extra_service_departures: int = 0

    @property
    def quantity_variable_count(self) -> int:
        return len(self.profiles)

    @property
    def variable_count(self) -> int:
        return 2 * len(self.profiles)

    @property
    def total_demand(self) -> float:
        return sum(value for _, value in self.demand_by_shipment)

    def objective_coefficients(self, objective: str) -> list[float]:
        if objective == "cost":
            continuous = [profile.unit_cost_usd for profile in self.profiles]
        elif objective == "time":
            continuous = [
                profile.lead_time_h / self.total_demand for profile in self.profiles
            ]
        else:
            raise ValueError(f"Unknown exact endpoint objective: {objective!r}.")
        return continuous + [0.0] * len(self.profiles)

    def gateway_share_coefficients(self, gateway: str) -> list[float]:
        """Return coefficients for the fraction of demand using ``gateway``."""
        continuous = [
            (1.0 / self.total_demand)
            if _profile_gateway(self, profile) == gateway
            else 0.0
            for profile in self.profiles
        ]
        return continuous + [0.0] * len(self.profiles)


def _profile_gateway(model: ExactEndpointModel, profile: PathProfile) -> str:
    return _gateway_for_arc_ids(model.data, profile.arc_ids)


def _gateway_for_arc_ids(data: ModelData, arc_ids: Iterable[str]) -> str:
    for arc_id in arc_ids:
        arc = data.arcs[arc_id]
        for node_id in (arc.origin, arc.destination):
            if data.nodes[node_id].kind.value == "us_port":
                return node_id
    return "unclassified"


def build_exact_endpoint_model(
    data: ModelData,
    config: ModelConfig,
    *,
    max_paths_per_gateway: int = 4,
    required_paths_by_shipment: dict[str, tuple[tuple[str, ...], ...]] | None = None,
    max_extra_service_departures: int = 0,
) -> ExactEndpointModel:
    """Build the path-allocation MILP using the evaluator's route accounting."""
    if max_extra_service_departures < 0:
        raise ValueError("max_extra_service_departures must be non-negative.")
    library = build_gateway_aware_library(
        data,
        max_legs=config.constraints.max_legs_per_path,
        max_mode_changes=config.constraints.max_mode_changes,
        max_paths_per_gateway=max_paths_per_gateway,
        disabled_gateways=set(config.scenario.disabled_gateways),
    )
    if required_paths_by_shipment:
        nested_library: dict[str, tuple[tuple[str, ...], ...]] = {}
        for shipment_id, generated_paths in library.items():
            merged: list[tuple[str, ...]] = []
            counts: dict[str, int] = {}
            for path in (
                *required_paths_by_shipment.get(shipment_id, ()),
                *generated_paths,
            ):
                if path in merged:
                    continue
                gateway = _gateway_for_arc_ids(data, path)
                if counts.get(gateway, 0) >= max_paths_per_gateway:
                    continue
                merged.append(path)
                counts[gateway] = counts.get(gateway, 0) + 1
            nested_library[shipment_id] = tuple(merged)
        library = nested_library
    profiles: list[PathProfile] = []
    resource_capacities: dict[tuple[str, str, str], float] = {}
    for shipment_id in sorted(library):
        profile_index = 0
        for arc_ids in library[shipment_id]:
            scheduled_positions = tuple(
                index
                for index, arc_id in enumerate(arc_ids)
                if config.constraints.enforce_timetable and data.arcs[arc_id].service_ids
            )
            choices = product(
                range(max_extra_service_departures + 1),
                repeat=len(scheduled_positions),
            )
            seen_variants: set[
                tuple[float, float, tuple[tuple[tuple[str, str, str], float], ...]]
            ] = set()
            for selected_ranks in choices:
                ranks = [0] * len(arc_ids)
                for position, rank in zip(scheduled_positions, selected_ranks):
                    ranks[position] = rank
                rank_tuple = tuple(ranks)
                unit_allocation = RouteAllocation(
                    shipment_id,
                    1.0,
                    arc_ids,
                    service_departure_ranks=rank_tuple,
                )
                components = route_component_rows(data, [unit_allocation], config)
                if len(components) != 1:
                    raise RuntimeError("A unit path must produce exactly one route profile.")
                component = components[0]
                resource_coefficients: dict[tuple[str, str, str], float] = {}
                if config.constraints.enforce_capacity:
                    for row in capacity_usage_rows(data, [unit_allocation], config):
                        key = (
                            str(row["resource_type"]),
                            str(row["resource_id"]),
                            str(row["period"]),
                        )
                        resource_coefficients[key] = float(row["used_feu"])
                        capacity = float(row["capacity_feu"])
                        if key in resource_capacities and not math.isclose(
                            resource_capacities[key],
                            capacity,
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        ):
                            raise RuntimeError(f"Inconsistent capacity for resource {key}.")
                        resource_capacities[key] = capacity
                resource_tuple = tuple(sorted(resource_coefficients.items()))
                variant_signature = (
                    float(component["total_cost_usd"]),
                    float(component["lead_time_h"]),
                    resource_tuple,
                )
                if variant_signature in seen_variants:
                    continue
                seen_variants.add(variant_signature)
                profiles.append(
                    PathProfile(
                        shipment_id=shipment_id,
                        path_index=profile_index,
                        arc_ids=arc_ids,
                        service_departure_ranks=rank_tuple,
                        unit_cost_usd=float(component["total_cost_usd"]),
                        lead_time_h=float(component["lead_time_h"]),
                        resource_coefficients=resource_tuple,
                    )
                )
                profile_index += 1
    demand_by_shipment = tuple(
        (
            shipment_id,
            data.shipments[shipment_id].quantity_feu * config.scenario.demand_multiplier,
        )
        for shipment_id in sorted(data.shipments)
    )
    return ExactEndpointModel(
        data=data,
        config=config,
        profiles=tuple(profiles),
        demand_by_shipment=demand_by_shipment,
        resource_capacities=tuple(sorted(resource_capacities.items())),
        max_extra_service_departures=max_extra_service_departures,
    )


def _status_name(status_code: int) -> str:
    return {
        0: "optimal_within_tolerance",
        1: "time_or_iteration_limit",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }.get(status_code, "unknown")


def _solver_call(
    result: object, stage: int, objective: str, elapsed_seconds: float
) -> SolverCall:
    status_code = int(getattr(result, "status", 4))
    value = getattr(result, "fun", None)
    return SolverCall(
        stage=stage,
        objective=objective,
        status_code=status_code,
        status=_status_name(status_code),
        success=bool(getattr(result, "success", False)),
        objective_value=float(value) if value is not None else None,
        dual_bound=(
            float(getattr(result, "mip_dual_bound"))
            if getattr(result, "mip_dual_bound", None) is not None
            else None
        ),
        relative_gap=(
            float(getattr(result, "mip_gap"))
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        node_count=(
            int(getattr(result, "mip_node_count"))
            if getattr(result, "mip_node_count", None) is not None
            else None
        ),
        elapsed_seconds=elapsed_seconds,
        message=str(getattr(result, "message", "")),
    )


def _linear_system(
    model: ExactEndpointModel,
    *,
    primary_coefficients: list[float] | None = None,
    primary_upper_bound: float | None = None,
    additional_upper_bounds: Iterable[tuple[list[float], float]] = (),
) -> tuple[object, object, object, object]:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised on the server
        raise RuntimeError(
            "Exact endpoint validation requires the optional 'exact' dependencies."
        ) from exc

    quantity_count = model.quantity_variable_count
    variable_count = model.variable_count
    demand = dict(model.demand_by_shipment)
    profiles_by_shipment: dict[str, list[int]] = {}
    for index, profile in enumerate(model.profiles):
        profiles_by_shipment.setdefault(profile.shipment_id, []).append(index)

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_row(coefficients: dict[int, float], low: float, high: float) -> None:
        rows.append(coefficients)
        lower.append(low)
        upper.append(high)

    for shipment_id, indices in profiles_by_shipment.items():
        shipment_demand = demand[shipment_id]
        add_row({index: 1.0 for index in indices}, shipment_demand, shipment_demand)
        max_paths = (
            config_max_paths(model.config) if model.config.constraints.allow_split else 1
        )
        add_row(
            {quantity_count + index: 1.0 for index in indices},
            -math.inf,
            float(max_paths),
        )
        minimum_quantity = shipment_demand * model.config.constraints.min_path_share
        for index in indices:
            add_row(
                {index: 1.0, quantity_count + index: -shipment_demand},
                -math.inf,
                0.0,
            )
            add_row(
                {index: -1.0, quantity_count + index: minimum_quantity},
                -math.inf,
                0.0,
            )

    if model.config.constraints.enforce_capacity:
        coefficients_by_resource: dict[tuple[str, str, str], dict[int, float]] = {
            key: {} for key, _ in model.resource_capacities
        }
        for index, profile in enumerate(model.profiles):
            for key, coefficient in profile.resource_coefficients:
                coefficients_by_resource[key][index] = coefficient
        for key, capacity in model.resource_capacities:
            add_row(coefficients_by_resource[key], -math.inf, capacity)

    if primary_coefficients is not None and primary_upper_bound is not None:
        add_row(
            {
                index: coefficient
                for index, coefficient in enumerate(primary_coefficients)
                if coefficient
            },
            -math.inf,
            primary_upper_bound,
        )

    for coefficients, upper_bound in additional_upper_bounds:
        if len(coefficients) != variable_count:
            raise ValueError("A bounded MILP coefficient vector has the wrong length.")
        # Endpoint-equivalent share models add cost and time rows whose right-
        # hand sides can differ by five orders of magnitude.  Normalize each
        # added objective row by its largest coefficient before handing it to
        # HiGHS.  This is an algebraically identical constraint, but avoids a
        # false presolve-infeasible result when two tight lexicographic bounds
        # intersect at an endpoint.
        nonzero = [abs(coefficient) for coefficient in coefficients if coefficient]
        row_scale = max(nonzero, default=1.0)
        add_row(
            {
                index: coefficient / row_scale
                for index, coefficient in enumerate(coefficients)
                if coefficient
            },
            -math.inf,
            upper_bound / row_scale,
        )

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, coefficients in enumerate(rows):
        for column_index, value in coefficients.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            values.append(value)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), variable_count),
    ).tocsr()

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    for index, profile in enumerate(model.profiles):
        upper_bounds[index] = demand[profile.shipment_id]
        if model.config.constraints.hard_due_time:
            shipment = model.data.shipments[profile.shipment_id]
            if shipment.due_h is not None and profile.lead_time_h > (
                shipment.due_h - shipment.release_h
            ) + model.config.constraints.allocation_tolerance_feu:
                upper_bounds[index] = 0.0
                upper_bounds[quantity_count + index] = 0.0
    integrality = np.concatenate(
        (np.zeros(quantity_count, dtype=int), np.ones(quantity_count, dtype=int))
    )
    return (
        integrality,
        Bounds(lower_bounds, upper_bounds),
        LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        np,
    )


def config_max_paths(config: ModelConfig) -> int:
    return config.constraints.max_paths_per_shipment


def _decode_allocations(
    model: ExactEndpointModel,
    values: Iterable[float],
) -> tuple[RouteAllocation, ...]:
    quantities = list(values)[: model.quantity_variable_count]
    by_shipment: dict[str, list[tuple[int, float]]] = {}
    base_tolerance = model.config.constraints.allocation_tolerance_feu
    demand = dict(model.demand_by_shipment)
    for index, (profile, quantity) in enumerate(zip(model.profiles, quantities)):
        # HiGHS can return tiny positive quantity residuals on paths whose
        # binary variable is numerically zero.  Such values are several orders
        # of magnitude below the model's minimum active-path quantity and must
        # not be decoded as genuine route allocations.  The relative term is
        # deliberately tiny (one millionth of shipment demand) and therefore
        # cannot suppress a feasible allocation under the 10% minimum-share
        # rule.
        cleanup_tolerance = max(
            base_tolerance,
            1e-6 * demand[profile.shipment_id],
        )
        if quantity > cleanup_tolerance:
            by_shipment.setdefault(profile.shipment_id, []).append((index, float(quantity)))
    allocations: list[RouteAllocation] = []
    for shipment_id in sorted(demand):
        active = by_shipment.get(shipment_id, [])
        if not active:
            continue
        residual = demand[shipment_id] - sum(quantity for _, quantity in active)
        cleanup_tolerance = max(base_tolerance, 1e-6 * demand[shipment_id])
        if abs(residual) <= cleanup_tolerance:
            largest = max(range(len(active)), key=lambda item: active[item][1])
            index, quantity = active[largest]
            active[largest] = (index, quantity + residual)
        for index, quantity in active:
            allocations.append(
                RouteAllocation(
                    shipment_id=shipment_id,
                    quantity_feu=quantity,
                    arc_ids=model.profiles[index].arc_ids,
                    service_departure_ranks=(
                        model.profiles[index].service_departure_ranks
                    ),
                )
            )
    return tuple(allocations)


def _resource_consistency_error(
    model: ExactEndpointModel,
    allocations: tuple[RouteAllocation, ...],
    values: Iterable[float],
) -> float:
    quantities = list(values)[: model.quantity_variable_count]
    model_usage: dict[tuple[str, str, str], float] = {
        key: 0.0 for key, _ in model.resource_capacities
    }
    for profile, quantity in zip(model.profiles, quantities):
        if quantity <= 0:
            continue
        for key, coefficient in profile.resource_coefficients:
            model_usage[key] += coefficient * quantity
    evaluator_usage = {
        (str(row["resource_type"]), str(row["resource_id"]), str(row["period"])): float(
            row["used_feu"]
        )
        for row in capacity_usage_rows(model.data, list(allocations), model.config)
    }
    keys = set(model_usage) | set(evaluator_usage)
    return max(
        (abs(model_usage.get(key, 0.0) - evaluator_usage.get(key, 0.0)) for key in keys),
        default=0.0,
    )


def _evaluate_vector_solution(
    model: ExactEndpointModel,
    values: Iterable[float],
) -> tuple[
    tuple[RouteAllocation, ...],
    EvaluationResult,
    float,
    float,
    float,
    bool,
]:
    values = tuple(float(value) for value in values)
    allocations = _decode_allocations(model, values)
    total_demand = sum(dict(model.demand_by_shipment).values())
    numerical_allocation_tolerance = max(
        model.config.constraints.allocation_tolerance_feu,
        1e-6 * total_demand,
    )
    evaluation_config = replace(
        model.config,
        constraints=replace(
            model.config.constraints,
            allocation_tolerance_feu=numerical_allocation_tolerance,
        ),
    )
    evaluation = evaluate_solution(model.data, list(allocations), evaluation_config)
    cost_from_vector = sum(
        coefficient * value
        for coefficient, value in zip(model.objective_coefficients("cost"), values)
    )
    time_from_vector = sum(
        coefficient * value
        for coefficient, value in zip(model.objective_coefficients("time"), values)
    )
    cost_error = abs(cost_from_vector - evaluation.total_cost_usd)
    time_error = abs(
        time_from_vector - evaluation.quantity_weighted_mean_delivery_time_h
    )
    resource_error = _resource_consistency_error(model, allocations, values)
    objective_consistent = (
        cost_error <= max(1e-4, 1e-8 * abs(evaluation.total_cost_usd))
        and time_error
        <= max(
            1e-7,
            1e-6 * abs(evaluation.quantity_weighted_mean_delivery_time_h),
        )
        and resource_error <= numerical_allocation_tolerance
    )
    return (
        allocations,
        evaluation,
        cost_error,
        time_error,
        resource_error,
        evaluation.feasible and objective_consistent,
    )


def solve_bounded_milp(
    model: ExactEndpointModel,
    *,
    objective_coefficients: list[float],
    objective_name: str,
    upper_bounds: Iterable[tuple[list[float], float]] = (),
    maximize: bool = False,
    time_limit_seconds: float = 900.0,
    mip_relative_gap: float = 1e-8,
    presolve: bool = True,
) -> BoundedMilpResult:
    """Solve one linear MILP without conflating timeout and infeasibility."""
    try:
        from scipy.optimize import milp
    except ImportError as exc:  # pragma: no cover - exercised on the server
        raise RuntimeError(
            "Exact endpoint validation requires: pip install -e '.[exact]'."
        ) from exc

    if len(objective_coefficients) != model.variable_count:
        raise ValueError("The bounded MILP objective vector has the wrong length.")
    signed = [
        -coefficient if maximize else coefficient
        for coefficient in objective_coefficients
    ]
    integrality, bounds, constraints, np = _linear_system(
        model,
        additional_upper_bounds=tuple(upper_bounds),
    )
    started = time.perf_counter()
    result = milp(
        c=np.asarray(signed),
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": float(mip_relative_gap),
            "presolve": bool(presolve),
        },
    )
    call = _solver_call(
        result,
        1,
        ("max_" if maximize else "min_") + objective_name,
        time.perf_counter() - started,
    )
    status_code = int(getattr(result, "status", 4))
    values = getattr(result, "x", None)
    if values is None:
        return BoundedMilpResult(
            objective=objective_name,
            maximize=maximize,
            solver_call=call,
            allocations=(),
            evaluation=None,
            objective_value=None,
            proven_optimal=False,
            proven_infeasible=status_code == 2,
            evaluator_consistent=False,
            evaluator_cost_abs_error=None,
            evaluator_time_abs_error=None,
            resource_max_abs_error_feu=None,
        )
    (
        allocations,
        evaluation,
        cost_error,
        time_error,
        resource_error,
        evaluator_consistent,
    ) = _evaluate_vector_solution(model, values)
    objective_value = sum(
        coefficient * float(value)
        for coefficient, value in zip(objective_coefficients, values)
    )
    return BoundedMilpResult(
        objective=objective_name,
        maximize=maximize,
        solver_call=call,
        allocations=allocations,
        evaluation=evaluation,
        objective_value=objective_value,
        proven_optimal=status_code == 0 and evaluator_consistent,
        proven_infeasible=False,
        evaluator_consistent=evaluator_consistent,
        evaluator_cost_abs_error=cost_error,
        evaluator_time_abs_error=time_error,
        resource_max_abs_error_feu=resource_error,
    )


def solve_feasibility(
    model: ExactEndpointModel,
    *,
    time_limit_seconds: float = 900.0,
    mip_relative_gap: float = 1e-8,
) -> BoundedMilpResult:
    """Find a feasible allocation or prove that none exists in the MILP."""
    return solve_bounded_milp(
        model,
        objective_coefficients=[0.0] * model.variable_count,
        objective_name="feasibility",
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
    )


def solve_lexicographic_endpoint(
    model: ExactEndpointModel,
    *,
    primary_objective: str,
    time_limit_seconds: float = 900.0,
    mip_relative_gap: float = 1e-8,
    primary_relative_tolerance: float = 1e-9,
    primary_absolute_tolerance: float = 1e-6,
) -> ExactEndpointResult:
    """Solve one two-stage endpoint and verify it with ``evaluate_solution``.

    A result is labelled lexicographically proven only when both HiGHS calls
    terminate with status 0 under the recorded MIP gap and the decoded solution
    matches the independent evaluator and capacity accounting.
    """
    try:
        from scipy.optimize import milp
    except ImportError as exc:  # pragma: no cover - exercised on the server
        raise RuntimeError(
            "Exact endpoint validation requires: pip install -e '.[exact]'."
        ) from exc

    secondary_objective = "time" if primary_objective == "cost" else "cost"
    primary = model.objective_coefficients(primary_objective)
    integrality, bounds, constraints, np = _linear_system(model)
    options = {
        "time_limit": float(time_limit_seconds),
        "mip_rel_gap": float(mip_relative_gap),
        "presolve": True,
    }
    started = time.perf_counter()
    stage_one = milp(
        c=np.asarray(primary),
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    calls = [
        _solver_call(
            stage_one, 1, primary_objective, time.perf_counter() - started
        )
    ]
    stage_one_proven = int(getattr(stage_one, "status", 4)) == 0
    primary_optimum = (
        float(stage_one.fun) if getattr(stage_one, "fun", None) is not None else None
    )
    primary_tolerance = None
    final = stage_one
    if stage_one_proven and primary_optimum is not None:
        primary_tolerance = primary_absolute_tolerance + (
            primary_relative_tolerance * abs(primary_optimum)
        )
        integrality, bounds, constraints, np = _linear_system(
            model,
            primary_coefficients=primary,
            primary_upper_bound=primary_optimum + primary_tolerance,
        )
        secondary = model.objective_coefficients(secondary_objective)
        started = time.perf_counter()
        stage_two = milp(
            c=np.asarray(secondary),
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        calls.append(
            _solver_call(
                stage_two, 2, secondary_objective, time.perf_counter() - started
            )
        )
        if getattr(stage_two, "x", None) is not None:
            final = stage_two

    values = getattr(final, "x", None)
    if values is None:
        return ExactEndpointResult(
            primary_objective=primary_objective,
            secondary_objective=secondary_objective,
            stage_calls=tuple(calls),
            allocations=(),
            evaluation=None,
            primary_optimum=primary_optimum,
            primary_tolerance=primary_tolerance,
            stage_one_proven=stage_one_proven,
            lexicographic_proven=False,
            evaluator_consistent=False,
            evaluator_cost_abs_error=None,
            evaluator_time_abs_error=None,
            resource_max_abs_error_feu=None,
        )

    (
        allocations,
        evaluation,
        cost_error,
        time_error,
        resource_error,
        evaluator_consistent,
    ) = _evaluate_vector_solution(model, values)
    stage_two_proven = len(calls) == 2 and calls[1].status_code == 0
    return ExactEndpointResult(
        primary_objective=primary_objective,
        secondary_objective=secondary_objective,
        stage_calls=tuple(calls),
        allocations=allocations,
        evaluation=evaluation,
        primary_optimum=primary_optimum,
        primary_tolerance=primary_tolerance,
        stage_one_proven=stage_one_proven,
        lexicographic_proven=(
            stage_one_proven and stage_two_proven and evaluator_consistent
        ),
        evaluator_consistent=evaluator_consistent,
        evaluator_cost_abs_error=cost_error,
        evaluator_time_abs_error=time_error,
        resource_max_abs_error_feu=resource_error,
    )
