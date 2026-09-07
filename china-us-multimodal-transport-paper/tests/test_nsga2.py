from china_us_multimodal.config import (
    ConstraintConfig,
    ModelConfig,
    PenaltyConfig,
    ScenarioConfig,
)
from china_us_multimodal.domain import Arc, ModelData, Mode, Node, NodeKind, Shipment
from china_us_multimodal.nsga2 import NSGA2Config, run_nsga2


def test_nsga2_returns_feasible_front_for_two_route_problem():
    data = ModelData(
        nodes={
            "a": Node("a", "A", NodeKind.CN_ORIGIN, "CN"),
            "b": Node("b", "B", NodeKind.DESTINATION, "US"),
        },
        arcs={
            "cheap": Arc("cheap", "a", "b", Mode.ROAD, 10, 1),
            "fast": Arc("fast", "a", "b", Mode.ROAD, 5, 2),
        },
        services={},
        shipments={"s": Shipment("s", "a", "b", 10)},
    )
    model = ModelConfig(
        "FEU",
        ("total_cost_usd", "quantity_weighted_mean_delivery_time_h"),
        ConstraintConfig(enforce_timetable=False, enforce_capacity=False),
        PenaltyConfig(),
        ScenarioConfig("test"),
    )
    algorithm = NSGA2Config(population_size=8, generations=3, seed=7)
    result = run_nsga2(
        data,
        model,
        algorithm,
        candidate_library={"s": (("cheap",), ("fast",))},
    )
    assert result.pareto_front
    assert all(item.evaluation and item.evaluation.feasible for item in result.pareto_front)
    objectives = {item.objectives for item in result.pareto_front}
    assert (10, 10) in objectives
    assert (20, 5) in objectives
    for individual in result.pareto_front:
        assert all(allocation.quantity_feu >= 1.0 for allocation in individual.allocations)
