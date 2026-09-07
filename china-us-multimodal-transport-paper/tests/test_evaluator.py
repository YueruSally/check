from china_us_multimodal.config import (
    ConstraintConfig,
    ModelConfig,
    OperationalCostConfig,
    PenaltyConfig,
    ScenarioConfig,
)
from china_us_multimodal.domain import (
    Arc,
    ModelData,
    Mode,
    Node,
    NodeKind,
    RouteAllocation,
    Service,
    Shipment,
    TransferRule,
)
from china_us_multimodal.evaluator import evaluate_solution
from china_us_multimodal.validation import validate_model_data


def test_cost_schedule_transfer_and_makespan_are_shared():
    nodes = {
        "xian": Node("xian", "Xi'an", NodeKind.CN_ORIGIN, "CN"),
        "shanghai": Node("shanghai", "Shanghai", NodeKind.CN_PORT, "CN"),
        "la": Node("la", "Los Angeles", NodeKind.US_PORT, "US"),
        "chicago": Node("chicago", "Chicago", NodeKind.DESTINATION, "US"),
    }
    arcs = {
        "cn-road": Arc("cn-road", "xian", "shanghai", Mode.ROAD, 10, 100),
        "ocean": Arc(
            "ocean", "shanghai", "la", Mode.OCEAN, 240, 1000,
            service_ids=("sailing",),
        ),
        "us-road": Arc("us-road", "la", "chicago", Mode.ROAD, 10, 200),
    }
    services = {"sailing": Service("sailing", "ocean", (24,), 168, 20)}
    shipment = Shipment("b1", "xian", "chicago", 1, due_h=300)
    transfers = {}
    for node, from_mode, to_mode, duration in [
        ("shanghai", Mode.ROAD, Mode.OCEAN, 4),
        ("la", Mode.OCEAN, Mode.ROAD, 6),
    ]:
        rule = TransferRule(node, from_mode, to_mode, duration, 250)
        transfers[(node, from_mode, to_mode)] = rule
    data = ModelData(nodes, arcs, services, {"b1": shipment}, transfers)
    validate_model_data(data)
    config = ModelConfig(
        "FEU",
        ("total_cost_usd", "quantity_weighted_mean_delivery_time_h"),
        ConstraintConfig(),
        PenaltyConfig(),
        ScenarioConfig("test"),
    )
    result = evaluate_solution(
        data,
        [RouteAllocation("b1", 1, ("cn-road", "ocean", "us-road"))],
        config,
    )
    assert result.feasible
    assert result.constraint_violation == 0
    assert result.total_cost_usd == 1800
    assert result.makespan_h == 280
    assert result.quantity_weighted_mean_delivery_time_h == 280


def test_shipment_specific_tardiness_penalty_overrides_fallback():
    nodes = {
        "a": Node("a", "A", NodeKind.CN_ORIGIN, "CN"),
        "b": Node("b", "B", NodeKind.DESTINATION, "US"),
    }
    arcs = {"road": Arc("road", "a", "b", Mode.ROAD, 10, 5)}
    shipment = Shipment("s", "a", "b", 2, due_h=5, tardiness_usd_per_feu_h=3)
    data = ModelData(nodes, arcs, {}, {"s": shipment})
    config = ModelConfig(
        "FEU",
        ("total_cost_usd", "quantity_weighted_mean_delivery_time_h"),
        ConstraintConfig(enforce_timetable=False, enforce_capacity=False),
        PenaltyConfig(tardiness_usd_per_feu_h=99),
        ScenarioConfig("test"),
    )
    result = evaluate_solution(data, [RouteAllocation("s", 2, ("road",))], config)
    assert result.total_cost_usd == 40


def test_path_below_ten_percent_is_infeasible():
    nodes = {
        "a": Node("a", "A", NodeKind.CN_ORIGIN, "CN"),
        "b": Node("b", "B", NodeKind.DESTINATION, "US"),
    }
    arcs = {
        "cheap": Arc("cheap", "a", "b", Mode.ROAD, 10, 1),
        "fast": Arc("fast", "a", "b", Mode.ROAD, 5, 2),
    }
    data = ModelData(nodes, arcs, {}, {"s": Shipment("s", "a", "b", 10)})
    config = ModelConfig(
        "FEU",
        ("total_cost_usd", "quantity_weighted_mean_delivery_time_h"),
        ConstraintConfig(enforce_timetable=False, enforce_capacity=False),
        PenaltyConfig(),
        ScenarioConfig("test"),
    )
    result = evaluate_solution(
        data,
        [
            RouteAllocation("s", 0.5, ("cheap",)),
            RouteAllocation("s", 9.5, ("fast",)),
        ],
        config,
    )
    assert not result.feasible
    assert any("10% minimum" in message for message in result.violations)


def test_west_coast_departure_delay_changes_wait_cost_and_arrival():
    nodes = {
        "xian": Node("xian", "Xi'an", NodeKind.CN_ORIGIN, "CN"),
        "shanghai": Node("shanghai", "Shanghai", NodeKind.CN_PORT, "CN"),
        "la_lb": Node("la_lb", "Los Angeles", NodeKind.US_PORT, "US"),
        "chicago": Node("chicago", "Chicago", NodeKind.DESTINATION, "US"),
    }
    arcs = {
        "cn-road": Arc("cn-road", "xian", "shanghai", Mode.ROAD, 10, 100),
        "ocean": Arc(
            "ocean", "shanghai", "la_lb", Mode.OCEAN, 240, 1000,
            service_ids=("sailing",),
        ),
        "us-road": Arc("us-road", "la_lb", "chicago", Mode.ROAD, 10, 200),
    }
    services = {"sailing": Service("sailing", "ocean", (24,), 168, 20)}
    transfers = {}
    for node, from_mode, to_mode, duration in [
        ("shanghai", Mode.ROAD, Mode.OCEAN, 4),
        ("la_lb", Mode.OCEAN, Mode.ROAD, 6),
    ]:
        rule = TransferRule(node, from_mode, to_mode, duration, 250)
        transfers[(node, from_mode, to_mode)] = rule
    data = ModelData(
        nodes,
        arcs,
        services,
        {"b1": Shipment("b1", "xian", "chicago", 1)},
        transfers,
    )
    config = ModelConfig(
        "FEU",
        ("total_cost_usd", "quantity_weighted_mean_delivery_time_h"),
        ConstraintConfig(),
        PenaltyConfig(),
        ScenarioConfig("delay", uswc_ocean_departure_delay_h=24),
        OperationalCostConfig(holding_usd_per_feu_h=2),
    )
    result = evaluate_solution(
        data,
        [RouteAllocation("b1", 1, ("cn-road", "ocean", "us-road"))],
        config,
    )
    assert result.feasible
    assert result.makespan_h == 304
    assert result.quantity_weighted_mean_delivery_time_h == 304
    assert result.total_cost_usd == 1868


def test_west_coast_port_capacity_multiplier_is_enforced():
    nodes = {
        "a": Node("a", "A", NodeKind.CN_ORIGIN, "CN"),
        "la_lb": Node(
            "la_lb",
            "Los Angeles",
            NodeKind.US_PORT,
            "US",
            capacity_feu_per_day=100,
        ),
        "b": Node("b", "B", NodeKind.DESTINATION, "US"),
    }
    arcs = {
        "in": Arc("in", "a", "la_lb", Mode.ROAD, 1, 1),
        "out": Arc("out", "la_lb", "b", Mode.ROAD, 1, 1),
    }
    data = ModelData(nodes, arcs, {}, {"s": Shipment("s", "a", "b", 60)})
    config = ModelConfig(
        "FEU",
        ("total_cost_usd", "quantity_weighted_mean_delivery_time_h"),
        ConstraintConfig(enforce_timetable=False, enforce_capacity=True),
        PenaltyConfig(),
        ScenarioConfig("capacity", uswc_port_capacity_multiplier=0.5),
    )
    result = evaluate_solution(
        data,
        [RouteAllocation("s", 60, ("in", "out"))],
        config,
    )
    assert not result.feasible
    assert any("Node la_lb day 0: 60>50 FEU" in item for item in result.violations)
