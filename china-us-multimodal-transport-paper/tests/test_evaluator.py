from china_us_multimodal.config import (
    ConstraintConfig,
    ModelConfig,
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
    services = {
        "sailing": Service("sailing", "ocean", (24,), 168, 20),
    }
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
        ("total_cost_usd", "makespan_h"),
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
    assert result.total_cost_usd == 1800
    assert result.makespan_h == 280
