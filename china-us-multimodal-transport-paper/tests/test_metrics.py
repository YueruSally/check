from china_us_multimodal.experiment import (
    BatchRun,
    BatchSolution,
    representative_rows,
    summarize_runs,
)
from china_us_multimodal.metrics import hypervolume_2d, nondominated_points, spacing


def test_nondominated_points_and_hypervolume() -> None:
    points = [(1.0, 5.0), (2.0, 4.0), (3.0, 6.0), (2.0, 4.0)]
    assert nondominated_points(points) == ((1.0, 5.0), (2.0, 4.0))
    assert abs(hypervolume_2d([(0.2, 0.8), (0.8, 0.2)]) - 0.45) < 1e-12
    assert spacing([(0.0, 1.0)]) == 0.0


def test_cross_seed_summary_uses_common_bounds() -> None:
    runs = [
        BatchRun("D1", "base", 1, 1.0, 8, 2, 8, 2, 10.0, 5.0, ((10.0, 7.0), (12.0, 5.0))),
        BatchRun("D1", "base", 2, 2.0, 8, 2, 8, 1, 11.0, 6.0, ((11.0, 6.0),)),
    ]
    run_rows, aggregate, union = summarize_runs(runs)
    assert len(run_rows) == 2
    assert aggregate[0]["success_rate"] == 1.0
    assert aggregate[0]["union_front_size"] == 3
    assert [row["source_seeds"] for row in union] == ["1", "2", "1"]


def test_representative_rows_preserve_allocations() -> None:
    solution = BatchSolution(
        total_cost_usd=10.0,
        quantity_weighted_mean_delivery_time_h=5.0,
        makespan_h=5.0,
        max_lead_time_h=4.0,
        total_tardiness_feu_h=0.0,
        allocations=(("S1", 2.5, ("A1", "A2")),),
    )
    run = BatchRun(
        "D1", "base", 7, 1.0, 8, 2, 8, 1, 10.0, 5.0,
        ((10.0, 5.0),), (solution,),
    )
    solutions, allocations = representative_rows([run])
    assert solutions[0]["roles"] == "min_cost|min_delivery_time|balanced"
    assert allocations[0]["arc_ids"] == "A1|A2"
