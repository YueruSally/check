from china_us_multimodal.experiment import BatchRun, summarize_runs
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
