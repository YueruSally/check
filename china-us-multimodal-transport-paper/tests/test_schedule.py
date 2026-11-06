from china_us_multimodal.domain import Service
from china_us_multimodal.schedule import next_departure, ranked_departure


def test_recurring_service_respects_cutoff():
    service = Service(
        id="rail-1",
        arc_id="a1",
        departures_h=(24.0, 72.0),
        repeat_every_h=168.0,
        capacity_feu=100.0,
        cutoff_h=2.0,
    )
    assert next_departure(service, 25.0).departure_h == 72.0


def test_ranked_departure_returns_later_feasible_events_without_new_services():
    service = Service(
        id="rail-1",
        arc_id="a1",
        departures_h=(24.0, 72.0),
        repeat_every_h=168.0,
        capacity_feu=100.0,
        cutoff_h=2.0,
    )
    assert ranked_departure([service], 25.0, rank=0).departure_h == 72.0
    assert ranked_departure([service], 25.0, rank=1).departure_h == 192.0
    assert ranked_departure([service], 25.0, rank=2).departure_h == 240.0


def test_rank_zero_keeps_input_order_when_services_depart_together():
    first = Service("z-first", "a1", (24.0,), 168.0, 100.0)
    second = Service("a-second", "a1", (24.0,), 168.0, 100.0)
    assert ranked_departure([first, second], 0.0, rank=0).service_id == "z-first"
    assert ranked_departure([first, second], 0.0, rank=1).service_id == "a-second"
