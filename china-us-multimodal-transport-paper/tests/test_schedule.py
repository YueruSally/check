from china_us_multimodal.domain import Service
from china_us_multimodal.schedule import next_departure


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
