from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.domain import Mode
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.path_library import build_gateway_aware_library
from china_us_multimodal.validation import validate_model_data


ROOT = Path(__file__).resolve().parents[1]


def test_generated_pilot_network_is_valid_and_case_selectable():
    data = load_model_data(ROOT / "data" / "pilot_v2")
    validate_model_data(data)
    assert (len(data.nodes), len(data.arcs), len(data.services), len(data.shipments)) == (
        9,
        28,
        17,
        120,
    )
    d1 = select_case(data, "D1")
    assert len(d1.shipments) == 40
    assert sum(item.quantity_feu for item in d1.shipments.values()) == 2227.5


def test_gateway_aware_martins_library_keeps_all_three_gateways():
    data = select_case(load_model_data(ROOT / "data" / "pilot_v2"), "D1")
    config = load_config(ROOT / "configs" / "nsga2_pilot.toml")
    library = build_gateway_aware_library(
        data,
        max_legs=config.constraints.max_legs_per_path,
        max_mode_changes=config.constraints.max_mode_changes,
        max_paths_per_gateway=4,
    )
    first = library[sorted(library)[0]]
    gateways = {
        data.arcs[path[1]].destination
        for path in first
        if len(path) >= 2 and data.arcs[path[1]].mode == Mode.OCEAN
    }
    assert gateways == {"la_lb", "seattle_tacoma", "ny_nj"}
