from __future__ import annotations

import argparse

from .config import load_config
from .evidence_workbook import read_baseline_inputs
from .io import load_model_data
from .validation import validate_model_data


def main() -> None:
    parser = argparse.ArgumentParser(description="China-US multimodal model utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evidence = subparsers.add_parser("validate-evidence")
    evidence.add_argument("workbook")
    network = subparsers.add_parser("validate-network")
    network.add_argument("data_directory")
    network.add_argument("--config", default="configs/baseline.toml")
    network.add_argument("--scenario")
    args = parser.parse_args()

    if args.command == "validate-evidence":
        values = read_baseline_inputs(args.workbook)
        print(f"Validated {len(values)} baseline evidence inputs.")
    else:
        config = load_config(args.config, args.scenario)
        data = load_model_data(args.data_directory)
        validate_model_data(data)
        print(
            f"Validated {len(data.nodes)} nodes, {len(data.arcs)} arcs, "
            f"{len(data.services)} services and {len(data.shipments)} shipments "
            f"for scenario {config.scenario.name}."
        )


if __name__ == "__main__":
    main()
