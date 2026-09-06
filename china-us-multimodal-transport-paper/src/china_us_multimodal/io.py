from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from .domain import Arc, ModelData, Mode, Node, NodeKind, Service, Shipment, TransferRule


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input table: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: str | None) -> float | None:
    return None if value is None or not value.strip() else float(value)


def _optional_text(value: str | None) -> str | None:
    return None if value is None or not value.strip() else value.strip()


def _boolean(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_model_data(directory: str | Path) -> ModelData:
    directory = Path(directory)
    nodes = {
        row["id"]: Node(
            id=row["id"],
            name=row["name"],
            kind=NodeKind(row["kind"]),
            country=row["country"],
            capacity_feu_per_day=_optional_float(row.get("capacity_feu_per_day")),
        )
        for row in _rows(directory / "nodes.csv")
    }

    arc_rows = _rows(directory / "arcs.csv")
    arcs = {
        row["id"]: Arc(
            id=row["id"],
            origin=row["origin"],
            destination=row["destination"],
            mode=Mode(row["mode"]),
            transit_time_h=float(row["transit_time_h"]),
            cost_usd_per_feu=float(row["cost_usd_per_feu"]),
            distance_km=_optional_float(row.get("distance_km")),
            capacity_feu_per_day=_optional_float(row.get("capacity_feu_per_day")),
            service_ids=tuple(filter(None, (row.get("service_ids") or "").split("|"))),
            planning_wait_h=float(row.get("planning_wait_h") or 0),
            via_panama=_boolean(row.get("via_panama")),
        )
        for row in arc_rows
    }

    services = {
        row["id"]: Service(
            id=row["id"],
            arc_id=row["arc_id"],
            departures_h=tuple(float(item) for item in row["departures_h"].split("|")),
            repeat_every_h=float(row["repeat_every_h"]),
            capacity_feu=float(row["capacity_feu"]),
            cutoff_h=float(row.get("cutoff_h") or 0),
        )
        for row in _rows(directory / "services.csv")
    }

    shipments = {
        row["id"]: Shipment(
            id=row["id"],
            origin=row["origin"],
            destination=row["destination"],
            quantity_feu=float(row["quantity_feu"]),
            release_h=float(row.get("release_h") or 0),
            due_h=_optional_float(row.get("due_h")),
            tardiness_usd_per_feu_h=_optional_float(
                row.get("tardiness_usd_per_feu_h")
            ),
            case_id=_optional_text(row.get("case_id")),
        )
        for row in _rows(directory / "shipments.csv")
    }

    transfer_rules = {}
    for row in _rows(directory / "transfers.csv"):
        rule = TransferRule(
            node_id=row["node_id"],
            from_mode=Mode(row["from_mode"]),
            to_mode=Mode(row["to_mode"]),
            duration_h=float(row["duration_h"]),
            cost_usd_per_feu=float(row["cost_usd_per_feu"]),
        )
        transfer_rules[(rule.node_id, rule.from_mode, rule.to_mode)] = rule

    return ModelData(nodes, arcs, services, shipments, transfer_rules)


def select_case(data: ModelData, case_id: str) -> ModelData:
    selected = {
        shipment_id: shipment
        for shipment_id, shipment in data.shipments.items()
        if shipment.case_id == case_id
    }
    if not selected:
        available = sorted({item.case_id for item in data.shipments.values() if item.case_id})
        raise ValueError(f"Unknown shipment case {case_id!r}; available cases: {available}.")
    return replace(data, shipments=selected)
