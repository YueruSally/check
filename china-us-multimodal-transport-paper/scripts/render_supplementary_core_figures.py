#!/usr/bin/env python3
"""Render the two exact NY/NJ value curves from the v1.4 core suite."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CASE_LABELS = {"D1": "Chicago", "D2": "Memphis", "D3": "Columbus"}
COLORS = {"D1": "#1f77b4", "D2": "#d95f02", "D3": "#2ca02c"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=Path("results/v14_core_exact_suite")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper/figures")
    )
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:  # pragma: no cover - server-only rendering
        raise RuntimeError("Install the analysis extra before rendering figures.") from exc

    rows = read_csv(args.results / "delay_value_curves.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.15), constrained_layout=True)
    for case_id in ("D1", "D2", "D3"):
        selected = sorted(
            (row for row in rows if row["case_id"] == case_id),
            key=lambda row: float(row["delay_h"]),
        )
        delays = [float(row["delay_h"]) for row in selected]
        costs = [float(row["minimum_cost_benefit_usd"]) for row in selected]
        times = [float(row["minimum_time_benefit_h"]) for row in selected]
        axes[0].plot(
            delays,
            costs,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=COLORS[case_id],
            label=CASE_LABELS[case_id],
        )
        axes[1].plot(
            delays,
            times,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=COLORS[case_id],
            label=CASE_LABELS[case_id],
        )
    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].set_title("(a) Minimum-cost value of NY/NJ")
    axes[1].set_title("(b) Minimum-time value of NY/NJ")
    axes[0].set_ylabel("Avoided cost (USD)")
    axes[1].set_ylabel("Avoided weighted mean time (h)")
    for axis in axes:
        axis.set_xlabel("Common West Coast sailing delay (h)")
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.legend(frameon=False)
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1_000_000:g}M")
    )
    fig.text(
        0.5,
        -0.03,
        "Panels compare distinct lexicographic endpoints; benefits are not simultaneous.",
        ha="center",
        fontsize=9,
    )
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(
            args.output / f"fig07_exact_ny_nj_value_curves.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("Exact NY/NJ value-curve figure written.", flush=True)


if __name__ == "__main__":
    main()
