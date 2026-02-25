#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NSGA-II (Batch-level encoding) — Scheme A + Program B timetable semantics
✅ UPDATED: Hypervolume (HV) is computed AFTER normalisation (min-max) using P* bounds.

Key changes (vs your original script):
1) run_nsga2_analytics(): no longer computes HV with run-specific ref_point.
   It only records front_hist (raw objectives tuples) per generation.
2) main: builds P* across all runs, derives mins/maxs, then computes
   HV_norm curves for each run using a FIXED ref point in normalised space.
   Recommended: ref=(1.1,1.1,1.1), Monte Carlo samples fixed with seed.
3) best_run selection is based on final HV_norm, ensuring consistency.
4) hv_curve / convergence plots use HV_norm (mean±std over runs).

✅ Additional robustness fixes (this version):
- Timetable row cleaning: skip invalid/truncated rows (e.g., Origin/Dest empty or startswith '...')
- extract_min_objectives(): uses finite filtering + np.nanmin + forward fill to avoid plot gaps
- 2D evolution overlays: filters non-finite points before plotting
"""

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ========================
# Global settings
# ========================

TIME_BUCKET_H = 1.0  # simulation granularity: 1 hour
CHINA_REGIONS = {"CN", "China"}
EUROPE_REGIONS = {"WE", "EE", "EU", "Europe"}

# ---- Time window handling:
# HARD_TIME_WINDOW=False means lateness is allowed but penalised in COST (Scheme A).
# HARD_TIME_WINDOW=True means lateness is forbidden (treated as infeasible).
HARD_TIME_WINDOW = False

# ---- HARD constraint penalty weights
PEN_MISS_TT = 5e7
PEN_MISS_ALLOC = 1e9
PEN_CAP_EXCESS_PER_TEU = 5e7
# ✅ Node capacity excess penalty
PEN_NODE_CAP_EXCESS_PER_TEU = 5e7

# ---- Scheme A: lateness and waiting are in objectives (cost/emission)
WAITING_COST_PER_TEU_HOUR_DEFAULT = 0.5
WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT = 0.0

# Lateness: cost per TEU per hour late (Scheme A)
LATE_COST_PER_TEU_HOUR = 1e6

# ---- Static roulette priors (same as Program A)
W_ADD = 0.25
W_DEL = 0.20
W_MOD = 0.35
W_MODE = 0.20
OPS = ["add", "del", "mod", "mode"]

# ---- Program A hyper-params (IMPORTANT: keep consistent)
ROULETTE_EMA_ALPHA = 0.35     # ✅ Program A: EMA(α≈0.35)
ROULETTE_MIN_PROB = 0.05
ROULETTE_SCORE_EPS = 1e-3

CROSSOVER_RATE = 0.92
MUTATION_RATE = 0.25          # per-child mutation chance

# ---- Path library diversity
PATHS_TOPK_PER_CRITERION = 30
PATH_LIB_CAP_TOTAL = 90
DFS_MAX_PATHS_PER_OD = 1200

# ---- Use both crossovers (structural + segment/common-node)
CROSSOVER_SEGMENT_PROB = 0.50  # within crossover trigger, 50% segment, 50% structural


# ========================
# Analytics speed knobs
# ========================
HV_EVERY = 5               # compute HV every HV_EVERY generations
HV_SAMPLES = 2000          # HV Monte Carlo samples
METRIC_EVERY = 5           # IGD+/Spacing every METRIC_EVERY generations

PSTAR_TAIL_GENS = 30       # P* uses the last tail gens
PSTAR_CAP_PER_GEN = 40     # cap points per generation
PSTAR_MAX_TOTAL = 50000    # global cap for P* builder

# ✅ NEW: HV normalised ref point (in [0,1] space; use >1 to avoid boundary effects)
HV_REF_NORM = (1.2, 1.2, 1.2)
HV_MC_SEED = 12345         # fixed MC seed for reproducible HV estimation

SCENARIO_GRID = [
    ("S0",  10, 0.5),
    ("S1",  10, 1.0),
    ("S2",  10, 2.0),
    ("S3",  20, 0.5),
    ("S4",  20, 1.0),
    ("S5",  20, 2.0),
    ("S6",  30, 0.5),
    ("S7",  30, 1.0),
    ("S8",  30, 2.0),
    ("S9",  50, 0.5),
    ("S10", 50, 1.0),
    ("S11", 50, 2.0),
]


# ========================
# Helpers
# ========================

def normalize_mode(mode_raw: str) -> str:
    m = str(mode_raw).strip().lower()
    if m in {"railway", "rail"}:
        return "rail"
    if m in {"road", "truck"}:
        return "road"
    if m in {"water", "ship", "sea"}:
        return "water"
    return m


def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def parse_distance_km(x) -> float:
    s = str(x)
    cleaned = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    return float(cleaned) if cleaned else 0.0


def unique_objective_tuples(
    objs: List[Tuple[float, float, float]],
    tol: float = 1e-9
) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for o in objs:
        dup = False
        for p in out:
            if all(abs(o[i] - p[i]) <= tol for i in range(3)):
                dup = True
                break
        if not dup:
            out.append(o)
    return out


def _is_bad_text_token(s: str) -> bool:
    """Detect clearly truncated/invalid tokens such as '...an'."""
    if s is None:
        return True
    t = str(s).strip()
    if t == "":
        return True
    if t.startswith("..."):
        return True
    return False


def _ffill_nan(arr: np.ndarray) -> np.ndarray:
    """
    Forward-fill NaNs for plotting continuity.
    If leading values are NaN, backfill using the first finite value if any; otherwise keep NaN.
    """
    x = np.array(arr, dtype=float).copy()
    if x.size == 0:
        return x

    finite_idx = np.where(np.isfinite(x))[0]
    if finite_idx.size == 0:
        return x  # all NaN/inf -> keep; caller may still plot nothing useful

    # convert inf to nan first (for consistency)
    x[~np.isfinite(x)] = np.nan

    # backfill leading NaNs with first finite value
    first = finite_idx[0]
    if first > 0 and np.isfinite(x[first]):
        x[:first] = x[first]

    # forward fill
    for i in range(1, len(x)):
        if np.isnan(x[i]) and np.isfinite(x[i - 1]):
            x[i] = x[i - 1]

    return x


def _finite_points_array(pts: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Convert points to np.array and keep only rows with all finite values.
    """
    if not pts:
        return np.empty((0, 3), dtype=float)
    arr = np.array(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    mask = np.all(np.isfinite(arr), axis=1)
    return arr[mask]


# ========================
# Data structures
# ========================

@dataclass
class Arc:
    from_node: str
    to_node: str
    mode: str
    distance: float
    capacity: float       # TEU / bucket (e.g., TEU/hour)
    cost_per_teu_km: float
    emission_per_teu_km: float  # gCO2 per TEU-km
    speed_kmh: float
    from_region: str = ""
    to_region: str = ""


@dataclass
class TimetableEntry:
    from_node: str
    to_node: str
    mode: str
    frequency_per_week: float
    first_departure_hour: float
    headway_hours: float


@dataclass
class Batch:
    batch_id: int
    origin: str
    destination: str
    quantity: float  # TEU
    ET: float
    LT: float


@dataclass
class Path:
    path_id: int
    origin: str
    destination: str
    nodes: List[str]
    modes: List[str]
    arcs: List[Arc]
    base_cost_per_teu: float
    base_emission_per_teu: float  # gCO2 per TEU
    base_travel_time_h: float

    def __eq__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return self.nodes == other.nodes and self.modes == other.modes

    def __hash__(self):
        return hash((tuple(self.nodes), tuple(self.modes)))


@dataclass
class PathAllocation:
    path: Path
    share: float

    def __repr__(self):
        chain = ""
        for i, node in enumerate(self.path.nodes[:-1]):
            mode = self.path.modes[i]
            chain += f"{node}--({mode})-->"
        chain += self.path.nodes[-1]
        return f"\n    {{ Structure: [{chain}], Share: {self.share:.2%} }}"


@dataclass(eq=False)
class Individual:
    # key=(origin, dest, batch_id)
    od_allocations: Dict[Tuple[str, str, int], List[PathAllocation]] = field(default_factory=dict)

    # objectives: (cost, emission, makespan)
    objectives: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

    # HARD-constraint handling
    penalty: float = 0.0
    feasible: bool = False          # hard-feasible (and lateness rule if HARD_TIME_WINDOW=True)
    feasible_hard: bool = False     # strict: also no lateness (always tracked)

    # diagnostics
    vio_breakdown: Dict[str, float] = field(default_factory=dict)


# ========================
# Merge & normalise shares
# ========================

def merge_and_normalize(allocs: List[PathAllocation]) -> List[PathAllocation]:
    if not allocs:
        return []

    merged: Dict[Path, float] = {}
    for a in allocs:
        merged[a.path] = merged.get(a.path, 0.0) + float(a.share)

    unique_allocs = [PathAllocation(path=p, share=s) for p, s in merged.items()]
    total = sum(a.share for a in unique_allocs)

    if total <= 1e-12:
        avg = 1.0 / max(1, len(unique_allocs))
        for a in unique_allocs:
            a.share = avg
    else:
        for a in unique_allocs:
            a.share /= total

    unique_allocs = [a for a in unique_allocs if a.share > 0.001]

    if unique_allocs:
        total2 = sum(a.share for a in unique_allocs)
        if abs(total2 - 1.0) > 1e-9:
            for a in unique_allocs:
                a.share /= total2

    return unique_allocs


# ========================
# Load data
# ========================
def load_carbon_tax_map(xls: pd.ExcelFile) -> Dict[str, float]:
    """
    Load regional carbon tax map.
    Expected sheet: Carbon_Tax
    Accepts columns like:
      - Region / region
      - CarbonTax / Carbon_Tax / CT / CT_$_per_tCO2 / CarbonTax_$_per_tCO2
    Returns:
      dict[region] = tax in $/tCO2
    """
    out: Dict[str, float] = {}
    if "Carbon_Tax" not in xls.sheet_names:
        print("[INFO] Carbon_Tax sheet not found. Carbon tax defaults to 0.")
        return out

    try:
        df = pd.read_excel(xls, "Carbon_Tax")
        if df.empty:
            print("[INFO] Carbon_Tax sheet is empty. Carbon tax defaults to 0.")
            return out

        region_col = None
        for c in ["Region", "region", "RegionCode", "region_code"]:
            if c in df.columns:
                region_col = c
                break

        tax_col = None
        for c in ["CarbonTax", "Carbon_Tax", "CT", "CT_$_per_tCO2", "CarbonTax_$_per_tCO2"]:
            if c in df.columns:
                tax_col = c
                break

        if region_col is None or tax_col is None:
            print("[WARN] Carbon_Tax sheet columns not recognised. Carbon tax defaults to 0.")
            return out

        for _, row in df.iterrows():
            r = str(row.get(region_col, "")).strip()
            if not r:
                continue
            out[r] = safe_float(row.get(tax_col), default=0.0)

        print(f"[INFO] Loaded carbon tax map for {len(out)} regions.")
        return out
    except Exception as e:
        print(f"[WARN] Failed to read Carbon_Tax sheet ({e}). Carbon tax defaults to 0.")
        return out


def load_emission_factor_map(xls: pd.ExcelFile) -> Dict[Tuple[str, str], float]:
    """
    Load emission factor overrides by (mode, region), unit gCO2 per TEU-km (or compatible).
    Expected sheet: Emission_Factors
    Accepts columns like:
      - Mode / mode
      - Region / region
      - EmissionFactor / Emission_gCO2_per_TEU_km / EF / value
    Returns:
      dict[(mode, region)] = emission factor
    """
    out: Dict[Tuple[str, str], float] = {}
    if "Emission_Factors" not in xls.sheet_names:
        print("[INFO] Emission_Factors sheet not found. Using Arcs_All emission factors.")
        return out

    try:
        df = pd.read_excel(xls, "Emission_Factors")
        if df.empty:
            print("[INFO] Emission_Factors sheet is empty. Using Arcs_All emission factors.")
            return out

        mode_col = next((c for c in ["Mode", "mode"] if c in df.columns), None)
        region_col = next((c for c in ["Region", "region", "RegionCode"] if c in df.columns), None)
        ef_col = next((c for c in ["EmissionFactor", "Emission_gCO2_per_TEU_km", "EF", "value"] if c in df.columns), None)

        if mode_col is None or region_col is None or ef_col is None:
            print("[WARN] Emission_Factors sheet columns not recognised. Using Arcs_All emission factors.")
            return out

        for _, row in df.iterrows():
            m = normalize_mode(row.get(mode_col, ""))
            r = str(row.get(region_col, "")).strip()
            if not m or not r:
                continue
            out[(m, r)] = safe_float(row.get(ef_col), default=0.0)

        print(f"[INFO] Loaded emission factors for {len(out)} (mode, region) pairs.")
        return out
    except Exception as e:
        print(f"[WARN] Failed to read Emission_Factors sheet ({e}). Using Arcs_All emission factors.")
        return out


def load_mode_speeds(xls: pd.ExcelFile) -> Dict[str, float]:
    """
    Load optional mode speed overrides from sheet Mode_Speeds.
    Expected columns:
      - Mode / mode
      - Speed_kmh / speed_kmh / Speed
    """
    out: Dict[str, float] = {}
    if "Mode_Speeds" not in xls.sheet_names:
        print("[INFO] Mode_Speeds sheet not found. Using defaults road=75, rail=50, water=30.")
        return out

    try:
        df = pd.read_excel(xls, "Mode_Speeds")
        if df.empty:
            print("[INFO] Mode_Speeds sheet is empty. Using defaults.")
            return out

        mode_col = next((c for c in ["Mode", "mode"] if c in df.columns), None)
        sp_col = next((c for c in ["Speed_kmh", "speed_kmh", "Speed"] if c in df.columns), None)
        if mode_col is None or sp_col is None:
            print("[WARN] Mode_Speeds columns not recognised. Using defaults.")
            return out

        for _, row in df.iterrows():
            m = normalize_mode(row.get(mode_col, ""))
            if not m:
                continue
            out[m] = safe_float(row.get(sp_col), default=0.0)

        print(f"[INFO] Loaded mode speed overrides: {out}")
        return out
    except Exception as e:
        print(f"[WARN] Failed to read Mode_Speeds ({e}). Using defaults.")
        return out


def load_transshipment_map(xls: pd.ExcelFile) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """
    Load transshipment cost/time map keyed by (node, in_mode, out_mode).
    Expected sheet: Transshipment
    Accepts columns like:
      - Node / NodeEN / EnglishName
      - InMode / FromMode / mode_in
      - OutMode / ToMode / mode_out
      - TransCost / Cost / trans_cost
      - TransTime_h / Time_h / trans_time_h
    Returns:
      dict[(node, in_mode, out_mode)] = {"cost_per_teu": ..., "time_h": ...}
    """
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    if "Transshipment" not in xls.sheet_names:
        print("[INFO] Transshipment sheet not found. Transshipment cost/time defaults to 0.")
        return out

    try:
        df = pd.read_excel(xls, "Transshipment")
        if df.empty:
            print("[INFO] Transshipment sheet is empty. Transshipment cost/time defaults to 0.")
            return out

        node_col = next((c for c in ["Node", "NodeEN", "EnglishName"] if c in df.columns), None)
        in_col = next((c for c in ["InMode", "FromMode", "mode_in"] if c in df.columns), None)
        out_col = next((c for c in ["OutMode", "ToMode", "mode_out"] if c in df.columns), None)
        cost_col = next((c for c in ["TransCost", "Cost", "trans_cost", "Cost_per_TEU"] if c in df.columns), None)
        time_col = next((c for c in ["TransTime_h", "Time_h", "trans_time_h", "Time"] if c in df.columns), None)

        if node_col is None or in_col is None or out_col is None:
            print("[WARN] Transshipment columns not recognised. Defaults to 0.")
            return out

        for _, row in df.iterrows():
            node = str(row.get(node_col, "")).strip()
            in_mode = normalize_mode(row.get(in_col, ""))
            out_mode = normalize_mode(row.get(out_col, ""))
            if not node or not in_mode or not out_mode:
                continue
            out[(node, in_mode, out_mode)] = {
                "cost_per_teu": safe_float(row.get(cost_col), default=0.0) if cost_col else 0.0,
                "time_h": safe_float(row.get(time_col), default=0.0) if time_col else 0.0,
            }

        print(f"[INFO] Loaded transshipment entries: {len(out)}")
        return out
    except Exception as e:
        print(f"[WARN] Failed to read Transshipment ({e}). Defaults to 0.")
        return out


def load_waiting_params(xls: pd.ExcelFile) -> Tuple[float, float]:
    wc = WAITING_COST_PER_TEU_HOUR_DEFAULT
    we = WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT

    if "Waiting_Costs" not in xls.sheet_names:
        print(f"[INFO] Waiting_Costs sheet not found. Using defaults: cost={wc}, emission={we}")
        return wc, we

    try:
        df = pd.read_excel(xls, "Waiting_Costs")

        def pick(colnames, default):
            for c in colnames:
                if c in df.columns:
                    vals = df[c].dropna().tolist()
                    if vals:
                        return safe_float(vals[0], default=default)
            return default

        wc = pick(["WaitingCost_per_TEU_h", "WaitCost_per_TEU_h"], wc)
        we = pick(["WaitEmission_gCO2_per_TEU_h", "WaitingEmission_gCO2_per_TEU_h"], we)
        print(f"[INFO] Loaded waiting params: cost={wc}, emission={we} (gCO2/TEU/h)")
        return wc, we
    except Exception as e:
        print(f"[WARN] Failed to read Waiting_Costs ({e}). Using defaults: cost={wc}, emission={we}")
        return wc, we


def load_network_from_extended(filename: str):

    xls = pd.ExcelFile(filename)

    # ✅ New maps first
    carbon_tax_map = load_carbon_tax_map(xls)
    emission_factor_map = load_emission_factor_map(xls)
    mode_speeds_map = load_mode_speeds(xls)
    trans_map = load_transshipment_map(xls)

    nodes_df = pd.read_excel(xls, "Nodes")
    node_names = nodes_df["EnglishName"].astype(str).str.strip().tolist()
    node_region = dict(
        zip(nodes_df["EnglishName"].astype(str).str.strip(),
            nodes_df["Region"].astype(str).str.strip())
    )

    # node capacities (TEU/day -> TEU/bucket)
    DAILY_HOURS = 24.0
    node_caps: Dict[str, float] = {}
    for _, row in nodes_df.iterrows():
        n = str(row.get("EnglishName", "")).strip()
        raw = None
        for col in ["NodeCap_TEU_per_day", "Capacity_TEU", "Throughput_TEU"]:
            if col in nodes_df.columns and (not pd.isna(row.get(col, np.nan))):
                raw = safe_float(row.get(col), default=None)
                break
        if raw is None:
            raw = 1e12  # unlimited if not provided
        node_caps[n] = float(raw) * (TIME_BUCKET_H / DAILY_HOURS)

    waiting_cost_per_teu_h, wait_emis_g_per_teu_h = load_waiting_params(xls)

    arcs_df = pd.read_excel(xls, "Arcs_All")
    arcs: List[Arc] = []

    # column aliases for cost/emission
    cost_cols = ["Cost_$_per_km", "Cost_per_km", "Cost"]
    emis_cols = ["Emission_gCO2_per_tkm", "Emission_gCO2_per_TEU_km", "EmissionFactor", "Emission"]

    for _, row in arcs_df.iterrows():
        mode = normalize_mode(row.get("Mode", "road"))

        # default speeds + optional override
        if mode == "road":
            speed = 75.0
        elif mode == "water":
            speed = 30.0
        else:
            speed = 50.0
        if mode in mode_speeds_map and mode_speeds_map[mode] > 0:
            speed = mode_speeds_map[mode]

        origin = str(row.get("OriginEN", "")).strip()
        dest = str(row.get("DestEN", "")).strip()
        if _is_bad_text_token(origin) or _is_bad_text_token(dest):
            continue

        from_region = str(node_region.get(origin, "")).strip()
        to_region = str(node_region.get(dest, "")).strip()

        distance = parse_distance_km(row.get("Distance_km", 0.0))

        if "Capacity_TEU" in arcs_df.columns and not pd.isna(row.get("Capacity_TEU", np.nan)):
            raw_cap = safe_float(row.get("Capacity_TEU"), default=1e9)
        else:
            raw_cap = 1e9
        capacity = raw_cap * (TIME_BUCKET_H / DAILY_HOURS)

        # base cost
        cpkm = 0.0
        for c in cost_cols:
            if c in arcs_df.columns:
                cpkm = safe_float(row.get(c), default=0.0)
                break

        # base emission from Arcs_All
        epkm = 0.0
        for c in emis_cols:
            if c in arcs_df.columns:
                epkm = safe_float(row.get(c), default=0.0)
                break

        # ✅ optional override by (mode, region) using from_region
        if (mode, from_region) in emission_factor_map:
            epkm = emission_factor_map[(mode, from_region)]

        arcs.append(Arc(
            from_node=origin,
            to_node=dest,
            mode=mode,
            distance=distance,
            capacity=capacity,
            cost_per_teu_km=cpkm,
            emission_per_teu_km=epkm,
            speed_kmh=speed,
            from_region=from_region,
            to_region=to_region
        ))

    # ✅ Timetable cleaning / guard against truncated rows
    tdf = pd.read_excel(xls, "Timetable")
    timetables: List[TimetableEntry] = []
    skipped_tt_rows = 0

    for ridx, row in tdf.iterrows():
        origin_raw = row.get("OriginEN", "")
        dest_raw = row.get("DestEN", "")
        mode_raw = row.get("Mode", "")

        origin = str(origin_raw).strip()
        dest = str(dest_raw).strip()
        mode_norm = normalize_mode(mode_raw)

        if _is_bad_text_token(origin) or _is_bad_text_token(dest):
            skipped_tt_rows += 1
            print(f"[WARN] Timetable row skipped (invalid Origin/Dest) at row={ridx}: Origin={origin_raw!r}, Dest={dest_raw!r}")
            continue

        if mode_norm not in {"road", "rail", "water"}:
            skipped_tt_rows += 1
            print(f"[WARN] Timetable row skipped (invalid Mode) at row={ridx}: Mode={mode_raw!r}")
            continue

        freq = safe_float(row.get("Frequency_per_week"), default=1.0)
        hd = row.get("Headway_Hours", np.nan)
        hd = 168.0 / max(freq, 1.0) if pd.isna(hd) else safe_float(hd, default=168.0)

        v = row.get("FirstDepartureHour", np.nan)
        fd = 0.0
        if not pd.isna(v):
            try:
                s = str(v).strip()
                fd = float(s.split(":")[0]) if ":" in s else float(s)
            except Exception:
                fd = 0.0

        timetables.append(TimetableEntry(
            from_node=origin,
            to_node=dest,
            mode=mode_norm,
            frequency_per_week=freq,
            first_departure_hour=fd,
            headway_hours=hd
        ))

    if skipped_tt_rows > 0:
        print(f"[WARN] Timetable cleaning skipped {skipped_tt_rows} invalid row(s).")

    bdf = pd.read_excel(xls, "Batches")
    bdf = augment_batches_to_20(bdf, node_region=node_region, random_seed=2026)

    batches: List[Batch] = []
    for _, row in bdf.iterrows():
        origin = str(row.get("OriginEN", "")).strip()
        dest = str(row.get("DestEN", "")).strip()
        o_reg = node_region.get(origin)
        d_reg = node_region.get(dest)

        if o_reg in CHINA_REGIONS and d_reg in EUROPE_REGIONS:
            batches.append(Batch(
                batch_id=int(row.get("BatchID", 0)),
                origin=origin,
                destination=dest,
                quantity=safe_float(row.get("QuantityTEU"), default=0.0),
                ET=safe_float(row.get("ET"), default=0.0),
                LT=safe_float(row.get("LT"), default=0.0)
            ))

    print(f"[INFO] Number of batches loaded: {len(batches)}")
    print(f"[INFO] Loaded node capacities for {len(node_caps)} nodes (TEU/bucket).")
    print(f"[INFO] Carbon tax regions: {len(carbon_tax_map)} | EF overrides: {len(emission_factor_map)} | "
          f"Mode speeds: {len(mode_speeds_map)} | Transshipment entries: {len(trans_map)}")

    return (
        node_names, node_region, node_caps,
        arcs, timetables, batches,
        waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
        carbon_tax_map, emission_factor_map, mode_speeds_map, trans_map
    )


def build_graph(arcs: List[Arc]) -> Dict[str, List[Tuple[str, Arc]]]:
    g: Dict[str, List[Tuple[str, Arc]]] = {}
    for a in arcs:
        g.setdefault(a.from_node, []).append((a.to_node, a))
    return g


def build_timetable_dict(timetables: List[TimetableEntry]) -> Dict[Tuple[str, str, str], List[TimetableEntry]]:
    tt_dict: Dict[Tuple[str, str, str], List[TimetableEntry]] = {}
    for t in timetables:
        tt_dict.setdefault((t.from_node, t.to_node, t.mode), []).append(t)
    return tt_dict


def build_arc_lookup(arcs: List[Arc]) -> Dict[Tuple[str, str, str], Arc]:
    mp: Dict[Tuple[str, str, str], Arc] = {}
    for a in arcs:
        k = (a.from_node, a.to_node, a.mode)
        if k not in mp:
            mp[k] = a
    return mp


# ========================
# Path library
# ========================

def random_dfs_paths(graph, origin, dest, max_len=12, max_paths=200) -> List[List[Arc]]:
    paths: List[List[Arc]] = []

    def dfs(node, cur_arcs, visited):
        if len(paths) >= max_paths or len(cur_arcs) > max_len:
            return
        if node == dest and cur_arcs:
            paths.append(cur_arcs.copy())
            return

        neighbors = graph.get(node, [])
        random.shuffle(neighbors)
        for nxt, arc in neighbors:
            if nxt in visited:
                continue
            dfs(nxt, cur_arcs + [arc], visited | {nxt})

    dfs(origin, [], {origin})
    return paths


def repair_arc_seq_with_road_fallback(
    arc_seq: List[Arc],
    tt_dict: Dict[Tuple[str, str, str], List[TimetableEntry]],
    arc_lookup: Dict[Tuple[str, str, str], Arc]
) -> Optional[List[Arc]]:
    new_seq: List[Arc] = []
    for arc in arc_seq:
        if arc.mode == "road":
            new_seq.append(arc)
            continue

        if tt_dict.get((arc.from_node, arc.to_node, arc.mode), []):
            new_seq.append(arc)
            continue

        k_road = (arc.from_node, arc.to_node, "road")
        if k_road in arc_lookup:
            new_seq.append(arc_lookup[k_road])
        else:
            return None

    return new_seq


def select_topk_by_cost_time_emis(paths: List[Path], k: int = 30, cap_total: int = 90) -> List[Path]:
    if not paths:
        return []

    by_cost = sorted(paths, key=lambda p: p.base_cost_per_teu)
    by_time = sorted(paths, key=lambda p: p.base_travel_time_h)
    by_emis = sorted(paths, key=lambda p: p.base_emission_per_teu)

    picked: List[Path] = []
    used = set()

    def add_list(lst, kk):
        nonlocal picked, used
        for p in lst[:kk]:
            if p not in used:
                picked.append(p)
                used.add(p)

    add_list(by_cost, k)
    add_list(by_time, k)
    add_list(by_emis, k)

    if cap_total is not None and len(picked) > cap_total:
        picked = picked[:cap_total]

    return picked


def build_path_library(node_names, arcs, batches, tt_dict, arc_lookup) -> Dict[Tuple[str, str], List[Path]]:
    graph = build_graph(arcs)
    path_lib: Dict[Tuple[str, str], List[Path]] = {}
    next_path_id = 0

    for b in batches:
        od = (b.origin, b.destination)
        if od in path_lib:
            continue

        arc_paths = random_dfs_paths(
            graph, b.origin, b.destination,
            max_len=12,
            max_paths=DFS_MAX_PATHS_PER_OD
        )
        paths_for_od: List[Path] = []

        for arc_seq in arc_paths:
            repaired = repair_arc_seq_with_road_fallback(arc_seq, tt_dict, arc_lookup)
            if repaired is None:
                continue

            nodes = [repaired[0].from_node] + [a.to_node for a in repaired]
            if len(set(nodes)) != len(nodes):
                continue
            modes = [a.mode for a in repaired]

            paths_for_od.append(Path(
                path_id=next_path_id,
                origin=b.origin,
                destination=b.destination,
                nodes=nodes,
                modes=modes,
                arcs=repaired,
                base_cost_per_teu=sum(a.cost_per_teu_km * a.distance for a in repaired),
                base_emission_per_teu=sum(a.emission_per_teu_km * a.distance for a in repaired),
                base_travel_time_h=sum(a.distance / max(a.speed_kmh, 1.0) for a in repaired),
            ))
            next_path_id += 1

        if paths_for_od:
            path_lib[od] = select_topk_by_cost_time_emis(
                paths_for_od,
                k=PATHS_TOPK_PER_CRITERION,
                cap_total=PATH_LIB_CAP_TOTAL
            )

    return path_lib


def sanity_check_path_lib(batches: List[Batch], path_lib: Dict[Tuple[str, str], List[Path]]):
    missing = []
    for b in batches:
        od = (b.origin, b.destination)
        if len(path_lib.get(od, [])) == 0:
            missing.append((b.batch_id, od))
    if missing:
        for bid, od in missing[:20]:
            print(f"[SANITY CHECK] ❌ missing paths for Batch {bid} OD={od}")
        raise RuntimeError("Path library missing some ODs. Infeasible forever (miss_alloc>0).")
    print("[SANITY CHECK] ✅ All batches have at least one usable path in path_lib.")


def repair_missing_allocations(ind: Individual, batches: List[Batch], path_lib: Dict[Tuple[str, str], List[Path]]):
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        if ind.od_allocations.get(key, []):
            continue
        paths = path_lib.get((b.origin, b.destination), [])
        if paths:
            ind.od_allocations[key] = [PathAllocation(path=paths[0], share=1.0)]


# ========================
# Simulation & evaluation
# ========================

def next_departure_time_programB_first_entry_only(t: float, entries: List[TimetableEntry]) -> float:
    """
    Program B semantics:
      - use entries[0] only
      - if t <= first_departure -> dep = first_departure
      - else dep = first_departure + ceil((t-first)/headway)*headway
    """
    e = entries[0]
    if t <= e.first_departure_hour:
        return e.first_departure_hour
    waited = (t - e.first_departure_hour)
    n = math.ceil(waited / max(e.headway_hours, 1e-6))
    return e.first_departure_hour + n * e.headway_hours


def simulate_path_time_capacity(
    path: Path,
    batch: Batch,
    flow_teu: float,
    tt_dict,
    arc_flow_map,
    node_flow_map,
    trans_map: Optional[Dict[Tuple[str, str, str], Dict[str, float]]] = None,
) -> Tuple[float, float, float, int]:

    t = float(batch.ET)
    total_wait = 0.0
    total_trans_h = 0.0
    miss_tt = 0
    trans_map = trans_map or {}

    prev_arc = None

    for arc in path.arcs:
        # ---- Transshipment time at mode-change node (before departure) ----
        if prev_arc is not None and prev_arc.mode != arc.mode:
            node = arc.from_node  # current arc starts from transfer node
            rec = trans_map.get((node, prev_arc.mode, arc.mode), None)
            if rec:
                trans_h = safe_float(rec.get("time_h"), default=0.0)
                if trans_h > 0:
                    t += trans_h
                    total_trans_h += trans_h

        # ---- Travel time on current arc ----
        travel_time = arc.distance / max(arc.speed_kmh, 1.0)

        # ---- Timetable handling (Program B semantics) ----
        if arc.mode == "road":
            entries = []
        else:
            entries = tt_dict.get((arc.from_node, arc.to_node, arc.mode), [])

        if (arc.mode != "road") and (not entries):
            miss_tt += 1
            return float("inf"), float("inf"), float("inf"), miss_tt

        dep = t if not entries else next_departure_time_programB_first_entry_only(t, entries)

        wait_here = max(0.0, dep - t)
        total_wait += wait_here

        arr = dep + travel_time

        # ---- Arc capacity accounting (departure bucket) ----
        start_slot = int(dep)
        arc_key = (arc.from_node, arc.to_node, arc.mode)
        slot_key = (arc_key, start_slot)
        arc_flow_map[slot_key] = arc_flow_map.get(slot_key, 0.0) + flow_teu

        # ---- Node capacity accounting (arrival bucket) ----
        arrive_node = arc.to_node
        arr_slot = int(arr)
        node_flow_map[(arrive_node, arr_slot)] = node_flow_map.get((arrive_node, arr_slot), 0.0) + flow_teu

        t = arr
        prev_arc = arc

    return (t - batch.ET), total_wait, total_trans_h, miss_tt


def evaluate_individual(
    ind: Individual,
    batches: List[Batch],
    arcs: List[Arc],
    tt_dict,
    waiting_cost_per_teu_h: float,
    wait_emis_g_per_teu_h: float,
    node_caps: Dict[str, float],
    # ✅ NEW (Document6 merged)
    carbon_tax_map: Optional[Dict[str, float]] = None,             # $/tCO2
    trans_map: Optional[Dict[Tuple[str, str, str], Dict[str, float]]] = None,
):
    total_cost = 0.0
    total_emission_g = 0.0
    makespan = 0.0

    arc_flow_map: Dict[Tuple[Tuple[str, str, str], int], float] = {}
    node_flow_map: Dict[Tuple[str, int], float] = {}

    arc_caps = {(a.from_node, a.to_node, a.mode): a.capacity for a in arcs}

    carbon_tax_map = carbon_tax_map or {}
    trans_map = trans_map or {}

    miss_alloc = 0
    miss_tt = 0
    cap_excess = 0.0
    node_cap_excess = 0.0

    late_h_total = 0.0
    late_teu_h_total = 0.0
    wait_h_total = 0.0
    wait_teu_h_total = 0.0

    # ✅ diagnostics extras
    trans_time_h_total = 0.0
    trans_teu_h_total = 0.0
    trans_cost_total = 0.0
    carbon_cost_total = 0.0

    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        allocs = ind.od_allocations.get(key, [])
        if not allocs:
            miss_alloc += 1
            continue

        batch_finish_time = b.ET

        for alloc in allocs:
            if alloc.share <= 1e-12:
                continue

            flow = alloc.share * b.quantity
            p = alloc.path

            # 1) Base transport cost + transport emission (arc sums)
            base_transport_cost = p.base_cost_per_teu * flow
            base_transport_emis_g = p.base_emission_per_teu * flow

            total_cost += base_transport_cost
            total_emission_g += base_transport_emis_g

            # 2) ✅ Carbon tax cost from transport emissions, region-specific
            #    Use arc.from_region as the taxation region (consistent with your model notes).
            carbon_cost_this_alloc = 0.0
            for arc in p.arcs:
                emis_g = arc.emission_per_teu_km * arc.distance * flow
                region = getattr(arc, "from_region", "") or ""
                ct = float(carbon_tax_map.get(region, 0.0))  # $/tCO2
                carbon_cost_this_alloc += (emis_g / 1e6) * ct  # g -> tCO2
            total_cost += carbon_cost_this_alloc
            carbon_cost_total += carbon_cost_this_alloc

            # 3) ✅ Transshipment cost (mode changes)
            trans_cost_this_alloc = 0.0
            for i in range(1, len(p.arcs)):
                prev_arc = p.arcs[i - 1]
                cur_arc = p.arcs[i]
                if prev_arc.mode != cur_arc.mode:
                    node = cur_arc.from_node
                    rec = trans_map.get((node, prev_arc.mode, cur_arc.mode), None)
                    if rec:
                        trans_cost_this_alloc += safe_float(rec.get("cost_per_teu"), default=0.0) * flow
            total_cost += trans_cost_this_alloc
            trans_cost_total += trans_cost_this_alloc

            # 4) Time simulation (✅ now includes transshipment time)
            travel_time, wait_h, trans_h, mtt = simulate_path_time_capacity(
                p, b, flow, tt_dict, arc_flow_map, node_flow_map, trans_map=trans_map
            )

            if math.isinf(travel_time):
                miss_tt += mtt
                continue

            # waiting terms (Scheme A)
            wait_teu_h = flow * wait_h
            wait_teu_h_total += wait_teu_h
            wait_h_total += alloc.share * wait_h

            total_cost += waiting_cost_per_teu_h * wait_teu_h
            total_emission_g += wait_emis_g_per_teu_h * wait_teu_h

            # transshipment time diagnostics (cost/time cost already handled by trans cost + time propagation)
            trans_time_h_total += alloc.share * trans_h
            trans_teu_h_total += flow * trans_h

            arrival_time = b.ET + travel_time
            batch_finish_time = max(batch_finish_time, arrival_time)

            if arrival_time > b.LT:
                late_h = arrival_time - b.LT
                late_h_total += alloc.share * late_h
                late_teu_h = flow * late_h
                late_teu_h_total += late_teu_h
                total_cost += LATE_COST_PER_TEU_HOUR * late_teu_h

        makespan = max(makespan, batch_finish_time)

    # Arc cap excess
    for (arc_key, slot), flow in arc_flow_map.items():
        cap = arc_caps.get(arc_key, 1e9)
        if flow > cap:
            cap_excess += (flow - cap)

    # Node cap excess
    for (node, slot), flow in node_flow_map.items():
        cap = node_caps.get(node, 1e12)
        if flow > cap:
            node_cap_excess += (flow - cap)

    penalty = (
        PEN_MISS_ALLOC * float(miss_alloc) +
        PEN_MISS_TT * float(miss_tt) +
        PEN_CAP_EXCESS_PER_TEU * float(cap_excess) +
        PEN_NODE_CAP_EXCESS_PER_TEU * float(node_cap_excess)
    )

    ind.objectives = (float(total_cost), float(total_emission_g), float(makespan))
    ind.penalty = float(penalty)

    hard_ok = (miss_alloc == 0 and miss_tt == 0 and cap_excess <= 1e-9 and node_cap_excess <= 1e-9)
    strict_no_late = (late_h_total <= 1e-9)

    ind.feasible_hard = bool(hard_ok and strict_no_late)
    ind.feasible = bool(hard_ok and (strict_no_late if HARD_TIME_WINDOW else True))

    ind.vio_breakdown = {
        "miss_alloc": float(miss_alloc),
        "miss_tt": float(miss_tt),
        "cap_excess": float(cap_excess),
        "node_cap_excess": float(node_cap_excess),
        "late_h": float(late_h_total),
        "late_teu_h": float(late_teu_h_total),
        "wait_h": float(wait_h_total),
        "wait_teu_h": float(wait_teu_h_total),
        # ✅ new diagnostics (won’t break old plotting unless you add them)
        "trans_h": float(trans_time_h_total),
        "trans_teu_h": float(trans_teu_h_total),
        "trans_cost": float(trans_cost_total),
        "carbon_cost": float(carbon_cost_total),
    }


# ========================
# GA operators
# ========================

def clone_gene(alloc: PathAllocation) -> PathAllocation:
    return PathAllocation(path=alloc.path, share=float(alloc.share))


def crossover_structural(ind1: Individual, ind2: Individual, batches: List[Batch]) -> Tuple[Individual, Individual]:
    child1 = Individual()
    child2 = Individual()

    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        g1 = ind1.od_allocations.get(key, [])
        g2 = ind2.od_allocations.get(key, [])

        if not g1 and not g2:
            continue

        if not g1:
            child1.od_allocations[key] = [clone_gene(x) for x in g2]
            child2.od_allocations[key] = [clone_gene(x) for x in g2]
            continue

        if not g2:
            child1.od_allocations[key] = [clone_gene(x) for x in g1]
            child2.od_allocations[key] = [clone_gene(x) for x in g1]
            continue

        cut1 = random.randint(0, len(g1))
        cut2 = random.randint(0, len(g2))

        c1 = [clone_gene(x) for x in g1[:cut1]] + [clone_gene(x) for x in g2[cut2:]]
        c2 = [clone_gene(x) for x in g2[:cut2]] + [clone_gene(x) for x in g1[cut1:]]

        child1.od_allocations[key] = merge_and_normalize(c1)
        child2.od_allocations[key] = merge_and_normalize(c2)

    return child1, child2


# --- Common-node segment crossover helpers

def path_from_arcs(new_arcs: List[Arc], origin: str, destination: str, path_id: int = -1) -> Optional[Path]:
    if not new_arcs:
        return None
    nodes = [new_arcs[0].from_node] + [a.to_node for a in new_arcs]
    if nodes[0] != origin or nodes[-1] != destination:
        return None
    if len(set(nodes)) != len(nodes):
        return None
    modes = [a.mode for a in new_arcs]
    base_cost = sum(a.cost_per_teu_km * a.distance for a in new_arcs)
    base_emis = sum(a.emission_per_teu_km * a.distance for a in new_arcs)
    base_time = sum(a.distance / max(a.speed_kmh, 1.0) for a in new_arcs)
    return Path(
        path_id=path_id,
        origin=origin,
        destination=destination,
        nodes=nodes,
        modes=modes,
        arcs=new_arcs,
        base_cost_per_teu=base_cost,
        base_emission_per_teu=base_emis,
        base_travel_time_h=base_time
    )


def rebuild_path_from_nodes_modes(
    origin: str,
    destination: str,
    nodes: List[str],
    modes: List[str],
    tt_dict: Dict[Tuple[str, str, str], List[TimetableEntry]],
    arc_lookup: Dict[Tuple[str, str, str], Arc],
    allow_road_fallback_if_no_timetable: bool = True,
) -> Optional[Path]:
    if not nodes or len(nodes) < 2:
        return None
    if nodes[0] != origin or nodes[-1] != destination:
        return None
    if len(modes) != len(nodes) - 1:
        return None
    if len(set(nodes)) != len(nodes):
        return None

    new_arcs: List[Arc] = []
    for i in range(len(modes)):
        u, v, m = nodes[i], nodes[i + 1], modes[i]
        k = (u, v, m)

        if k not in arc_lookup:
            return None

        arc = arc_lookup[k]

        if arc.mode != "road":
            entries = tt_dict.get((u, v, arc.mode), [])
            if not entries:
                if allow_road_fallback_if_no_timetable and (u, v, "road") in arc_lookup:
                    arc = arc_lookup[(u, v, "road")]
                else:
                    return None

        new_arcs.append(arc)

    return path_from_arcs(new_arcs, origin, destination, path_id=-1)


def find_common_internal_nodes(p1: Path, p2: Path) -> List[str]:
    if not p1.nodes or not p2.nodes:
        return []
    s1 = set(p1.nodes[1:-1])
    s2 = set(p2.nodes[1:-1])
    return list(s1.intersection(s2))


def perform_single_point_crossover_paths(
    pA: Path,
    pB: Path,
    join_node: str,
    tt_dict: Dict[Tuple[str, str, str], List[TimetableEntry]],
    arc_lookup: Dict[Tuple[str, str, str], Arc],
) -> Optional[Path]:
    if join_node not in pA.nodes or join_node not in pB.nodes:
        return None

    ia = pA.nodes.index(join_node)
    ib = pB.nodes.index(join_node)

    new_nodes = pA.nodes[:ia + 1] + pB.nodes[ib + 1:]
    new_modes = pA.modes[:ia] + pB.modes[ib:]

    return rebuild_path_from_nodes_modes(
        origin=pA.origin,
        destination=pA.destination,
        nodes=new_nodes,
        modes=new_modes,
        tt_dict=tt_dict,
        arc_lookup=arc_lookup,
        allow_road_fallback_if_no_timetable=True,
    )


def crossover_common_node(
    ind1: Individual,
    ind2: Individual,
    batches: List[Batch],
    tt_dict: Dict[Tuple[str, str, str], List[TimetableEntry]],
    arc_lookup: Dict[Tuple[str, str, str], Arc],
) -> Tuple[Individual, Individual]:
    child1 = Individual()
    child2 = Individual()

    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        g1 = ind1.od_allocations.get(key, [])
        g2 = ind2.od_allocations.get(key, [])

        if not g1 and not g2:
            continue

        if not g1:
            child1.od_allocations[key] = [clone_gene(x) for x in g2]
            child2.od_allocations[key] = [clone_gene(x) for x in g2]
            continue

        if not g2:
            child1.od_allocations[key] = [clone_gene(x) for x in g1]
            child2.od_allocations[key] = [clone_gene(x) for x in g1]
            continue

        c1_allocs = [clone_gene(x) for x in g1]
        c2_allocs = [clone_gene(x) for x in g2]

        p1 = random.choice(g1).path
        p2 = random.choice(g2).path

        common = find_common_internal_nodes(p1, p2)
        if common:
            join = random.choice(common)

            new_p_for_c1 = perform_single_point_crossover_paths(p1, p2, join, tt_dict, arc_lookup)
            new_p_for_c2 = perform_single_point_crossover_paths(p2, p1, join, tt_dict, arc_lookup)

            if new_p_for_c1 is not None:
                c1_allocs.append(PathAllocation(path=new_p_for_c1, share=0.20))
            if new_p_for_c2 is not None:
                c2_allocs.append(PathAllocation(path=new_p_for_c2, share=0.20))

        child1.od_allocations[key] = merge_and_normalize(c1_allocs)
        child2.od_allocations[key] = merge_and_normalize(c2_allocs)

    return child1, child2


def crossover_hybrid(
    p1: Individual,
    p2: Individual,
    batches: List[Batch],
    tt_dict: Dict[Tuple[str, str, str], List[TimetableEntry]],
    arc_lookup: Dict[Tuple[str, str, str], Arc],
) -> Tuple[Individual, Individual]:
    if random.random() < CROSSOVER_SEGMENT_PROB:
        c1, c2 = crossover_common_node(p1, p2, batches, tt_dict, arc_lookup)
        if not c1.od_allocations and not c2.od_allocations:
            return crossover_structural(p1, p2, batches)
        return c1, c2
    return crossover_structural(p1, p2, batches)


def random_initial_individual(batches: List[Batch], path_lib: Dict[Tuple[str, str], List[Path]], max_paths=3) -> Individual:
    ind = Individual()
    for b in batches:
        paths = path_lib.get((b.origin, b.destination), [])
        if not paths:
            continue
        k = random.randint(1, min(max_paths, len(paths)))
        chosen = random.sample(paths, k)
        raw = [PathAllocation(path=p, share=random.random()) for p in chosen]
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = merge_and_normalize(raw)
    return ind


def mutate_add(ind: Individual, batch: Batch, path_lib):
    key = (batch.origin, batch.destination, batch.batch_id)
    od = (batch.origin, batch.destination)
    allocs = ind.od_allocations.get(key, [])
    pool = path_lib.get(od, [])
    if not pool:
        return False

    cur = {a.path for a in allocs}
    candidates = [p for p in pool if p not in cur]

    if not candidates:
        if allocs:
            repl = random.choice(pool)
            allocs[random.randrange(len(allocs))] = PathAllocation(path=repl, share=0.2)
            ind.od_allocations[key] = merge_and_normalize(allocs)
            return True
        return False

    new_path = random.choice(candidates)
    allocs.append(PathAllocation(path=new_path, share=0.2))
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_del(ind: Individual, batch: Batch):
    key = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if len(allocs) <= 1:
        return False
    allocs.pop(random.randrange(len(allocs)))
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_mod(ind: Individual, batch: Batch):
    key = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if not allocs:
        return False
    target = random.choice(allocs)
    target.share *= random.uniform(0.5, 1.5)
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_mode(ind: Individual, batch: Batch, tt_dict, arc_lookup, max_trials: int = 20):
    key = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if not allocs:
        return False

    idx = random.randrange(len(allocs))
    old_alloc = allocs[idx]
    p = old_alloc.path
    if not p.arcs:
        return False

    arc_i = random.randrange(len(p.arcs))
    old_arc = p.arcs[arc_i]
    u, v = old_arc.from_node, old_arc.to_node

    modes_all = [m for m in ["road", "rail", "water"] if m != old_arc.mode]
    if not modes_all:
        return False

    for _ in range(max_trials):
        new_mode = random.choice(modes_all)
        k_arc = (u, v, new_mode)
        if k_arc not in arc_lookup:
            continue

        if new_mode != "road":
            if not tt_dict.get((u, v, new_mode), []):
                continue

        new_arcs = list(p.arcs)
        new_arcs[arc_i] = arc_lookup[k_arc]
        new_path = path_from_arcs(new_arcs, p.origin, p.destination, path_id=-1)
        if new_path is None:
            continue

        allocs_new = deepcopy(allocs)
        allocs_new[idx] = PathAllocation(path=new_path, share=old_alloc.share)
        ind.od_allocations[key] = merge_and_normalize(allocs_new)
        return True

    return False
def augment_batches_to_20(
    bdf: pd.DataFrame,
    node_region: Dict[str, str],
    random_seed: int = 2026
) -> pd.DataFrame:
    """
    如果Batches表少于20行，从node_region中筛选中国节点和欧洲节点，
    随机组合生成补充批次，直到共20行。
    """
    df = bdf.copy()

    required_cols = ["BatchID", "OriginEN", "DestEN", "QuantityTEU", "ET", "LT"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Batches sheet missing columns {missing}; skip augmentation.")
        return df

    if len(df) >= 20:
        print(f"[INFO] Batches rows={len(df)} (>=20). Skip augmentation.")
        return df

    # 从node_region筛选合法起点（中国）和终点（欧洲）
    china_nodes = [n for n, r in node_region.items() if r in CHINA_REGIONS]
    europe_nodes = [n for n, r in node_region.items() if r in EUROPE_REGIONS]

    if not china_nodes or not europe_nodes:
        print("[WARN] No valid China/Europe nodes found in node_region. Skip augmentation.")
        return df

    # 从已有批次学习数量和时间窗范围
    q_vals = pd.to_numeric(df["QuantityTEU"], errors="coerce").dropna()
    lt_vals = pd.to_numeric(df["LT"], errors="coerce").dropna()

    q_min = int(q_vals.min()) if len(q_vals) > 0 else 80
    q_max = int(q_vals.max()) if len(q_vals) > 0 else 150
    lt_min = int(lt_vals.min()) if len(lt_vals) > 0 else 360
    lt_max = int(lt_vals.max()) if len(lt_vals) > 0 else 480

    existing_ids = set(
        pd.to_numeric(df["BatchID"], errors="coerce").dropna().astype(int).tolist()
    )
    next_id = max(existing_ids) + 1 if existing_ids else 11

    rng = np.random.default_rng(random_seed)
    num_new = 20 - len(df)
    new_rows = []

    for i in range(num_new):
        origin = str(rng.choice(china_nodes))
        dest = str(rng.choice(europe_nodes))
        new_rows.append({
            "BatchID": next_id + i,
            "OriginEN": origin,
            "DestEN": dest,
            "QuantityTEU": int(rng.integers(q_min, q_max + 1)),
            "ET": 0,
            "LT": int(rng.integers(lt_min, lt_max + 1)),
        })

    df_new = pd.DataFrame(new_rows)
    df_out = pd.concat([df, df_new], ignore_index=True)
    print(f"[INFO] Batches augmented: {len(df)} -> {len(df_out)} rows.")
    print(f"[INFO] China node pool ({len(china_nodes)}): {china_nodes}")
    print(f"[INFO] Europe node pool ({len(europe_nodes)}): {europe_nodes}")
    return df_out

def adjust_batches_to_target_k(batches, node_region, target_k, batch_cls, random_seed=2026):
    if target_k is None:
        return list(batches)
    target_k = int(target_k)
    b_sorted = sorted(list(batches), key=lambda b: int(getattr(b, "batch_id", 0)))
    if len(b_sorted) >= target_k:
        out = b_sorted[:target_k]
        print(f"[SCENARIO] batches truncated: {len(b_sorted)} -> {len(out)} (target={target_k})")
        return out
    china_nodes = [n for n, r in node_region.items() if r in CHINA_REGIONS]
    europe_nodes = [n for n, r in node_region.items() if r in EUROPE_REGIONS]
    if not china_nodes or not europe_nodes:
        print("[WARN] No valid China/Europe nodes. Return original batches.")
        return b_sorted
    rng = np.random.default_rng(int(random_seed))
    q_vals = [float(getattr(b, "quantity", 0.0)) for b in b_sorted if float(getattr(b, "quantity", 0.0)) > 0]
    lt_vals = [float(getattr(b, "LT", 0.0)) for b in b_sorted if float(getattr(b, "LT", 0.0)) > 0]
    et_vals = [float(getattr(b, "ET", 0.0)) for b in b_sorted]
    q_min = int(min(q_vals)) if q_vals else 80
    q_max = int(max(q_vals)) if q_vals else 150
    lt_min = int(min(lt_vals)) if lt_vals else 360
    lt_max = int(max(lt_vals)) if lt_vals else 480
    et_default = float(np.median(et_vals)) if et_vals else 0.0
    existing_ids = [int(getattr(b, "batch_id", 0)) for b in b_sorted]
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    need = target_k - len(b_sorted)
    new_batches = []
    for i in range(need):
        new_batches.append(batch_cls(
            batch_id=next_id + i,
            origin=str(rng.choice(china_nodes)),
            destination=str(rng.choice(europe_nodes)),
            quantity=float(int(rng.integers(q_min, q_max + 1))),
            ET=et_default,
            LT=float(int(rng.integers(lt_min, lt_max + 1)))
        ))
    out = b_sorted + new_batches
    print(f"[SCENARIO] batches augmented: {len(b_sorted)} -> {len(out)} (target={target_k})")
    return out


def apply_capacity_scale(arcs, node_caps, capacity_scale):
    s = float(capacity_scale)
    for a in arcs:
        a.capacity *= s
    for k in list(node_caps.keys()):
        node_caps[k] *= s
    print(f"[SCENARIO] Capacity scale applied: x{s}")

# ========================
# Adaptive roulette
# ========================

def _normalise_probs(scores: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(v, 0.0) for v in scores.values())
    if s <= 1e-12:
        k = len(scores)
        return {op: 1.0 / k for op in scores.keys()}
    return {op: max(v, 0.0) / s for op, v in scores.items()}


def is_improved(before: Individual, after: Individual, eps_pen=1e-9, eps_obj=1e-12) -> bool:
    if after.feasible and (not before.feasible):
        return True
    if (not after.feasible) and (not before.feasible):
        return after.penalty < before.penalty - eps_pen

    sb = float(sum(before.objectives))
    sa = float(sum(after.objectives))
    return sa < sb - eps_obj


class AdaptiveRoulette:
    def __init__(self, ops: List[str], init_probs: Dict[str, float], ema_alpha: float = 0.10,
                 min_prob: float = 0.05, score_eps: float = 1e-3):
        self.ops = list(ops)
        self.ema_alpha = float(ema_alpha)
        self.min_prob = float(min_prob)
        self.score_eps = float(score_eps)

        base = {op: float(init_probs.get(op, 1.0 / len(self.ops))) for op in self.ops}
        base = _normalise_probs(base)
        self.quality = {op: base[op] for op in self.ops}
        self.prob = self._quality_to_prob()

    def _quality_to_prob(self) -> Dict[str, float]:
        scores = {op: (self.quality[op] + self.score_eps) for op in self.ops}
        p = _normalise_probs(scores)

        k = len(self.ops)
        mp = min(self.min_prob, 1.0 / k - 1e-9) if k > 1 else 1.0
        remain = 1.0 - k * mp
        p = {op: mp + remain * p[op] for op in self.ops}

        s = sum(p.values())
        return {op: p[op] / s for op in self.ops}

    def sample(self) -> str:
        r = random.random()
        cum = 0.0
        for op in self.ops:
            cum += self.prob[op]
            if r <= cum:
                return op
        return self.ops[-1]

    def update(self, op: str, success: bool):
        y = 1.0 if success else 0.0
        q = self.quality.get(op, 0.0)
        self.quality[op] = (1.0 - self.ema_alpha) * q + self.ema_alpha * y
        self.prob = self._quality_to_prob()


def apply_mutation_op(ind: Individual, op: str, batch: Batch, path_lib, tt_dict, arc_lookup) -> bool:
    if op == "add":
        return mutate_add(ind, batch, path_lib)
    if op == "del":
        return mutate_del(ind, batch)
    if op == "mod":
        return mutate_mod(ind, batch)
    if op == "mode":
        return mutate_mode(ind, batch, tt_dict, arc_lookup)
    return False


def mutate_roulette_adaptive(
    ind: Individual,
    batches: List[Batch],
    path_lib, tt_dict, arc_lookup,
    roulette: AdaptiveRoulette,
    parent_snapshot: Individual,
    arcs: List[Arc],
    waiting_cost_per_teu_h: float,
    wait_emis_g_per_teu_h: float,
    node_caps: Dict[str, float],
    # ✅ NEW
    carbon_tax_map: Optional[Dict[str, float]] = None,
    trans_map: Optional[Dict[Tuple[str, str, str], Dict[str, float]]] = None,
) -> Tuple[str, bool, bool]:
    batch = random.choice(batches)
    op = roulette.sample()
    ok = apply_mutation_op(ind, op, batch, path_lib, tt_dict, arc_lookup)
    if not ok:
        roulette.update(op, success=False)
        return op, False, False

    repair_missing_allocations(ind, batches, path_lib)
    evaluate_individual(
    ind, batches, arcs, tt_dict,
    waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps,
    carbon_tax_map=carbon_tax_map, trans_map=trans_map
)
    success = is_improved(parent_snapshot, ind)
    roulette.update(op, success=success)
    return op, True, success


# ========================
# NSGA-II components
# ========================

def dominates(a: Individual, b: Individual) -> bool:
    if a.feasible and not b.feasible:
        return True
    if b.feasible and not a.feasible:
        return False

    if a.feasible and b.feasible:
        return (all(x <= y for x, y in zip(a.objectives, b.objectives)) and
                any(x < y for x, y in zip(a.objectives, b.objectives)))

    if a.penalty < b.penalty - 1e-12:
        return True
    if b.penalty < a.penalty - 1e-12:
        return False

    return (all(x <= y for x, y in zip(a.objectives, b.objectives)) and
            any(x < y for x, y in zip(a.objectives, b.objectives)))


def non_dominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    S: Dict[Individual, List[Individual]] = {p: [] for p in pop}
    n: Dict[Individual, int] = {p: 0 for p in pop}
    fronts: List[List[Individual]] = [[]]

    for p in pop:
        for q in pop:
            if p is q:
                continue
            if dominates(p, q):
                S[p].append(q)
            elif dominates(q, p):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: List[Individual] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    return fronts[:-1]


def crowding_distance(front: List[Individual]) -> Dict[Individual, float]:
    l = len(front)
    d: Dict[Individual, float] = {ind: 0.0 for ind in front}
    if l == 0:
        return d

    m = len(front[0].objectives)
    for i in range(m):
        front.sort(key=lambda x: x.objectives[i])
        d[front[0]] = float("inf")
        d[front[-1]] = float("inf")
        rng = front[-1].objectives[i] - front[0].objectives[i]
        if abs(rng) <= 1e-12:
            continue
        for j in range(1, l - 1):
            d[front[j]] += (front[j + 1].objectives[i] - front[j - 1].objectives[i]) / rng
    return d


def tournament_select(pop: List[Individual], dists: Dict[Individual, float], ranks: Dict[Individual, int]) -> Individual:
    a, b = random.sample(pop, 2)
    if ranks[a] < ranks[b]:
        return a
    if ranks[b] < ranks[a]:
        return b
    if dists[a] > dists[b]:
        return a
    return b


# ========================
# HV calculator (Monte Carlo)
# ========================

class HypervolumeCalculator:
    """
    Monte Carlo approximation of HV in [0, ref]^m.
    ✅ Supports calculate_points(points) for normalised tuples.
    """
    def __init__(self, ref_point: Tuple[float, float, float], num_samples=2000, seed: Optional[int] = None):
        self.ref_point = np.array(ref_point, dtype=float)
        self.num_samples = int(num_samples)
        self.ideal_point = np.zeros(3, dtype=float)

        # fixed samples if seed provided (reproducible)
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.samples = rng.uniform(low=self.ideal_point, high=self.ref_point, size=(self.num_samples, 3))
        else:
            self.samples = np.random.uniform(low=self.ideal_point, high=self.ref_point, size=(self.num_samples, 3))

    def calculate_points(self, points: List[Tuple[float, float, float]]) -> float:
        if not points:
            return 0.0
        front_objs = np.array(points, dtype=float)

        valid_mask = np.all(front_objs <= self.ref_point, axis=1)
        valid_objs = front_objs[valid_mask]
        if len(valid_objs) == 0:
            return 0.0

        S = self.samples[:, np.newaxis, :]
        O = valid_objs[np.newaxis, :, :]
        dominated = np.all(O <= S, axis=2)
        dominated_samples = np.any(dominated, axis=1)
        return float(np.sum(dominated_samples) / float(self.num_samples))


def unique_individuals_by_objectives(front: List[Individual], tol: float = 1e-3) -> List[Individual]:
    uniq: List[Individual] = []
    seen: List[Tuple[float, float, float]] = []
    for ind in front:
        obj = ind.objectives
        is_dup = False
        for o in seen:
            if (abs(obj[0] - o[0]) <= tol and abs(obj[1] - o[1]) <= tol and abs(obj[2] - o[2]) <= tol):
                is_dup = True
                break
        if not is_dup:
            seen.append(obj)
            uniq.append(ind)
    return uniq


# ========================
# Metrics: P*, IGD+, Spacing
# ========================

def dominates_obj(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    return all(a[i] <= b[i] for i in range(3)) and any(a[i] < b[i] for i in range(3))


def nondominated_set(points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    pts = unique_objective_tuples(points, tol=1e-9)
    nd = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if dominates_obj(q, p):
                dominated = True
                break
        if not dominated:
            nd.append(p)
    return nd


def normalize_points(points: List[Tuple[float, float, float]], mins, maxs):
    out = []
    for p in points:
        pp = []
        for i in range(3):
            rng = maxs[i] - mins[i]
            if rng <= 1e-12:
                pp.append(0.0)
            else:
                pp.append((p[i] - mins[i]) / rng)
        out.append(tuple(pp))
    return out


def clip_points(points: List[Tuple[float, float, float]], ref: Tuple[float, float, float]):
    out = []
    for p in points:
        out.append(tuple(min(max(p[i], 0.0), ref[i]) for i in range(3)))
    return out


def igd_plus(P_star: List[Tuple[float, float, float]], A: List[Tuple[float, float, float]]) -> float:
    if not P_star or not A:
        return float("inf")
    P = np.array(P_star, dtype=float)
    Q = np.array(A, dtype=float)

    dists = []
    for p in P:
        diff = Q - p
        diff = np.maximum(diff, 0.0)
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dists.append(float(np.min(dist)))
    return float(np.mean(dists))


def spacing_metric(A: List[Tuple[float, float, float]]) -> float:
    if A is None or len(A) < 2:
        return 0.0
    Q = np.array(A, dtype=float)
    n = Q.shape[0]
    dmin = []
    for i in range(n):
        diff = Q - Q[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dist[i] = np.inf
        dmin.append(float(np.min(dist)))
    dmin = np.array(dmin)
    return float(np.sqrt(np.sum((dmin - np.mean(dmin)) ** 2) / max(1, n - 1)))


# ========================
# fast P* builder
# ========================

def build_P_star_fast(run_front_hist,
                      tail_gens=PSTAR_TAIL_GENS,
                      cap_per_gen=PSTAR_CAP_PER_GEN,
                      max_total=PSTAR_MAX_TOTAL) -> List[Tuple[float, float, float]]:
    pts: List[Tuple[float, float, float]] = []
    for r in range(len(run_front_hist)):
        hist = run_front_hist[r]
        tail = hist[-tail_gens:] if (tail_gens is not None and tail_gens > 0) else hist
        for gen_front in tail:
            if not gen_front:
                continue
            pts.extend(gen_front[:cap_per_gen])
            if len(pts) >= max_total:
                break
        if len(pts) >= max_total:
            break
    return nondominated_set(pts)


# ========================
# Plot helpers (mutation + metrics)
# ========================

def aggregate_mutation_over_runs(mut_runs: List[dict], generations: int):
    share_runs = []
    rate_runs = []
    eff_runs = []
    prob_runs = []

    for tr in mut_runs:
        attempt = np.vstack([tr["attempt"][op] for op in OPS]).astype(float)
        success = np.vstack([tr["success"][op] for op in OPS]).astype(float)
        total_attempt = np.sum(attempt, axis=0)

        share = attempt / np.maximum(total_attempt, 1.0)
        rate = success / np.maximum(attempt, 1.0)
        eff = share * rate

        share_runs.append(share)
        rate_runs.append(rate)
        eff_runs.append(eff)

        if "prob" in tr:
            prob = np.vstack([tr["prob"][op] for op in OPS]).astype(float)
            prob_runs.append(prob)

    share_runs = np.stack(share_runs, axis=0)
    rate_runs = np.stack(rate_runs, axis=0)
    eff_runs = np.stack(eff_runs, axis=0)

    out = {
        "share_mean": np.mean(share_runs, axis=0),
        "share_std": np.std(share_runs, axis=0),
        "rate_mean": np.mean(rate_runs, axis=0),
        "rate_std": np.std(rate_runs, axis=0),
        "eff_mean": np.mean(eff_runs, axis=0),
        "eff_std": np.std(eff_runs, axis=0),
    }

    if prob_runs:
        prob_runs = np.stack(prob_runs, axis=0)
        out["prob_mean"] = np.mean(prob_runs, axis=0)
        out["prob_std"] = np.std(prob_runs, axis=0)

    return out


def plot_mutation_attempt_stacked(gen, share_mean, save="mutation_attempt_stacked.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.stackplot(
        gen,
        share_mean[0], share_mean[1], share_mean[2], share_mean[3],
        labels=["Add", "Delete", "Modify", "Mode"],
        alpha=0.85
    )
    plt.margins(0, 0)
    plt.ylim(0, 1.0)
    plt.xlabel("Generation")
    plt.ylabel("Attempt Share")
    plt.title("Mutation Roulette: Attempt Share (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print("Saved:", save)


def plot_mutation_adaptive_prob_stacked(gen, prob_mean, save="mutation_adaptive_prob_stacked.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.stackplot(
        gen,
        prob_mean[0], prob_mean[1], prob_mean[2], prob_mean[3],
        labels=["Add", "Delete", "Modify", "Mode"],
        alpha=0.85
    )
    plt.margins(0, 0)
    plt.ylim(0, 1.0)
    plt.xlabel("Generation")
    plt.ylabel("Selection Probability")
    plt.title("Adaptive Roulette: Operator Selection Probability (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print("Saved:", save)


def plot_mutation_success_rate_2x2(gen, rate_mean, rate_std, save="mutation_success_rate.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300, sharex=True, sharey=True)
    axes = axes.ravel()

    for i, op in enumerate(OPS):
        ax = axes[i]
        m = rate_mean[i]
        s = rate_std[i]
        ax.plot(gen, m, linewidth=2.5, label=f"{op} mean")
        ax.fill_between(gen, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), alpha=0.2, label="± std")
        ax.set_title(op.upper())
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Mutation Operators: Success Rate (mean ± std over runs)", y=0.98)
    fig.text(0.5, 0.03, "Generation", ha="center")
    fig.text(0.03, 0.5, "Success Rate", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.06, 1, 0.95])
    plt.savefig(save)
    plt.close(fig)
    print("Saved:", save)


def plot_mutation_effective_contribution(gen, eff_mean, save="mutation_effective_contribution.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    for i, op in enumerate(OPS):
        plt.plot(gen, eff_mean[i], linewidth=2.0, label=op)
    plt.ylim(0, 1.0)
    plt.xlabel("Generation")
    plt.ylabel("Attempt × Success")
    plt.title("Mutation Operators: Effective Contribution (mean over runs)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print("Saved:", save)


def plot_convergence_curves(gen, hv_mean, hv_std, igd_mean, igd_std, save="convergence_hv_igd.png"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)
    ax.plot(gen, hv_mean, linewidth=2.0, label="HV (normalised, mean)")
    ax.fill_between(gen, hv_mean - hv_std, hv_mean + hv_std, alpha=0.2, label="HV ± std")
    ax.set_xlabel("Generation")
    ax.set_ylabel("HV (normalised)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(gen, igd_mean, linewidth=2.0, linestyle="--", label="IGD+ (mean)")
    ax2.fill_between(gen, np.maximum(igd_mean - igd_std, 0), igd_mean + igd_std, alpha=0.15, label="IGD+ ± std")
    ax2.set_ylabel("IGD+ (lower is better)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Convergence: HV (normalised) and IGD+ (mean ± std over runs)")
    plt.tight_layout()
    plt.savefig(save)
    plt.close(fig)
    print("Saved:", save)


def plot_spacing_curve(gen, sp_mean, sp_std, save="diversity_spacing.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, sp_mean, linewidth=2.0, label="Spacing (mean)")
    plt.fill_between(gen, np.maximum(sp_mean - sp_std, 0), sp_mean + sp_std, alpha=0.2, label="± std")
    plt.xlabel("Generation")
    plt.ylabel("Spacing (lower is better)")
    plt.title("Diversity: Spacing (mean ± std over runs)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print("Saved:", save)


def plot_feasible_ratio(gen, fr_mean, fr_std, save="feasible_ratio.png",
                        title="Constraint Handling: Feasible Ratio (mean ± std over runs)",
                        ylabel="Feasible Ratio"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, fr_mean, linewidth=2.0, label="Feasible Ratio (mean)", zorder=3)
    plt.fill_between(gen,
                     np.clip(fr_mean - fr_std, 0, 1),
                     np.clip(fr_mean + fr_std, 0, 1),
                     alpha=0.2, label="± std", zorder=2)

    plt.ylim(-0.02, 1.02)

    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print("Saved:", save)


def plot_violation_breakdown_stacked(gen, vio_mean_dict_mean, save="violation_breakdown.png"):
    miss_alloc = np.array(vio_mean_dict_mean["miss_alloc"])
    miss_tt = np.array(vio_mean_dict_mean["miss_tt"])
    cap_excess = np.array(vio_mean_dict_mean["cap_excess"])
    node_cap_excess = np.array(vio_mean_dict_mean["node_cap_excess"])
    late_h = np.array(vio_mean_dict_mean["late_h"])
    wait_h = np.array(vio_mean_dict_mean["wait_h"])

    plt.figure(figsize=(12, 4), dpi=300)
    plt.bar(gen, miss_alloc, label="miss_alloc")
    plt.bar(gen, miss_tt, bottom=miss_alloc, label="miss_tt")
    plt.bar(gen, cap_excess, bottom=miss_alloc + miss_tt, label="cap_excess")
    plt.bar(gen, node_cap_excess, bottom=miss_alloc + miss_tt + cap_excess, label="node_cap_excess")
    plt.bar(gen, late_h, bottom=miss_alloc + miss_tt + cap_excess + node_cap_excess, label="late_h")
    plt.bar(gen, wait_h, bottom=miss_alloc + miss_tt + cap_excess + node_cap_excess + late_h, label="wait_h")

    plt.xlabel("Generation")
    plt.ylabel("Mean Violation Components (stacked)")
    plt.title("Diagnostics Breakdown (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print("Saved:", save)


# ========================
# Extra outputs requested
# ========================

def plot_hv_curve(gen, hv_mean, hv_std=None, save="hv_curve.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, hv_mean, linewidth=2.2, label="HV (normalised, mean)", zorder=3)
    if hv_std is not None:
        plt.fill_between(gen, hv_mean - hv_std, hv_mean + hv_std, alpha=0.2, label="HV ± std", zorder=2)
    plt.xlabel("Generation")
    plt.ylabel("HV (normalised)")
    plt.title("Hypervolume Curve (normalised, mean ± std over runs)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print("Saved:", save)


def plot_feasible_ratio_curve(gen, fr_mean, fr_std=None, frs_mean=None, frs_std=None, save="feasible_ratio_curve.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, fr_mean, linewidth=2.2, label="Feasible Ratio (soft, mean)", zorder=3)
    if fr_std is not None:
        plt.fill_between(gen, np.clip(fr_mean - fr_std, 0, 1), np.clip(fr_mean + fr_std, 0, 1),
                         alpha=0.2, label="soft ± std", zorder=2)

    if frs_mean is not None:
        plt.plot(gen, frs_mean, linewidth=2.0, linestyle="--", label="Feasible Ratio (strict, mean)", zorder=3)
        if frs_std is not None:
            plt.fill_between(gen, np.clip(frs_mean - frs_std, 0, 1), np.clip(frs_mean + frs_std, 0, 1),
                             alpha=0.12, label="strict ± std", zorder=1)

    plt.ylim(-0.02, 1.02)
    plt.xlabel("Generation")
    plt.ylabel("Feasible Ratio")
    plt.title("Feasible Ratio Curve (mean ± std over runs)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print("Saved:", save)


def _pick_overlay_gens(generations: int, n: int = 5) -> List[int]:
    if generations <= 1:
        return [0]
    cand = [0, generations // 4, generations // 2, (3 * generations) // 4, generations - 1]
    cand = sorted(list(dict.fromkeys([int(x) for x in cand if 0 <= int(x) < generations])))
    return cand


def plot_2d_evolution_overlays(
    front_hist: List[List[Tuple[float, float, float]]],
    generations: int,
    save_prefix: str = "evolution2d",
    gens_to_plot: Optional[List[int]] = None
):
    if not front_hist:
        print("[WARN] Empty front_hist, skip 2D evolution overlays.")
        return

    if gens_to_plot is None:
        gens_to_plot = _pick_overlay_gens(generations, n=5)

    pairs = [
        (0, 1, "Cost", "Emission (gCO2)", f"{save_prefix}_cost_emis.png"),
        (0, 2, "Cost", "Time",           f"{save_prefix}_cost_time.png"),
        (1, 2, "Emission (gCO2)", "Time", f"{save_prefix}_emis_time.png"),
    ]

    for xi, yi, xl, yl, outname in pairs:
        plt.figure(figsize=(6.6, 5.2), dpi=300)
        plotted_any = False
        for g in gens_to_plot:
            pts = front_hist[g] if (0 <= g < len(front_hist)) else []
            arr = _finite_points_array(pts)
            if arr.shape[0] == 0:
                continue
            plt.scatter(arr[:, xi], arr[:, yi], s=18, alpha=0.70, label=f"Gen {g}")
            plotted_any = True

        if not plotted_any:
            plt.close()
            print(f"[WARN] No finite points for {outname}, skipped.")
            continue

        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"2D Evolution Overlay (Best Run): {xl} vs {yl}")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(outname)
        plt.close()
        print("Saved:", outname)


def extract_min_objectives(front_hist: List[List[Tuple[float, float, float]]], generations: int):
    """
    ✅ Robust version:
    - ignore non-finite points (nan/inf)
    - use np.nanmin
    - forward-fill NaNs to avoid matplotlib line breaks ('gaps')
    """
    min_cost = np.full(generations, np.nan, dtype=float)
    min_emis = np.full(generations, np.nan, dtype=float)
    min_time = np.full(generations, np.nan, dtype=float)

    gmax = min(generations, len(front_hist))
    for g in range(gmax):
        pts = front_hist[g]
        if not pts:
            continue

        arr = _finite_points_array(pts)
        if arr.shape[0] == 0:
            continue

        # use nanmin for robustness (though finite filter already applied)
        try:
            min_cost[g] = float(np.nanmin(arr[:, 0]))
            min_emis[g] = float(np.nanmin(arr[:, 1]))
            min_time[g] = float(np.nanmin(arr[:, 2]))
        except Exception:
            continue

    # ✅ forward fill to avoid plotting gaps
    min_cost = _ffill_nan(min_cost)
    min_emis = _ffill_nan(min_emis)
    min_time = _ffill_nan(min_time)

    return min_cost, min_emis, min_time


def plot_min_objective_curves(gen, min_cost, min_emis, min_time,
                              save_cost="min_cost_curve.png",
                              save_emis="min_emission_curve.png",
                              save_time="min_time_curve.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, min_cost, linewidth=2.2)
    plt.xlabel("Generation")
    plt.ylabel("Min Cost")
    plt.title("Best-run Min Cost per Generation (front0 unique)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_cost)
    plt.close()
    print("Saved:", save_cost)

    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, min_emis, linewidth=2.2)
    plt.xlabel("Generation")
    plt.ylabel("Min Emission (gCO2)")
    plt.title("Best-run Min Emission per Generation (front0 unique)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_emis)
    plt.close()
    print("Saved:", save_emis)

    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, min_time, linewidth=2.2)
    plt.xlabel("Generation")
    plt.ylabel("Min Time")
    plt.title("Best-run Min Time per Generation (front0 unique)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_time)
    plt.close()
    print("Saved:", save_time)


def export_gen_metrics_best_run(
    front_hist: List[List[Tuple[float, float, float]]],
    hv_hist: List[float],
    fr_hist: List[float],
    frs_hist: List[float],
    vio_hist: Dict[str, List[float]],
    generations: int,
    out_csv: str = "gen_metrics_best_run.csv"
):
    min_cost, min_emis, min_time = extract_min_objectives(front_hist, generations)

    df = pd.DataFrame({
        "gen": np.arange(generations, dtype=int),
        "HV_norm": np.array(hv_hist[:generations], dtype=float),
        "feasible_ratio_soft": np.array(fr_hist[:generations], dtype=float),
        "feasible_ratio_strict": np.array(frs_hist[:generations], dtype=float),
        "min_cost": min_cost,
        "min_emission_gCO2": min_emis,
        "min_time": min_time,
    })

    if vio_hist:
        for k, series in vio_hist.items():
            if isinstance(series, list) and len(series) >= generations:
                df[f"vio_mean_{k}"] = np.array(series[:generations], dtype=float)

    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return df


# ========================
# Pareto 3D plots
# ========================

def plot_pareto_3d_final_only(pareto_points: List[Tuple[float, float, float]], save: str, title: str):
    if not pareto_points:
        print("[WARN] No final Pareto points to plot.")
        return
    A = _finite_points_array(pareto_points)
    if A.shape[0] == 0:
        print("[WARN] No finite final Pareto points to plot.")
        return
    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], marker="o", s=30, alpha=0.9)
    ax.set_xlabel("Cost")
    ax.set_ylabel("Emission (gCO2)")
    ax.set_zlabel("Time")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save)
    plt.close(fig)
    print("Saved:", save)


def plot_pareto_3d_all_generations(
    all_gen_points_with_gen: List[Tuple[float, float, float, int]],
    final_pareto_points: Optional[List[Tuple[float, float, float]]] = None,
    save: str = "pareto_3d_allgens_best_run.png",
    title: str = "Pareto Points Across All Generations (Best Run)",
    cmap_name: str = "turbo",
):
    if not all_gen_points_with_gen:
        print("[WARN] No all-generation Pareto points to plot.")
        return

    raw = np.array([(c, e, t, g) for (c, e, t, g) in all_gen_points_with_gen], dtype=float)
    mask = np.all(np.isfinite(raw[:, :3]), axis=1)
    raw = raw[mask]
    if raw.shape[0] == 0:
        print("[WARN] No finite all-generation Pareto points to plot.")
        return

    P = raw[:, :3]
    G = raw[:, 3]

    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=G, cmap=cmap_name, s=10, alpha=0.55)
    cbar = plt.colorbar(sc, ax=ax, pad=0.10, fraction=0.04)
    cbar.set_label("Generation")
    gmin, gmax = int(np.min(G)), int(np.max(G))
    ticks = np.linspace(gmin, gmax, num=6).astype(int)
    cbar.set_ticks(ticks)

    if final_pareto_points:
        F = _finite_points_array(final_pareto_points)
        if F.shape[0] > 0:
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], marker="^", s=40, alpha=0.95, label="Final feasible Pareto")
            ax.legend(loc="best")

    ax.set_xlabel("Cost")
    ax.set_ylabel("Emission (gCO2)")
    ax.set_zlabel("Time")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save)
    plt.close(fig)
    print("Saved:", save)


# ========================
# Runner (one run)
# ========================

def run_nsga2_one_run(filename, pop_size, generations, seed,
                       target_batch_k=None, capacity_scale=1.0):
    """
    ✅ IMPORTANT:
    This function NO LONGER computes HV.
    It records front_hist_objs (raw objective tuples) so main can
    compute HV after normalisation using P* bounds.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("Loading data...")
    (node_names, node_region, node_caps,
        arcs, timetables, batches,
        waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
        carbon_tax_map, emission_factor_map, mode_speeds_map, trans_map) = load_network_from_extended(filename)
    batches = adjust_batches_to_target_k(
        batches, node_region, target_batch_k, Batch, random_seed=seed)
    apply_capacity_scale(arcs, node_caps, capacity_scale)

    tt_dict = build_timetable_dict(timetables)
    arc_lookup = build_arc_lookup(arcs)

    print("Building path library...")
    path_lib = build_path_library(node_names, arcs, batches, tt_dict, arc_lookup)
    sanity_check_path_lib(batches, path_lib)

    init_probs = _normalise_probs({"add": W_ADD, "del": W_DEL, "mod": W_MOD, "mode": W_MODE})
    roulette = AdaptiveRoulette(
        ops=OPS,
        init_probs=init_probs,
        ema_alpha=ROULETTE_EMA_ALPHA,
        min_prob=ROULETTE_MIN_PROB,
        score_eps=ROULETTE_SCORE_EPS
    )

    population: List[Individual] = []
    for _ in range(pop_size):
        ind = random_initial_individual(batches, path_lib)
        repair_missing_allocations(ind, batches, path_lib)
        evaluate_individual(
            ind, batches, arcs, tt_dict,
            waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps,
            carbon_tax_map=carbon_tax_map, trans_map=trans_map
        )
        population.append(ind)

    front_hist_objs: List[List[Tuple[float, float, float]]] = []
    feasible_ratio_hist: List[float] = []
    feasible_ratio_strict_hist: List[float] = []
    vio_mean_hist: Dict[str, List[float]] = {k: [] for k in ["miss_alloc", "miss_tt", "cap_excess", "node_cap_excess", "late_h", "wait_h"]}

    mut_tracker = {
        "attempt": {op: [0] * generations for op in OPS},
        "success": {op: [0] * generations for op in OPS},
        "prob":    {op: [0.0] * generations for op in OPS},
    }

    for gen in range(generations):
        for op in OPS:
            mut_tracker["prob"][op][gen] = float(roulette.prob[op])

        fronts = non_dominated_sort(population)
        front0 = fronts[0]

        feasible_front0 = [ind for ind in front0 if ind.feasible]
        base_front = feasible_front0 if feasible_front0 else front0
        front0_unique = unique_individuals_by_objectives(base_front, tol=1e-3)

        current_front_objs = [ind.objectives for ind in front0_unique]
        front_hist_objs.append(current_front_objs)

        feasible_ratio_hist.append(sum(1 for ind in population if ind.feasible) / float(len(population)))
        feasible_ratio_strict_hist.append(sum(1 for ind in population if ind.feasible_hard) / float(len(population)))

        for k in vio_mean_hist.keys():
            vals = [ind.vio_breakdown.get(k, 0.0) for ind in population]
            vio_mean_hist[k].append(float(np.mean(vals)) if vals else 0.0)

        if gen % 10 == 0 or gen == generations - 1:
            feas_n = len(feasible_front0)
            best_pen = min(ind.penalty for ind in front0_unique) if front0_unique else float("inf")
            best_cost = min(ind.objectives[0] for ind in front0_unique) if front0_unique else float("inf")
            print(f"Gen {gen:03d} | Front0={len(front0):2d} | FeasFront0={feas_n:2d} | "
                  f"FeasRatio={feasible_ratio_hist[-1]:.2%} | StrictFeas={feasible_ratio_strict_hist[-1]:.2%} | "
                  f"BestCost={best_cost:.3e} | BestPenalty={best_pen:.3e}")

        ranks: Dict[Individual, int] = {}
        for r, fr in enumerate(fronts):
            for ind in fr:
                ranks[ind] = r

        dists: Dict[Individual, float] = {}
        for fr in fronts:
            dists.update(crowding_distance(fr))

        mating_pool: List[Individual] = []
        while len(mating_pool) < pop_size:
            mating_pool.append(tournament_select(population, dists, ranks))

        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(mating_pool, 2)

            if random.random() < CROSSOVER_RATE:
                c1, c2 = crossover_hybrid(p1, p2, batches, tt_dict, arc_lookup)
            else:
                c1 = random_initial_individual(batches, path_lib)
                c2 = random_initial_individual(batches, path_lib)

            repair_missing_allocations(c1, batches, path_lib)
            repair_missing_allocations(c2, batches, path_lib)
            evaluate_individual(c1, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps, carbon_tax_map=carbon_tax_map, trans_map=trans_map)
            evaluate_individual(c2, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps, carbon_tax_map=carbon_tax_map, trans_map=trans_map)

            if random.random() < MUTATION_RATE:
                snap = deepcopy(c1)
                op, ok, suc = mutate_roulette_adaptive(
                    c1, batches, path_lib, tt_dict, arc_lookup,
                    roulette, snap, arcs,
                    waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
                    node_caps,
                    carbon_tax_map=carbon_tax_map, trans_map=trans_map
                )
                mut_tracker["attempt"][op][gen] += 1
                mut_tracker["success"][op][gen] += (1 if suc else 0)

            if random.random() < MUTATION_RATE:
                snap = deepcopy(c2)
                op, ok, suc = mutate_roulette_adaptive(
                    c2, batches, path_lib, tt_dict, arc_lookup,
                    roulette, snap, arcs,
                    waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
                    node_caps,
                    carbon_tax_map=carbon_tax_map, trans_map=trans_map
                )
                mut_tracker["attempt"][op][gen] += 1
                mut_tracker["success"][op][gen] += (1 if suc else 0)

            repair_missing_allocations(c1, batches, path_lib)
            repair_missing_allocations(c2, batches, path_lib)
            evaluate_individual(c1, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps, carbon_tax_map=carbon_tax_map, trans_map=trans_map)
            evaluate_individual(c2, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps, carbon_tax_map=carbon_tax_map, trans_map=trans_map)

            offspring.append(c1)
            offspring.append(c2)

        combined = population + offspring
        fronts2 = non_dominated_sort(combined)

        new_pop: List[Individual] = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend(fr)
            else:
                d = crowding_distance(fr)
                fr.sort(key=lambda x: d[x], reverse=True)
                new_pop.extend(fr[:pop_size - len(new_pop)])
                break

        population = new_pop

    final_fronts = non_dominated_sort(population)
    f0 = final_fronts[0]
    feasible_f0 = [ind for ind in f0 if ind.feasible]
    pareto = unique_individuals_by_objectives(feasible_f0, tol=1e-3)

    return (population, pareto, batches,
            front_hist_objs,
            feasible_ratio_hist, feasible_ratio_strict_hist,
            vio_mean_hist, mut_tracker)


# ========================
# Output helpers
# ========================

def print_pure_structure(ind: Individual, batches: List[Batch], sol_name="Solution"):
    print(f"\n===== {sol_name} Final Structure (Node+Mode | Share) =====")
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        allocs = ind.od_allocations.get(key, [])
        if not allocs:
            continue
        print(f"\nBatch {b.batch_id}: {b.origin} -> {b.destination}, Q={b.quantity}\n")
        for a in allocs:
            print(a)


def save_pareto_solutions(pareto: List[Individual], batches: List[Batch], filename: str = "result.txt"):
    pareto = unique_individuals_by_objectives(pareto, tol=1e-3)
    pareto = [ind for ind in pareto if ind.feasible]

    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== NSGA-II Pareto Solutions (Feasible Only) =====\n\n")
        if not pareto:
            f.write("NO FEASIBLE SOLUTION FOUND.\n")
            f.write("Check: path_lib coverage, timetable coverage, capacity scale, node capacity, or HARD_TIME_WINDOW.\n")
            print(f"Saved 0 feasible Pareto solutions to {filename}")
            return

        for i, ind in enumerate(pareto):
            cost, emis, t = ind.objectives
            f.write(f"===== Pareto Sol {i} =====\n")
            f.write(f"Objectives: Cost={cost:.6f}, Emission_gCO2={emis:.6f}, Time={t:.6f}, "
                    f"Penalty={ind.penalty:.6f}, Feasible={ind.feasible}, FeasibleHardNoLate={ind.feasible_hard}, "
                    f"Breakdown={ind.vio_breakdown}\n\n")

            for b in batches:
                key = (b.origin, b.destination, b.batch_id)
                allocs = ind.od_allocations.get(key, [])
                if not allocs:
                    continue
                f.write(f"Batch {b.batch_id}: {b.origin} -> {b.destination}, Q={b.quantity}\n\n")
                for a in allocs:
                    f.write(str(a) + "\n")
                f.write("\n")
            f.write("\n")
    print(f"Saved {len(pareto)} feasible Pareto solutions to {filename}")


from pathlib import Path as FSPath
import json as _json

def export_pareto_points_json(pareto: List[Individual], batches: List[Batch], out_json: str):
    FSPath(FSPath(out_json).parent).mkdir(parents=True, exist_ok=True)

    out = []
    for ind in pareto:
        sol = {
            "objectives": {
                "cost": float(ind.objectives[0]),
                "emission_gCO2": float(ind.objectives[1]),
                "time_h": float(ind.objectives[2]),
                "penalty": float(ind.penalty),
            },
            "feasible_soft": bool(ind.feasible),
            "feasible_strict": bool(ind.feasible_hard),
            "vio_breakdown": {k: float(v) for k, v in (ind.vio_breakdown or {}).items()},
            "allocations": []
        }

        for b in batches:
            key = (b.origin, b.destination, b.batch_id)
            allocs = ind.od_allocations.get(key, [])
            blk = {
                "batch_id": int(b.batch_id),
                "origin": b.origin,
                "destination": b.destination,
                "paths": []
            }
            for a in allocs:
                blk["paths"].append({
                    "share": float(a.share),
                    "nodes": list(a.path.nodes),
                    "modes": list(a.path.modes),
                })
            sol["allocations"].append(blk)

        out.append(sol)

    with open(out_json, "w", encoding="utf-8") as f:
        _json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved Pareto JSON:", out_json)


# ========================
# Main (multi-runs) + plots
# ========================

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", default="data.xlsx")
    parser.add_argument("--out",   default="OUT_NSGA2_GRID")
    parser.add_argument("--runs",  type=int, default=30)
    parser.add_argument("--pop",   type=int, default=75)
    parser.add_argument("--gen",   type=int, default=160)
    parser.add_argument("--seed_base", type=int, default=1000)
    parser.add_argument("--single_scenario", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    scenarios = SCENARIO_GRID
    if args.single_scenario:
        scenarios = [s for s in SCENARIO_GRID if s[0] == args.single_scenario]

    all_summary_rows = []
    t_all0 = time.perf_counter()

    # ════════════════════════════════════════
    # 外层：场景循环
    # ════════════════════════════════════════
    for scenario_id, batch_k, cap_scale in scenarios:
        print(f"\n{'='*80}")
        print(f"[SCENARIO] {scenario_id} | batch_k={batch_k} | cap_scale={cap_scale}")
        print(f"{'='*80}")

        scen_out = os.path.join(args.out, scenario_id)
        os.makedirs(scen_out, exist_ok=True)

        run_front_hist, run_paretos, run_batches_list = [], [], []
        run_feasible_ratio, run_feasible_ratio_strict = [], []
        run_vio_mean, mut_runs, run_rows = [], [], []

        # ────────────────────────────────────
        # 内层：runs循环
        # ────────────────────────────────────
        for run_id in range(args.runs):
            seed = args.seed_base + run_id
            print(f"\n[{scenario_id}] RUN {run_id}/{args.runs-1} | seed={seed}")
            t0 = time.perf_counter()

            pop, pareto, batches, front_hist, fr_hist, frs_hist, vio_hist, mut_tracker = run_nsga2_one_run(
                filename=args.excel,
                pop_size=args.pop,
                generations=args.gen,
                seed=seed,
                target_batch_k=int(batch_k),
                capacity_scale=float(cap_scale),
            )
            runtime_s = float(time.perf_counter() - t0)

            run_front_hist.append(front_hist)
            run_paretos.append(pareto)
            run_batches_list.append(batches)
            run_feasible_ratio.append(fr_hist)
            run_feasible_ratio_strict.append(frs_hist)
            run_vio_mean.append(vio_hist)
            mut_runs.append(mut_tracker)
            run_rows.append({
                "scenario_id": scenario_id,
                "batch_k": int(batch_k),
                "capacity_scale": float(cap_scale),
                "run_id": int(run_id),
                "seed": int(seed),
                "runtime_s": runtime_s,
                "runtime_min": runtime_s / 60.0,
                "final_feasible_ratio": float(fr_hist[-1]) if fr_hist else 0.0,
                "final_strict_feasible_ratio": float(frs_hist[-1]) if frs_hist else 0.0,
                "final_pareto_size": int(len(pareto)),
            })
            print(f"[{scenario_id}][RUN {run_id}] Pareto={len(pareto)} | "
                  f"Feas={fr_hist[-1]:.2%} | {runtime_s:.1f}s")

        # ────────────────────────────────────
        # 后验 P* + HV/IGD+/Spacing
        # ────────────────────────────────────
        P_star = build_P_star_fast(run_front_hist, PSTAR_TAIL_GENS, PSTAR_CAP_PER_GEN, PSTAR_MAX_TOTAL)
        print(f"[{scenario_id}] P* size={len(P_star)}")

        if P_star:
            P_arr = np.array(P_star, float)
            mins, maxs = np.min(P_arr, axis=0), np.max(P_arr, axis=0)
        else:
            mins, maxs = np.zeros(3), np.ones(3)

        hv_calc = HypervolumeCalculator(HV_REF_NORM, HV_SAMPLES, HV_MC_SEED)
        Pn = normalize_points(P_star, mins, maxs) if P_star else []

        hv_mat, igd_mat, sp_mat = [], [], []
        for r in range(args.runs):
            hv_h, igd_h, sp_h = [], [], []
            last_hv = 0.0
            for gi, gf in enumerate(run_front_hist[r]):
                finite = [tuple(x) for x in _finite_points_array(gf)]
                An = clip_points(normalize_points(finite, mins, maxs), HV_REF_NORM) if finite else []
                if gi % HV_EVERY == 0:
                    last_hv = hv_calc.calculate_points(An) if An else 0.0
                hv_h.append(last_hv)
                igd_h.append(igd_plus(Pn, An) if (Pn and An) else float("inf"))
                sp_h.append(spacing_metric(An) if An else 0.0)
            hv_mat.append(hv_h)
            igd_mat.append(igd_h)
            sp_mat.append(sp_h)

        hv_mat  = np.array(hv_mat,  float)
        igd_mat = np.array(igd_mat, float)
        sp_mat  = np.array(sp_mat,  float)
        fr_mat  = np.array(run_feasible_ratio, float)
        frs_mat = np.array(run_feasible_ratio_strict, float)
        gen_arr = np.arange(args.gen)

        # ────────────────────────────────────
        # 最佳run选取
        # ────────────────────────────────────
        best_run = int(np.nanargmax(hv_mat[:, -1]))
        best_pareto    = run_paretos[best_run]
        best_batches   = run_batches_list[best_run]
        best_front_hist = run_front_hist[best_run]

        # ────────────────────────────────────
        # 保存CSV矩阵
        # ────────────────────────────────────
        pd.DataFrame(hv_mat).to_csv(os.path.join(scen_out, "hv_by_run_by_gen.csv"), index=False)
        pd.DataFrame(igd_mat).to_csv(os.path.join(scen_out, "igd_by_run_by_gen.csv"), index=False)
        pd.DataFrame(sp_mat).to_csv(os.path.join(scen_out, "spacing_by_run_by_gen.csv"), index=False)
        pd.DataFrame(fr_mat).to_csv(os.path.join(scen_out, "feasible_ratio_by_run_by_gen.csv"), index=False)
        pd.DataFrame(frs_mat).to_csv(os.path.join(scen_out, "strict_feasible_ratio_by_run_by_gen.csv"), index=False)

        pd.DataFrame({
            "gen": gen_arr,
            "hv_mean": np.nanmean(hv_mat, 0), "hv_std": np.nanstd(hv_mat, 0),
        }).to_csv(os.path.join(scen_out, "hv_curve_mean_std.csv"), index=False)
        pd.DataFrame({
            "gen": gen_arr,
            "igd_mean": np.nanmean(igd_mat, 0), "igd_std": np.nanstd(igd_mat, 0),
        }).to_csv(os.path.join(scen_out, "igd_curve_mean_std.csv"), index=False)
        pd.DataFrame({
            "gen": gen_arr,
            "spacing_mean": np.nanmean(sp_mat, 0), "spacing_std": np.nanstd(sp_mat, 0),
        }).to_csv(os.path.join(scen_out, "spacing_curve_mean_std.csv"), index=False)

        # ────────────────────────────────────
        # 曲线图（复用原有函数）
        # ────────────────────────────────────
        plot_hv_curve(gen_arr, np.nanmean(hv_mat, 0), np.nanstd(hv_mat, 0),
                      save=os.path.join(scen_out, "hv_curve.png"))
        plot_convergence_curves(gen_arr,
                                np.nanmean(hv_mat, 0), np.nanstd(hv_mat, 0),
                                np.nanmean(igd_mat, 0), np.nanstd(igd_mat, 0),
                                save=os.path.join(scen_out, "convergence_hv_igd.png"))
        plot_spacing_curve(gen_arr, np.nanmean(sp_mat, 0), np.nanstd(sp_mat, 0),
                           save=os.path.join(scen_out, "diversity_spacing.png"))
        plot_feasible_ratio_curve(gen_arr,
                                  np.nanmean(fr_mat, 0), np.nanstd(fr_mat, 0),
                                  frs_mean=np.nanmean(frs_mat, 0),
                                  frs_std=np.nanstd(frs_mat, 0),
                                  save=os.path.join(scen_out, "feasible_ratio_curve.png"))

        # mutation诊断（仅最佳run）
        mut_agg = aggregate_mutation_over_runs(mut_runs, args.gen)
        plot_mutation_attempt_stacked(gen_arr, mut_agg["share_mean"],
                                      save=os.path.join(scen_out, "mutation_attempt_stacked.png"))
        plot_mutation_success_rate_2x2(gen_arr, mut_agg["rate_mean"], mut_agg["rate_std"],
                                       save=os.path.join(scen_out, "mutation_success_rate.png"))
        if "prob_mean" in mut_agg:
            plot_mutation_adaptive_prob_stacked(gen_arr, mut_agg["prob_mean"],
                                               save=os.path.join(scen_out, "mutation_adaptive_prob_stacked.png"))

        # 最佳run Pareto输出
        save_pareto_solutions(best_pareto, best_batches,
                              filename=os.path.join(scen_out, "result.txt"))
        export_pareto_points_json(best_pareto, best_batches,
                                  out_json=os.path.join(scen_out, "nsga_pareto_points.json"))

        best_final_pts = [ind.objectives for ind in best_pareto if ind.feasible]
        best_final_pts = unique_objective_tuples(best_final_pts, tol=1e-9)
        plot_pareto_3d_final_only(best_final_pts,
                                  save=os.path.join(scen_out, f"{scenario_id}_pareto_3d.png"),
                                  title=f"{scenario_id} Pareto (best run #{best_run})")
                                  
        

        if best_front_hist:
            plot_2d_evolution_overlays(best_front_hist, args.gen,
                                       save_prefix=os.path.join(scen_out, f"{scenario_id}_evolution2d"),
                                       gens_to_plot=_pick_overlay_gens(args.gen))
            min_cost, min_emis, min_time = extract_min_objectives(best_front_hist, args.gen)
            plot_min_objective_curves(gen_arr, min_cost, min_emis, min_time,
                                      save_cost=os.path.join(scen_out, "min_cost_curve.png"),
                                      save_emis=os.path.join(scen_out, "min_emission_curve.png"),
                                      save_time=os.path.join(scen_out, "min_time_curve.png"))
            export_gen_metrics_best_run(
                front_hist=best_front_hist,
                hv_hist=list(hv_mat[best_run, :]),
                fr_hist=run_feasible_ratio[best_run],
                frs_hist=run_feasible_ratio_strict[best_run],
                vio_hist=run_vio_mean[best_run],
                generations=args.gen,
                out_csv=os.path.join(scen_out, "gen_metrics_best_run.csv")
            )

        # ────────────────────────────────────
        # 场景Excel汇总
        # ────────────────────────────────────
        df_runs = pd.DataFrame(run_rows)
        df_runs["final_HV_norm"]  = hv_mat[:, -1]
        df_runs["final_IGD_plus"] = igd_mat[:, -1]
        df_runs["final_Spacing"]  = sp_mat[:, -1]

        summary_row = {
            "scenario_id": scenario_id,
            "batch_k": int(batch_k),
            "capacity_scale": float(cap_scale),
            "runs": int(args.runs),
            "pop": int(args.pop),
            "gen": int(args.gen),
            "P_star_size": int(len(P_star)),
            "HV_mean":  float(np.nanmean(hv_mat[:, -1])),
            "HV_std":   float(np.nanstd(hv_mat[:, -1])),
            "IGD_mean": float(np.nanmean(igd_mat[:, -1])),
            "IGD_std":  float(np.nanstd(igd_mat[:, -1])),
            "Spacing_mean": float(np.nanmean(sp_mat[:, -1])),
            "Spacing_std":  float(np.nanstd(sp_mat[:, -1])),
            "final_feas_ratio_mean": float(np.nanmean(df_runs["final_feasible_ratio"])),
            "runtime_s_mean": float(np.nanmean(df_runs["runtime_s"])),
            "runtime_s_std":  float(np.nanstd(df_runs["runtime_s"])),
            "best_run_idx":   int(best_run),
            "best_run_HV":    float(hv_mat[best_run, -1]),
        }
        all_summary_rows.append(summary_row)

        scen_xlsx = os.path.join(scen_out, "scenario_results.xlsx")
        with pd.ExcelWriter(scen_xlsx, engine="openpyxl") as w:
            df_runs.to_excel(w, sheet_name="RunSummary", index=False)
            pd.DataFrame([summary_row]).to_excel(w, sheet_name="Summary", index=False)
        print(f"Saved: {scen_xlsx}")
        print(pd.DataFrame([summary_row]).to_string(index=False))

    # ════════════════════════════════════════
    # 全部场景汇总
    # ════════════════════════════════════════
    total_s = float(time.perf_counter() - t_all0)
    df_all = pd.DataFrame(all_summary_rows)
    order_map = {sid: i for i, (sid, _, _) in enumerate(SCENARIO_GRID)}
    df_all = df_all.sort_values(
        by="scenario_id",
        key=lambda s: s.map(lambda x: order_map.get(x, 999))
    ).reset_index(drop=True)

    all_csv  = os.path.join(args.out, "all_scenarios_summary.csv")
    all_xlsx = os.path.join(args.out, "all_scenarios_summary.xlsx")
    df_all.to_csv(all_csv, index=False)
    with pd.ExcelWriter(all_xlsx, engine="openpyxl") as w:
        df_all.to_excel(w, sheet_name="AllScenarios", index=False)

    print(f"\n[DONE] Total wall time = {total_s/60.0:.2f} min")
    print(f"Saved: {all_csv}")
    print(f"Saved: {all_xlsx}")