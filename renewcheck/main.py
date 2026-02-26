#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NSGA-II (Batch-level encoding) — Scheme A + Program B timetable semantics
✅ UPDATED: Hypervolume (HV) is computed AFTER normalisation (min-max) using P* bounds.

修复清单 (vs 上一版本):
1. 新增 CHINA_BORDER_NODES 常量
2. 新增 china_border_monotone_ok() 函数（口岸单调性约束）
3. random_dfs_paths() 恢复 node_region 参数，加入走廊剪枝
4. build_path_library() 恢复 node_region 参数，加入 region_monotone_ok 过滤
5. augment_batches_to_20() 排除口岸节点作为批次起点
6. mutate_roulette_adaptive() 修复 evaluate_individual 缩进错误
7. path_from_arcs() 正确传递 node_region 给 china_border_monotone_ok
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

TIME_BUCKET_H = 1.0

# ✅ 基于实际数据：Region 列的真实值为 CN / KZ / KG / UZ / RU / BY / EE / WE
CHINA_REGIONS = {"CN"}
EUROPE_REGIONS = {"EE", "WE"}          # EE=东欧, WE=西欧
TRANSIT_REGIONS = {"KZ", "KG", "UZ", "RU", "BY"}  # 中亚+俄罗斯+白俄过渡区域

# RegionGroup 走廊方向顺序（单调向前，不允许回退）
# CN=0 → CA=1（中亚）→ RU=2（俄/白俄）→ EE=3（东欧）→ WE=4（西欧）
CORRIDOR_ORDER: Dict[str, int] = {"CN": 0, "CA": 1, "RU": 2, "EE": 3, "WE": 4}

# ✅ 口岸节点集合：从 Node_Border 表动态加载（见 load_network_from_extended）
# 这里保留一个默认值作为后备，实际运行时会被数据覆盖
CHINA_BORDER_NODES: set = {"Erenhot", "Manzhouli", "Khorgos", "Lianyungang",
                            "Chongqing", "Yiwu"}

# 节点 → RegionGroup 映射（从数据动态加载，见 load_network_from_extended）
NODE_GROUP: Dict[str, str] = {}

HARD_TIME_WINDOW = False

PEN_MISS_TT = 5e7
PEN_MISS_ALLOC = 1e9
PEN_CAP_EXCESS_PER_TEU = 5e7
PEN_NODE_CAP_EXCESS_PER_TEU = 5e7

WAITING_COST_PER_TEU_HOUR_DEFAULT = 0.5
WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT = 0.0
LATE_COST_PER_TEU_HOUR = 1e6

W_ADD = 0.25
W_DEL = 0.20
W_MOD = 0.35
W_MODE = 0.20
OPS = ["add", "del", "mod", "mode"]

ROULETTE_EMA_ALPHA = 0.35
ROULETTE_MIN_PROB = 0.05
ROULETTE_SCORE_EPS = 1e-3

CROSSOVER_RATE = 0.92
MUTATION_RATE = 0.25

PATHS_TOPK_PER_CRITERION = 15
PATH_LIB_CAP_TOTAL = 45
DFS_MAX_PATHS_PER_OD = 150   # 走廊约束已保证路径质量，无需大量搜索

CROSSOVER_SEGMENT_PROB = 0.50

# ========================
# Analytics speed knobs
# ========================
HV_EVERY = 5
HV_SAMPLES = 2000
METRIC_EVERY = 5

PSTAR_TAIL_GENS = 30
PSTAR_CAP_PER_GEN = 40
PSTAR_MAX_TOTAL = 50000

HV_REF_NORM = (1.2, 1.2, 1.2)
HV_MC_SEED = 12345


# ========================
# Corridor constraint helpers
# ========================

def china_border_monotone_ok(nodes: List[str], node_region: Dict[str, str]) -> bool:
    """
    口岸单调性约束：
    1. 经过口岸节点后，不得再访问任何中国节点（内地或其他口岸）
    2. 一旦离开中国区域，不得再回到中国区域
    特殊情况：若起点本身是口岸节点（如 Yiwu 出发），
    则第一个节点不触发 passed_border，允许先访问其他节点再出境。
    """
    passed_border = False
    left_china = False
    start_is_border = (len(nodes) > 0 and nodes[0] in CHINA_BORDER_NODES)

    for i, n in enumerate(nodes):
        r = str(node_region.get(n, "")).strip()
        in_china = (r in CHINA_REGIONS)
        is_border = (n in CHINA_BORDER_NODES)

        # 规则1：离开中国后不得折返
        if left_china and in_china:
            return False

        if in_china:
            # 规则2：经过口岸后，不得再访问任何中国节点（含内地和其他口岸）
            if passed_border:
                return False
            # 标记经过口岸（起点是口岸时，第一个节点不触发）
            if is_border and not (i == 0 and start_is_border):
                passed_border = True
        else:
            left_china = True

    return True


def region_monotone_ok(nodes: List[str], node_region: Dict[str, str]) -> bool:
    """
    走廊方向单调性：沿 CN→CA→RU→EE→WE 方向前进，RegionGroup 只能向前不能回退。
    允许在同一 RegionGroup 内部跳转（如 WE 内部 Hamburg→Berlin），
    但不允许回到更早的 RegionGroup（如 WE→RU、EE→KZ 等）。
    未知 RegionGroup 的节点（如 Ningbo/Shanghai 水运中转港）跳过检查。
    """
    max_level = -1  # 已到达的最高走廊级别

    for n in nodes:
        grp = NODE_GROUP.get(n, "")
        level = CORRIDOR_ORDER.get(grp, -1)
        if level < 0:
            continue  # 未知节点跳过（不影响检查）
        if level < max_level:
            return False  # 回退了走廊方向
        max_level = level

    return True


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


def norm_region(x: str) -> str:
    """
    保留数据原始区域标签（CN/KZ/KG/UZ/RU/BY/EE/WE）。
    仅做基本清理和别名归一化。
    """
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", ""}:
        return ""
    # 处理常见别名
    sl = s.lower()
    if sl in {"china", "prc", "chn"}:
        return "CN"
    if sl in {"we", "west europe", "western europe"}:
        return "WE"
    if sl in {"ee", "east europe", "eastern europe"}:
        return "EE"
    # 直接返回大写原值
    return s.upper()


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
    if s is None:
        return True
    t = str(s).strip()
    if t == "":
        return True
    if t.startswith("..."):
        return True
    return False


def _ffill_nan(arr: np.ndarray) -> np.ndarray:
    x = np.array(arr, dtype=float).copy()
    if x.size == 0:
        return x
    finite_idx = np.where(np.isfinite(x))[0]
    if finite_idx.size == 0:
        return x
    x[~np.isfinite(x)] = np.nan
    first = finite_idx[0]
    if first > 0 and np.isfinite(x[first]):
        x[:first] = x[first]
    for i in range(1, len(x)):
        if np.isnan(x[i]) and np.isfinite(x[i - 1]):
            x[i] = x[i - 1]
    return x


def _finite_points_array(pts: List[Tuple[float, float, float]]) -> np.ndarray:
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
    capacity: float
    cost_per_teu_km: float
    emission_per_teu_km: float
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
    quantity: float
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
    base_emission_per_teu: float
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
    od_allocations: Dict[Tuple[str, str, int], List[PathAllocation]] = field(default_factory=dict)
    objectives: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
    penalty: float = 0.0
    feasible: bool = False
    feasible_hard: bool = False
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
    out: Dict[str, float] = {}
    if "Carbon_Tax" not in xls.sheet_names:
        print("[INFO] Carbon_Tax sheet not found. Carbon tax defaults to 0.")
        return out
    try:
        df = pd.read_excel(xls, "Carbon_Tax")
        if df.empty:
            return out
        region_col = next((c for c in ["Region", "region", "RegionCode", "region_code"] if c in df.columns), None)
        tax_col = next((c for c in ["CarbonTax", "Carbon_Tax", "CT", "CT_$_per_tCO2", "CarbonTax_$_per_tCO2"] if c in df.columns), None)
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
    out: Dict[Tuple[str, str], float] = {}
    if "Emission_Factors" not in xls.sheet_names:
        print("[INFO] Emission_Factors sheet not found. Using Arcs_All emission factors.")
        return out
    try:
        df = pd.read_excel(xls, "Emission_Factors")
        if df.empty:
            return out
        mode_col = next((c for c in ["Mode", "mode"] if c in df.columns), None)
        region_col = next((c for c in ["Region", "region", "RegionCode"] if c in df.columns), None)
        ef_col = next((c for c in ["EmissionFactor", "Emission_gCO2_per_TEU_km", "EF", "value"] if c in df.columns), None)
        if mode_col is None or region_col is None or ef_col is None:
            print("[WARN] Emission_Factors sheet columns not recognised.")
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
        print(f"[WARN] Failed to read Emission_Factors sheet ({e}).")
        return out


def load_mode_speeds(xls: pd.ExcelFile) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "Mode_Speeds" not in xls.sheet_names:
        return out
    try:
        df = pd.read_excel(xls, "Mode_Speeds")
        if df.empty:
            return out
        mode_col = next((c for c in ["Mode", "mode"] if c in df.columns), None)
        sp_col = next((c for c in ["Speed_kmh", "speed_kmh", "Speed"] if c in df.columns), None)
        if mode_col is None or sp_col is None:
            return out
        for _, row in df.iterrows():
            m = normalize_mode(row.get(mode_col, ""))
            if not m:
                continue
            out[m] = safe_float(row.get(sp_col), default=0.0)
        print(f"[INFO] Loaded mode speed overrides: {out}")
        return out
    except Exception as e:
        print(f"[WARN] Failed to read Mode_Speeds ({e}).")
        return out


def load_transshipment_map(xls: pd.ExcelFile) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    if "Transshipment" not in xls.sheet_names:
        print("[INFO] Transshipment sheet not found. Transshipment cost/time defaults to 0.")
        return out
    try:
        df = pd.read_excel(xls, "Transshipment")
        if df.empty:
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
        print(f"[WARN] Failed to read Waiting_Costs ({e}).")
        return wc, we


def load_network_from_extended(filename: str):
    global CHINA_BORDER_NODES, NODE_GROUP  # ✅ 声明在函数最前
    xls = pd.ExcelFile(filename)

    carbon_tax_map = load_carbon_tax_map(xls)
    emission_factor_map = load_emission_factor_map(xls)
    mode_speeds_map = load_mode_speeds(xls)
    trans_map = load_transshipment_map(xls)

    nodes_df = pd.read_excel(xls, "Nodes")
    node_names = nodes_df["EnglishName"].astype(str).str.strip().tolist()

    # ✅ 使用 norm_region 统一标签
    node_region = {
        str(name).strip(): norm_region(reg)
        for name, reg in zip(nodes_df["EnglishName"], nodes_df["Region"])
    }

    # ✅ 加载 NODE_GROUP（RegionGroup 列），用于走廊方向单调性约束
    if "RegionGroup" in nodes_df.columns:
        NODE_GROUP = {
            str(name).strip(): str(grp).strip()
            for name, grp in zip(nodes_df["EnglishName"], nodes_df["RegionGroup"])
            if str(grp).strip() not in ("", "nan", "None")
        }
    # 补充弧中出现但 Nodes 表缺失的节点（如 Ningbo/Shanghai 为中国港口节点）
    SEAPORT_NODES = {"Ningbo", "Shanghai"}  # 中国海港中转节点，不作为批次起点
    for n in SEAPORT_NODES:
        if n not in node_region:
            node_region[n] = "CN"
            NODE_GROUP[n] = "CN"
            if n not in node_names:
                node_names.append(n)
    # 将海港节点视为口岸节点，防止被选为补充批次起点（在下面 Node_Border 加载后再 update）
    print("[CHECK] Example regions:",
          {k: node_region.get(k) for k in ["Erenhot", "Manzhouli", "Lianyungang", "Chengdu", "Hamburg", "Ningbo"]
           if k in node_region})
    print("[CHECK] Unique region labels:", sorted(set(node_region.values())))
    print(f"[CHECK] NODE_GROUP sample: {dict(list(NODE_GROUP.items())[:5])}")

    # ✅ 从 Node_Border 表加载真实口岸节点集合，更新全局 CHINA_BORDER_NODES
    if "Node_Border" in xls.sheet_names:
        try:
            nb_df = pd.read_excel(xls, "Node_Border")
            loaded_borders = set(
                str(row["EnglishName"]).strip()
                for _, row in nb_df.iterrows()
                if str(row.get("Region", "")).strip() == "CN"
                and int(row.get("IsBorderNode", 0)) == 1
            )
            if loaded_borders:
                CHINA_BORDER_NODES = loaded_borders
                print(f"[INFO] Loaded China border nodes ({len(CHINA_BORDER_NODES)}): {sorted(CHINA_BORDER_NODES)}")
        except Exception as e:
            print(f"[WARN] Failed to load Node_Border: {e}. Using default CHINA_BORDER_NODES.")
    else:
        print(f"[INFO] Node_Border sheet not found. Using default CHINA_BORDER_NODES: {sorted(CHINA_BORDER_NODES)}")
    # 将海港中转节点也视为口岸节点（防止成为补充批次起点，防止口岸约束绕过）
    CHINA_BORDER_NODES.update({"Ningbo", "Shanghai"})

    DAILY_HOURS = 24.0
    node_caps: Dict[str, float] = {}
    for _, row in nodes_df.iterrows():
        n = str(row.get("EnglishName", "")).strip()
        # ✅ 优先读 NodeCap_TEUday（每天），其次 NodeCap_TEUh×24
        if "NodeCap_TEUday" in nodes_df.columns and not pd.isna(row.get("NodeCap_TEUday", np.nan)):
            raw = safe_float(row.get("NodeCap_TEUday"), default=None)
            if raw is not None:
                node_caps[n] = float(raw)
                continue
        if "NodeCap_TEUh" in nodes_df.columns and not pd.isna(row.get("NodeCap_TEUh", np.nan)):
            raw = safe_float(row.get("NodeCap_TEUh"), default=None)
            if raw is not None:
                node_caps[n] = float(raw) * DAILY_HOURS
                continue
        for col in ["NodeCap_TEU_per_day", "Capacity_TEU", "Throughput_TEU"]:
            if col in nodes_df.columns and not pd.isna(row.get(col, np.nan)):
                raw = safe_float(row.get(col), default=None)
                if raw is not None:
                    node_caps[n] = float(raw)
                    break
        else:
            node_caps[n] = 1e12

    waiting_cost_per_teu_h, wait_emis_g_per_teu_h = load_waiting_params(xls)

    arcs_df = pd.read_excel(xls, "Arcs_All")
    arcs: List[Arc] = []

    cost_cols = ["Cost_$_per_km", "Cost_per_km", "Cost"]
    emis_cols = ["Emission_gCO2_per_tkm", "Emission_gCO2_per_TEU_km", "EmissionFactor", "Emission"]

    for _, row in arcs_df.iterrows():
        mode = normalize_mode(row.get("Mode", "road"))
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

        # ✅ 容量：优先读 Capacity_TEUday（每天），其次 Capacity_TEUh×24，保持与批次量级一致
        if "Capacity_TEUday" in arcs_df.columns and not pd.isna(row.get("Capacity_TEUday", np.nan)):
            capacity = safe_float(row.get("Capacity_TEUday"), default=1e9)
        elif "Capacity_TEUh" in arcs_df.columns and not pd.isna(row.get("Capacity_TEUh", np.nan)):
            capacity = safe_float(row.get("Capacity_TEUh"), default=1e9) * 24.0
        elif "Capacity_TEU" in arcs_df.columns and not pd.isna(row.get("Capacity_TEU", np.nan)):
            capacity = safe_float(row.get("Capacity_TEU"), default=1e9)
        else:
            capacity = 1e9

        cpkm = 0.0
        for c in cost_cols:
            if c in arcs_df.columns:
                cpkm = safe_float(row.get(c), default=0.0)
                break

        epkm = 0.0
        for c in emis_cols:
            if c in arcs_df.columns:
                epkm = safe_float(row.get(c), default=0.0)
                break

        if (mode, from_region) in emission_factor_map:
            epkm = emission_factor_map[(mode, from_region)]

        arcs.append(Arc(
            from_node=origin, to_node=dest, mode=mode,
            distance=distance, capacity=capacity,
            cost_per_teu_km=cpkm, emission_per_teu_km=epkm,
            speed_kmh=speed, from_region=from_region, to_region=to_region
        ))

    if arcs:
        dists = np.array([a.distance for a in arcs], dtype=float)
        costs = np.array([a.cost_per_teu_km for a in arcs], dtype=float)
        print("[CHECK] dist min/median/max:", float(np.min(dists)), float(np.median(dists)), float(np.max(dists)))
        print("[CHECK] dist==0 count:", int(np.sum(dists <= 1e-9)))
        print("[CHECK] cost_per_km min/median/max:", float(np.min(costs)), float(np.median(costs)), float(np.max(costs)))

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
            from_node=origin, to_node=dest, mode=mode_norm,
            frequency_per_week=freq, first_departure_hour=fd, headway_hours=hd
        ))

    if skipped_tt_rows > 0:
        print(f"[WARN] Timetable cleaning skipped {skipped_tt_rows} invalid row(s).")

    bdf = pd.read_excel(xls, "Batches")
    # ✅ 先记录原始批次行数，augment 时对补充批次排除口岸起点
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

def random_dfs_paths(graph, origin, dest, node_region, max_len=12, max_paths=200,
                     timeout_sec=8.0) -> List[List[Arc]]:
    """
    迭代式随机路径搜索（栈模拟DFS），带超时保护。
    约束：
    - china_border_monotone_ok: 口岸单调性
    - region_monotone_ok: 走廊方向单调性 CN→CA→RU→EE→WE
    """
    import time
    deadline = time.time() + timeout_sec
    paths: List[List[Arc]] = []
    found_set: set = set()  # 去重

    # 每次随机从起点出发做一次深度优先遍历
    # 使用迭代栈：(node, arc_list, visited_set, node_list)
    attempts = 0
    max_attempts = max_paths * 20  # 最多尝试次数

    while len(paths) < max_paths and attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1

        # 单次随机游走（贪心随机DFS，找到一条路就收）
        node = origin
        cur_arcs: List[Arc] = []
        visited = {origin}
        cur_nodes = [origin]
        ok = True

        for _ in range(max_len):
            if node == dest:
                break
            neighbors = graph.get(node, [])
            if not neighbors:
                ok = False
                break

            # 随机打乱，优先选择走廊方向正确的邻居
            random.shuffle(neighbors)
            moved = False
            for nxt, arc in neighbors:
                if nxt in visited:
                    continue
                new_nodes = cur_nodes + [nxt]
                if not china_border_monotone_ok(new_nodes, node_region):
                    continue
                if not region_monotone_ok(new_nodes, node_region):
                    continue
                cur_arcs.append(arc)
                visited.add(nxt)
                cur_nodes.append(nxt)
                node = nxt
                moved = True
                break

            if not moved:
                ok = False
                break

        if ok and node == dest and cur_arcs:
            # 去重（用节点序列作为 key）
            key = tuple(cur_nodes)
            if key not in found_set:
                found_set.add(key)
                paths.append(cur_arcs)

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


def build_path_library(
    node_names, node_region, arcs, batches, tt_dict, arc_lookup
) -> Dict[Tuple[str, str], List[Path]]:
    """✅ 恢复 node_region 参数，加入走廊过滤"""
    graph = build_graph(arcs)
    path_lib: Dict[Tuple[str, str], List[Path]] = {}
    next_path_id = 0

    for b in batches:
        od = (b.origin, b.destination)
        if od in path_lib:
            continue

        arc_paths = random_dfs_paths(
            graph, b.origin, b.destination,
            node_region=node_region,
            max_len=12,
            max_paths=DFS_MAX_PATHS_PER_OD
        )
        paths_for_od: List[Path] = []

        for arc_seq in arc_paths:
            repaired = repair_arc_seq_with_road_fallback(arc_seq, tt_dict, arc_lookup)
            if repaired is None:
                continue

            nodes = [repaired[0].from_node] + [a.to_node for a in repaired]

            # 去重检查
            if len(set(nodes)) != len(nodes):
                continue

            # ✅ 走廊单调性过滤
            if not region_monotone_ok(nodes, node_region):
                continue

            # ✅ 口岸单调性过滤（双重保险）
            if not china_border_monotone_ok(nodes, node_region):
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

    # ✅ 强制后置过滤：清除任何仍然违反口岸约束的路径
    removed = 0
    for od in list(path_lib.keys()):
        before = len(path_lib[od])
        path_lib[od] = [
            p for p in path_lib[od]
            if china_border_monotone_ok(p.nodes, node_region)
        ]
        removed += before - len(path_lib[od])
        if not path_lib[od]:
            del path_lib[od]
    if removed > 0:
        print(f"[WARN] Post-filter removed {removed} paths violating border monotonicity.")
    else:
        print("[INFO] Post-filter: all paths pass border monotonicity check. ✅")

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
        if prev_arc is not None and prev_arc.mode != arc.mode:
            node = arc.from_node
            rec = trans_map.get((node, prev_arc.mode, arc.mode), None)
            if rec:
                trans_h = safe_float(rec.get("time_h"), default=0.0)
                if trans_h > 0:
                    t += trans_h
                    total_trans_h += trans_h

        travel_time = arc.distance / max(arc.speed_kmh, 1.0)

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

        # ✅ 容量 slot 按天分桶（容量单位 TEU/day）
        start_slot = int(dep // 24)
        arc_key = (arc.from_node, arc.to_node, arc.mode)
        slot_key = (arc_key, start_slot)
        arc_flow_map[slot_key] = arc_flow_map.get(slot_key, 0.0) + flow_teu

        arrive_node = arc.to_node
        arr_slot = int(arr // 24)
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
    carbon_tax_map: Optional[Dict[str, float]] = None,
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

            base_transport_cost = p.base_cost_per_teu * flow
            base_transport_emis_g = p.base_emission_per_teu * flow
            total_cost += base_transport_cost
            total_emission_g += base_transport_emis_g

            carbon_cost_this_alloc = 0.0
            for arc in p.arcs:
                emis_g = arc.emission_per_teu_km * arc.distance * flow
                region = getattr(arc, "from_region", "") or ""
                ct = float(carbon_tax_map.get(region, 0.0))
                carbon_cost_this_alloc += (emis_g / 1e6) * ct
            total_cost += carbon_cost_this_alloc
            carbon_cost_total += carbon_cost_this_alloc

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

            travel_time, wait_h, trans_h, mtt = simulate_path_time_capacity(
                p, b, flow, tt_dict, arc_flow_map, node_flow_map, trans_map=trans_map
            )

            if math.isinf(travel_time):
                miss_tt += mtt
                continue

            wait_teu_h = flow * wait_h
            wait_teu_h_total += wait_teu_h
            wait_h_total += alloc.share * wait_h
            total_cost += waiting_cost_per_teu_h * wait_teu_h
            total_emission_g += wait_emis_g_per_teu_h * wait_teu_h

            trans_time_h_total += alloc.share * trans_h
            trans_teu_h_total += flow * trans_h

            arrival_time = b.ET + travel_time
            batch_finish_time = max(batch_finish_time, arrival_time)

            if arrival_time > b.LT:
                late_h = arrival_time - b.LT
                late_h_total += alloc.share * late_h
                late_teu_h = flow * late_h
                late_teu_h_total += late_teu_h
                # ✅ 迟到不加入 total_cost，而是计入 penalty（见下方）

        makespan = max(makespan, batch_finish_time)

    for (arc_key, slot), flow in arc_flow_map.items():
        cap = arc_caps.get(arc_key, 1e9)
        if flow > cap:
            cap_excess += (flow - cap)

    for (node, slot), flow in node_flow_map.items():
        cap = node_caps.get(node, 1e12)
        if flow > cap:
            node_cap_excess += (flow - cap)

    # ✅ penalty 包含迟到惩罚（不加入 objectives）
    late_penalty = LATE_COST_PER_TEU_HOUR * late_teu_h_total
    penalty = (
        PEN_MISS_ALLOC * float(miss_alloc) +
        PEN_MISS_TT * float(miss_tt) +
        PEN_CAP_EXCESS_PER_TEU * float(cap_excess) +
        PEN_NODE_CAP_EXCESS_PER_TEU * float(node_cap_excess) +
        late_penalty
    )

    ind.objectives = (float(total_cost), float(total_emission_g), float(makespan))
    ind.penalty = float(penalty)

    hard_ok = (miss_alloc == 0 and miss_tt == 0 and cap_excess <= 1e-9 and node_cap_excess <= 1e-9)
    strict_no_late = (late_h_total <= 1e-9)
    ind.feasible_hard = bool(hard_ok and strict_no_late)
    # ✅ 迟到惩罚已移入 penalty，feasible 必须要求无迟到，否则 dominates 无法淘汰迟到解
    ind.feasible = bool(hard_ok and strict_no_late)

    ind.vio_breakdown = {
        "miss_alloc": float(miss_alloc),
        "miss_tt": float(miss_tt),
        "cap_excess": float(cap_excess),
        "node_cap_excess": float(node_cap_excess),
        "late_h": float(late_h_total),
        "late_teu_h": float(late_teu_h_total),
        "wait_h": float(wait_h_total),
        "wait_teu_h": float(wait_teu_h_total),
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


def path_from_arcs(new_arcs: List[Arc], origin: str, destination: str, path_id: int = -1,
                   node_region: Optional[Dict[str, str]] = None) -> Optional[Path]:
    if not new_arcs:
        return None
    nodes = [new_arcs[0].from_node] + [a.to_node for a in new_arcs]
    if nodes[0] != origin or nodes[-1] != destination:
        return None
    if len(set(nodes)) != len(nodes):
        return None
    # ✅ 口岸单调性检查
    if node_region is not None:
        if not china_border_monotone_ok(nodes, node_region):
            return None
        if not region_monotone_ok(nodes, node_region):
            return None
    modes = [a.mode for a in new_arcs]
    base_cost = sum(a.cost_per_teu_km * a.distance for a in new_arcs)
    base_emis = sum(a.emission_per_teu_km * a.distance for a in new_arcs)
    base_time = sum(a.distance / max(a.speed_kmh, 1.0) for a in new_arcs)
    return Path(
        path_id=path_id, origin=origin, destination=destination,
        nodes=nodes, modes=modes, arcs=new_arcs,
        base_cost_per_teu=base_cost, base_emission_per_teu=base_emis,
        base_travel_time_h=base_time
    )


def rebuild_path_from_nodes_modes(
    origin: str, destination: str,
    nodes: List[str], modes: List[str],
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
    pA: Path, pB: Path, join_node: str,
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
        origin=pA.origin, destination=pA.destination,
        nodes=new_nodes, modes=new_modes,
        tt_dict=tt_dict, arc_lookup=arc_lookup,
        allow_road_fallback_if_no_timetable=True,
    )


def crossover_common_node(
    ind1: Individual, ind2: Individual, batches: List[Batch],
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
    p1: Individual, p2: Individual, batches: List[Batch],
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


def greedy_initial_individual(batches: List[Batch], path_lib: Dict[Tuple[str, str], List[Path]]) -> Individual:
    """✅ 贪心初始化：每个批次选择基础旅行时间最短的路径，最大化初始可行率"""
    ind = Individual()
    for b in batches:
        paths = path_lib.get((b.origin, b.destination), [])
        if not paths:
            continue
        # 按基础旅行时间排序，选最快路径
        best = min(paths, key=lambda p: p.base_travel_time_h)
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = [PathAllocation(path=best, share=1.0)]
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
    df = bdf.copy()
    required_cols = ["BatchID", "OriginEN", "DestEN", "QuantityTEU", "ET", "LT"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Batches sheet missing columns {missing}; skip augmentation.")
        return df
    if len(df) >= 20:
        print(f"[INFO] Batches rows={len(df)} (>=20). Skip augmentation.")
        return df

    # ✅ 排除口岸节点作为起点，只选中国内地城市
    china_nodes = [
        n for n, r in node_region.items()
        if r in CHINA_REGIONS and n not in CHINA_BORDER_NODES
    ]
    europe_nodes = [n for n, r in node_region.items() if r in EUROPE_REGIONS]

    if not china_nodes or not europe_nodes:
        print("[WARN] No valid China/Europe nodes found in node_region. Skip augmentation.")
        return df

    q_vals = pd.to_numeric(df["QuantityTEU"], errors="coerce").dropna()
    q_min = int(q_vals.min()) if len(q_vals) > 0 else 80
    q_max = int(q_vals.max()) if len(q_vals) > 0 else 150

    # ✅ LT 参考原始批次中合理的时间窗（排除过短的异常值）
    lt_vals = pd.to_numeric(df["LT"], errors="coerce").dropna()
    # 只取 LT >= 300h 的（物理上从中国到欧洲最快也要 ~150-200h）
    lt_reasonable = lt_vals[lt_vals >= 300]
    if len(lt_reasonable) > 0:
        lt_min = int(lt_reasonable.min())
        lt_max = int(lt_reasonable.max())
    else:
        # 无合理参考值，使用保守默认值
        lt_min = 360
        lt_max = 504  # 21天

    existing_ids = set(pd.to_numeric(df["BatchID"], errors="coerce").dropna().astype(int).tolist())
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
    # ✅ 修复缩进：evaluate_individual 与上面代码对齐
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
    def __init__(self, ref_point: Tuple[float, float, float], num_samples=2000, seed: Optional[int] = None):
        self.ref_point = np.array(ref_point, dtype=float)
        self.num_samples = int(num_samples)
        self.ideal_point = np.zeros(3, dtype=float)
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
# Plot helpers
# ========================

def aggregate_mutation_over_runs(mut_runs: List[dict], generations: int):
    share_runs, rate_runs, eff_runs, prob_runs = [], [], [], []
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
        "share_mean": np.mean(share_runs, axis=0), "share_std": np.std(share_runs, axis=0),
        "rate_mean": np.mean(rate_runs, axis=0), "rate_std": np.std(rate_runs, axis=0),
        "eff_mean": np.mean(eff_runs, axis=0), "eff_std": np.std(eff_runs, axis=0),
    }
    if prob_runs:
        prob_runs = np.stack(prob_runs, axis=0)
        out["prob_mean"] = np.mean(prob_runs, axis=0)
        out["prob_std"] = np.std(prob_runs, axis=0)
    return out


def plot_mutation_attempt_stacked(gen, share_mean, save="mutation_attempt_stacked.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.stackplot(gen, share_mean[0], share_mean[1], share_mean[2], share_mean[3],
                  labels=["Add", "Delete", "Modify", "Mode"], alpha=0.85)
    plt.margins(0, 0); plt.ylim(0, 1.0)
    plt.xlabel("Generation"); plt.ylabel("Attempt Share")
    plt.title("Mutation Roulette: Attempt Share (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout(); plt.savefig(save); plt.close()
    print("Saved:", save)


def plot_mutation_adaptive_prob_stacked(gen, prob_mean, save="mutation_adaptive_prob_stacked.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.stackplot(gen, prob_mean[0], prob_mean[1], prob_mean[2], prob_mean[3],
                  labels=["Add", "Delete", "Modify", "Mode"], alpha=0.85)
    plt.margins(0, 0); plt.ylim(0, 1.0)
    plt.xlabel("Generation"); plt.ylabel("Selection Probability")
    plt.title("Adaptive Roulette: Operator Selection Probability (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout(); plt.savefig(save); plt.close()
    print("Saved:", save)


def plot_mutation_success_rate_2x2(gen, rate_mean, rate_std, save="mutation_success_rate.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300, sharex=True, sharey=True)
    axes = axes.ravel()
    for i, op in enumerate(OPS):
        ax = axes[i]
        m = rate_mean[i]; s = rate_std[i]
        ax.plot(gen, m, linewidth=2.5, label=f"{op} mean")
        ax.fill_between(gen, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), alpha=0.2)
        ax.set_title(op.upper()); ax.grid(True, linestyle=":", alpha=0.6)
    fig.suptitle("Mutation Operators: Success Rate (mean ± std over runs)", y=0.98)
    fig.text(0.5, 0.03, "Generation", ha="center")
    fig.text(0.03, 0.5, "Success Rate", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.06, 1, 0.95])
    plt.savefig(save); plt.close(fig)
    print("Saved:", save)


def plot_mutation_effective_contribution(gen, eff_mean, save="mutation_effective_contribution.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    for i, op in enumerate(OPS):
        plt.plot(gen, eff_mean[i], linewidth=2.0, label=op)
    plt.ylim(0, 1.0); plt.xlabel("Generation"); plt.ylabel("Attempt × Success")
    plt.title("Mutation Operators: Effective Contribution (mean over runs)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout(); plt.savefig(save); plt.close()
    print("Saved:", save)


def plot_convergence_curves(gen, hv_mean, hv_std, igd_mean, igd_std, save="convergence_hv_igd.png"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)
    ax.plot(gen, hv_mean, linewidth=2.0, label="HV (normalised, mean)")
    ax.fill_between(gen, hv_mean - hv_std, hv_mean + hv_std, alpha=0.2, label="HV ± std")
    ax.set_xlabel("Generation"); ax.set_ylabel("HV (normalised)")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(gen, igd_mean, linewidth=2.0, linestyle="--", label="IGD+ (mean)")
    ax2.fill_between(gen, np.maximum(igd_mean - igd_std, 0), igd_mean + igd_std, alpha=0.15)
    ax2.set_ylabel("IGD+ (lower is better)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.title("Convergence: HV (normalised) and IGD+ (mean ± std over runs)")
    plt.tight_layout(); plt.savefig(save); plt.close(fig)
    print("Saved:", save)


def plot_spacing_curve(gen, sp_mean, sp_std, save="diversity_spacing.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, sp_mean, linewidth=2.0, label="Spacing (mean)")
    plt.fill_between(gen, np.maximum(sp_mean - sp_std, 0), sp_mean + sp_std, alpha=0.2, label="± std")
    plt.xlabel("Generation"); plt.ylabel("Spacing (lower is better)")
    plt.title("Diversity: Spacing (mean ± std over runs)")
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend()
    plt.tight_layout(); plt.savefig(save); plt.close()
    print("Saved:", save)


def plot_feasible_ratio(gen, fr_mean, fr_std, save="feasible_ratio.png",
                        title="Constraint Handling: Feasible Ratio (mean ± std over runs)",
                        ylabel="Feasible Ratio"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, fr_mean, linewidth=2.0, label="Feasible Ratio (mean)", zorder=3)
    plt.fill_between(gen, np.clip(fr_mean - fr_std, 0, 1), np.clip(fr_mean + fr_std, 0, 1),
                     alpha=0.2, label="± std", zorder=2)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Generation"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend()
    plt.tight_layout(); plt.savefig(save); plt.close()
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
    plt.xlabel("Generation"); plt.ylabel("Mean Violation Components (stacked)")
    plt.title("Diagnostics Breakdown (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout(); plt.savefig(save); plt.close()
    print("Saved:", save)


def plot_hv_curve(gen, hv_mean, hv_std=None, save="hv_curve.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, hv_mean, linewidth=2.2, label="HV (normalised, mean)", zorder=3)
    if hv_std is not None:
        plt.fill_between(gen, hv_mean - hv_std, hv_mean + hv_std, alpha=0.2, label="HV ± std", zorder=2)
    plt.xlabel("Generation"); plt.ylabel("HV (normalised)")
    plt.title("Hypervolume Curve (normalised, mean ± std over runs)")
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend()
    plt.tight_layout(); plt.savefig(save); plt.close()
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
    plt.xlabel("Generation"); plt.ylabel("Feasible Ratio")
    plt.title("Feasible Ratio Curve (mean ± std over runs)")
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend()
    plt.tight_layout(); plt.savefig(save); plt.close()
    print("Saved:", save)


def _pick_overlay_gens(generations: int, n: int = 5) -> List[int]:
    if generations <= 1:
        return [0]
    cand = [0, generations // 4, generations // 2, (3 * generations) // 4, generations - 1]
    cand = sorted(list(dict.fromkeys([int(x) for x in cand if 0 <= int(x) < generations])))
    return cand


def plot_2d_evolution_overlays(front_hist, generations, save_prefix="evolution2d", gens_to_plot=None):
    if not front_hist:
        print("[WARN] Empty front_hist, skip 2D evolution overlays.")
        return
    if gens_to_plot is None:
        gens_to_plot = _pick_overlay_gens(generations, n=5)
    pairs = [
        (0, 1, "Cost", "Emission (gCO2)", f"{save_prefix}_cost_emis.png"),
        (0, 2, "Cost", "Time", f"{save_prefix}_cost_time.png"),
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
            continue
        plt.xlabel(xl); plt.ylabel(yl)
        plt.title(f"2D Evolution Overlay (Best Run): {xl} vs {yl}")
        plt.grid(True, linestyle=":", alpha=0.5); plt.legend(frameon=True)
        plt.tight_layout(); plt.savefig(outname); plt.close()
        print("Saved:", outname)


def extract_min_objectives(front_hist, generations):
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
        try:
            min_cost[g] = float(np.nanmin(arr[:, 0]))
            min_emis[g] = float(np.nanmin(arr[:, 1]))
            min_time[g] = float(np.nanmin(arr[:, 2]))
        except Exception:
            continue
    return _ffill_nan(min_cost), _ffill_nan(min_emis), _ffill_nan(min_time)


def plot_min_objective_curves(gen, min_cost, min_emis, min_time,
                              save_cost="min_cost_curve.png",
                              save_emis="min_emission_curve.png",
                              save_time="min_time_curve.png"):
    for data, ylabel, title, save in [
        (min_cost, "Min Cost (feasible only)", "Best-run Min Cost per Generation (feasible solutions)", save_cost),
        (min_emis, "Min Emission gCO2 (feasible only)", "Best-run Min Emission per Generation (feasible solutions)", save_emis),
        (min_time, "Min Time h (feasible only)", "Best-run Min Time per Generation (feasible solutions)", save_time),
    ]:
        plt.figure(figsize=(10, 4), dpi=300)
        # NaN 段不绘制，体现早期无可行解阶段
        plt.plot(gen, data, linewidth=2.2)
        # 标注首次出现可行解的代数
        finite_mask = np.isfinite(data)
        if finite_mask.any():
            first_feas_gen = int(np.where(finite_mask)[0][0])
            if first_feas_gen > 0:
                plt.axvline(x=first_feas_gen, color='green', linestyle='--', alpha=0.6,
                            label=f'First feasible gen {first_feas_gen}')
                plt.legend(fontsize=9)
        plt.xlabel("Generation"); plt.ylabel(ylabel); plt.title(title)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout(); plt.savefig(save); plt.close()
        print("Saved:", save)


def export_gen_metrics_best_run(front_hist, hv_hist, fr_hist, frs_hist, vio_hist, generations,
                                out_csv="gen_metrics_best_run.csv"):
    min_cost, min_emis, min_time = extract_min_objectives(front_hist, generations)
    df = pd.DataFrame({
        "gen": np.arange(generations, dtype=int),
        "HV_norm": np.array(hv_hist[:generations], dtype=float),
        "feasible_ratio_soft": np.array(fr_hist[:generations], dtype=float),
        "feasible_ratio_strict": np.array(frs_hist[:generations], dtype=float),
        "min_cost": min_cost, "min_emission_gCO2": min_emis, "min_time": min_time,
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

def plot_pareto_3d_final_only(pareto_points, save, title):
    if not pareto_points:
        print("[WARN] No final Pareto points to plot.")
        return
    A = _finite_points_array(pareto_points)
    if A.shape[0] == 0:
        return
    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], marker="o", s=30, alpha=0.9)
    ax.set_xlabel("Cost"); ax.set_ylabel("Emission (gCO2)"); ax.set_zlabel("Time")
    ax.set_title(title); ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout(); plt.savefig(save); plt.close(fig)
    print("Saved:", save)


def plot_pareto_3d_all_generations(all_gen_points_with_gen, final_pareto_points=None,
                                   save="pareto_3d_allgens_best_run.png",
                                   title="Pareto Points Across All Generations (Best Run)",
                                   cmap_name="turbo"):
    if not all_gen_points_with_gen:
        return
    raw = np.array([(c, e, t, g) for (c, e, t, g) in all_gen_points_with_gen], dtype=float)
    mask = np.all(np.isfinite(raw[:, :3]), axis=1)
    raw = raw[mask]
    if raw.shape[0] == 0:
        return
    P = raw[:, :3]; G = raw[:, 3]
    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=G, cmap=cmap_name, s=10, alpha=0.55)
    cbar = plt.colorbar(sc, ax=ax, pad=0.10, fraction=0.04)
    cbar.set_label("Generation")
    gmin, gmax = int(np.min(G)), int(np.max(G))
    cbar.set_ticks(np.linspace(gmin, gmax, num=6).astype(int))
    if final_pareto_points:
        F = _finite_points_array(final_pareto_points)
        if F.shape[0] > 0:
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], marker="^", s=40, alpha=0.95, label="Final feasible Pareto")
            ax.legend(loc="best")
    ax.set_xlabel("Cost"); ax.set_ylabel("Emission (gCO2)"); ax.set_zlabel("Time")
    ax.set_title(title); ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout(); plt.savefig(save); plt.close(fig)
    print("Saved:", save)


# ========================
# Runner (one run)
# ========================

def run_nsga2_analytics(filename="data.xlsx", pop_size=75, generations=160):
    print("Loading data...")
    (node_names, node_region, node_caps,
     arcs, timetables, batches,
     waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
     carbon_tax_map, emission_factor_map, mode_speeds_map, trans_map) = load_network_from_extended(filename)

    tt_dict = build_timetable_dict(timetables)
    arc_lookup = build_arc_lookup(arcs)

    print("Building path library...")
    # ✅ 传入 node_region
    path_lib = build_path_library(node_names, node_region, arcs, batches, tt_dict, arc_lookup)
    sanity_check_path_lib(batches, path_lib)

    init_probs = _normalise_probs({"add": W_ADD, "del": W_DEL, "mod": W_MOD, "mode": W_MODE})
    roulette = AdaptiveRoulette(
        ops=OPS, init_probs=init_probs,
        ema_alpha=ROULETTE_EMA_ALPHA, min_prob=ROULETTE_MIN_PROB, score_eps=ROULETTE_SCORE_EPS
    )

    population: List[Individual] = []
    n_greedy = max(1, pop_size // 3)  # 前1/3用贪心初始化（最快路径）
    for i in range(pop_size):
        if i < n_greedy:
            ind = greedy_initial_individual(batches, path_lib)
        else:
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
    vio_mean_hist: Dict[str, List[float]] = {
        k: [] for k in ["miss_alloc", "miss_tt", "cap_excess", "node_cap_excess", "late_h", "wait_h"]
    }
    mut_tracker = {
        "attempt": {op: [0] * generations for op in OPS},
        "success": {op: [0] * generations for op in OPS},
        "prob": {op: [0.0] * generations for op in OPS},
    }

    for gen in range(generations):
        for op in OPS:
            mut_tracker["prob"][op][gen] = float(roulette.prob[op])

        fronts = non_dominated_sort(population)
        front0 = fronts[0]
        feasible_front0 = [ind for ind in front0 if ind.feasible]
        base_front = feasible_front0 if feasible_front0 else front0
        front0_unique = unique_individuals_by_objectives(base_front, tol=1e-3)

        front_hist_objs.append([
            ind.objectives for ind in front0_unique if ind.feasible
        ])
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
            evaluate_individual(c1, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps,
                                carbon_tax_map=carbon_tax_map, trans_map=trans_map)
            evaluate_individual(c2, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps,
                                carbon_tax_map=carbon_tax_map, trans_map=trans_map)

            if random.random() < MUTATION_RATE:
                snap = deepcopy(c1)
                op, ok, suc = mutate_roulette_adaptive(
                    c1, batches, path_lib, tt_dict, arc_lookup,
                    roulette, snap, arcs,
                    waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps,
                    carbon_tax_map=carbon_tax_map, trans_map=trans_map
                )
                mut_tracker["attempt"][op][gen] += 1
                mut_tracker["success"][op][gen] += (1 if suc else 0)

            if random.random() < MUTATION_RATE:
                snap = deepcopy(c2)
                op, ok, suc = mutate_roulette_adaptive(
                    c2, batches, path_lib, tt_dict, arc_lookup,
                    roulette, snap, arcs,
                    waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps,
                    carbon_tax_map=carbon_tax_map, trans_map=trans_map
                )
                mut_tracker["attempt"][op][gen] += 1
                mut_tracker["success"][op][gen] += (1 if suc else 0)

            repair_missing_allocations(c1, batches, path_lib)
            repair_missing_allocations(c2, batches, path_lib)
            evaluate_individual(c1, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps,
                                carbon_tax_map=carbon_tax_map, trans_map=trans_map)
            evaluate_individual(c2, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, node_caps,
                                carbon_tax_map=carbon_tax_map, trans_map=trans_map)

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
            print(f"Saved 0 feasible Pareto solutions to {filename}")
            return
        for i, ind in enumerate(pareto):
            cost, emis, t = ind.objectives
            f.write(f"===== Pareto Sol {i} =====\n")
            f.write(f"Objectives: Cost={cost:.6f}, Emission_gCO2={emis:.6f}, Time={t:.6f}, "
                    f"Penalty={ind.penalty:.6f}, Feasible={ind.feasible}, "
                    f"FeasibleHardNoLate={ind.feasible_hard}, Breakdown={ind.vio_breakdown}\n\n")
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
                "batch_id": int(b.batch_id), "origin": b.origin, "destination": b.destination, "paths": []
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
    filename = "data.xlsx"
    pop_size = 75
    generations = 160
    runs = 30

    run_front_hist = []
    run_feasible_ratio = []
    run_feasible_ratio_strict = []
    run_vio_mean = []
    mut_runs = []
    run_rows = []
    run_paretos: List[List[Individual]] = []
    run_batches_list: List[List[Batch]] = []

    print(f"[CONFIG] HARD_TIME_WINDOW={HARD_TIME_WINDOW}")
    print(f"[CONFIG] CHINA_BORDER_NODES={CHINA_BORDER_NODES}")
    print(f"[CONFIG] Waiting default: cost_per_teu_h={WAITING_COST_PER_TEU_HOUR_DEFAULT}, "
          f"emis_g_per_teu_h={WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT}")
    print(f"[CONFIG] Lateness in cost: LATE_COST_PER_TEU_HOUR={LATE_COST_PER_TEU_HOUR:.3e}")
    print(f"[CONFIG] Penalties: MISS_ALLOC={PEN_MISS_ALLOC:.2e}, MISS_TT={PEN_MISS_TT:.2e}, "
          f"ARC_CAP={PEN_CAP_EXCESS_PER_TEU:.2e}, NODE_CAP={PEN_NODE_CAP_EXCESS_PER_TEU:.2e}")
    print(f"[CONFIG] Path lib: topK={PATHS_TOPK_PER_CRITERION}, cap_total={PATH_LIB_CAP_TOTAL}, "
          f"dfs_pool={DFS_MAX_PATHS_PER_OD}")
    print(f"[CONFIG] GA rates: CROSSOVER_RATE={CROSSOVER_RATE}, MUTATION_RATE={MUTATION_RATE}, "
          f"segment_prob={CROSSOVER_SEGMENT_PROB}")
    print(f"[CONFIG] HV: ref={HV_REF_NORM}, samples={HV_SAMPLES}, HV_EVERY={HV_EVERY}, MC_seed={HV_MC_SEED}")

    for run_id in range(runs):
        seed = 1000 + run_id
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n========== RUN {run_id}/{runs-1}, seed={seed} ==========")
        t0 = time.perf_counter()

        pop, pareto, batches, front_hist, fr_hist, fr_strict_hist, vio_hist, mut_tracker = run_nsga2_analytics(
            filename=filename, pop_size=pop_size, generations=generations
        )

        t1 = time.perf_counter()
        runtime_s = float(t1 - t0)

        run_front_hist.append(front_hist)
        run_feasible_ratio.append(fr_hist)
        run_feasible_ratio_strict.append(fr_strict_hist)
        run_vio_mean.append(vio_hist)
        mut_runs.append(mut_tracker)
        run_paretos.append(pareto)
        run_batches_list.append(batches)

        run_rows.append({
            "run_id": run_id, "seed": seed,
            "runtime_s": runtime_s, "runtime_min": runtime_s / 60.0,
            "final_feasible_ratio": float(fr_hist[-1]) if fr_hist else 0.0,
            "final_strict_feasible_ratio": float(fr_strict_hist[-1]) if fr_strict_hist else 0.0,
            "final_pareto_size": int(len(pareto))
        })
        print(f"[RUN {run_id}] Pareto={len(pareto)}, Runtime={runtime_s:.2f}s, "
              f"FinalFeas={fr_hist[-1]:.2%}, StrictFeas={fr_strict_hist[-1]:.2%}")

    df_runs = pd.DataFrame(run_rows).sort_values("run_id").reset_index(drop=True)

    # ========================
    # Build P* and normalisation
    # ========================
    t_ps = time.perf_counter()
    P_star = build_P_star_fast(run_front_hist, tail_gens=PSTAR_TAIL_GENS,
                               cap_per_gen=PSTAR_CAP_PER_GEN, max_total=PSTAR_MAX_TOTAL)
    print(f"\n[P*] Size = {len(P_star)} | time={time.perf_counter()-t_ps:.2f}s")

    if len(P_star) > 0:
        P_arr = np.array(P_star, dtype=float)
        mins = np.min(P_arr, axis=0)
        maxs = np.max(P_arr, axis=0)
    else:
        mins = np.zeros(3, dtype=float)
        maxs = np.ones(3, dtype=float)

    # ========================
    # HV (normalised)
    # ========================
    hv_calc_norm = HypervolumeCalculator(ref_point=HV_REF_NORM, num_samples=HV_SAMPLES, seed=HV_MC_SEED)
    hv_norm_runs = []
    for r in range(runs):
        hv_hist = []
        last_hv = 0.0
        for gi, gen_front in enumerate(run_front_hist[r]):
            if gi % HV_EVERY == 0:
                finite_pts = [tuple(x) for x in _finite_points_array(gen_front)]
                An = normalize_points(finite_pts, mins, maxs) if finite_pts else []
                An = clip_points(An, HV_REF_NORM)
                last_hv = hv_calc_norm.calculate_points(An) if An else 0.0
            hv_hist.append(last_hv)
        hv_norm_runs.append(hv_hist)

    hv_runs = np.array(hv_norm_runs, dtype=float)
    hv_mean, hv_std = np.mean(hv_runs, axis=0), np.std(hv_runs, axis=0)
    df_runs["final_HV_norm"] = hv_runs[:, -1]

    print("\n=========== RUN TIME SUMMARY ===========")
    print(df_runs.to_string(index=False))

    with pd.ExcelWriter("run_time_summary.xlsx", engine="openpyxl") as writer:
        df_runs.to_excel(writer, sheet_name="RunSummary", index=False)
    print("Saved: run_time_summary.xlsx")

    # ========================
    # IGD+/Spacing
    # ========================
    Pn = normalize_points(P_star, mins, maxs) if P_star else []
    igd_runs, sp_runs = [], []
    for r in range(runs):
        igd_hist, sp_hist = [], []
        last_igd, last_sp = float("inf"), 0.0
        for gi, gen_front in enumerate(run_front_hist[r]):
            if gi % METRIC_EVERY == 0:
                finite_pts = [tuple(x) for x in _finite_points_array(gen_front)]
                An = normalize_points(finite_pts, mins, maxs) if finite_pts else []
                last_igd = igd_plus(Pn, An) if (Pn and An) else float("inf")
                last_sp = spacing_metric(An) if An else 0.0
            igd_hist.append(last_igd)
            sp_hist.append(last_sp)
        igd_runs.append(igd_hist)
        sp_runs.append(sp_hist)

    igd_runs = np.array(igd_runs, dtype=float)
    sp_runs = np.array(sp_runs, dtype=float)
    fr_runs = np.array(run_feasible_ratio, dtype=float)
    frs_runs = np.array(run_feasible_ratio_strict, dtype=float)

    gen = np.arange(generations)
    igd_mean, igd_std = np.mean(igd_runs, axis=0), np.std(igd_runs, axis=0)
    sp_mean, sp_std = np.mean(sp_runs, axis=0), np.std(sp_runs, axis=0)
    fr_mean, fr_std = np.mean(fr_runs, axis=0), np.std(fr_runs, axis=0)
    frs_mean, frs_std = np.mean(frs_runs, axis=0), np.std(frs_runs, axis=0)

    vio_keys = ["miss_alloc", "miss_tt", "cap_excess", "node_cap_excess", "late_h", "wait_h"]
    vio_mean_dict_mean = {k: [0.0] * generations for k in vio_keys}
    for k in vio_keys:
        mat = np.array([run_vio_mean[r][k] for r in range(runs)], dtype=float)
        vio_mean_dict_mean[k] = list(np.mean(mat, axis=0))

    mut_agg = aggregate_mutation_over_runs(mut_runs, generations)

    def summarize_final(mat):
        mat = np.array(mat, dtype=float)
        final = mat[:, -1]
        return float(np.min(final)), float(np.max(final)), float(np.mean(final)), float(np.std(final))

    hv_best, hv_worst, hv_m, hv_s = summarize_final(hv_runs)
    igd_best, igd_worst, igd_m, igd_s = summarize_final(igd_runs)
    sp_best, sp_worst, sp_m, sp_s = summarize_final(sp_runs)
    fr_best, fr_worst, fr_m, fr_s = summarize_final(fr_runs)
    frs_best, frs_worst, frs_m, frs_s = summarize_final(frs_runs)

    best_run_idx = int(np.argmax(hv_runs[:, -1]))
    best_run_hv = float(hv_runs[best_run_idx, -1])

    print("\n=========== SUMMARY OVER RUNS (Final Generation) ===========")
    print(f"Runs: {runs}")
    print(f"HV_norm (↑) | best={hv_best:.4f}, worst={hv_worst:.4f}, mean={hv_m:.4f}, std={hv_s:.4f}")
    print(f"IGD+    (↓) | best={igd_best:.4f}, worst={igd_worst:.4f}, mean={igd_m:.4f}, std={igd_s:.4f}")
    print(f"Spacing (↓) | best={sp_best:.4f}, worst={sp_worst:.4f}, mean={sp_m:.4f}, std={sp_s:.4f}")
    print(f"FeasRatio soft  | best={fr_best:.2%}, worst={fr_worst:.2%}, mean={fr_m:.2%}, std={fr_s:.2%}")
    print(f"FeasRatio strict| best={frs_best:.2%}, worst={frs_worst:.2%}, mean={frs_m:.2%}, std={frs_s:.2%}")
    print(f"Best run: #{best_run_idx}, HV_norm={best_run_hv:.4f}")
    print("=============================================================\n")

    best_pareto = run_paretos[best_run_idx]
    best_batches = run_batches_list[best_run_idx]
    best_front_hist = run_front_hist[best_run_idx]

    if best_pareto and best_batches:
        for i, ind in enumerate(best_pareto[:3]):
            print_pure_structure(ind, best_batches, f"BestRun Pareto Sol {i}")
        save_pareto_solutions(best_pareto, best_batches, filename="result.txt")
        export_pareto_points_json(best_pareto, best_batches, out_json="nsga_pareto_points.json")

    best_final_points = []
    if best_pareto:
        best_final_points = unique_objective_tuples(
            [ind.objectives for ind in best_pareto if ind.feasible], tol=1e-9
        )

    plot_pareto_3d_final_only(best_final_points, save="pareto_3d_best_run_final.png",
                               title=f"Pareto Front (Final Gen) - Best Run #{best_run_idx}")

    all_points_best_with_gen = []
    if best_front_hist:
        seen = set()
        for g, gen_pts in enumerate(best_front_hist):
            for p in gen_pts:
                key = (round(p[0], 6), round(p[1], 6), round(p[2], 6), g)
                if key not in seen:
                    seen.add(key)
                    all_points_best_with_gen.append((p[0], p[1], p[2], g))

    plot_pareto_3d_all_generations(
        all_gen_points_with_gen=all_points_best_with_gen,
        final_pareto_points=best_final_points if best_final_points else None,
        save="pareto_3d_allgens_best_run.png",
        title=f"Pareto Points Across {generations} Generations (Best Run #{best_run_idx})"
    )

    # Mutation diagnostics
    plot_mutation_attempt_stacked(gen, mut_agg["share_mean"])
    plot_mutation_success_rate_2x2(gen, mut_agg["rate_mean"], mut_agg["rate_std"])
    plot_mutation_effective_contribution(gen, mut_agg["eff_mean"])
    if "prob_mean" in mut_agg:
        plot_mutation_adaptive_prob_stacked(gen, mut_agg["prob_mean"])

    # Convergence & diversity
    plot_convergence_curves(gen, hv_mean, hv_std, igd_mean, igd_std)
    plot_spacing_curve(gen, sp_mean, sp_std)
    plot_feasible_ratio(gen, fr_mean, fr_std, save="feasible_ratio_soft.png",
                        title="Constraint Handling: Feasible Ratio (soft)", ylabel="Feasible Ratio (soft)")
    plot_feasible_ratio(gen, frs_mean, frs_std, save="feasible_ratio_strict.png",
                        title="Constraint Handling: Feasible Ratio (strict no-late)", ylabel="Feasible Ratio (strict)")
    plot_violation_breakdown_stacked(gen, vio_mean_dict_mean)
    plot_hv_curve(gen, hv_mean, hv_std)
    plot_feasible_ratio_curve(gen, fr_mean, fr_std, frs_mean=frs_mean, frs_std=frs_std)

    if best_front_hist:
        plot_2d_evolution_overlays(best_front_hist, generations, save_prefix="evolution2d_best_run",
                                   gens_to_plot=_pick_overlay_gens(generations))
        min_cost, min_emis, min_time = extract_min_objectives(best_front_hist, generations)
        plot_min_objective_curves(gen, min_cost, min_emis, min_time)

    export_gen_metrics_best_run(
        front_hist=best_front_hist,
        hv_hist=list(hv_runs[best_run_idx, :]),
        fr_hist=run_feasible_ratio[best_run_idx],
        frs_hist=run_feasible_ratio_strict[best_run_idx],
        vio_hist=run_vio_mean[best_run_idx],
        generations=generations
    )

    print("\n✅ All outputs saved.")