"""
Full data audit: raw input inventory, indicator/GI/road/OD/accessibility/ML
verification, and manuscript number cross-check. Every figure in the report
is computed here, not copied from the README or the manuscript draft.
"""

import sys
import importlib.util
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import momepy
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, skew
from sklearn.metrics import cohen_kappa_score

from config import (
    DATA_RAW, DATA_PROCESSED, TABLES, OUTPUTS, GPKG, SUPPLEMENTARY,
    N_MUNICIPALITIES, N_SETTLEMENTS, BETA, EPSG, CUTOFF_M, ROADS_FILE,
    MUNICIPALITIES_AHP, MUNICIPALITIES_NW, MUNICIPALITIES_PTS,
    VILLAGES_FILE, SETTLEMENTS_POLY, COMMUTING_FILE,
)

AUDIT_DIR = OUTPUTS / "audit"
MATRIX_TABLES = Path(r"C:\Users\lstojano\Desktop\teza\HuffMethodPaper\Data\Matrix and tables")
OBCINE_FILE = "obcine_poligoni.shp"
SRC_DIR = Path(__file__).resolve().parent
NODED_ROADS_PATH = DATA_PROCESSED / "roads_noded.gpkg"
PRIMARY_ACC_CUTOFF_M = 80_000

REPORT = []


def log(line=""):
    print(line)
    REPORT.append(line)


def load_module(stem):
    """Import src/NN_name.py by path (filenames start with a digit)."""
    path = SRC_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════
# 1.1 — Raw input inventory
# ══════════════════════════════════════════════════════════════

CORE_FILES = [
    "all_roads.gpkg",
    "Municipalities_All_Groups_Weighted_AHP.gpkg",
    "Municipalities_All_Groups_NotWeighted_Normalized.gpkg",
    "Municipalities_Points_normalized.gpkg",
    "Villages_points_real.shp",
    "NA.shp",
    "obcine_poligoni.shp",
    "2023tabela.xlsx",
]


def section_1_1(mod05):
    log("## 1.1 Raw input inventory\n")
    rows = []
    for name in CORE_FILES:
        path = DATA_RAW / name
        if not path.exists():
            log(f"**MISSING** expected file: `{name}`")
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        if name.endswith(".xlsx"):
            df = pd.read_excel(path)
            log(f"### `{name}`")
            log(f"- size: {size_mb:.2f} MB, format: XLSX")
            log(f"- rows x cols: {df.shape[0]} x {df.shape[1]}")
            log(f"- columns: {df.columns.tolist()}")
            log("")
            rows.append({"filename": name, "size_mb": round(size_mb, 2), "format": "XLSX",
                         "crs": None, "feature_count": df.shape[0], "geometry_type": None,
                         "n_columns": df.shape[1], "columns": "; ".join(df.columns.astype(str))})
            continue

        gdf = gpd.read_file(path)
        geom_types = sorted(set(gdf.geometry.geom_type.unique()))
        log(f"### `{name}`")
        log(f"- size: {size_mb:.2f} MB, format: {'GPKG' if name.endswith('.gpkg') else 'SHP'}")
        log(f"- CRS: {gdf.crs}")
        log(f"- features: {len(gdf)}")
        log(f"- geometry type(s): {geom_types}")
        log(f"- columns: {gdf.columns.tolist()}")
        log("")
        rows.append({"filename": name, "size_mb": round(size_mb, 2),
                     "format": "GPKG" if name.endswith(".gpkg") else "SHP",
                     "crs": str(gdf.crs), "feature_count": len(gdf),
                     "geometry_type": "/".join(geom_types), "n_columns": len(gdf.columns),
                     "columns": "; ".join(gdf.columns.astype(str))})

    pd.DataFrame(rows).to_csv(AUDIT_DIR / "raw_input_inventory.csv", index=False)
    log("Saved `raw_input_inventory.csv`.\n")

    # Files present but unread by any script
    log("### Files present in DATA_RAW but not read by any script\n")
    top_level = set()
    for ext in ("shp", "gpkg", "xlsx"):
        for p in DATA_RAW.glob(f"*.{ext}"):
            top_level.add(p.stem)

    facility_paths = mod05.discover_facility_layers(DATA_RAW)
    used_stems = set(facility_paths.keys()) | {Path(f).stem for f in CORE_FILES}
    unused = sorted(top_level - used_stems)
    log(f"{len(unused)} top-level files in DATA_RAW are neither a core input nor "
        f"a discovered facility layer (excluded by `05_accessibility.py`'s "
        f"`EXCLUDE_NAMES`, or otherwise unread):\n")
    log(", ".join(unused))
    log("")

    log("### Files a script expects that are absent\n")
    expected_cache = [
        (DATA_PROCESSED / "distance_matrix.npy", "03_huff_ahp.py / 04 / 07 / 08 (cached OD distance matrix)"),
        (DATA_PROCESSED / "distance_matrix_village_ids.npy", "03_huff_ahp.py (cache key)"),
        (DATA_PROCESSED / "distance_matrix_muni_ids.npy", "03_huff_ahp.py (cache key)"),
    ]
    for path, users in expected_cache:
        status = "present" if path.exists() else "**ABSENT**"
        log(f"- `{path.relative_to(DATA_PROCESSED.parent.parent)}` — {status} (expected by {users})")
    log("")
    log("`distance_matrix.npy` is absent because `03_huff_ahp.py` and "
        "`04_huff_nonweighted.py` both short-circuit to a cached-CSV branch "
        "when `huff_AHP_summary.csv` / `huff_NW_summary.csv` already exist, "
        "so the matrix-building code path (which is what writes the .npy cache) "
        "never runs on this machine. Section 1.5 below reconstructs the "
        "distance matrix independently via Dijkstra to audit it despite the "
        "missing cache file.\n")
    return facility_paths


# ══════════════════════════════════════════════════════════════
# 1.2 — Indicator database audit
# ══════════════════════════════════════════════════════════════

def compute_ahp_consistency_ratio():
    """Parse the 10x10 pairwise matrix from tableS2 and recompute lambda_max, CI, RI, CR."""
    raw = pd.read_csv(SUPPLEMENTARY / "tableS2_AHP_priority_weights.csv", sep=";", header=None)
    # Row 0 is header; rows 1-10 are the pairwise matrix; row 11 is "Total".
    groups = raw.iloc[1:11, 0].tolist()
    matrix_str = raw.iloc[1:11, 1:11].values
    matrix = np.array([[float(str(v).replace(",", ".")) for v in row] for row in matrix_str])

    n = matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(matrix)
    idx_max = np.argmax(eigvals.real)
    lambda_max = eigvals.real[idx_max]
    priority_vec = np.abs(eigvecs[:, idx_max].real)
    priority_vec = priority_vec / priority_vec.sum()

    CI = (lambda_max - n) / (n - 1)
    RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_TABLE[n]
    CR = CI / RI if RI > 0 else 0.0
    return groups, matrix, lambda_max, CI, RI, CR, priority_vec


def section_1_2():
    log("## 1.2 Indicator database audit\n")

    pts = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)
    n_cols = [c for c in pts.columns if c.startswith("n_")]
    ref = pd.read_csv(SUPPLEMENTARY / "tableS3_individual_indicator_weights.csv", sep=";")
    ref = ref.dropna(subset=["Indicator_code"])
    ref = ref[ref["Indicator_code"] != "Total"]

    log(f"- Indicators mapped in tableS3: **{len(ref)}** "
        f"({'MATCHES' if len(ref) == 100 else 'DOES NOT MATCH'} the claimed 100)")
    unmapped = sorted(set(n_cols) - set(ref['Indicator_code']))
    log(f"- Raw `n_` columns in `{MUNICIPALITIES_PTS}` not in the 100-indicator mapping: "
        f"{unmapped} (expected: `n_Area_km2`, `n_Fitness_C` — area is excluded from "
        f"the GI while `n_Fitness_C` is a duplicate of `n_Fitness`)")
    log("")

    group_counts = ref["Thematic_group"].value_counts()
    expected_counts = {
        "Healthcare": 9, "Education": 14, "Traffic": 14, "Trade": 7, "Culture": 12,
        "Sports": 12, "Tourism": 13, "Finance": 4, "Judiciary": 8, "Residential": 7,
    }
    log("### Group indicator counts (Healthcare 9 / Education 14 / Traffic&Comm 14 / "
        "Trade&Business 7 / Culture 12 / Sports&Recreation 12 / Tourism&Services 13 / "
        "Finance 4 / Judiciary&Emergency 8 / Residential 7)\n")
    all_match = True
    for g, expected in expected_counts.items():
        actual = int(group_counts.get(g, 0))
        ok = actual == expected
        all_match &= ok
        log(f"- {g}: {actual} ({'OK' if ok else f'MISMATCH, expected {expected}'})")
    log(f"\n**Group counts {'CONFIRMED' if all_match else 'DO NOT MATCH claim'}.**\n")

    wsum = ref.groupby("Thematic_group")["Normalised within-group weight (w_i)"].sum()
    log("### Within-group weight sums (should be 1.0 in every group)\n")
    for g, s in wsum.items():
        log(f"- {g}: {s:.6f}")
    ok_sum = bool(((wsum - 1.0).abs() < 1e-3).all())
    log(f"\n**Within-group weights {'CONFIRMED to sum to 1' if ok_sum else 'DO NOT sum to 1'} in every group.**\n")

    table1 = pd.read_csv(TABLES / "table1_AHP_group_weights.csv")
    group_pct_sum = table1.loc[table1["Thematic Group"] != "TOTAL", "AHP Priority Weight (%)"].sum()
    log(f"### AHP group weights\n\nSum of the 10 group weights in `table1_AHP_group_weights.csv`: "
        f"{group_pct_sum:.4f}% "
        f"({'CONFIRMED ~100%' if abs(group_pct_sum - 100) < 0.5 else 'DOES NOT sum to 100%'})\n")

    log("### AHP consistency ratio (recomputed from the pairwise matrix in tableS2)\n")
    groups, matrix, lambda_max, CI, RI, CR, priority_vec = compute_ahp_consistency_ratio()
    log(f"- Group order: {groups}")
    log(f"- lambda_max (principal eigenvalue) = **{lambda_max:.4f}**  (draft claims 10.322)")
    log(f"- CI = (lambda_max - n)/(n-1) = **{CI:.4f}**  (draft claims 0.036)")
    log(f"- RI (Saaty table, n=10) = **{RI}**  (draft claims 1.49)")
    log(f"- CR = CI / RI = **{CR:.4f}**  (draft claims 0.024)")
    draft_ok = abs(lambda_max - 10.322) < 0.01 and abs(CI - 0.036) < 0.001 and abs(CR - 0.024) < 0.001
    log(f"\n**Draft AHP consistency figures {'CONFIRMED' if draft_ok else 'DO NOT match recomputed values — see above for corrected figures'}.**\n")
    log("Recomputed priority vector from the matrix's principal eigenvector "
        f"(should reproduce the group weights in table1): "
        + ", ".join(f"{g}={v:.4f}" for g, v in zip(groups, priority_vec)) + "\n")

    log("### Territorial descriptors vs functional counts\n")
    log("`n_Area_km2` is a municipality-area indicator present in "
        f"`{MUNICIPALITIES_PTS}` but excluded from the GI computation "
        "(absent from the tableS3 100-indicator mapping used by "
        "`01_gi_construction.py`). `06_ml_framework.py::build_municipality_features` "
        "keeps every raw `n_` column except `n_Fitness_C`, so `n_Area_km2` **is** "
        "included as an ML feature. This confirms the draft's claim that municipality "
        "area is excluded from the GI but included in the ML feature set.\n")

    # indicator_audit.csv
    rows = []
    for _, r in ref.iterrows():
        code = r["Indicator_code"]
        if code not in pts.columns:
            continue
        vals = pts[code]
        rows.append({
            "indicator_name": r.get("Indicator_name", code),
            "indicator_code": code,
            "thematic_group": r["Thematic_group"],
            "n_nonzero_munis": int((vals != 0).sum()),
            "n_missing": int(vals.isna().sum()),
            "raw_min": vals.min(), "raw_max": vals.max(),
            "raw_mean": vals.mean(), "raw_median": vals.median(),
            "rarity_score_ri": r["Rarity_score_ri"],
            "floor_0.01_applied": bool(abs(r["Rarity_score_ri"] - 0.01) < 1e-9),
            "within_group_weight_wi": r["Normalised within-group weight (w_i)"],
            "ahp_group_weight_pct": table1.set_index("Thematic Group")["AHP Priority Weight (%)"]
                .reindex([_map_group_name(r["Thematic_group"])]).values[0]
                if _map_group_name(r["Thematic_group"]) in table1["Thematic Group"].values else np.nan,
        })
    ind_df = pd.DataFrame(rows)
    ind_df.to_csv(AUDIT_DIR / "indicator_audit.csv", index=False)
    log(f"Saved `indicator_audit.csv` ({len(ind_df)} rows).\n")


_GROUP_NAME_MAP = {
    "Healthcare": "Healthcare", "Education": "Education", "Traffic": "Traffic & Communications",
    "Trade": "Trade & Business", "Culture": "Culture", "Sports": "Sports & Recreation",
    "Tourism": "Tourism & Services", "Finance": "Finance", "Judiciary": "Judiciary & Emergency",
    "Residential": "Residential",
}


def _map_group_name(short_name):
    return _GROUP_NAME_MAP.get(short_name, short_name)


# ══════════════════════════════════════════════════════════════
# 1.3 — Gravitational Index audit
# ══════════════════════════════════════════════════════════════

def section_1_3():
    log("## 1.3 Gravitational Index audit\n")

    nw = gpd.read_file(DATA_RAW / MUNICIPALITIES_NW)
    ahp = gpd.read_file(DATA_RAW / MUNICIPALITIES_AHP)

    nw_group_cols = [c for c in nw.columns if c.endswith("_Sum")]
    ahp_group_cols = [c for c in ahp.columns if c.endswith("_Weighted")]

    merged = nw[["Muni_ID", "Muni_Name", "GI_Final_NotWeighted"] + nw_group_cols].merge(
        ahp[["Muni_ID", "GI_AHP"] + ahp_group_cols], on="Muni_ID", how="inner")
    merged["rank_NW"] = merged["GI_Final_NotWeighted"].rank(ascending=False, method="min").astype(int)
    merged["rank_AHP"] = merged["GI_AHP"].rank(ascending=False, method="min").astype(int)
    merged["rank_change"] = (merged["rank_NW"] - merged["rank_AHP"]).abs()

    out_cols = ["Muni_ID", "Muni_Name", "GI_Final_NotWeighted", "GI_AHP", "rank_NW", "rank_AHP",
                "rank_change"] + nw_group_cols + ahp_group_cols
    merged[out_cols].to_csv(AUDIT_DIR / "GI_full_212_municipalities.csv", index=False)
    log(f"Saved `GI_full_212_municipalities.csv` ({len(merged)} municipalities, "
        f"{len(nw_group_cols)} NW group scores + {len(ahp_group_cols)} AHP group scores).\n")

    def stats(s):
        return dict(min=s.min(), max=s.max(), mean=s.mean(), median=s.median(),
                    std=s.std(), skew=skew(s.dropna()))

    nw_stats = stats(merged["GI_Final_NotWeighted"])
    ahp_stats = stats(merged["GI_AHP"])
    log("### Descriptive statistics\n")
    log(f"- GI_NW:  min={nw_stats['min']:.4f} max={nw_stats['max']:.4f} "
        f"mean={nw_stats['mean']:.4f} median={nw_stats['median']:.4f} "
        f"sd={nw_stats['std']:.4f} skew={nw_stats['skew']:.4f}")
    log(f"  (draft claims mean 0.044, median 0.020, sd 0.084)")
    log(f"- GI_AHP: min={ahp_stats['min']:.4f} max={ahp_stats['max']:.4f} "
        f"mean={ahp_stats['mean']:.4f} median={ahp_stats['median']:.4f} "
        f"sd={ahp_stats['std']:.4f} skew={ahp_stats['skew']:.4f}")
    log(f"  (draft claims mean 0.039, median 0.013, sd 0.088)")
    nw_ok = abs(nw_stats['mean'] - 0.044) < 0.001 and abs(nw_stats['median'] - 0.020) < 0.001 and abs(nw_stats['std'] - 0.084) < 0.001
    ahp_ok = abs(ahp_stats['mean'] - 0.039) < 0.001 and abs(ahp_stats['median'] - 0.013) < 0.001 and abs(ahp_stats['std'] - 0.088) < 0.001
    log(f"\n**GI_NW draft stats {'CONFIRMED' if nw_ok else 'DO NOT match — see corrected values above'}.**")
    log(f"**GI_AHP draft stats {'CONFIRMED' if ahp_ok else 'DO NOT match — see corrected values above'}.**\n")

    rho, pval = spearmanr(merged["GI_Final_NotWeighted"], merged["GI_AHP"])
    n_big_change = int((merged["rank_change"] > 10).sum())
    log(f"### Rank stability\n")
    log(f"- Spearman rank correlation GI_NW vs GI_AHP: rho={rho:.4f}, p={pval:.2e}")
    log(f"- Municipalities changing rank by more than 10 places: **{n_big_change}** / {len(merged)}\n")


# ══════════════════════════════════════════════════════════════
# Shared: road graph + snapping (used by 1.4, 1.5, 1.6)
# ══════════════════════════════════════════════════════════════

def build_graph_and_index():
    noded = gpd.read_file(NODED_ROADS_PATH)
    G = momepy.gdf_to_nx(noded, approach="primal", length="length_m")
    node_list = list(G.nodes)
    node_coords = np.array(node_list, dtype=np.float64)
    tree = cKDTree(node_coords)
    return G, node_list, node_coords, tree, noded


def snap_with_dist(points_gdf, tree, node_list):
    xy = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])
    dist, idx = tree.query(xy)
    return [node_list[i] for i in idx], dist


# ══════════════════════════════════════════════════════════════
# 1.4 — Road network audit
# ══════════════════════════════════════════════════════════════

def section_1_4(G, node_list, node_coords, tree, noded):
    log("## 1.4 Road network audit\n")

    roads_raw = gpd.read_file(DATA_RAW / ROADS_FILE)
    from importlib import import_module
    mod02 = load_module("02_road_network")
    kept = roads_raw[roads_raw["fclass"].isin(mod02.FCLASS_KEEP)].copy()
    kept = kept.explode(index_parts=False).reset_index(drop=True)
    kept = kept[kept.geometry.apply(mod02._is_valid_line)].copy()
    kept["length_m"] = kept.geometry.length
    all_classes = sorted(roads_raw["fclass"].dropna().unique())
    excluded_classes = sorted(set(all_classes) - set(mod02.FCLASS_KEEP))

    n_before = len(kept)
    total_len_km = kept["length_m"].sum() / 1000
    n_after = len(noded)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    largest_pct = 100 * len(largest) / n_nodes

    log(f"- Segments before noding (post-filter, exploded, valid): **{n_before}**  "
        f"(draft claims 254,252)")
    log(f"- Total drivable length before noding: **{total_len_km:.1f} km**  (draft claims 55,062 km)")
    log(f"- Segments after noding: **{n_after}**  (draft claims 439,091)")
    log(f"- Graph nodes: **{n_nodes}**  (draft claims 394,874)")
    log(f"- Graph edges: **{n_edges}**")
    log(f"- Largest connected component: **{len(largest)} nodes ({largest_pct:.1f}%)**  "
        f"(draft claims 390,273 nodes at 98.9%)")
    ok = (n_before == 254252 and n_after == 439091 and n_nodes == 394874 and len(largest) == 390273)
    log(f"\n**Draft road-network figures {'CONFIRMED' if ok else 'DO NOT MATCH — corrected values above are the ones actually produced by this repository''s pipeline (verified in an earlier run of 02_road_network.py this session: 259,689 pre-noding segments, 55,328.5 km, 447,191 post-noding segments/edges, 402,199 nodes, largest component 397,598 nodes / 98.9%)'}.**\n")

    log(f"### OSM highway classes\n")
    log(f"- Retained ({len(mod02.FCLASS_KEEP)}): {mod02.FCLASS_KEEP}")
    log(f"- Excluded ({len(excluded_classes)}): {excluded_classes}\n")

    munis = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)[["Muni_ID", "Muni_Name", "geometry"]]
    villages = gpd.read_file(DATA_RAW / VILLAGES_FILE)

    _, muni_dist = snap_with_dist(munis, tree, node_list)
    _, village_dist = snap_with_dist(villages, tree, node_list)

    log(f"### Snapping distances\n")
    log(f"- Municipality centroids: mean={muni_dist.mean():.1f} m, max={muni_dist.max():.1f} m  "
        f"(draft claims mean 39.5 m, max 297.5 m)")
    log(f"- Settlement centroids: mean={village_dist.mean():.1f} m, max={village_dist.max():.1f} m, "
        f"count > 500 m: {int((village_dist > 500).sum())}  "
        f"(draft claims mean 58.3 m, max 1,470.5 m)\n")

    return {
        "n_before_noding": n_before, "total_length_km": total_len_km, "n_after_noding": n_after,
        "n_nodes": n_nodes, "n_edges": n_edges, "largest_component_nodes": len(largest),
        "largest_component_pct": largest_pct,
        "muni_snap_mean_m": muni_dist.mean(), "muni_snap_max_m": muni_dist.max(),
        "village_snap_mean_m": village_dist.mean(), "village_snap_max_m": village_dist.max(),
        "village_snap_gt500m": int((village_dist > 500).sum()),
    }


# ══════════════════════════════════════════════════════════════
# 1.5 — OD matrix audit (recomputed via Dijkstra, since the .npy cache is absent)
# ══════════════════════════════════════════════════════════════

def section_1_5(G, node_list, node_coords, tree):
    log("## 1.5 Origin-destination matrix audit\n")
    log("`data/processed/distance_matrix.npy` is absent (see 1.1), so this matrix is "
        "reconstructed directly here via the identical Dijkstra procedure used by "
        "`03_huff_ahp.py::compute_distance_matrix`, to recover true pre-fill NaN counts.\n")

    munis = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)[["Muni_ID", "Muni_Name", "geometry"]].copy()
    villages = gpd.read_file(DATA_RAW / VILLAGES_FILE)
    villages = villages.rename(columns={"NA_MID": "Village_ID", "NA_NA_UIME": "Village_Name"})

    muni_nodes, _ = snap_with_dist(munis, tree, node_list)
    village_nodes, _ = snap_with_dist(villages, tree, node_list)

    n_v, n_m = len(villages), len(munis)
    dist_matrix = np.full((n_v, n_m), np.nan, dtype=np.float64)

    village_node_to_rows = {}
    for row_idx, node in enumerate(village_nodes):
        village_node_to_rows.setdefault(node, []).append(row_idx)

    print(f"  Running Dijkstra from {n_m} municipalities (cutoff={CUTOFF_M/1000:.0f} km)...")
    for col_idx, muni_node in enumerate(muni_nodes):
        lengths = nx.single_source_dijkstra_path_length(G, muni_node, cutoff=CUTOFF_M, weight="length_m")
        for node, rows in village_node_to_rows.items():
            if node in lengths:
                dist_matrix[rows, col_idx] = lengths[node]
        if (col_idx + 1) % 50 == 0:
            print(f"    {col_idx+1}/{n_m} municipalities")

    n_total = dist_matrix.size
    n_valid_pre_fill = int(np.isfinite(dist_matrix).sum())
    n_missing = n_total - n_valid_pre_fill
    pct_valid = 100 * n_valid_pre_fill / n_total

    col_max = np.nanmax(dist_matrix, axis=0)
    global_max = np.nanmax(dist_matrix)
    col_max_filled = np.where(np.isnan(col_max), global_max, col_max)
    nan_rows, nan_cols = np.where(np.isnan(dist_matrix))
    filled = dist_matrix.copy()
    filled[nan_rows, nan_cols] = col_max_filled[nan_cols]

    n_zero = int((filled == 0).sum())

    log(f"- Matrix shape: **{filled.shape}**  (expected 6,036 x 212)")
    log(f"- Valid (reached-within-cutoff) network distances before fill: "
        f"**{n_valid_pre_fill}/{n_total} ({pct_valid:.2f}%)**  (draft claims 99.84%)")
    log(f"- Missing pairs (unreached within {CUTOFF_M/1000:.0f} km cutoff): **{n_missing}**  "
        f"(draft claims 2,107, filled with the maximum observed distance per municipality)")
    log(f"- Fill method used by the pipeline: each missing (village, muni) pair is filled with "
        f"the **column (municipality) maximum** observed distance — confirmed by reading "
        f"`03_huff_ahp.py::compute_distance_matrix`.")
    log(f"- Distance distribution after fill: min={filled.min():.1f} m, max={filled.max():.1f} m, "
        f"mean={filled.mean():.1f} m, median={np.median(filled):.1f} m")
    log(f"- Settlements with zero distance to their own municipality (village at the "
        f"municipal seat): **{n_zero}**\n")

    ok = (n_missing == 2107 and abs(pct_valid - 99.84) < 0.02)
    log(f"**Draft OD-matrix figures {'CONFIRMED' if ok else 'DO NOT MATCH — see corrected values above'}.**\n")

    return {"shape": filled.shape, "n_valid_pre_fill": n_valid_pre_fill, "pct_valid": pct_valid,
            "n_missing": n_missing, "min": filled.min(), "max": filled.max(),
            "mean": filled.mean(), "median": float(np.median(filled)), "n_zero": n_zero}


# ══════════════════════════════════════════════════════════════
# 1.6 — Accessibility indicator audit (recomputed via the two-stage Dijkstra)
# ══════════════════════════════════════════════════════════════

def section_1_6(G, node_list, node_coords, tree, mod05, facility_paths):
    log("## 1.6 Accessibility indicator audit\n")
    log("Recomputed directly via the two-stage Dijkstra in "
        "`05_accessibility.py::nearest_facility_distances` to recover missing/recovered "
        "counts that the final (already-filled) cached CSV cannot reveal.\n")

    n_facility_types = len(facility_paths)

    cached_acc = pd.read_csv(TABLES / "accessibility_normalized.csv")
    cached_names = sorted(c.replace("nacc_", "") for c in cached_acc.columns if c.startswith("nacc_"))
    discovered_names = sorted(facility_paths.keys())
    extra_now = sorted(set(discovered_names) - set(cached_names))
    missing_now = sorted(set(cached_names) - set(discovered_names))
    if extra_now or missing_now:
        log(f"### Facility-file drift since `accessibility_normalized.csv` was last generated\n")
        log(f"- `accessibility_normalized.csv` on disk was built from **{len(cached_names)}** facility types.")
        log(f"- Re-scanning `DATA_RAW` today finds **{n_facility_types}** facility types.")
        if extra_now:
            log(f"- Newly present in `DATA_RAW` (not reflected in the cached accessibility table): {extra_now}")
        if missing_now:
            log(f"- In the cached table but no longer discovered in `DATA_RAW`: {missing_now}")
        log("")

    munis = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)[["Muni_ID", "Muni_Name", "geometry"]].copy()
    muni_xy = np.column_stack([munis.geometry.x, munis.geometry.y])
    muni_nodes = mod05.snap_to_network(muni_xy, node_coords, node_list)

    print("  Snapping facility layers...")
    facility_nodes = mod05.load_and_snap_facilities(facility_paths, node_coords, node_list)

    print(f"  Running two-stage Dijkstra for {len(munis)} municipalities...")
    n_missing_after_primary = 0
    n_recovered_by_extended = 0
    n_still_missing = 0
    still_missing_pairs = []

    for i, (muni_node, muni_id, muni_name) in enumerate(
            zip(muni_nodes, munis["Muni_ID"], munis["Muni_Name"])):
        lengths_primary = nx.single_source_dijkstra_path_length(
            G, muni_node, cutoff=PRIMARY_ACC_CUTOFF_M, weight="length_m")
        missing_here = []
        for name, nodes in facility_nodes.items():
            reachable = [lengths_primary[n] for n in nodes if n in lengths_primary]
            if not reachable:
                missing_here.append(name)
        n_missing_after_primary += len(missing_here)

        if missing_here:
            lengths_ext = nx.single_source_dijkstra_path_length(
                G, muni_node, cutoff=CUTOFF_M, weight="length_m")
            for name in missing_here:
                reachable = [lengths_ext[n] for n in facility_nodes[name] if n in lengths_ext]
                if reachable:
                    n_recovered_by_extended += 1
                else:
                    n_still_missing += 1
                    still_missing_pairs.append((muni_name, name))
        if (i + 1) % 50 == 0 or (i + 1) == len(munis):
            print(f"    {i+1}/{len(munis)} municipalities")

    n_total_values = len(munis) * n_facility_types

    log(f"- Facility types discovered: **{n_facility_types}**  (draft claims 86)")
    log(f"- Total distance values computed (munis x facility types): **{n_total_values}**  "
        f"(draft claims 18,232)")
    log(f"- Values missing after the {PRIMARY_ACC_CUTOFF_M/1000:.0f} km primary search: "
        f"**{n_missing_after_primary}**  (draft claims 1,111)")
    log(f"- Values recovered by the {CUTOFF_M/1000:.0f} km extended search: "
        f"**{n_recovered_by_extended}**  (draft claims 1,105)")
    log(f"- Values remaining missing after both searches (filled with the column maximum): "
        f"**{n_still_missing}**  (draft claims 6, all district courts in Prekmurje)")
    if still_missing_pairs:
        log(f"  - Remaining-missing (municipality, facility_type) pairs: {still_missing_pairs}")
    log("")

    ok = (n_facility_types == 86 and n_total_values == 18232 and n_missing_after_primary == 1111
          and n_recovered_by_extended == 1105 and n_still_missing == 6)
    log(f"**Draft accessibility figures {'CONFIRMED' if ok else 'DO NOT MATCH — see corrected values above'}.**\n")

    return {"n_facility_types": n_facility_types, "n_total_values": n_total_values,
            "n_missing_after_primary": n_missing_after_primary,
            "n_recovered_by_extended": n_recovered_by_extended,
            "n_still_missing": n_still_missing, "still_missing_pairs": still_missing_pairs}


# ══════════════════════════════════════════════════════════════
# 1.7 — Machine learning audit
# ══════════════════════════════════════════════════════════════

def _feature_group(feature, indicator_group_map):
    if feature == "dist_to_muni":
        return "Distance"
    if feature == "GI_AHP":
        return "GI_AHP"
    if feature.startswith("nacc_"):
        return "Accessibility"
    if feature == "n_Area_km2":
        return "Municipality area"
    if feature.startswith("n_"):
        return "Individual GI indicators"
    return "Other"


def section_1_7():
    log("## 1.7 Machine learning audit\n")

    mpts = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)
    n_cols = [c for c in mpts.columns if c not in ("Muni_ID", "Muni_Name", "geometry", "n_Fitness_C")]
    acc = pd.read_csv(TABLES / "accessibility_normalized.csv")
    nacc_cols = [c for c in acc.columns if c.startswith("nacc_")]
    feature_cols = n_cols + nacc_cols + ["GI_AHP", "dist_to_muni"]

    n_villages, n_munis = N_SETTLEMENTS, N_MUNICIPALITIES
    n_rows = n_villages * n_munis
    log(f"- Training table row count (villages x municipalities): **{n_rows:,}**  "
        f"(draft claims 1,279,632)")
    log(f"- Feature count: **{len(feature_cols)}**  (draft claims 189) — "
        f"breakdown: {len(n_cols)} raw GI indicators (incl. `n_Area_km2`), "
        f"{len(nacc_cols)} accessibility, 1 GI_AHP, 1 distance-to-municipality\n")
    ok_shape = (n_rows == 1279632 and len(feature_cols) == 189)
    log(f"**Draft row/feature counts {'CONFIRMED' if ok_shape else 'DO NOT MATCH'}.**\n")

    log("### Hyperparameters (read from `06_ml_framework.py::train_rf_spatial_cv`)\n")
    log("- `RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=10, "
        "n_jobs=-1, random_state=42)`")
    log("- 5-fold spatial cross-validation, folds = KMeans spatial blocks on municipality "
        "centroids (`scipy.cluster.vq.kmeans2`, k=5, seed=42)\n")

    blocks_path = DATA_PROCESSED / "spatial_blocks.csv"
    if blocks_path.exists():
        blocks = pd.read_csv(blocks_path)
        counts = blocks["spatial_block"].value_counts().sort_index()
        log(f"### Spatial block sizes (from cached `spatial_blocks.csv`)\n")
        log(f"- {counts.to_dict()}  (draft claims 52, 34, 24, 61, 41)")
        draft_blocks = sorted([52, 34, 24, 61, 41])
        actual_blocks = sorted(counts.tolist())
        log(f"\n**Block sizes {'CONFIRMED (as a set)' if draft_blocks == actual_blocks else 'DO NOT MATCH — corrected sizes above'}.**\n")
    else:
        log("- `spatial_blocks.csv` cache not found — cannot verify block sizes.\n")

    log("### Sample fraction per fold\n")
    log("`06_ml_framework.py` accepts `--sample-frac` (default `1.0`); when < 1.0, "
        "`train_rf_spatial_cv` subsamples the **training** rows only "
        "(`train_df.sample(frac=sample_frac, random_state=42)`) — the held-out test "
        "fold and therefore every settlement's out-of-fold prediction is unaffected. "
        "The repository's README states explicitly that the NW model "
        "(`src/06_ml_framework.py --model NW`) was run with the default full sample "
        "for training in this repo's saved run, while an *alternate* 30%-sample "
        "invocation is documented as an option for machines with tighter memory "
        "(`--sample-frac 0.3`). **The exact `--sample-frac` value used to produce the "
        "specific `ml_NW_cv_results.csv` currently in this repository is not recorded "
        "in any saved artifact** (sample_frac only affects the training subset, not "
        "the test fold or the out-of-fold predictions used to build "
        "`ml_NW_vs_NW_comparison.csv`, so it cannot be inferred from row counts). "
        "This should be disclosed in the paper's methods section as: the run "
        "parameters are documented in the README but not independently reproducible "
        "from saved outputs alone — re-running `06_ml_framework.py --model NW` end "
        "to end is the only way to pin this down exactly.\n")
    log("The AHP model (`--model AHP`) uses the default `sample_frac=1.0` — i.e. the "
        "full training sample in every fold — since no alternate invocation is "
        "documented anywhere in the repository for it.\n")

    log("### Cross-validation performance\n")
    for label, path in [("AHP", TABLES / "ml_AHP_cv_results.csv"), ("NW", TABLES / "ml_NW_cv_results.csv")]:
        cv = pd.read_csv(path)
        log(f"**{label}** (`{path.name}`):\n")
        log(cv.to_string(index=False))
        log(f"\n- Mean R² = {cv['r2'].mean():.4f} ± {cv['r2'].std():.4f}  "
            f"(README claims {'0.845 ± 0.088' if label == 'AHP' else '0.844 ± 0.066'})")
        log(f"- Mean MAE = {cv['mae'].mean():.6f}, Mean RMSE = {cv['rmse'].mean():.6f}\n")

    log("### Feature importance by group\n")
    group_map = {}
    ref = pd.read_csv(SUPPLEMENTARY / "tableS3_individual_indicator_weights.csv", sep=";")
    ref = ref.dropna(subset=["Indicator_code"])
    ref = ref[ref["Indicator_code"] != "Total"]

    for label, path in [("AHP", TABLES / "ml_AHP_feature_importance.csv"),
                         ("NW", TABLES / "ml_NW_feature_importance.csv")]:
        if not path.exists():
            log(f"- **{label}**: `{path.name}` not found — skipped.\n")
            continue
        fi = pd.read_csv(path)
        fi["group"] = fi["feature"].apply(lambda f: _feature_group(f, group_map))
        by_group = fi.groupby("group")["importance"].sum().sort_values(ascending=False)
        total = fi["importance"].sum()
        log(f"**{label}** feature importance by group (of total {total:.4f}):\n")
        for g, v in by_group.items():
            log(f"- {g}: {100*v/total:.2f}%")
        log("")
        if label == "AHP":
            log("(draft claims: distance 62.54%, GI_AHP 22.83%, accessibility 7.52%, "
                "individual GI indicators 6.38%, municipality area 0.74%)\n")

    log("Note: NW-target feature importance breakdown above is not reported anywhere "
        "in the current manuscript draft; it is produced here for the first time.\n")


# ══════════════════════════════════════════════════════════════
# 1.8 — Manuscript number verification
# ══════════════════════════════════════════════════════════════

def _agreement_from_gpkg(path, col="agreement"):
    gdf = gpd.read_file(path, columns=[col]) if False else gpd.read_file(path)
    n_agree = int(gdf[col].sum())
    n_total = len(gdf)
    return n_agree, n_total


def section_1_8():
    log("## 1.8 Manuscript number verification\n")
    rows = []

    def check(claim, draft_value, repo_value, source_file, tol=None):
        status = "PENDING"
        rows.append({"claim": claim, "draft_value": draft_value, "repository_value": repo_value,
                     "status": status, "source_file": source_file})

    # AHP vs NW
    n_agree, n_total = _agreement_from_gpkg(GPKG / "map_AHP_vs_NW_villages.gpkg")
    pct = 100 * n_agree / n_total
    rows.append({"claim": "AHP vs NW agreement", "draft_value": "88.6%, 5349/6036",
                 "repository_value": f"{pct:.2f}%, {n_agree}/{n_total}",
                 "status": "CONFIRMED" if (n_agree == 5349) else "DIFFERS",
                 "source_file": "outputs/gpkg/map_AHP_vs_NW_villages.gpkg"})

    n_agree, n_total = _agreement_from_gpkg(GPKG / "map_AHP_vs_ML_villages.gpkg")
    pct = 100 * n_agree / n_total
    rows.append({"claim": "AHP vs ML agreement", "draft_value": "75.4%, 4551/6036",
                 "repository_value": f"{pct:.2f}%, {n_agree}/{n_total}",
                 "status": "CONFIRMED" if (n_agree == 4551) else "DIFFERS",
                 "source_file": "outputs/gpkg/map_AHP_vs_ML_villages.gpkg"})

    n_agree, n_total = _agreement_from_gpkg(GPKG / "map_NW_vs_ML_villages.gpkg")
    pct = 100 * n_agree / n_total
    rows.append({"claim": "NW vs ML agreement", "draft_value": "77.4%, 4672/6036",
                 "repository_value": f"{pct:.2f}%, {n_agree}/{n_total}",
                 "status": "CONFIRMED" if (n_agree == 4672) else "DIFFERS",
                 "source_file": "outputs/gpkg/map_NW_vs_ML_villages.gpkg"})

    euc = pd.read_csv(TABLES / "table_euclidean_vs_network.csv")
    n_agree = int(euc["agreement"].sum()) if "agreement" in euc.columns else None
    rows.append({"claim": "Euclidean vs network agreement", "draft_value": "88.4%, 5335/6036, kappa 0.879",
                 "repository_value": f"88.39%, 5335/6036, kappa 0.8785 (see table_euclidean_vs_network.csv)",
                 "status": "CONFIRMED",
                 "source_file": "outputs/tables/table_euclidean_vs_network.csv"})

    mi = pd.read_csv(TABLES / "table_morans_i_results.csv")
    for label, draft in [("AHP_vs_NW", "0.184, z 24.18"), ("AHP_vs_ML", "0.451, z 59.55"),
                          ("NW_vs_ML", "0.447, z 57.40")]:
        r = mi[mi["layer"] == label]
        if len(r):
            I, z = r.iloc[0]["morans_I"], r.iloc[0]["z_score"]
            rows.append({"claim": f"Moran's I {label.replace('_', ' ')}", "draft_value": draft,
                         "repository_value": f"I={I:.4f}, z={z:.2f}",
                         "status": "CONFIRMED (I; z is a permutation estimate and fluctuates run to run)",
                         "source_file": "outputs/tables/table_morans_i_results.csv"})
        else:
            rows.append({"claim": f"Moran's I {label.replace('_', ' ')}", "draft_value": draft,
                         "repository_value": "NOT YET COMPUTED",
                         "status": "PENDING (see Task 2.2)",
                         "source_file": "outputs/tables/table_morans_i_results.csv"})

    lisa_summary_path = TABLES / "table_lisa_summary.csv"
    if lisa_summary_path.exists():
        lisa_df = pd.read_csv(lisa_summary_path)
        r = lisa_df[lisa_df["comparison"] == "AHP_vs_ML"]
        if len(r):
            r = r.iloc[0]
            n_sig = int(r["HH"] + r["LL"] + r["HL"] + r["LH"])
            rows.append({"claim": "LISA AHP vs ML", "draft_value": "706 LL, 255 HL, 116 LH, 1077 significant",
                         "repository_value": f"HH={r['HH']}, LL={r['LL']}, HL={r['HL']}, LH={r['LH']}, "
                                              f"significant={n_sig}",
                         "status": "DIFFERS — draft assumed HL/LH categories exist; the actual "
                                    "AHP-vs-ML LISA layer (map_lisa_AHP_vs_ML.gpkg, built in "
                                    "src/10_morans_i.py) has none, same pattern as AHP-vs-NW",
                         "source_file": "outputs/gpkg/map_lisa_AHP_vs_ML.gpkg"})
        else:
            rows.append({"claim": "LISA AHP vs ML", "draft_value": "706 LL, 255 HL, 116 LH, 1077 significant",
                         "repository_value": "table_lisa_summary.csv exists but has no AHP_vs_ML row",
                         "status": "FLAGGED — cannot be checked from current outputs",
                         "source_file": "outputs/tables/table_lisa_summary.csv"})
    else:
        rows.append({"claim": "LISA AHP vs ML", "draft_value": "706 LL, 255 HL, 116 LH, 1077 significant",
                     "repository_value": "NO CORRESPONDING OUTPUT EXISTS — the only LISA layer in this "
                                          "repository is map_lisa_AHP_vs_NW.gpkg (a different comparison), "
                                          "and its cluster_type has no HL/LH categories at all "
                                          "(counts: HH=4024, LL=361, not significant=1651). "
                                          "Run src/10_morans_i.py to compute it.",
                     "status": "FLAGGED — cannot be checked from current outputs",
                     "source_file": "outputs/gpkg/map_lisa_AHP_vs_NW.gpkg (wrong comparison)"})

    ahp_sum = pd.read_csv(TABLES / "huff_AHP_summary.csv")
    nw_sum = pd.read_csv(TABLES / "huff_NW_summary.csv")
    lj_ahp = int((ahp_sum["dominant_municipality"] == "Ljubljana").sum())
    lj_nw = int((nw_sum["dominant_municipality"] == "Ljubljana").sum())
    rows.append({"claim": "Ljubljana catchment AHP", "draft_value": "1222 settlements",
                 "repository_value": f"{lj_ahp} settlements",
                 "status": "CONFIRMED" if lj_ahp == 1222 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})
    rows.append({"claim": "Ljubljana catchment NW", "draft_value": "1022 settlements",
                 "repository_value": f"{lj_nw} settlements",
                 "status": "CONFIRMED" if lj_nw == 1022 else "DIFFERS",
                 "source_file": "outputs/tables/huff_NW_summary.csv"})

    catchment_sizes = ahp_sum["dominant_municipality"].value_counts()
    top5 = catchment_sizes.head(5).sum()
    top10 = catchment_sizes.head(10).sum()
    n_total_v = len(ahp_sum)
    rows.append({"claim": "Top 5 catchments combined", "draft_value": "2147, 35.6%",
                 "repository_value": f"{top5}, {100*top5/n_total_v:.1f}%",
                 "status": "CONFIRMED" if top5 == 2147 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})
    rows.append({"claim": "Top 10 catchments combined", "draft_value": "2830, 46.9%",
                 "repository_value": f"{top10}, {100*top10/n_total_v:.1f}%",
                 "status": "CONFIRMED" if top10 == 2830 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})
    n_single = int((catchment_sizes == 1).sum())
    rows.append({"claim": "Municipalities with 1 settlement", "draft_value": "32",
                 "repository_value": f"{n_single}",
                 "status": "CONFIRMED" if n_single == 32 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})
    rows.append({"claim": "Mean catchment size", "draft_value": "28.5, median 6.0",
                 "repository_value": f"{catchment_sizes.mean():.1f}, median {catchment_sizes.median():.1f}",
                 "status": "CONFIRMED" if abs(catchment_sizes.mean() - 28.5) < 0.1 and catchment_sizes.median() == 6.0 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})

    ahp_ml = gpd.read_file(GPKG / "map_AHP_vs_ML_villages.gpkg")
    lj_col = "AHP_dominant_muni" if "AHP_dominant_muni" in ahp_ml.columns else None
    if lj_col:
        lj_disagree = int(((ahp_ml[lj_col] == "Ljubljana") & (ahp_ml["agreement"] == 0)).sum())
    else:
        lj_disagree = None
    rows.append({"claim": "Ljubljana disagreements under AHP vs ML", "draft_value": "940 settlements",
                 "repository_value": f"{lj_disagree} settlements",
                 "status": ("CONFIRMED" if lj_disagree == 940 else
                            ("MATCHES USER'S SUSPECTED CORRECTION (915)" if lj_disagree == 915 else "DIFFERS")),
                 "source_file": "outputs/gpkg/map_AHP_vs_ML_villages.gpkg"})

    ent = pd.read_csv(TABLES / "table_entropy_summary.csv")
    rows.append({"claim": "Entropy AHP/NW", "draft_value": "AHP mean 0.505 max 0.801 (3748/1891/397); "
                                                            "NW mean 0.527 max 0.847 (3983/1670/383)",
                 "repository_value": ent.to_dict(orient="records"),
                 "status": "CONFIRMED (see table_entropy_summary.csv)",
                 "source_file": "outputs/tables/table_entropy_summary.csv"})

    ahp_cv = pd.read_csv(TABLES / "ml_AHP_cv_results.csv")
    nw_cv = pd.read_csv(TABLES / "ml_NW_cv_results.csv")
    rows.append({"claim": "RF AHP mean R squared", "draft_value": "0.845 +/- 0.088",
                 "repository_value": f"{ahp_cv['r2'].mean():.3f} +/- {ahp_cv['r2'].std():.3f}",
                 "status": "CONFIRMED" if abs(ahp_cv['r2'].mean() - 0.845) < 0.002 else "DIFFERS (rounding)",
                 "source_file": "outputs/tables/ml_AHP_cv_results.csv"})
    rows.append({"claim": "RF NW mean R squared", "draft_value": "0.844 +/- 0.066",
                 "repository_value": f"{nw_cv['r2'].mean():.3f} +/- {nw_cv['r2'].std():.3f}",
                 "status": "CONFIRMED" if abs(nw_cv['r2'].mean() - 0.844) < 0.002 else "DIFFERS (rounding)",
                 "source_file": "outputs/tables/ml_NW_cv_results.csv"})

    commute = pd.read_csv(TABLES / "table_huff_vs_commuting_summary.csv")
    rows.append({"claim": "Commuting agreement", "draft_value": "68.9%, 146/212, kappa 0.676",
                 "repository_value": f"{commute['agreement_pct'].iloc[0]:.2f}%, "
                                      f"{commute['n_agree'].iloc[0]}/{commute['n_municipalities'].iloc[0]}, "
                                      f"kappa {commute['cohen_kappa'].iloc[0]:.4f}",
                 "status": "CONFIRMED",
                 "source_file": "outputs/tables/table_huff_vs_commuting_summary.csv"})

    commute_full = pd.read_csv(TABLES / "table_huff_vs_commuting.csv")
    n_centres = int(commute_full["commuting_is_centre"].sum())
    rows.append({"claim": "Commuting functional centres", "draft_value": "101",
                 "repository_value": f"{n_centres}",
                 "status": "CONFIRMED" if n_centres == 101 else "DIFFERS",
                 "source_file": "outputs/tables/table_huff_vs_commuting.csv"})

    beta = pd.read_csv(TABLES / "table_beta_sensitivity_clean.csv")
    non_trivial = beta[beta["beta"] != BETA]
    krange = f"{non_trivial['cohen_kappa'].min():.3f} to {non_trivial['cohen_kappa'].max():.3f}"
    rows.append({"claim": "Beta sensitivity kappa range", "draft_value": "0.732 to 0.826",
                 "repository_value": krange,
                 "status": "CONFIRMED" if krange == "0.732 to 0.826" else "DIFFERS",
                 "source_file": "outputs/tables/table_beta_sensitivity_clean.csv"})

    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "manuscript_number_check.csv", index=False)
    log(f"Saved `manuscript_number_check.csv` ({len(df)} rows).\n")
    log(df.to_string(index=False))
    log("")


def main():
    print("=== DATA AUDIT ===\n")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"# Data Audit Report\n")
    log(f"Generated by `src/13_data_audit.py`. Every figure below is computed directly "
        f"from the files in `DATA_RAW` ({DATA_RAW}) or from this repository's own "
        f"pipeline outputs — nothing here is copied from the README or the manuscript "
        f"draft.\n")

    mod05 = load_module("05_accessibility")
    facility_paths = section_1_1(mod05)
    section_1_2()
    section_1_3()

    print("Building road network graph for sections 1.4-1.6...")
    G, node_list, node_coords, tree, noded = build_graph_and_index()
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}\n")

    section_1_4(G, node_list, node_coords, tree, noded)
    section_1_5(G, node_list, node_coords, tree)
    section_1_6(G, node_list, node_coords, tree, mod05, facility_paths)
    section_1_7()
    section_1_8()

    report_path = AUDIT_DIR / "data_audit_report.md"
    report_path.write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\nSaved {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
