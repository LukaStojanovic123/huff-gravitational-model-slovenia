"""
Step 17 of the pipeline: independently recompute a large sample of the
pipeline's own numbers from source and check them against each other and
against the manuscript, as a final self-check before publication.

What this script does: everything else in the pipeline computes a result
once and writes it to a file. This script is different — it is the
pipeline checking its own work. Section 1.1 inventories exactly which raw
files were used. Section 1.2 rebuilds the indicator rarity weights from
first principles, the same check 01_gi_construction.py already runs, but
repeated here as part of one consolidated audit. Section 1.3 re-derives
GI descriptive statistics directly from the raw municipality layers.
Sections 1.4-1.6 rebuild the road network, the settlement-to-municipality
distance matrix, and the accessibility scores from scratch — using a
freshly rebuilt graph, not any cached intermediate — and compare the
result to what 02_road_network.py, 03_huff_ahp.py and 05_accessibility.py
actually produced. Section 1.7 checks the ML training table's shape and
the spatial cross-validation results. Section 1.8 is the one section that
does not recompute anything: it holds a fixed list of specific numbers
quoted in the manuscript draft, written directly into this script, and
checks each one against the pipeline's actual current output, reporting a
plain CONFIRMED / DIFFERS per claim.
Every number quoted anywhere in this report is computed in this script,
not copied from the README, the manuscript, or any other document.

Reads: nearly everything — the raw data files directly, and the outputs
of 01 through 17.

Writes: data_audit_report.md, raw_input_inventory.csv, indicator_audit.csv,
GI_full_212_municipalities.csv, manuscript_number_check.csv, and
supplementary/tableS7_road_network_statistics.csv (assembled from values
this section's own section 1.4 already independently recomputes, not a
separate computation).

Runs seventeenth, second to last — after every script whose numbers it
checks, before only 18_output_manifest.py.
"""

import re
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
    DATA_RAW, DATA_PROCESSED, TABLES, OUTPUTS, GPKG, SUPPLEMENTARY, REPO_ROOT,
    N_MUNICIPALITIES, N_SETTLEMENTS, BETA, EPSG, CUTOFF_M, ROADS_FILE,
    MUNICIPALITIES_AHP, MUNICIPALITIES_NW, MUNICIPALITIES_PTS,
    VILLAGES_FILE, SETTLEMENTS_POLY, COMMUTING_FILE,
)
from crs_utils import ensure_crs

AUDIT_DIR = OUTPUTS / "audit"
DATA_EXTERNAL = REPO_ROOT / "data" / "external"
OBCINE_FILE = "obcine_poligoni.shp"

OUTPUT_FILES = [
    "audit/raw_input_inventory.csv",
    "audit/indicator_audit.csv",
    "audit/GI_full_212_municipalities.csv",
    "audit/manuscript_number_check.csv",
    "audit/data_audit_report.md",
    "supplementary/tableS7_road_network_statistics.csv",
]
SRC_DIR = Path(__file__).resolve().parent
NODED_ROADS_PATH = DATA_PROCESSED / "roads_noded.gpkg"
PRIMARY_ACC_CUTOFF_M = 80_000

FINAL_VALUES_PATH = REPO_ROOT / "docs" / "final_manuscript_values.md"

REPORT = []
CHECK_FAILURES = []


def record_failure(section, message):
    """Record a failed check so main() can exit non-zero once every section has run.

    Printing a DIFFERS/MISMATCH line in the report is not enough on its
    own — this repository shipped a broken check for weeks because
    nothing forced anyone to notice the report said so. Every check that
    reaches a genuine fail verdict below calls this, so the script exits
    loudly instead of quietly succeeding regardless of what it found.
    """
    CHECK_FAILURES.append(f"[{section}] {message}")


def load_final_manuscript_values():
    """Read docs/final_manuscript_values.md once, or return None if it is unavailable.

    This is the one document this repository maintains as an ongoing
    source of truth for "what does the manuscript currently claim" —
    several checks below used to compare against a value hardcoded
    directly in this script ("draft claims X"), frozen at whatever the
    manuscript said when that line was last edited. The manuscript's own
    numbers have already moved at least once since this script was
    written; a hardcoded comparison has no way to notice that. Reading
    this file at call time means the comparison always uses whatever
    final_manuscript_values.md currently says, and a missing file or a
    missing specific value is reported as exactly that — not silently
    replaced with an invented number.
    """
    if not FINAL_VALUES_PATH.exists():
        return None
    return FINAL_VALUES_PATH.read_text(encoding="utf-8")


def find_reference_value(text, pattern, group=1):
    """Search final_manuscript_values.md's text for one labelled value.

    `pattern` is a regex with at least one capture group; returns the
    text of `group`, or None if `text` is None (file missing) or the
    pattern isn't found in it (this specific value isn't recorded there).
    Callers must treat None as "no reference value exists" and say so
    plainly, never fall back to a guessed number.
    """
    if text is None:
        return None
    m = re.search(pattern, text)
    return m.group(group) if m else None


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
    "gis_osm_roads_free_1.shp",
    "Municipalities_All_Groups_Weighted_AHP.gpkg",
    "Municipalities_All_Groups_NotWeighted_Normalized.gpkg",
    "Municipalities_Points_normalized.gpkg",
    "Villages_points_real.shp",
    "NA.shp",
    "obcine_poligoni.shp",
    "2023tabela.xlsx",
]


def section_1_1(mod05):
    """List every raw file the pipeline actually reads, and flag any file in DATA_RAW that is not one of them."""
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
        f"one of the 86 facility layers listed in `data/facility_layers.txt`:\n")
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
    # No sep= argument: tableS2 is comma-delimited, like tableS1, S3 and S4.
    # A stale sep=";" here (left over from before all four supplementary
    # tables were standardised to commas) made this call read the entire
    # header as one unsplit column instead of ten.
    #
    # nrows=11 stops reading after the pairwise matrix (row 0 is its
    # header, rows 1-10 are the matrix itself) — this file also contains a
    # second, differently-shaped table further down (the derived priority
    # weights, with an extra "PRIORITY WEIGHTS PER GROUP" column this
    # function does not need), and pandas' default parser refuses to read
    # past a row with a different column count than the rows before it.
    raw = pd.read_csv(SUPPLEMENTARY / "tableS2_AHP_priority_weights.csv",
                       header=None, nrows=11)
    # Row 0 is header; rows 1-10 are the pairwise matrix.
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
    """Independently recompute each indicator's rarity weight from raw data and check it against tableS3, plus recheck the AHP consistency ratio."""
    log("## 1.2 Indicator database audit\n")

    pts = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)
    pts = ensure_crs(pts, EPSG, label=MUNICIPALITIES_PTS)
    n_cols = [c for c in pts.columns if c.startswith("n_")]
    # No sep= argument: tableS3 is comma-delimited — see the same note in
    # compute_ahp_consistency_ratio above and in 01_gi_construction.py.
    ref = pd.read_csv(SUPPLEMENTARY / "tableS3_individual_indicator_weights.csv")
    ref = ref.dropna(subset=["Indicator_code"])
    ref = ref[ref["Indicator_code"] != "Total"]

    log(f"- Indicators mapped in tableS3: **{len(ref)}** "
        f"({'MATCHES' if len(ref) == 100 else 'DOES NOT MATCH'} the claimed 100)")
    unmapped = sorted(set(n_cols) - set(ref['Indicator_code']))
    log(f"- Raw `n_` columns in `{MUNICIPALITIES_PTS}` not in the 100-indicator mapping: "
        f"{unmapped} (expected: `n_Area_km2`, `n_Fitness_C` — area is excluded from "
        f"the GI while `n_Fitness_C` is a duplicate of `n_Fitness`)")
    log("")

    # Independently recompute the rarity weights from the raw indicator data
    # and the categorisation input, using the same formula as
    # 01_gi_construction.py::compute_rarity_weights but implemented
    # separately here, and compare against what that script actually wrote
    # to tableS3. This is the genuine cross-check tableS3 needs now that it
    # is a computed pipeline output rather than a hand-maintained file —
    # the same principle already applied elsewhere in this audit (the road
    # network, OD matrix and accessibility scores are all independently
    # rebuilt from scratch and compared against their scripts' own output,
    # rather than just re-displaying it).
    cats = pd.read_csv(DATA_EXTERNAL / "indicator_categories.csv")
    cats = cats.dropna(subset=["Indicator_code"])
    cats = cats[cats["Indicator_code"] != "Total"]
    group_map = dict(zip(cats["Indicator_code"], cats["Thematic_group"]))

    recompute_rows = []
    for code, group in group_map.items():
        if code not in pts.columns:
            continue
        nonzero = int((pts[code] != 0).sum())
        ri = max(1 - nonzero / N_MUNICIPALITIES, 0.01)
        recompute_rows.append({"Indicator_code": code, "Thematic_group": group,
                                "Nonzero_munis": nonzero, "ri": ri})
    recomputed = pd.DataFrame(recompute_rows)
    recomputed["sqrt_ri"] = np.sqrt(recomputed["ri"])
    recomputed["wi"] = recomputed.groupby("Thematic_group")["sqrt_ri"].transform(lambda s: s / s.sum())

    cross_check = recomputed.merge(
        ref[["Indicator_code", "Rarity_score_ri", "Sqrt_ri", "Normalised within-group weight (w_i)"]],
        on="Indicator_code", how="outer", indicator=True)
    unmatched = cross_check[cross_check["_merge"] != "both"]
    max_ri_diff = (cross_check["ri"] - cross_check["Rarity_score_ri"]).abs().max()
    max_wi_diff = (cross_check["wi"] - cross_check["Normalised within-group weight (w_i)"]).abs().max()
    weights_confirmed = len(unmatched) == 0 and max_ri_diff < 1e-9 and max_wi_diff < 1e-9
    log(f"- Independently recomputed rarity weights vs `tableS3` "
        f"(`01_gi_construction.py`'s own output): "
        f"max |ri diff| = {max_ri_diff:.2e}, max |wi diff| = {max_wi_diff:.2e}, "
        f"{len(unmatched)} indicator(s) present in only one of the two. "
        f"**{'CONFIRMED — tableS3 matches an independent recomputation' if weights_confirmed else 'DOES NOT MATCH — investigate 01_gi_construction.py or this recomputation'}.**\n")

    if not weights_confirmed:
        record_failure("1.2", f"tableS3 rarity-weight cross-check: max |ri diff| = {max_ri_diff:.2e}, "
                               f"max |wi diff| = {max_wi_diff:.2e}, {len(unmatched)} unmatched indicator(s)")

    group_counts = ref["Thematic_group"].value_counts()
    # Keys must match tableS3's actual Thematic_group spelling exactly
    # ("Traffic and Communications", not the short form "Traffic") — see
    # _map_group_name's docstring below for why this matters and what went
    # wrong the last time these fell out of sync.
    expected_counts = {
        "Healthcare": 9, "Education": 14, "Traffic and Communications": 14,
        "Trade and Business": 7, "Culture": 12, "Sports and Recreation": 12,
        "Tourism and Services": 13, "Finance": 4, "Judiciary and Emergency": 8,
        "Residential": 7,
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

    if not all_match:
        record_failure("1.2", f"Group indicator counts do not match the expected breakdown "
                               f"(actual: {group_counts.to_dict()})")

    wsum = ref.groupby("Thematic_group")["Normalised within-group weight (w_i)"].sum()
    log("### Within-group weight sums (should be 1.0 in every group)\n")
    for g, s in wsum.items():
        log(f"- {g}: {s:.6f}")
    ok_sum = bool(((wsum - 1.0).abs() < 1e-3).all())
    log(f"\n**Within-group weights {'CONFIRMED to sum to 1' if ok_sum else 'DO NOT sum to 1'} in every group.**\n")

    if not ok_sum:
        record_failure("1.2", f"Within-group weight sums do not all equal 1.0: {wsum.to_dict()}")

    table1 = pd.read_csv(TABLES / "table1_AHP_group_weights.csv")
    group_pct_sum = table1.loc[table1["Thematic Group"] != "TOTAL", "AHP Priority Weight (%)"].sum()
    log(f"### AHP group weights\n\nSum of the 10 group weights in `table1_AHP_group_weights.csv`: "
        f"{group_pct_sum:.4f}% "
        f"({'CONFIRMED ~100%' if abs(group_pct_sum - 100) < 0.5 else 'DOES NOT sum to 100%'})\n")
    if abs(group_pct_sum - 100) >= 0.5:
        record_failure("1.2", f"AHP group weights sum to {group_pct_sum:.4f}%, not ~100%")

    log("### AHP consistency ratio (recomputed from the pairwise matrix in tableS2)\n")
    groups, matrix, lambda_max, CI, RI, CR, priority_vec = compute_ahp_consistency_ratio()
    log(f"- Group order: {groups}")
    log(f"- lambda_max (principal eigenvalue) = **{lambda_max:.4f}**  (draft claims 10.322)")
    log(f"- CI = (lambda_max - n)/(n-1) = **{CI:.4f}**  (draft claims 0.036)")
    log(f"- RI (Saaty table, n=10) = **{RI}**  (draft claims 1.49)")
    log(f"- CR = CI / RI = **{CR:.4f}**  (draft claims 0.024)")
    draft_ok = abs(lambda_max - 10.322) < 0.01 and abs(CI - 0.036) < 0.001 and abs(CR - 0.024) < 0.001
    log(f"\n**Draft AHP consistency figures {'CONFIRMED' if draft_ok else 'DO NOT match recomputed values — see above for corrected figures'}.**\n")
    if not draft_ok:
        record_failure("1.2", f"AHP consistency ratio: recomputed lambda_max={lambda_max:.4f}, "
                               f"CI={CI:.4f}, CR={CR:.4f} vs draft 10.322/0.036/0.024")
    log("Recomputed priority vector from the matrix's principal eigenvector "
        f"(should reproduce the group weights in table1): "
        + ", ".join(f"{g}={v:.4f}" for g, v in zip(groups, priority_vec)) + "\n")

    log("### Territorial descriptors vs functional counts\n")
    area_in_pts = "n_Area_km2" in pts.columns
    area_in_tableS3 = "n_Area_km2" in set(ref["Indicator_code"])
    territorial_ok = area_in_pts and not area_in_tableS3
    log(f"- `n_Area_km2` present in `{MUNICIPALITIES_PTS}`: **{area_in_pts}**")
    log(f"- `n_Area_km2` present in tableS3's 100-indicator mapping (i.e. included in the GI): "
        f"**{area_in_tableS3}**")
    log(f"\n**Territorial-descriptor exclusion {'CONFIRMED' if territorial_ok else 'DOES NOT hold'}: "
        f"`n_Area_km2` is {'correctly excluded from' if territorial_ok else 'unexpectedly included in or missing from'} "
        "the GI's 100-indicator set.**\n")
    if not territorial_ok:
        record_failure("1.2", f"Territorial-descriptor exclusion: n_Area_km2 in pts={area_in_pts}, "
                               f"in tableS3={area_in_tableS3} (expected True/False)")

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
    "Healthcare": "Healthcare", "Education": "Education",
    "Traffic and Communications": "Traffic & Communications",
    "Trade and Business": "Trade & Business", "Culture": "Culture",
    "Sports and Recreation": "Sports & Recreation",
    "Tourism and Services": "Tourism & Services", "Finance": "Finance",
    "Judiciary and Emergency": "Judiciary & Emergency", "Residential": "Residential",
}


def _map_group_name(name):
    """Convert a thematic group name from tableS3's spelling ("X and Y") to table1's ("X & Y").

    tableS3 (and the raw indicator categorisation it's built from) spells
    out group names in full ("Traffic and Communications"); table1's own
    "Thematic Group" column abbreviates the same names with an ampersand
    ("Traffic & Communications"). Both are the correct spelling for their
    own table — this just translates between them so this script can look
    a tableS3 group up in table1. This mapping, and the group-count check
    in section_1_2 below, both went stale for the same reason once before:
    an earlier commit standardised tableS3's group names from short
    abbreviations ("Traffic") to the full "and"-form without updating
    either place that still expected the old short form, so every lookup
    for five of the ten groups silently returned nothing.
    """
    return _GROUP_NAME_MAP.get(name, name)


# ══════════════════════════════════════════════════════════════
# 1.3 — Gravitational Index audit
# ══════════════════════════════════════════════════════════════

def section_1_3():
    """Re-derive GI descriptive statistics and rankings straight from the raw municipality layers."""
    log("## 1.3 Gravitational Index audit\n")

    nw = gpd.read_file(DATA_RAW / MUNICIPALITIES_NW)
    nw = ensure_crs(nw, EPSG, label=MUNICIPALITIES_NW)
    ahp = gpd.read_file(DATA_RAW / MUNICIPALITIES_AHP)
    ahp = ensure_crs(ahp, EPSG, label=MUNICIPALITIES_AHP)

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
    if not nw_ok:
        record_failure("1.3", f"GI_NW descriptive stats: mean={nw_stats['mean']:.4f}, "
                               f"median={nw_stats['median']:.4f}, sd={nw_stats['std']:.4f} "
                               "vs draft 0.044/0.020/0.084")
    if not ahp_ok:
        record_failure("1.3", f"GI_AHP descriptive stats: mean={ahp_stats['mean']:.4f}, "
                               f"median={ahp_stats['median']:.4f}, sd={ahp_stats['std']:.4f} "
                               "vs draft 0.039/0.013/0.088")

    rho, pval = spearmanr(merged["GI_Final_NotWeighted"], merged["GI_AHP"])
    n_big_change = int((merged["rank_change"] > 10).sum())
    log(f"### Rank stability\n")
    log(f"- Spearman rank correlation GI_NW vs GI_AHP: rho={rho:.4f}, p={pval:.2e}")
    log(f"- Municipalities changing rank by more than 10 places: **{n_big_change}** / {len(merged)}\n")


# ══════════════════════════════════════════════════════════════
# Shared: road graph + snapping (used by 1.4, 1.5, 1.6)
# ══════════════════════════════════════════════════════════════

def build_graph_and_index():
    """Rebuild the road network graph and its spatial index, shared by sections 1.4-1.6."""
    noded = gpd.read_file(NODED_ROADS_PATH)
    G = momepy.gdf_to_nx(noded, approach="primal", length="length_m")
    node_list = list(G.nodes)
    node_coords = np.array(node_list, dtype=np.float64)
    tree = cKDTree(node_coords)
    return G, node_list, node_coords, tree, noded


def snap_with_dist(points_gdf, tree, node_list):
    """Attach points to their nearest road-network node, also returning the snapping distance."""
    xy = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])
    dist, idx = tree.query(xy)
    return [node_list[i] for i in idx], dist


# ══════════════════════════════════════════════════════════════
# 1.4 — Road network audit
# ══════════════════════════════════════════════════════════════

def section_1_4(G, node_list, node_coords, tree, noded):
    """Compare the freshly rebuilt road graph's segment, node and length counts against 02_road_network.py's own output."""
    log("## 1.4 Road network audit\n")

    roads_raw = gpd.read_file(DATA_RAW / ROADS_FILE)
    roads_raw = ensure_crs(roads_raw, EPSG, label=ROADS_FILE)
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

    # `noded`/`G` here are read from the SAVED roads_noded.gpkg, which (since the
    # Stage 2A largest-component fix in 02_road_network.py) already contains only
    # the largest connected component — not the full pre-filter noded graph. So
    # n_after/n_nodes below are largest-component figures, not the pre-filter
    # 439,091/394,874 the manuscript states; those two are not the same quantity
    # and comparing them directly would be an apples-to-oranges check.
    n_after = len(noded)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    largest_pct = 100 * len(largest) / n_nodes

    log(f"- Segments before noding (post-filter, exploded, valid): **{n_before}**  "
        f"(manuscript: 254,252)")
    log(f"- Total drivable length before noding: **{total_len_km:.1f} km**  (manuscript: 55,062 km)")
    log(f"- Segments in saved roads_noded.gpkg (largest connected component only): **{n_after}**")
    log(f"- Graph nodes (largest connected component only): **{n_nodes}**  "
        f"(manuscript's largest component: 390,273 nodes, 98.9% of 394,874)")
    log(f"- Graph edges: **{n_edges}**")
    log(f"- Connected components in the saved file: **{len(components)}** "
        f"({'as expected — the file is pre-filtered to one component' if len(components) == 1 else 'UNEXPECTED — should be 1'})")
    ok = (n_before == 254252 and n_nodes == 390273 and len(components) == 1)
    log(f"\n**Road-network figures {'CONFIRMED' if ok else 'DO NOT MATCH'}** against the manuscript's "
        f"pre-filter segment count (254,252) and largest-component node count (390,273). "
        f"Note the manuscript's own stated connectivity percentage (98.9%) is itself a rounding "
        f"error: 390,273/394,874 = 98.835%, which rounds to 98.8%, not 98.9%.\n")
    if not ok:
        record_failure("1.4", f"Road-network figures: n_before={n_before}, n_nodes={n_nodes}, "
                               f"components={len(components)} vs expected 254252/390273/1")

    log(f"### OSM highway classes\n")
    log(f"- Retained ({len(mod02.FCLASS_KEEP)}): {mod02.FCLASS_KEEP}")
    log(f"- Excluded ({len(excluded_classes)}): {excluded_classes}\n")

    munis = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)[["Muni_ID", "Muni_Name", "geometry"]]
    munis = ensure_crs(munis, EPSG, label=MUNICIPALITIES_PTS)
    villages = gpd.read_file(DATA_RAW / VILLAGES_FILE)
    villages = ensure_crs(villages, EPSG, label=VILLAGES_FILE)

    _, muni_dist = snap_with_dist(munis, tree, node_list)
    _, village_dist = snap_with_dist(villages, tree, node_list)

    log(f"### Snapping distances\n")
    fm_text = load_final_manuscript_values()
    n_gt500 = int((village_dist > 500).sum())

    def _check_against_reference(section, label, computed, pattern, tol, is_int=False):
        """Compare one computed value against its labelled line in final_manuscript_values.md.

        Prints CONFIRMED/DIFFERS when a reference value is found, or says
        plainly that no reference value exists rather than inventing one
        to compare against.
        """
        raw = find_reference_value(fm_text, pattern)
        if raw is None:
            log(f"- {label}: {computed}  (no reference value in docs/final_manuscript_values.md)")
            return
        ref = int(raw.replace(",", "")) if is_int else float(raw.replace(",", ""))
        matches = (computed == ref) if is_int else (abs(computed - ref) < tol)
        log(f"- {label}: {computed}  (final_manuscript_values.md: {raw}) "
            f"— {'CONFIRMED' if matches else 'DIFFERS'}")
        if not matches:
            record_failure(section, f"{label}: recomputed {computed}, "
                                     f"final_manuscript_values.md says {raw}")

    _check_against_reference("1.4", "Municipality snapping distance, mean (m)", round(muni_dist.mean(), 1),
                              r"Municipality snapping distance, mean:\s*([\d,.]+)\s*m", 0.05)
    _check_against_reference("1.4", "Municipality snapping distance, max (m)", round(muni_dist.max(), 1),
                              r"Municipality snapping distance, max:\s*([\d,.]+)\s*m", 0.05)
    _check_against_reference("1.4", "Settlement snapping distance, mean (m)", round(village_dist.mean(), 1),
                              r"Settlement snapping distance, mean:\s*([\d,.]+)\s*m", 0.05)
    _check_against_reference("1.4", "Settlement snapping distance, max (m)", round(village_dist.max(), 1),
                              r"Settlement snapping distance, max:\s*([\d,.]+)\s*m", 0.05)
    _check_against_reference("1.4", "Settlements snapped > 500 m", n_gt500,
                              r"Settlements snapped > 500 m:\s*(\d+)", 0, is_int=True)
    log("")

    # Supplementary Table S7 — road network statistics, assembled from the
    # values this section already independently recomputed above (including
    # the largest-component length, summed fresh from roads_noded.gpkg's
    # own length_m column). Two exceptions, both clearly labelled rather
    # than silently presented as equally verified: total nodes before the
    # largest-component filter (394,874) and that filter's percentage
    # (98.8%, the audit-corrected figure — see the CONFIRMED/DIFFERS check
    # above for why not 98.9%) are not recoverable from `roads_noded.gpkg`
    # as saved, since that file is already filtered to a single component —
    # there is no pre-filter graph left to recompute them from without
    # re-noding the raw segments from scratch. They are carried from
    # final_manuscript_values.md / data_audit_report.md's own citation, not
    # independently verified here.
    s7_rows = [
        {"metric": "Road classes retained", "value": f"{len(mod02.FCLASS_KEEP)}: " + "; ".join(mod02.FCLASS_KEEP)},
        {"metric": "Road classes excluded", "value": f"{len(excluded_classes)}: " + "; ".join(excluded_classes)},
        {"metric": "Filtered segments (pre-noding)", "value": n_before},
        {"metric": "Total drivable length, pre-noding (km)", "value": round(total_len_km, 1)},
        {"metric": "Noded segments (largest connected component, saved file)", "value": n_after},
        {"metric": "Graph edges (largest connected component)", "value": n_edges},
        {"metric": "Total nodes before largest-component filter (cited, not independently recomputed here — see note above)", "value": 394874},
        {"metric": "Graph nodes, largest connected component", "value": n_nodes},
        {"metric": "Largest component as % of pre-filter nodes (cited, not independently recomputed here — see note above)", "value": 98.8},
        {"metric": "Largest component drivable length (km)", "value": round(noded["length_m"].sum() / 1000, 1)},
        {"metric": "Municipality snapping distance, mean (m)", "value": round(muni_dist.mean(), 1)},
        {"metric": "Municipality snapping distance, max (m)", "value": round(muni_dist.max(), 1)},
        {"metric": "Settlement snapping distance, mean (m)", "value": round(village_dist.mean(), 1)},
        {"metric": "Settlement snapping distance, max (m)", "value": round(village_dist.max(), 1)},
        {"metric": "Settlements snapped > 500 m", "value": n_gt500},
    ]
    s7_df = pd.DataFrame(s7_rows)
    s7_path = SUPPLEMENTARY / "tableS7_road_network_statistics.csv"
    SUPPLEMENTARY.mkdir(parents=True, exist_ok=True)
    s7_df.to_csv(s7_path, index=False)
    log(f"Saved `{s7_path.name}` (Supplementary Table S7).\n")

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
    """Recompute the settlement-to-municipality distance matrix via Dijkstra and check it against 03_huff_ahp.py's summary."""
    log("## 1.5 Origin-destination matrix audit\n")
    log("`data/processed/distance_matrix.npy` is absent (see 1.1), so this matrix is "
        "reconstructed directly here via the identical Dijkstra procedure used by "
        "`03_huff_ahp.py::compute_distance_matrix`, to recover true pre-fill NaN counts.\n")

    munis = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)[["Muni_ID", "Muni_Name", "geometry"]].copy()
    munis = ensure_crs(munis, EPSG, label=MUNICIPALITIES_PTS)
    villages = gpd.read_file(DATA_RAW / VILLAGES_FILE)
    villages = ensure_crs(villages, EPSG, label=VILLAGES_FILE)
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

    fm_text_1_5 = load_final_manuscript_values()
    shape_match = re.search(r"Shape:\s*([\d,]+)\s*villages x\s*([\d,]+)\s*municipalities", fm_text_1_5) if fm_text_1_5 else None
    if shape_match:
        ref_v, ref_m = int(shape_match.group(1).replace(",", "")), int(shape_match.group(2).replace(",", ""))
        shape_ok = filled.shape == (ref_v, ref_m)
        log(f"- Matrix shape: **{filled.shape}**  (final_manuscript_values.md: {ref_v:,} villages x "
            f"{ref_m:,} municipalities) — {'CONFIRMED' if shape_ok else 'DIFFERS'}")
        if not shape_ok:
            record_failure("1.5", f"Matrix shape: recomputed {filled.shape}, "
                                   f"final_manuscript_values.md says ({ref_v}, {ref_m})")
    else:
        log(f"- Matrix shape: **{filled.shape}**  (no reference value in docs/final_manuscript_values.md)")
    log(f"- Valid (reached-within-cutoff) network distances before fill: "
        f"**{n_valid_pre_fill}/{n_total} ({pct_valid:.2f}%)**")
    log(f"- Missing pairs (unreached within {CUTOFF_M/1000:.0f} km cutoff): **{n_missing}**  "
        f"(filled with the maximum observed distance per municipality)")
    log(f"- Fill method used by the pipeline: each missing (village, muni) pair is filled with "
        f"the **column (municipality) maximum** observed distance — confirmed by reading "
        f"`03_huff_ahp.py::compute_distance_matrix`.")
    log(f"- Distance distribution after fill: min={filled.min():.1f} m, max={filled.max():.1f} m, "
        f"mean={filled.mean():.1f} m, median={np.median(filled):.1f} m")
    log(f"- Settlements with zero distance to their own municipality (village at the "
        f"municipal seat): **{n_zero}**\n")

    # Compared against docs/final_manuscript_values.md, not a hardcoded
    # literal — this used to compare against 2,107 missing pairs / 99.84%,
    # which was the pre-road-network-fix figure and had been silently
    # stale (reporting DO NOT MATCH on every run without anyone treating
    # it as a failure) since the network was corrected. See
    # docs/reproducibility_note.md's OD-matrix discussion for why the
    # missing-pair count changed.
    missing_ref = re.search(r"Missing pairs.*?:\s*([\d,]+)", fm_text_1_5) if fm_text_1_5 else None
    valid_pct_ref = re.search(r"Valid.*?pairs:\s*([\d.]+)%", fm_text_1_5) if fm_text_1_5 else None
    if missing_ref and valid_pct_ref:
        ref_missing = int(missing_ref.group(1).replace(",", ""))
        ref_pct = float(valid_pct_ref.group(1))
        ok = (n_missing == ref_missing and abs(pct_valid - ref_pct) < 0.02)
        log(f"**OD-matrix figures vs final_manuscript_values.md ({ref_missing} missing pairs, "
            f"{ref_pct}% valid): {'CONFIRMED' if ok else 'DIFFERS'}.**\n")
        if not ok:
            record_failure("1.5", f"OD-matrix figures: n_missing={n_missing} ({pct_valid:.2f}% valid) "
                                   f"vs final_manuscript_values.md's {ref_missing} ({ref_pct}% valid)")
    else:
        log("**OD-matrix figures: no reference value in docs/final_manuscript_values.md.**\n")

    return {"shape": filled.shape, "n_valid_pre_fill": n_valid_pre_fill, "pct_valid": pct_valid,
            "n_missing": n_missing, "min": filled.min(), "max": filled.max(),
            "mean": filled.mean(), "median": float(np.median(filled)), "n_zero": n_zero}


# ══════════════════════════════════════════════════════════════
# 1.6 — Accessibility indicator audit (recomputed via the two-stage Dijkstra)
# ══════════════════════════════════════════════════════════════

def section_1_6(G, node_list, node_coords, tree, mod05, facility_paths):
    """Recompute the accessibility scores via the same two-stage Dijkstra as 05_accessibility.py and check them against its output."""
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
    munis = ensure_crs(munis, EPSG, label=MUNICIPALITIES_PTS)
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

    log(f"- Facility types discovered: **{n_facility_types}**")
    log(f"- Total distance values computed (munis x facility types): **{n_total_values}**")
    log(f"- Values missing after the {PRIMARY_ACC_CUTOFF_M/1000:.0f} km primary search: "
        f"**{n_missing_after_primary}**")
    log(f"- Values recovered by the {CUTOFF_M/1000:.0f} km extended search: "
        f"**{n_recovered_by_extended}**")
    log(f"- Values remaining missing after both searches (filled with the column maximum): "
        f"**{n_still_missing}**")
    if still_missing_pairs:
        log(f"  - Remaining-missing (municipality, facility_type) pairs: {still_missing_pairs}")
    log("")

    # facility-type count and total-value count have a maintained reference
    # in docs/final_manuscript_values.md and gate record_failure below. The
    # missing/recovered/still-missing breakdown has no reference value
    # there (only ever compared against a stale pre-road-network-fix
    # literal, 1111/1105/6) — reported for visibility, per "say so rather
    # than inventing a target," but not gated until that reference exists.
    fm_text_1_6 = load_final_manuscript_values()
    types_ref = re.search(r"Facility types \(accessibility\):\s*(\d+)", fm_text_1_6) if fm_text_1_6 else None
    values_ref = re.search(r"Accessibility values computed.*?:\s*([\d,]+)", fm_text_1_6) if fm_text_1_6 else None
    if types_ref and values_ref:
        ref_types, ref_values = int(types_ref.group(1)), int(values_ref.group(1).replace(",", ""))
        ok = (n_facility_types == ref_types and n_total_values == ref_values)
        log(f"**Accessibility figures vs final_manuscript_values.md ({ref_types} types, "
            f"{ref_values:,} values): {'CONFIRMED' if ok else 'DIFFERS'}.**")
        if not ok:
            record_failure("1.6", f"Accessibility figures: types={n_facility_types}, "
                                   f"total_values={n_total_values} vs final_manuscript_values.md's "
                                   f"{ref_types}/{ref_values}")
    else:
        log("**Accessibility figures: no reference value in docs/final_manuscript_values.md.**")
    log(f"(missing/recovered/still-missing breakdown — {n_missing_after_primary}/"
        f"{n_recovered_by_extended}/{n_still_missing} — has no maintained reference value yet; "
        "not gated.)\n")

    return {"n_facility_types": n_facility_types, "n_total_values": n_total_values,
            "n_missing_after_primary": n_missing_after_primary,
            "n_recovered_by_extended": n_recovered_by_extended,
            "n_still_missing": n_still_missing, "still_missing_pairs": still_missing_pairs}


# ══════════════════════════════════════════════════════════════
# 1.7 — Machine learning audit
# ══════════════════════════════════════════════════════════════

def _feature_group(feature, indicator_group_map):
    """Sort one ML feature name into a thematic category, for the ML audit's feature breakdown.

    GI_AHP and GI_NW are each their own model's single composite score
    feature, not an "Other" — a previous version of this function only
    special-cased "GI_AHP" by name, so ml_NW_feature_importance.csv's
    GI_NW row silently fell through to "Other" instead of getting its own
    category, understating "Other" and hiding the composite score's real
    contribution whenever the NW model's breakdown was audited.
    """
    if feature == "dist_to_muni":
        return "Distance"
    if feature in ("GI_AHP", "GI_NW"):
        return feature
    if feature.startswith("nacc_"):
        return "Accessibility"
    if feature == "n_Area_km2":
        return "Municipality area"
    if feature.startswith("n_"):
        return "Individual GI indicators"
    return "Other"


def section_1_7():
    """Check the ML training table's row/feature counts and the spatial cross-validation results against 06_ml_framework.py's own output."""
    log("## 1.7 Machine learning audit\n")

    mpts = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)
    mpts = ensure_crs(mpts, EPSG, label=MUNICIPALITIES_PTS)
    n_cols = [c for c in mpts.columns if c not in ("Muni_ID", "Muni_Name", "geometry", "n_Fitness_C")]
    acc = pd.read_csv(TABLES / "accessibility_normalized.csv")
    nacc_cols = [c for c in acc.columns if c.startswith("nacc_")]
    feature_cols = n_cols + nacc_cols + ["GI_AHP", "dist_to_muni"]

    area_in_features = "n_Area_km2" in n_cols
    log(f"- `n_Area_km2` present in the ML feature set (`06_ml_framework.py::build_municipality_features` "
        f"keeps every raw `n_` column except `n_Fitness_C`): **{area_in_features}** "
        f"({'CONFIRMED — complements 1.2, which excludes it from the GI' if area_in_features else 'MISSING — was excluded here too'})\n")
    if not area_in_features:
        record_failure("1.7", "n_Area_km2 expected in the ML feature set (excluded from the GI, "
                               "but should still be an ML feature) — it is absent")

    # Row count is derived from the actual OD matrix on disk, not from
    # N_SETTLEMENTS * N_MUNICIPALITIES — those are the same two config.py
    # constants the expected total is defined from, so multiplying them
    # together and comparing the result to a number derived the same way
    # can never disagree: it is arithmetic on two hardcoded literals, not
    # a check of anything this pipeline actually computed. Reading
    # huff_od_matrix.csv's real shape means a genuine problem upstream
    # (a dropped settlement, a broken merge) would actually show up here.
    od = pd.read_csv(TABLES / "huff_od_matrix.csv")
    n_villages_actual = len(od)
    n_munis_actual = len([c for c in od.columns if c.startswith("Pij_")])
    n_rows = n_villages_actual * n_munis_actual
    log(f"- Training table row count (from `huff_od_matrix.csv`: "
        f"{n_villages_actual:,} villages x {n_munis_actual} municipalities): **{n_rows:,}**")
    log(f"- Feature count: **{len(feature_cols)}** — "
        f"breakdown: {len(n_cols)} raw GI indicators (incl. `n_Area_km2`), "
        f"{len(nacc_cols)} accessibility, 1 GI_AHP, 1 distance-to-municipality\n")

    fm_text_1_7a = load_final_manuscript_values()
    shape_ref = re.search(r"Training table:\s*([\d,]+)\s*rows x\s*([\d,]+)\s*features",
                           fm_text_1_7a) if fm_text_1_7a else None
    if shape_ref:
        ref_rows, ref_features = int(shape_ref.group(1).replace(",", "")), int(shape_ref.group(2))
        ok_shape = (n_rows == ref_rows and len(feature_cols) == ref_features)
        log(f"**Row/feature counts vs final_manuscript_values.md ({ref_rows:,} rows x {ref_features} "
            f"features): {'CONFIRMED' if ok_shape else 'DIFFERS'}.**\n")
        if not ok_shape:
            record_failure("1.7", f"Row/feature counts: recomputed {n_rows:,} rows x {len(feature_cols)} "
                                   f"features, final_manuscript_values.md says {ref_rows:,} rows x "
                                   f"{ref_features} features")
    else:
        log("**Row/feature counts: no reference value in docs/final_manuscript_values.md.**\n")

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
        log(f"- {counts.to_dict()}")
        actual_blocks = sorted(counts.tolist())
        # Compared against docs/final_manuscript_values.md's own {block: size}
        # dict, not a hardcoded literal — the previous hardcoded target
        # (52, 34, 24, 61, 41) was stale from before the road-network fix
        # and had been silently reporting DO NOT MATCH on every run.
        fm_text_1_7b = load_final_manuscript_values()
        blocks_ref = re.search(r"Spatial CV blocks.*?:\s*\{([^}]+)\}", fm_text_1_7b) if fm_text_1_7b else None
        if blocks_ref:
            ref_blocks = sorted(int(v.strip()) for v in re.findall(r":\s*(\d+)", blocks_ref.group(1)))
            blocks_ok = actual_blocks == ref_blocks
            log(f"\n**Block sizes vs final_manuscript_values.md ({ref_blocks}): "
                f"{'CONFIRMED (as a set)' if blocks_ok else 'DIFFERS'}.**\n")
            if not blocks_ok:
                record_failure("1.7", f"Spatial block sizes: {actual_blocks} vs "
                                       f"final_manuscript_values.md's {ref_blocks}")
        else:
            log("\n**Block sizes: no reference value in docs/final_manuscript_values.md.**\n")
    else:
        log("- `spatial_blocks.csv` cache not found — cannot verify block sizes.\n")

    log("### Sample fraction per fold\n")
    log("`06_ml_framework.py` accepts `--sample-frac` (default `1.0`); when < 1.0, "
        "`train_rf_spatial_cv` subsamples the **training** rows only "
        "(`train_df.sample(frac=sample_frac, random_state=42)`) — the held-out test "
        "fold and therefore every settlement's out-of-fold prediction is unaffected. "
        "The documented reproduction commands "
        "(`06_ml_framework.py --model AHP --sample-frac 1.0`, then separately "
        "`--model NW --sample-frac 1.0` — run as two invocations, not `--model both`, "
        "to avoid holding both models' feature tables in memory at once; see "
        "`docs/reproducibility_note.md`) run both models on the full training sample "
        "in every fold. An earlier version of this repository ran the NW model on a "
        "30% sample for memory reasons; that is no longer the documented or "
        "recommended way to reproduce this repository's saved results, and should not "
        "be assumed for whichever `ml_NW_cv_results.csv` is currently on disk unless "
        "it was regenerated with the commands above.\n")

    log("### Cross-validation performance\n")
    for label, path in [("AHP", TABLES / "ml_AHP_cv_results.csv"), ("NW", TABLES / "ml_NW_cv_results.csv")]:
        cv = pd.read_csv(path)
        log(f"**{label}** (`{path.name}`):\n")
        log(cv.to_string(index=False))
        log(f"\n- Mean R² = {cv['r2'].mean():.4f} ± {cv['r2'].std():.4f}  "
            f"(see `docs/final_manuscript_values.md` for the currently published figure)")
        log(f"- Mean MAE = {cv['mae'].mean():.6f}, Mean RMSE = {cv['rmse'].mean():.6f}\n")

    log("### Feature importance by group\n")
    group_map = {}
    ref = pd.read_csv(SUPPLEMENTARY / "tableS3_individual_indicator_weights.csv")
    ref = ref.dropna(subset=["Indicator_code"])
    ref = ref[ref["Indicator_code"] != "Total"]

    def _parse_feature_importance_reference(text, gi_label):
        """Extract the five feature-group percentages for one model from final_manuscript_values.md."""
        if text is None:
            return None
        pattern = (r"Feature importance by group:\s*Distance\s*([\d.]+)%,\s*"
                   + re.escape(gi_label) + r"\s*([\d.]+)%,\s*Accessibility\s*([\d.]+)%,\s*"
                   r"Individual GI indicators\s*([\d.]+)%,\s*Municipality area\s*([\d.]+)%")
        m = re.search(pattern, text)
        if not m:
            return None
        return {"Distance": float(m.group(1)), gi_label: float(m.group(2)),
                "Accessibility": float(m.group(3)), "Individual GI indicators": float(m.group(4)),
                "Municipality area": float(m.group(5))}

    fm_text_1_7 = load_final_manuscript_values()

    for label, gi_label, path in [("AHP", "GI_AHP", TABLES / "ml_AHP_feature_importance.csv"),
                                   ("NW", "GI_NW", TABLES / "ml_NW_feature_importance.csv")]:
        if not path.exists():
            log(f"- **{label}**: `{path.name}` not found — skipped.\n")
            continue
        fi = pd.read_csv(path)
        fi["group"] = fi["feature"].apply(lambda f: _feature_group(f, group_map))
        by_group = fi.groupby("group")["importance"].sum().sort_values(ascending=False)
        total = fi["importance"].sum()
        computed_pct = {g: 100 * v / total for g, v in by_group.items()}
        log(f"**{label}** feature importance by group (of total {total:.4f}):\n")
        for g, v in by_group.items():
            log(f"- {g}: {100*v/total:.2f}%")
        log("")

        ref_pct = _parse_feature_importance_reference(fm_text_1_7, gi_label)
        if ref_pct is None:
            log(f"(no reference value in docs/final_manuscript_values.md for the {label} model)\n")
            continue
        diffs = {g: abs(computed_pct.get(g, 0.0) - v) for g, v in ref_pct.items()}
        max_diff = max(diffs.values())
        matches = max_diff < 0.5
        log(f"(final_manuscript_values.md: " + ", ".join(f"{g} {v:.2f}%" for g, v in ref_pct.items())
            + f") — {'CONFIRMED' if matches else 'DIFFERS'} (max |diff| = {max_diff:.2f} pp)\n")
        if not matches:
            record_failure("1.7", f"{label} feature importance by group: max |diff| = {max_diff:.2f} pp "
                                   f"vs final_manuscript_values.md ({diffs})")


# ══════════════════════════════════════════════════════════════
# 1.8 — Manuscript number verification
# ══════════════════════════════════════════════════════════════

def _agreement_from_gpkg(path, col="agreement"):
    """Read one comparison map and return how many of its settlements agree, out of the total."""
    gdf = gpd.read_file(path)
    n_agree = int(gdf[col].sum())
    n_total = len(gdf)
    return n_agree, n_total


def section_1_8():
    """Check a fixed list of specific numbers quoted in the manuscript draft against the pipeline's current output, and separately against the maintained reference doc.

    Two different comparisons, on purpose. `draft_value` is what the
    *original* manuscript draft claimed — informational only, since this
    repository's rerun is expected to differ from it permanently (the
    road-network provenance gap documented at length in
    docs/reproducibility_note.md's "Two residual discrepancies, honestly"
    section is not a bug to fix). `reference_value`, pulled from
    docs/final_manuscript_values.md — the document this repository
    actually maintains as current truth — is what gates `record_failure`:
    if this run's own output no longer matches what was already verified
    and written down there, that is a real regression, not an old,
    understood gap.
    """
    log("## 1.8 Manuscript number verification\n")
    rows = []
    fm_text_1_8 = load_final_manuscript_values()

    def ref(pattern, group=1):
        return find_reference_value(fm_text_1_8, pattern, group)

    def refs(pattern, groups=1):
        """Like ref(), but returns every match (as tuples) for a pattern that repeats in the doc."""
        if fm_text_1_8 is None:
            return []
        return re.findall(pattern, fm_text_1_8)

    def add_ref_check(row, ref_value_str, matches):
        """Attach a reference_value / ref_status pair to a row already appended to `rows`.

        matches=None means no reference value could be found — noted, but
        never gates record_failure, per the same "say so rather than
        inventing a target" rule used elsewhere in this file.
        """
        row["reference_value"] = ref_value_str if ref_value_str is not None else "(none in final_manuscript_values.md)"
        # `matches is True` / `is False` deliberately avoided: most callers
        # pass the result of a numpy/pandas comparison (numpy.bool_), and
        # `numpy.bool_(True) is True` is False — identity, not equality —
        # which silently sent every genuine numpy-bool match (and, more
        # dangerously, every genuine numpy-bool MISMATCH) into "no
        # reference value" instead of being recorded at all. Caught by
        # actually running this section end to end rather than trusting
        # the code read right.
        if matches is None:
            row["ref_status"] = "no reference value"
        elif bool(matches):
            row["ref_status"] = "CONFIRMED vs reference"
        else:
            row["ref_status"] = "DIFFERS vs reference"

    # AHP vs NW
    n_agree, n_total = _agreement_from_gpkg(GPKG / "map_AHP_vs_NW_villages.gpkg")
    pct = 100 * n_agree / n_total
    rows.append({"claim": "AHP vs NW agreement", "draft_value": "88.6%, 5349/6036",
                 "repository_value": f"{pct:.2f}%, {n_agree}/{n_total}",
                 "status": "CONFIRMED" if (n_agree == 5349) else "DIFFERS",
                 "source_file": "outputs/gpkg/map_AHP_vs_NW_villages.gpkg"})
    r_agree = ref(r"AHP vs NW: n_agree ([\d,]+) / [\d,]+")
    add_ref_check(rows[-1], f"{r_agree}/6036" if r_agree else None,
                  None if r_agree is None else n_agree == int(r_agree.replace(",", "")))

    n_agree, n_total = _agreement_from_gpkg(GPKG / "map_AHP_vs_ML_villages.gpkg")
    pct = 100 * n_agree / n_total
    rows.append({"claim": "AHP vs ML agreement", "draft_value": "75.4%, 4551/6036",
                 "repository_value": f"{pct:.2f}%, {n_agree}/{n_total}",
                 "status": "CONFIRMED" if (n_agree == 4551) else "DIFFERS",
                 "source_file": "outputs/gpkg/map_AHP_vs_ML_villages.gpkg"})
    r_agree = ref(r"AHP vs ML: n_agree ([\d,]+) / [\d,]+")
    add_ref_check(rows[-1], f"{r_agree}/6036" if r_agree else None,
                  None if r_agree is None else n_agree == int(r_agree.replace(",", "")))

    n_agree, n_total = _agreement_from_gpkg(GPKG / "map_NW_vs_ML_villages.gpkg")
    pct = 100 * n_agree / n_total
    rows.append({"claim": "NW vs ML agreement", "draft_value": "77.4%, 4672/6036",
                 "repository_value": f"{pct:.2f}%, {n_agree}/{n_total}",
                 "status": "CONFIRMED" if (n_agree == 4672) else "DIFFERS",
                 "source_file": "outputs/gpkg/map_NW_vs_ML_villages.gpkg"})
    r_agree = ref(r"NW vs ML: n_agree ([\d,]+) / [\d,]+")
    add_ref_check(rows[-1], f"{r_agree}/6036" if r_agree else None,
                  None if r_agree is None else n_agree == int(r_agree.replace(",", "")))

    euc = pd.read_csv(TABLES / "table_euclidean_vs_network.csv")
    n_agree = int(euc["agreement"].sum())
    n_total_euc = len(euc)
    pct_euc = 100.0 * n_agree / n_total_euc
    kappa_euc = cohen_kappa_score(euc["network_dominant_muni"], euc["euclidean_dominant_muni"])
    rows.append({"claim": "Euclidean vs network agreement", "draft_value": "88.4%, 5335/6036, kappa 0.879",
                 "repository_value": f"{pct_euc:.2f}%, {n_agree}/{n_total_euc}, kappa {kappa_euc:.4f}",
                 "status": "CONFIRMED" if n_agree == 5335 else "DIFFERS",
                 "source_file": "outputs/tables/table_euclidean_vs_network.csv"})
    r_agree = ref(r"Euclidean vs network-distance agreement: n_agree ([\d,]+) / [\d,]+")
    add_ref_check(rows[-1], f"{r_agree}/6036" if r_agree else None,
                  None if r_agree is None else n_agree == int(r_agree.replace(",", "")))

    mi = pd.read_csv(TABLES / "table_morans_i_results.csv")
    mi_ref_patterns = {
        "AHP_vs_NW": r"AHP vs NW: n_agree.*?Moran's I ([\d.]+)",
        "AHP_vs_ML": r"AHP vs ML: n_agree.*?Moran's I ([\d.]+)",
        "NW_vs_ML": r"NW vs ML: n_agree.*?Moran's I ([\d.]+)",
    }
    for label, draft, draft_I in [("AHP_vs_NW", "0.184, z 24.31", 0.184),
                                   ("AHP_vs_ML", "0.451, z 59.55", 0.451),
                                   ("NW_vs_ML", "0.447, z 57.40", 0.447)]:
        r = mi[mi["layer"] == label]
        if not len(r):
            # table_morans_i_results.csv exists (the read above would already
            # have raised FileNotFoundError otherwise) but is missing a row
            # 12_morans_lisa.py always produces in one pass — a partial or
            # stale file, not a "not yet run" state a soft status can paper
            # over.
            raise RuntimeError(
                f"outputs/tables/table_morans_i_results.csv has no '{label}' row — "
                f"rows present: {sorted(mi['layer'].unique())}. Re-run src/12_morans_lisa.py; "
                "it should always produce all four comparisons in one pass."
            )
        I, z = r.iloc[0]["morans_I"], r.iloc[0]["z_score"]
        i_matches = abs(I - draft_I) < 0.002
        rows.append({"claim": f"Moran's I {label.replace('_', ' ')}", "draft_value": draft,
                     "repository_value": f"I={I:.4f}, z={z:.2f}",
                     "status": ("CONFIRMED (I; z is a permutation estimate and fluctuates run to run)"
                                if i_matches else
                                f"DIFFERS (I) — draft {draft_I}, repository {I:.4f}; "
                                "z is a permutation estimate and fluctuates run to run"),
                     "source_file": "outputs/tables/table_morans_i_results.csv"})
        r_I = ref(mi_ref_patterns[label])
        add_ref_check(rows[-1], f"I={r_I}" if r_I else None,
                      None if r_I is None else abs(I - float(r_I)) < 0.002)

    # No soft fallback here: this check needs table_lisa_summary.csv with an
    # AHP_vs_ML row to mean anything at all. A previous version of this
    # block handled a missing file by printing a "FLAGGED — cannot be
    # checked" row containing a paragraph of narrative about a repository
    # state from before all four LISA comparisons existed (a single
    # AHP-vs-NW-only layer, with specific stale counts baked into the
    # text) — that branch could never fire once 12_morans_lisa.py started
    # producing all four comparisons, but the stale prose would still have
    # printed as if current had it ever been reached again. Raising here
    # instead means a genuinely missing prerequisite stops the run loudly,
    # rather than being described by a paragraph nobody kept up to date.
    lisa_summary_path = TABLES / "table_lisa_summary.csv"
    if not lisa_summary_path.exists():
        raise RuntimeError(
            f"{lisa_summary_path} not found — run src/12_morans_lisa.py before "
            "src/17_data_audit.py; section 1.8's LISA AHP vs ML check has no "
            "meaningful fallback without it."
        )
    lisa_df = pd.read_csv(lisa_summary_path)
    r = lisa_df[lisa_df["comparison"] == "AHP_vs_ML"]
    if not len(r):
        raise RuntimeError(
            f"{lisa_summary_path} exists but has no AHP_vs_ML row — "
            f"comparisons present: {sorted(lisa_df['comparison'].unique())}. "
            "Re-run src/12_morans_lisa.py; it should always produce all four comparisons."
        )
    r = r.iloc[0]
    n_sig = int(r["HH"] + r["LL"] + r["HL"] + r["LH"])
    draft_ll, draft_hl, draft_lh, draft_sig = 706, 255, 116, 1077
    counts_match = (r["LL"] == draft_ll and r["HL"] == draft_hl
                     and r["LH"] == draft_lh and n_sig == draft_sig)
    rows.append({"claim": "LISA AHP vs ML", "draft_value": "706 LL, 255 HL, 116 LH, 1077 significant",
                 "repository_value": f"HH={r['HH']}, LL={r['LL']}, HL={r['HL']}, LH={r['LH']}, "
                                      f"significant={n_sig}",
                 "status": ("CONFIRMED" if counts_match else
                            f"DIFFERS — draft LL={draft_ll}/HL={draft_hl}/LH={draft_lh}/"
                            f"sig={draft_sig} vs repository LL={r['LL']}/HL={r['HL']}/"
                            f"LH={r['LH']}/sig={n_sig}"),
                 "source_file": "outputs/gpkg/map_lisa_AHP_vs_ML.gpkg"})
    lisa_ref = re.search(r"AHP vs ML: HH (\d+), LL (\d+), HL (\d+), LH (\d+)", fm_text_1_8) if fm_text_1_8 else None
    if lisa_ref:
        r_hh, r_ll, r_hl, r_lh = (int(x) for x in lisa_ref.groups())
        ref_match = (r["HH"] == r_hh and r["LL"] == r_ll and r["HL"] == r_hl and r["LH"] == r_lh)
        add_ref_check(rows[-1], f"HH={r_hh}, LL={r_ll}, HL={r_hl}, LH={r_lh}", ref_match)
    else:
        add_ref_check(rows[-1], None, None)

    ahp_sum = pd.read_csv(TABLES / "huff_AHP_summary.csv")
    nw_sum = pd.read_csv(TABLES / "huff_NW_summary.csv")
    lj_ahp = int((ahp_sum["dominant_municipality"] == "Ljubljana").sum())
    lj_nw = int((nw_sum["dominant_municipality"] == "Ljubljana").sum())
    rows.append({"claim": "Ljubljana catchment AHP", "draft_value": "1222 settlements",
                 "repository_value": f"{lj_ahp} settlements",
                 "status": "CONFIRMED" if lj_ahp == 1222 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})
    r_lj_ahp = ref(r"Top 15, AHP weighting: Ljubljana ([\d,]+)")
    add_ref_check(rows[-1], f"{r_lj_ahp} settlements" if r_lj_ahp else None,
                  None if r_lj_ahp is None else lj_ahp == int(r_lj_ahp.replace(",", "")))

    rows.append({"claim": "Ljubljana catchment NW", "draft_value": "1022 settlements",
                 "repository_value": f"{lj_nw} settlements",
                 "status": "CONFIRMED" if lj_nw == 1022 else "DIFFERS",
                 "source_file": "outputs/tables/huff_NW_summary.csv"})
    r_lj_nw = ref(r"Top 15, NW weighting: Ljubljana ([\d,]+)")
    add_ref_check(rows[-1], f"{r_lj_nw} settlements" if r_lj_nw else None,
                  None if r_lj_nw is None else lj_nw == int(r_lj_nw.replace(",", "")))

    catchment_sizes = ahp_sum["dominant_municipality"].value_counts()
    top5 = catchment_sizes.head(5).sum()
    top10 = catchment_sizes.head(10).sum()
    n_total_v = len(ahp_sum)
    rows.append({"claim": "Top 5 catchments combined", "draft_value": "2147, 35.6%",
                 "repository_value": f"{top5}, {100*top5/n_total_v:.1f}%",
                 "status": "CONFIRMED" if top5 == 2147 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})
    top5_ref = re.search(r"Top 5 catchments combined \(AHP\): ([\d,]+) settlements, ([\d.]+)%", fm_text_1_8) if fm_text_1_8 else None
    if top5_ref:
        r_top5, r_top5_pct = int(top5_ref.group(1).replace(",", "")), float(top5_ref.group(2))
        add_ref_check(rows[-1], f"{r_top5}, {r_top5_pct}%",
                      top5 == r_top5 and abs(100*top5/n_total_v - r_top5_pct) < 0.05)
    else:
        add_ref_check(rows[-1], None, None)

    rows.append({"claim": "Top 10 catchments combined", "draft_value": "2830, 46.9%",
                 "repository_value": f"{top10}, {100*top10/n_total_v:.1f}%",
                 "status": "CONFIRMED" if top10 == 2830 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})
    top10_ref = re.search(r"Top 10 catchments combined \(AHP\): ([\d,]+) settlements, ([\d.]+)%", fm_text_1_8) if fm_text_1_8 else None
    if top10_ref:
        r_top10, r_top10_pct = int(top10_ref.group(1).replace(",", "")), float(top10_ref.group(2))
        add_ref_check(rows[-1], f"{r_top10}, {r_top10_pct}%",
                      top10 == r_top10 and abs(100*top10/n_total_v - r_top10_pct) < 0.05)
    else:
        add_ref_check(rows[-1], None, None)

    n_single = int((catchment_sizes == 1).sum())
    rows.append({"claim": "Municipalities with 1 settlement", "draft_value": "32",
                 "repository_value": f"{n_single}",
                 "status": "CONFIRMED" if n_single == 32 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})
    r_single = ref(r"Single-settlement municipalities: (\d+)")
    add_ref_check(rows[-1], r_single, None if r_single is None else n_single == int(r_single))

    rows.append({"claim": "Mean catchment size", "draft_value": "28.5, median 6.0",
                 "repository_value": f"{catchment_sizes.mean():.1f}, median {catchment_sizes.median():.1f}",
                 "status": "CONFIRMED" if abs(catchment_sizes.mean() - 28.5) < 0.1 and catchment_sizes.median() == 6.0 else "DIFFERS",
                 "source_file": "outputs/tables/huff_AHP_summary.csv"})
    r_mean = ref(r"Mean catchment size: ([\d.]+)")
    r_median = ref(r"Median catchment size: ([\d.]+)")
    if r_mean and r_median:
        add_ref_check(rows[-1], f"{r_mean}, median {r_median}",
                      abs(catchment_sizes.mean() - float(r_mean)) < 0.1
                      and catchment_sizes.median() == float(r_median))
    else:
        add_ref_check(rows[-1], None, None)

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
    add_ref_check(rows[-1], None, None)  # not itemised anywhere in final_manuscript_values.md

    ent = pd.read_csv(TABLES / "table_entropy_summary.csv")
    ent_ahp = ent[ent["GI_scenario"] == "AHP"].iloc[0]
    ent_nw = ent[ent["GI_scenario"] == "NW"].iloc[0]
    draft_mean_ahp, draft_mean_nw = 0.51, 0.53
    means_match = (abs(ent_ahp["mean"] - draft_mean_ahp) < 0.01
                   and abs(ent_nw["mean"] - draft_mean_nw) < 0.01)
    # Manuscript (Section 4.5 prose) states only "mean ~0.51 (AHP) / ~0.53 (NW), range near 0 to
    # ~0.80" — it gives no precise max or low/medium/high class-count breakdown for either
    # scenario. The 0.801/0.847 max and exact class counts previously hardcoded here as the
    # "draft" target were fabricated by an earlier version of this check, not read from the
    # manuscript (see docs/audit-history/reconciliation.csv rows 63-64).
    rows.append({"claim": "Entropy AHP/NW", "draft_value": "AHP mean ~0.51, NW mean ~0.53 "
                                                            "(Section 4.5 prose only; no manuscript "
                                                            "max or class-count breakdown exists)",
                 "repository_value": f"AHP mean={ent_ahp['mean']:.4f} max={ent_ahp['max']:.4f} "
                                      f"(low={ent_ahp['n_low']}/med={ent_ahp['n_medium']}/"
                                      f"high={ent_ahp['n_high']}); "
                                      f"NW mean={ent_nw['mean']:.4f} max={ent_nw['max']:.4f} "
                                      f"(low={ent_nw['n_low']}/med={ent_nw['n_medium']}/"
                                      f"high={ent_nw['n_high']})",
                 "status": ("CONFIRMED (means only; no manuscript value exists for max or class counts)"
                            if means_match else
                            f"DIFFERS — mean AHP {ent_ahp['mean']:.4f} vs draft ~{draft_mean_ahp}, "
                            f"mean NW {ent_nw['mean']:.4f} vs draft ~{draft_mean_nw}"),
                 "source_file": "outputs/tables/table_entropy_summary.csv"})
    ahp_ent_ref = re.search(r"Entropy, AHP weighting: min [\-\d.]+, max ([\d.]+), mean ([\d.]+)", fm_text_1_8) if fm_text_1_8 else None
    nw_ent_ref = re.search(r"Entropy, NW weighting: min [\-\d.]+, max ([\d.]+), mean ([\d.]+)", fm_text_1_8) if fm_text_1_8 else None
    if ahp_ent_ref and nw_ent_ref:
        r_ahp_max, r_ahp_mean = float(ahp_ent_ref.group(1)), float(ahp_ent_ref.group(2))
        r_nw_max, r_nw_mean = float(nw_ent_ref.group(1)), float(nw_ent_ref.group(2))
        ent_ref_match = (abs(ent_ahp["mean"] - r_ahp_mean) < 0.001 and abs(ent_ahp["max"] - r_ahp_max) < 0.001
                          and abs(ent_nw["mean"] - r_nw_mean) < 0.001 and abs(ent_nw["max"] - r_nw_max) < 0.001)
        add_ref_check(rows[-1], f"AHP mean={r_ahp_mean} max={r_ahp_max}; NW mean={r_nw_mean} max={r_nw_max}",
                      ent_ref_match)
    else:
        add_ref_check(rows[-1], None, None)

    ahp_cv = pd.read_csv(TABLES / "ml_AHP_cv_results.csv")
    nw_cv = pd.read_csv(TABLES / "ml_NW_cv_results.csv")
    r2_refs = refs(r"Mean R² ± sd: ([\d.]+) ± ([\d.]+)") if fm_text_1_8 else []

    rows.append({"claim": "RF AHP mean R squared", "draft_value": "0.845 +/- 0.088",
                 "repository_value": f"{ahp_cv['r2'].mean():.3f} +/- {ahp_cv['r2'].std():.3f}",
                 "status": "CONFIRMED" if abs(ahp_cv['r2'].mean() - 0.845) < 0.002 else "DIFFERS (rounding)",
                 "source_file": "outputs/tables/ml_AHP_cv_results.csv"})
    if len(r2_refs) >= 1:
        r_ahp_r2, r_ahp_sd = (float(x) for x in r2_refs[0])
        add_ref_check(rows[-1], f"{r_ahp_r2} +/- {r_ahp_sd}",
                      abs(ahp_cv['r2'].mean() - r_ahp_r2) < 0.002)
    else:
        add_ref_check(rows[-1], None, None)

    rows.append({"claim": "RF NW mean R squared", "draft_value": "0.844 +/- 0.066",
                 "repository_value": f"{nw_cv['r2'].mean():.3f} +/- {nw_cv['r2'].std():.3f}",
                 "status": "CONFIRMED" if abs(nw_cv['r2'].mean() - 0.844) < 0.002 else "DIFFERS (rounding)",
                 "source_file": "outputs/tables/ml_NW_cv_results.csv"})
    if len(r2_refs) >= 2:
        r_nw_r2, r_nw_sd = (float(x) for x in r2_refs[1])
        add_ref_check(rows[-1], f"{r_nw_r2} +/- {r_nw_sd}",
                      abs(nw_cv['r2'].mean() - r_nw_r2) < 0.002)
    else:
        add_ref_check(rows[-1], None, None)

    commute = pd.read_csv(TABLES / "table_huff_vs_commuting_summary.csv")
    n_agree_com = int(commute['n_agree'].iloc[0])
    n_muni_com = int(commute['n_municipalities'].iloc[0])
    kappa_com = commute['cohen_kappa'].iloc[0]
    commute_matches = (n_agree_com == 146 and n_muni_com == 212 and abs(kappa_com - 0.676) < 0.002)
    rows.append({"claim": "Commuting agreement", "draft_value": "68.9%, 146/212, kappa 0.676",
                 "repository_value": f"{commute['agreement_pct'].iloc[0]:.2f}%, "
                                      f"{n_agree_com}/{n_muni_com}, kappa {kappa_com:.4f}",
                 "status": "CONFIRMED" if commute_matches else "DIFFERS",
                 "source_file": "outputs/tables/table_huff_vs_commuting_summary.csv"})
    commute_ref = re.search(r"Agreement: ([\d.]+)% \(([\d,]+) / ([\d,]+) municipalities\)", fm_text_1_8) if fm_text_1_8 else None
    kappa_ref = ref(r"Cohen's kappa: ([\d.]+)")
    if commute_ref and kappa_ref:
        r_pct, r_n_agree, r_n_muni = float(commute_ref.group(1)), int(commute_ref.group(2)), int(commute_ref.group(3))
        r_kappa = float(kappa_ref)
        add_ref_check(rows[-1], f"{r_pct}%, {r_n_agree}/{r_n_muni}, kappa {r_kappa}",
                      n_agree_com == r_n_agree and n_muni_com == r_n_muni and abs(kappa_com - r_kappa) < 0.002)
    else:
        add_ref_check(rows[-1], None, None)

    commute_full = pd.read_csv(TABLES / "table_huff_vs_commuting.csv")
    n_centres = int(commute_full["commuting_is_centre"].sum())
    rows.append({"claim": "Commuting functional centres", "draft_value": "101",
                 "repository_value": f"{n_centres}",
                 "status": "CONFIRMED" if n_centres == 101 else "DIFFERS",
                 "source_file": "outputs/tables/table_huff_vs_commuting.csv"})
    r_centres = ref(r"Functional centres \(Huff self-flow dominant\): (\d+)")
    add_ref_check(rows[-1], r_centres, None if r_centres is None else n_centres == int(r_centres))

    beta = pd.read_csv(TABLES / "table_beta_sensitivity_clean.csv")
    non_trivial = beta[beta["beta"] != BETA]
    krange = f"{non_trivial['cohen_kappa'].min():.3f} to {non_trivial['cohen_kappa'].max():.3f}"
    rows.append({"claim": "Beta sensitivity kappa range", "draft_value": "0.732 to 0.826",
                 "repository_value": krange,
                 "status": "CONFIRMED" if krange == "0.732 to 0.826" else "DIFFERS",
                 "source_file": "outputs/tables/table_beta_sensitivity_clean.csv"})
    beta_kappas = re.findall(r"β=[123]\.[05] → [\d.]+% / ([\d.]+)", fm_text_1_8) if fm_text_1_8 else []
    if len(beta_kappas) == 3:
        r_kappas = sorted(float(k) for k in beta_kappas)
        r_krange = f"{r_kappas[0]:.3f} to {r_kappas[-1]:.3f}"
        add_ref_check(rows[-1], r_krange,
                      abs(non_trivial['cohen_kappa'].min() - r_kappas[0]) < 0.002
                      and abs(non_trivial['cohen_kappa'].max() - r_kappas[-1]) < 0.002)
    else:
        add_ref_check(rows[-1], None, None)

    # One closing sweep over every row collected above, rather than an
    # individual record_failure call at each of the ~20 individual claims
    # in this section. Gates on ref_status (this run vs
    # docs/final_manuscript_values.md, the maintained reference), not on
    # `status` (this run vs the original manuscript draft) — the draft
    # comparison is expected to differ permanently for reasons already
    # documented in docs/reproducibility_note.md, so treating it as a
    # failure would make this script fail every run forever regardless of
    # whether anything is actually wrong. A row with no reference value at
    # all ("no reference value") is reported but never gates failure,
    # per the same "say so rather than inventing a target" rule used
    # throughout this file.
    for row in rows:
        if row["ref_status"] == "DIFFERS vs reference":
            record_failure("1.8", f"{row['claim']}: repository {row['repository_value']}, "
                                   f"final_manuscript_values.md {row['reference_value']} "
                                   f"(draft was {row['draft_value']})")

    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "manuscript_number_check.csv", index=False)
    log(f"Saved `manuscript_number_check.csv` ({len(df)} rows).\n")
    log(df.to_string(index=False))
    log("")


def main():
    print("=== DATA AUDIT ===\n")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"# Data Audit Report\n")
    log(f"Generated by `src/17_data_audit.py`. Every figure below is computed directly "
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

    if CHECK_FAILURES:
        print(f"\n=== {len(CHECK_FAILURES)} CHECK(S) FAILED ===")
        for f in CHECK_FAILURES:
            print(f"  - {f}")
        print("\nSee the sections above and data_audit_report.md for full detail.")
        sys.exit(1)

    print("Done. All checks passed.")


if __name__ == "__main__":
    main()
