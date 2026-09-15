"""
Step 11 of the pipeline: assemble every paper-ready table, figure and map
layer into outputs/, and report which of them are actually present.

What this script does: three things. First, it builds or copies the six
main manuscript tables and four supplementary tables into
outputs/tables/ and outputs/supplementary/, computing the ones that are
pure derivations directly from other outputs rather than trusting a
possibly stale pre-existing copy (see consolidate_tables below). Second,
it builds the four headline village-level agreement maps (AHP vs NW Huff,
AHP Huff vs its Random Forest, NW Huff vs its Random Forest, and the two
Random Forest models against each other) as GPKG layers for QGIS, adding
the map_class/is_ljubljana_source fields QGIS needs for consistent
styling. Third, it draws the feature-importance bar chart for the AHP
Random Forest and prints a checklist of every file the full pipeline is
expected to have produced by this point, so a missing upstream step is
obvious rather than silently absent.

Reads: whichever upstream outputs each table/map needs — see the specific
functions below and the EXPECTED_OUTPUTS checklist for the full list.

Writes: table1-6, tableS1-S4, the four agreement map GPKGs, and
fig07_feature_importance.png/.pdf, listed in full in OUTPUT_FILES below.

Runs eleventh, after every other numbered table- or map-producing script
it depends on (03, 04, 06, 07). 13_morans_lisa.py in turn depends on the
maps this script builds — it must always run after this script, never
before (see 13_morans_lisa.py's own staleness check for what happens if
that order is violated).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    DATA_RAW, DATA_PROCESSED, TABLES, FIGURES, GPKG, SUPPLEMENTARY, OUTPUTS, EPSG, REPO_ROOT,
    MUNICIPALITIES_AHP, MUNICIPALITIES_NW,
)
from crs_utils import ensure_crs

AUDIT = OUTPUTS / "audit"
SRC_DIR = Path(__file__).resolve().parent
DATA_EXTERNAL = REPO_ROOT / "data" / "external"

OUTPUT_FILES = [
    "tables/table1_AHP_group_weights.csv",
    "tables/table2_top20_GI_NotWeighted.csv",
    "tables/table3_top20_GI_AHP.csv",
    "tables/table4_top15_catchments.csv",
    "tables/table5_beta_sensitivity.csv",
    "tables/table6_cv_performance.csv",
    "supplementary/tableS1_indicators_sources.csv",
    "supplementary/tableS4_beta_sensitivity.csv",
    "figures/fig07_feature_importance.png",
    "figures/fig07_feature_importance.pdf",
    "gpkg/map_AHP_vs_NW_villages.gpkg",
    "gpkg/map_NW_vs_ML_villages.gpkg",
    "gpkg/map_AHP_vs_ML_villages.gpkg",
    "gpkg/map_ML_AHP_vs_ML_NW_villages.gpkg",
]


def build_map_class(gdf, destination_col, threshold):
    """Classify each disagreeing settlement by which municipality gains it.

    A settlement keeps the label "agree" if both models assigned it to the
    same municipality. Otherwise it is labelled with the name of the
    municipality gaining it under `destination_col`, but only if that
    municipality gains at least `threshold` settlements overall in this
    comparison — below that, it is grouped into "other centre" so the map
    legend does not end up with dozens of one- or two-settlement colours.
    """
    disagreeing = gdf["agreement"] == 0
    dest_counts = gdf.loc[disagreeing, destination_col].value_counts()
    big_centres = set(dest_counts[dest_counts >= threshold].index)

    map_class = pd.Series("agree", index=gdf.index)
    is_big = gdf[destination_col].isin(big_centres)
    map_class = map_class.mask(disagreeing & is_big, gdf[destination_col])
    map_class = map_class.mask(disagreeing & ~is_big, "other centre")
    return map_class


def safe_copy(src, dst, label):
    """Copy src to dst if src exists; report status either way."""
    if not src.exists():
        print(f"  SKIP  {label}: source not found ({src})")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"  OK    {label} -> {dst.name}")
    return True


def first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    return None


def build_top20_gi_table(gpkg_path, gi_col, out_path, label):
    """Rank all 212 municipalities by a GI column and save the top 20.

    Computed directly from the raw municipality layer each time this runs
    (not copied from a pre-existing file), so this table can never go
    stale relative to the actual GI data.
    """
    gdf = gpd.read_file(gpkg_path)
    gdf = ensure_crs(gdf, EPSG, label=gpkg_path.name)
    ranked = gdf[["Muni_Name", gi_col]].sort_values(gi_col, ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", ranked.index + 1)
    ranked = ranked.rename(columns={"Muni_Name": "Municipality"})
    ranked.head(20).to_csv(out_path, index=False)
    print(f"  OK    {label} -> {out_path.name} (computed from {gpkg_path.name})")


def build_top15_catchment_table(out_path, label):
    """Build the top-15 catchment size table, AHP and NW side by side.

    Computed directly from huff_AHP_summary.csv / huff_NW_summary.csv each
    time this runs, so it always reflects the current Huff results.
    """
    ahp = pd.read_csv(TABLES / "huff_AHP_summary.csv")
    nw = pd.read_csv(TABLES / "huff_NW_summary.csv")
    ahp_counts = ahp["dominant_municipality"].value_counts().head(15).reset_index()
    ahp_counts.columns = ["Municipality (AHP)", "Villages (AHP)"]
    nw_counts = nw["dominant_municipality"].value_counts().head(15).reset_index()
    nw_counts.columns = ["Municipality (NW)", "Villages (NW)"]
    combined = pd.concat([ahp_counts, nw_counts], axis=1)
    combined.insert(0, "Rank", range(1, len(combined) + 1))
    combined.to_csv(out_path, index=False)
    print(f"  OK    {label} -> {out_path.name} (computed from huff_AHP_summary.csv / huff_NW_summary.csv)")


def consolidate_tables():
    """Build/copy the paper's final tables (table1-6, tableS1-S4) into place.

    table1 and tableS1 are genuine inputs (the AHP pairwise-comparison result
    and the indicator source citations) that no script computes from data —
    those are copied from data/external/, which is committed to the repo, not
    an absolute path on one machine. table2-4 are pure rankings/derivations
    with no ambiguity, so they're computed here directly from repo data
    instead of being copied from a pre-existing file.
    """
    print("Consolidating paper tables...")

    # table1: the AHP group priority weights are a pairwise-comparison judgment,
    # not a data-derived result — a genuine input, not something to recompute.
    # Same warning as tableS1 below: this is a plain copy from
    # data/external/. Any future correction to this table must be made to
    # data/external/table1_AHP_group_weights_FIXED.csv, not to
    # outputs/tables/table1_AHP_group_weights.csv directly — the latter is
    # silently overwritten by the former on every run.
    safe_copy(DATA_EXTERNAL / "table1_AHP_group_weights_FIXED.csv",
              TABLES / "table1_AHP_group_weights.csv", "table1_AHP_group_weights")

    build_top20_gi_table(DATA_RAW / MUNICIPALITIES_NW, "GI_Final_NotWeighted",
                          TABLES / "table2_top20_GI_NotWeighted.csv", "table2_top20_GI_NotWeighted")

    build_top20_gi_table(DATA_RAW / MUNICIPALITIES_AHP, "GI_AHP",
                          TABLES / "table3_top20_GI_AHP.csv", "table3_top20_GI_AHP")

    build_top15_catchment_table(TABLES / "table4_top15_catchments.csv", "table4_top15_catchments")

    # table_beta_sensitivity_clean.csv (07_beta_sensitivity.py's actual output) must win over
    # the unsuffixed table_beta_sensitivity.csv name. That unsuffixed name was a legacy
    # pre-remediation artifact (quarantined to data/quarantine/ — see its README) that this
    # first_existing() call silently preferred over the pipeline's own fresh output for the
    # entire Stage 2B/3 rerun, meaning table5/tableS4 kept serving stale numbers even after
    # everything upstream was fixed. Order matters here: always prefer the script's own
    # current output; only fall back to a legacy name if the script has never been run.
    beta_src = first_existing(TABLES / "table_beta_sensitivity_clean.csv",
                               TABLES / "table_beta_sensitivity.csv")
    if beta_src is not None:
        safe_copy(beta_src, TABLES / "table5_beta_sensitivity.csv", "table5_beta_sensitivity")
        safe_copy(beta_src, SUPPLEMENTARY / "tableS4_beta_sensitivity.csv", "tableS4_beta_sensitivity")
    else:
        print("  SKIP  table5_beta_sensitivity / tableS4_beta_sensitivity: "
              "no table_beta_sensitivity(_clean).csv found — run src/07_beta_sensitivity.py first")

    safe_copy(TABLES / "ml_AHP_cv_results.csv",
              TABLES / "table6_cv_performance.csv", "table6_cv_performance")

    # tableS1: indicator source citations — reference metadata, not a computed result.
    # This is a plain copy, not a regeneration: a correction ever applied to
    # tableS1 must be made in data/external/tableS1_indicators_sources.csv
    # itself, not just in the outputs/supplementary/ copy — a delimiter and
    # group-name fix was once made only to the outputs/ copy, and the very
    # next pipeline run silently overwrote it back to the old, wrong
    # version from data/external/, since that is what this call actually
    # reads from. Found and fixed during this remediation's own Stage 2.4
    # pipeline verification.
    safe_copy(DATA_EXTERNAL / "tableS1_indicators_sources.csv",
              SUPPLEMENTARY / "tableS1_indicators_sources.csv", "tableS1_indicators_sources")

    for name in ("tableS2_AHP_priority_weights.csv", "tableS3_individual_indicator_weights.csv"):
        status = "OK   " if (SUPPLEMENTARY / name).exists() else "MISSING"
        print(f"  {status} {name} (expected already in outputs/supplementary/)")

    print()


def load_indicator_group_map():
    """Map each indicator code to its thematic group, read from tableS3 (see 01_gi_construction.py)."""
    ref_path = SUPPLEMENTARY / "tableS3_individual_indicator_weights.csv"
    if not ref_path.exists():
        return {}
    # No sep= argument: tableS3 is comma-delimited. See the matching note
    # in 01_gi_construction.py::load_indicator_group_map for why this
    # matters — a stale sep=";" here previously made this call fail.
    ref = pd.read_csv(ref_path)
    ref = ref.dropna(subset=["Indicator_code"])
    ref = ref[ref["Indicator_code"] != "Total"]
    return dict(zip(ref["Indicator_code"], ref["Thematic_group"]))


def classify_feature(feature, group_map):
    """Label one ML feature name for the feature-importance chart's colour legend."""
    if feature.startswith("nacc_"):
        return "Accessibility"
    if feature == "GI_AHP":
        return "GI_AHP"
    if feature == "dist_to_muni":
        return "Distance"
    if feature.startswith("n_"):
        return group_map.get(feature, "Other indicator")
    return "Other"


def plot_feature_importance():
    """Draw the AHP Random Forest's top-30 feature importance bar chart, coloured by thematic group."""
    fi_path = TABLES / "ml_AHP_feature_importance.csv"
    if not fi_path.exists():
        print("  SKIP  fig07_feature_importance: "
              "ml_AHP_feature_importance.csv not found — run src/06_ml_framework.py first")
        return

    print("Generating feature importance figure...")
    fi = pd.read_csv(fi_path).sort_values("importance", ascending=False).head(30).reset_index(drop=True)

    group_map = load_indicator_group_map()
    fi["group"] = fi["feature"].apply(lambda f: classify_feature(f, group_map))

    groups = sorted(fi["group"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {g: cmap(i % 10) for i, g in enumerate(groups)}
    colors = fi["group"].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title("Top 30 features — AHP Random Forest")
    ax.legend(handles=[mpatches.Patch(color=color_map[g], label=g) for g in groups],
              loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig07_feature_importance.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES / "fig07_feature_importance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig07_feature_importance.png / .pdf ({len(fi)} features)")
    print()


def export_agreement_maps(data_raw, tables_path, gpkg_path):
    """Export three village-polygon agreement GPKG layers.

    Also adds the map_class / is_ljubljana_source QGIS symbology fields
    inline, right before each layer is saved, using build_map_class() above,
    so a plain rerun of this script can never regenerate Maps A/B/C without
    them. These fields used to be added by a separate script that had to run
    *after* this one — any rerun of this script would silently wipe them
    again, which is why that logic now lives here instead.
    """
    print("Exporting agreement maps...")

    ahp_path = tables_path / "huff_AHP_summary.csv"
    nw_path = tables_path / "huff_NW_summary.csv"
    if not (ahp_path.exists() and nw_path.exists()):
        print("  SKIP  agreement maps: huff_AHP_summary.csv / huff_NW_summary.csv missing — "
              "run src/03_huff_ahp.py and src/04_huff_nonweighted.py first")
        return

    print("  Loading village polygons...")
    na = gpd.read_file(data_raw / "NA.shp")
    na = ensure_crs(na, EPSG, label="NA.shp")
    na = na[["NA_MID", "NA_UIME", "POV_KM2", "geometry"]].copy()

    # ── Map 1: AHP vs NW ─────────────────────────────────────
    print("  Building Map 1: AHP vs NW...")
    ahp_sum = pd.read_csv(ahp_path)[
        ["Village_ID", "dominant_municipality", "dominant_Pij"]].rename(
        columns={"dominant_municipality": "AHP_dominant_muni",
                 "dominant_Pij": "AHP_dominant_Pij"})
    nw_sum = pd.read_csv(nw_path)[
        ["Village_ID", "dominant_municipality", "dominant_Pij"]].rename(
        columns={"dominant_municipality": "NW_dominant_muni",
                 "dominant_Pij": "NW_dominant_Pij"})

    map1 = na.merge(ahp_sum, left_on="NA_MID", right_on="Village_ID", how="left")
    map1 = map1.merge(nw_sum, left_on="NA_MID", right_on="Village_ID", how="left")
    map1["agreement"] = (map1["AHP_dominant_muni"] == map1["NW_dominant_muni"]).astype(int)
    map1["agreement_label"] = map1["agreement"].map(
        {1: "AHP and NW agree", 0: "AHP and NW disagree"})
    map1.drop(columns=["Village_ID_x", "Village_ID_y"], errors="ignore", inplace=True)
    map1["map_class"] = build_map_class(map1, "AHP_dominant_muni", 20)
    map1["is_ljubljana_source"] = (map1["AHP_dominant_muni"] == "Ljubljana")
    map1.to_file(gpkg_path / "map_AHP_vs_NW_villages.gpkg", driver="GPKG")
    agree1 = map1["agreement"].sum()
    print(f"    Saved: {len(map1)} villages, {agree1} agree ({agree1 / len(map1) * 100:.1f}%)")

    # ── Map 2: NW vs ML ──────────────────────────────────────
    nw_ml_path = tables_path / "ml_NW_vs_NW_comparison.csv"
    if nw_ml_path.exists():
        print("  Building Map 2: NW vs ML...")
        ml_nw = pd.read_csv(nw_ml_path)[
            ["Village_ID", "Village_Name", "ml_dominant_muni", "ml_dominant_Pij",
             "huff_dominant_muni", "huff_dominant_Pij", "agreement"]].rename(
            columns={"huff_dominant_muni": "NW_dominant_muni",
                     "huff_dominant_Pij": "NW_dominant_Pij"})
        ml_nw["agreement_label"] = ml_nw["agreement"].map(
            {1: "NW Huff and ML agree", 0: "NW Huff and ML disagree"})

        map2 = na.merge(ml_nw, left_on="NA_MID", right_on="Village_ID", how="left")
        map2.drop(columns=["Village_ID"], errors="ignore", inplace=True)
        map2["map_class"] = build_map_class(map2, "ml_dominant_muni", 40)
        map2["is_ljubljana_source"] = (map2["NW_dominant_muni"] == "Ljubljana")
        map2.to_file(gpkg_path / "map_NW_vs_ML_villages.gpkg", driver="GPKG")
        agree2 = map2["agreement"].sum()
        print(f"    Saved: {len(map2)} villages, {agree2} agree ({agree2 / len(map2) * 100:.1f}%)")
    else:
        print(f"  SKIP  Map 2 (NW vs ML): {nw_ml_path.name} not found — "
              "run src/06_ml_framework.py (Model 2 / NW) first")

    # ── Map 3: AHP vs ML ─────────────────────────────────────
    ahp_ml_path = tables_path / "ml_AHP_vs_AHP_comparison.csv"
    if ahp_ml_path.exists():
        print("  Building Map 3: AHP vs ML...")
        ml_ahp = pd.read_csv(ahp_ml_path)[
            ["Village_ID", "Village_Name", "ml_dominant_muni", "ml_dominant_Pij",
             "huff_dominant_muni", "huff_dominant_Pij", "agreement"]].rename(
            columns={"huff_dominant_muni": "AHP_dominant_muni",
                     "huff_dominant_Pij": "AHP_dominant_Pij"})
        ml_ahp["agreement_label"] = ml_ahp["agreement"].map(
            {1: "AHP Huff and ML agree", 0: "AHP Huff and ML disagree"})

        map3 = na.merge(ml_ahp, left_on="NA_MID", right_on="Village_ID", how="left")
        map3.drop(columns=["Village_ID"], errors="ignore", inplace=True)
        map3["map_class"] = build_map_class(map3, "ml_dominant_muni", 40)
        map3["is_ljubljana_source"] = (map3["AHP_dominant_muni"] == "Ljubljana")
        map3.to_file(gpkg_path / "map_AHP_vs_ML_villages.gpkg", driver="GPKG")
        agree3 = map3["agreement"].sum()
        print(f"    Saved: {len(map3)} villages, {agree3} agree ({agree3 / len(map3) * 100:.1f}%)")
    else:
        print(f"  SKIP  Map 3 (AHP vs ML): {ahp_ml_path.name} not found — "
              "run src/06_ml_framework.py (Model 1 / AHP) first")

    # ── Map 4: RF(AHP-target) vs RF(NW-target) ────────────────
    # The comparison the original three never covered: do the two separately
    # trained RF models converge on the same catchment structure? See
    # docs/ml_model_design_note.md for why these are two distinct
    # models, not one model compared twice.
    if ahp_ml_path.exists() and nw_ml_path.exists():
        print("  Building Map 4: RF(AHP-target) vs RF(NW-target)...")
        ml_ahp4 = pd.read_csv(ahp_ml_path)[
            ["Village_ID", "ml_dominant_muni", "ml_dominant_Pij"]].rename(
            columns={"ml_dominant_muni": "ML_AHP_dominant_muni",
                     "ml_dominant_Pij": "ML_AHP_dominant_Pij"})
        ml_nw4 = pd.read_csv(nw_ml_path)[
            ["Village_ID", "ml_dominant_muni", "ml_dominant_Pij"]].rename(
            columns={"ml_dominant_muni": "ML_NW_dominant_muni",
                     "ml_dominant_Pij": "ML_NW_dominant_Pij"})

        map4 = na.merge(ml_ahp4, left_on="NA_MID", right_on="Village_ID", how="left")
        map4 = map4.merge(ml_nw4, left_on="NA_MID", right_on="Village_ID", how="left",
                           suffixes=("", "_nw"))
        map4.drop(columns=[c for c in map4.columns if c.startswith("Village_ID")],
                  errors="ignore", inplace=True)
        map4["agreement"] = (map4["ML_AHP_dominant_muni"] == map4["ML_NW_dominant_muni"]).astype(int)
        map4["agreement_label"] = map4["agreement"].map(
            {1: "RF(AHP) and RF(NW) agree", 0: "RF(AHP) and RF(NW) disagree"})
        map4["map_class"] = build_map_class(map4, "ML_NW_dominant_muni", 40)
        map4["is_ljubljana_source"] = (map4["ML_AHP_dominant_muni"] == "Ljubljana")
        map4.to_file(gpkg_path / "map_ML_AHP_vs_ML_NW_villages.gpkg", driver="GPKG")
        agree4 = map4["agreement"].sum()
        print(f"    Saved: {len(map4)} villages, {agree4} agree ({agree4 / len(map4) * 100:.1f}%)")
    else:
        print("  SKIP  Map 4 (RF-AHP vs RF-NW): both ml_*_comparison.csv files needed")

    print()


# Every output the full pipeline (all 18 numbered scripts) is expected to
# produce, for the final completeness checklist below. Grouped by output
# directory. This list is separate from each script's own OUTPUT_FILES
# declaration (used by 19_output_manifest.py to check which script wrote
# which file) — this one exists purely to answer "is everything here,"
# not "who made it."
EXPECTED_OUTPUTS = {
    "data/processed": [
        DATA_PROCESSED / "roads_noded.gpkg",
        DATA_PROCESSED / "distance_matrix.npy",
        DATA_PROCESSED / "distance_matrix_village_ids.npy",
        DATA_PROCESSED / "distance_matrix_muni_ids.npy",
    ],
    "outputs/tables": [
        TABLES / "table_GI_summary_stats.csv",
        TABLES / "table_top20_GI_both.csv",
        TABLES / "huff_AHP_summary.csv",
        TABLES / "huff_NW_summary.csv",
        TABLES / "accessibility_raw_distances.csv",
        TABLES / "accessibility_normalized.csv",
        TABLES / "ml_AHP_feature_importance.csv",
        TABLES / "ml_AHP_cv_results.csv",
        TABLES / "ml_AHP_vs_AHP_comparison.csv",
        TABLES / "ml_NW_feature_importance.csv",
        TABLES / "ml_NW_cv_results.csv",
        TABLES / "ml_NW_vs_NW_comparison.csv",
        TABLES / "table_euclidean_vs_network.csv",
        TABLES / "table_entropy_summary.csv",
        TABLES / "table_morans_i_results.csv",
        TABLES / "table_huff_vs_commuting.csv",
        TABLES / "table_huff_vs_commuting_summary.csv",
        TABLES / "table1_AHP_group_weights.csv",
        TABLES / "table2_top20_GI_NotWeighted.csv",
        TABLES / "table3_top20_GI_AHP.csv",
        TABLES / "table4_top15_catchments.csv",
        TABLES / "table5_beta_sensitivity.csv",
        TABLES / "table6_cv_performance.csv",
        # Three-way agreement, LISA, disagreement synthesis, RF catchment
        # structure, feature importance comparison (src/13, 14, 15).
        TABLES / "table_three_way_agreement.csv",
        TABLES / "table_join_counts.csv",
        TABLES / "table_lisa_summary.csv",
        TABLES / "table_disagreement_synthesis.csv",
        TABLES / "table_disagreement_destinations.csv",
        TABLES / "table_ml_catchment_sizes.csv",
        TABLES / "table_feature_importance_comparison.csv",
    ],
    "outputs/supplementary": [
        SUPPLEMENTARY / "tableS1_indicators_sources.csv",
        SUPPLEMENTARY / "tableS2_AHP_priority_weights.csv",
        SUPPLEMENTARY / "tableS3_individual_indicator_weights.csv",
        SUPPLEMENTARY / "tableS4_beta_sensitivity.csv",
    ],
    "outputs/figures": [
        FIGURES / "fig07_beta_sensitivity.png",
        FIGURES / "fig07_feature_importance_AHP.png",
        FIGURES / "fig07_feature_importance.png",
        FIGURES / "fig07_feature_importance.pdf",
        FIGURES / "fig08_shap_bar_AHP.png",
        FIGURES / "fig08_shap_summary_AHP.png",
        # SHAP dependence (src/16) and feature importance comparison (src/15).
        FIGURES / "fig_shap_dependence_AHP.png",
        FIGURES / "fig_shap_dependence_AHP.pdf",
        FIGURES / "fig_feature_importance_comparison.png",
        FIGURES / "fig_feature_importance_comparison.pdf",
    ],
    "outputs/gpkg": [
        GPKG / "map_AHP_vs_NW_villages.gpkg",
        GPKG / "map_NW_vs_ML_villages.gpkg",
        GPKG / "map_AHP_vs_ML_villages.gpkg",
        GPKG / "map_ML_AHP_vs_ML_NW_villages.gpkg",
        GPKG / "map_euclidean_vs_network_villages.gpkg",
        GPKG / "map_entropy_AHP_villages.gpkg",
        GPKG / "map_entropy_NW_villages.gpkg",
        GPKG / "map_lisa_AHP_vs_NW.gpkg",
        GPKG / "fig_huff_vs_commuting_municipalities.gpkg",
        # LISA for all three comparisons (src/13).
        GPKG / "map_lisa_AHP_vs_ML.gpkg",
        GPKG / "map_lisa_NW_vs_ML.gpkg",
        # Three-way disagreement synthesis (src/14).
        GPKG / "map_disagreement_count_villages.gpkg",
        # Study area, GI choropleths, catchment layers (src/17).
        GPKG / "fig01_study_area.gpkg",
        GPKG / "fig03_GI_NW_municipalities.gpkg",
        GPKG / "fig04_GI_AHP_municipalities.gpkg",
        GPKG / "fig05_catchments_AHP.gpkg",
        GPKG / "fig06_catchments_NW.gpkg",
        GPKG / "fig10_catchments_ML.gpkg",
    ],
    "outputs/audit": [
        AUDIT / "data_audit_report.md",
        AUDIT / "raw_input_inventory.csv",
        AUDIT / "indicator_audit.csv",
        AUDIT / "GI_full_212_municipalities.csv",
        AUDIT / "manuscript_number_check.csv",
        AUDIT / "output_manifest.csv",
    ],
}


def print_checklist():
    """Print, for every file in EXPECTED_OUTPUTS, whether it currently exists on disk."""
    print("=== OUTPUT CHECKLIST ===")
    total = 0
    present = 0
    for group, paths in EXPECTED_OUTPUTS.items():
        print(f"\n{group}/")
        for p in paths:
            total += 1
            ok = p.exists()
            present += int(ok)
            print(f"  {'OK     ' if ok else 'MISSING'} {p.name}")
    print(f"\n{present}/{total} expected output files present.")


def main():
    print("=== EXPORT OUTPUTS ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    GPKG.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    consolidate_tables()
    export_agreement_maps(DATA_RAW, TABLES, GPKG)
    plot_feature_importance()
    print_checklist()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
