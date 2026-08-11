"""
Export all paper-ready outputs — 6 tables, 3 supplementary tables,
9 figures as PNG and PDF, all GPKG layers for QGIS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import DATA_RAW, DATA_PROCESSED, TABLES, FIGURES, GPKG, SUPPLEMENTARY

TABLES_AND_CHARTS = Path(r"C:\Users\lstojano\Desktop\teza\HuffMethodPaper\Data\tables and charts")
HUFFMETHODPAPER_ROOT = Path(r"C:\Users\lstojano\Desktop\teza\HuffMethodPaper")


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


def consolidate_tables():
    """Copy/rename the paper's final tables (table1-6, tableS1-S4) into place."""
    print("Consolidating paper tables...")

    # table1: the FIXED original already combines AHP priority weight + indicator count
    safe_copy(TABLES_AND_CHARTS / "table1_AHP_group_weights_FIXED.csv",
              TABLES / "table1_AHP_group_weights.csv", "table1_AHP_group_weights")

    safe_copy(TABLES_AND_CHARTS / "table2_top20_GI_NotWeighted.csv",
              TABLES / "table2_top20_GI_NotWeighted.csv", "table2_top20_GI_NotWeighted")

    safe_copy(TABLES_AND_CHARTS / "table3_top20_GI_AHP.csv",
              TABLES / "table3_top20_GI_AHP.csv", "table3_top20_GI_AHP")

    safe_copy(TABLES_AND_CHARTS / "table4_top15_catchment_AHP_vs_NW.csv",
              TABLES / "table4_top15_catchments.csv", "table4_top15_catchments")

    beta_src = first_existing(TABLES / "table_beta_sensitivity.csv",
                               TABLES / "table_beta_sensitivity_clean.csv")
    if beta_src is not None:
        safe_copy(beta_src, TABLES / "table5_beta_sensitivity.csv", "table5_beta_sensitivity")
        safe_copy(beta_src, SUPPLEMENTARY / "tableS4_beta_sensitivity.csv", "tableS4_beta_sensitivity")
    else:
        print("  SKIP  table5_beta_sensitivity / tableS4_beta_sensitivity: "
              "no table_beta_sensitivity(.csv|_clean.csv) found — run src/07_beta_sensitivity.py first")

    safe_copy(TABLES / "ml_AHP_cv_results.csv",
              TABLES / "table6_cv_performance.csv", "table6_cv_performance")

    safe_copy(HUFFMETHODPAPER_ROOT / "tableS1_indicators_sources.csv",
              SUPPLEMENTARY / "tableS1_indicators_sources.csv", "tableS1_indicators_sources")

    for name in ("tableS2_AHP_priority_weights.csv", "tableS3_individual_indicator_weights.csv"):
        status = "OK   " if (SUPPLEMENTARY / name).exists() else "MISSING"
        print(f"  {status} {name} (expected already in outputs/supplementary/)")

    print()


def load_indicator_group_map():
    """Indicator_code -> Thematic_group, from tableS3 (see src/01_gi_construction.py)."""
    ref_path = SUPPLEMENTARY / "tableS3_individual_indicator_weights.csv"
    if not ref_path.exists():
        return {}
    ref = pd.read_csv(ref_path, sep=";")
    ref = ref.dropna(subset=["Indicator_code"])
    ref = ref[ref["Indicator_code"] != "Total"]
    return dict(zip(ref["Indicator_code"], ref["Thematic_group"]))


def classify_feature(feature, group_map):
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
    """Top 30 features bar chart, coloured by thematic group."""
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
    """Export three village-polygon agreement GPKG layers."""
    print("Exporting agreement maps...")

    ahp_path = tables_path / "huff_AHP_summary.csv"
    nw_path = tables_path / "huff_NW_summary.csv"
    if not (ahp_path.exists() and nw_path.exists()):
        print("  SKIP  agreement maps: huff_AHP_summary.csv / huff_NW_summary.csv missing — "
              "run src/03_huff_ahp.py and src/04_huff_nonweighted.py first")
        return

    print("  Loading village polygons...")
    na = gpd.read_file(data_raw / "NA.shp")
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
        map3.to_file(gpkg_path / "map_AHP_vs_ML_villages.gpkg", driver="GPKG")
        agree3 = map3["agreement"].sum()
        print(f"    Saved: {len(map3)} villages, {agree3} agree ({agree3 / len(map3) * 100:.1f}%)")
    else:
        print(f"  SKIP  Map 3 (AHP vs ML): {ahp_ml_path.name} not found — "
              "run src/06_ml_framework.py (Model 1 / AHP) first")

    print()


# Every output the full pipeline (01-12) is expected to produce, for the
# final completeness checklist. Grouped by output directory.
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
    ],
    "outputs/gpkg": [
        GPKG / "map_AHP_vs_NW_villages.gpkg",
        GPKG / "map_NW_vs_ML_villages.gpkg",
        GPKG / "map_AHP_vs_ML_villages.gpkg",
        GPKG / "map_euclidean_vs_network_villages.gpkg",
        GPKG / "map_entropy_AHP_villages.gpkg",
        GPKG / "map_entropy_NW_villages.gpkg",
        GPKG / "map_lisa_AHP_vs_NW.gpkg",
        GPKG / "fig_huff_vs_commuting_municipalities.gpkg",
    ],
}


def print_checklist():
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

    consolidate_tables()
    export_agreement_maps(DATA_RAW, TABLES, GPKG)
    plot_feature_importance()
    print_checklist()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
