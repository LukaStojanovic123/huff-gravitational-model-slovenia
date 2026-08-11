"""
Export all paper-ready outputs — 6 tables, 3 supplementary tables,
9 figures as PNG and PDF, all GPKG layers for QGIS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED, TABLES, FIGURES, GPKG, SUPPLEMENTARY


def export_agreement_maps(data_raw, tables_path, gpkg_path):
    """Export three village-polygon agreement GPKG layers."""
    import geopandas as gpd
    import pandas as pd

    print("Loading village polygons...")
    na = gpd.read_file(data_raw / "NA.shp")
    na = na[["NA_MID", "NA_UIME", "POV_KM2", "geometry"]].copy()

    # ── Map 1: AHP vs NW ─────────────────────────────────────
    print("Building Map 1: AHP vs NW...")
    ahp_sum = pd.read_csv(tables_path / "huff_summary.csv")[
        ["Village_ID", "dominant_municipality", "dominant_Pij"]].rename(
        columns={"dominant_municipality": "AHP_dominant_muni",
                 "dominant_Pij": "AHP_dominant_Pij"})
    nw_sum = pd.read_csv(tables_path / "huff_NW_summary.csv")[
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
    print(f"  Saved: {len(map1)} villages, {agree1} agree ({agree1/len(map1)*100:.1f}%)")

    # ── Map 2: NW vs ML ──────────────────────────────────────
    print("Building Map 2: NW vs ML...")
    ml_nw = pd.read_csv(tables_path / "ml_NW_vs_NW_comparison.csv")[
        ["Village_ID", "Village_Name", "ml_dominant_muni", "ml_dominant_Pij",
         "huff_dominant_muni", "huff_dominant_Pij", "agreement", "agreement_label"]].rename(
        columns={"huff_dominant_muni": "NW_dominant_muni",
                 "huff_dominant_Pij": "NW_dominant_Pij",
                 "agreement_label": "agreement_label"})
    ml_nw["agreement_label"] = ml_nw["agreement"].map(
        {1: "NW Huff and ML agree", 0: "NW Huff and ML disagree"})

    map2 = na.merge(ml_nw, left_on="NA_MID", right_on="Village_ID", how="left")
    map2.drop(columns=["Village_ID"], errors="ignore", inplace=True)
    map2.to_file(gpkg_path / "map_NW_vs_ML_villages.gpkg", driver="GPKG")
    agree2 = map2["agreement"].sum()
    print(f"  Saved: {len(map2)} villages, {agree2} agree ({agree2/len(map2)*100:.1f}%)")

    # ── Map 3: AHP vs ML ─────────────────────────────────────
    print("Building Map 3: AHP vs ML...")
    ml_ahp = pd.read_csv(tables_path / "ml_AHP_vs_AHP_comparison.csv")[
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
    print(f"  Saved: {len(map3)} villages, {agree3} agree ({agree3/len(map3)*100:.1f}%)")

    print("All three agreement maps saved to outputs/gpkg/")


def main():
    GPKG.mkdir(parents=True, exist_ok=True)
    export_agreement_maps(DATA_RAW, TABLES, GPKG)


if __name__ == "__main__":
    main()
