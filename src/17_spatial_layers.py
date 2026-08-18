"""
Build the spatial layers referenced in the manuscript's figure plan that
script 12 never creates: the study-area overview, GI choropleth layers
(NW and AHP), and catchment layers (AHP Huff, NW Huff, RF) with both a
per-settlement and a dissolved-by-municipality version.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import geopandas as gpd

from config import (
    DATA_RAW, GPKG, TABLES, MUNICIPALITIES_AHP, MUNICIPALITIES_NW,
    VILLAGES_FILE, SETTLEMENTS_POLY, EPSG,
)

OBCINE_FILE = "obcine_poligoni.shp"


def build_study_area():
    print("--- 3.1 Study area layer ---")
    obcine = gpd.read_file(DATA_RAW / OBCINE_FILE)[["SIFRA", "NAZIV", "geometry"]].rename(
        columns={"SIFRA": "muni_id", "NAZIV": "muni_name"})
    settlements = gpd.read_file(DATA_RAW / VILLAGES_FILE)[["NA_MID", "NA_NA_UIME", "geometry"]].rename(
        columns={"NA_MID": "settlement_id", "NA_NA_UIME": "settlement_name"})

    assert obcine.crs.to_epsg() == EPSG and settlements.crs.to_epsg() == EPSG, "CRS mismatch"

    out_path = GPKG / "fig01_study_area.gpkg"
    obcine.to_file(out_path, layer="municipalities", driver="GPKG")
    settlements.to_file(out_path, layer="settlements", driver="GPKG")
    print(f"  Saved {out_path} (layers: municipalities [{len(obcine)}], "
          f"settlements [{len(settlements)}])")
    print()


def build_gi_choropleths():
    print("--- 3.2 GI choropleth layers ---")
    obcine = gpd.read_file(DATA_RAW / OBCINE_FILE)[["SIFRA", "NAZIV", "geometry"]].rename(
        columns={"SIFRA": "muni_id", "NAZIV": "muni_name"})

    nw = gpd.read_file(DATA_RAW / MUNICIPALITIES_NW)
    nw_group_cols = [c for c in nw.columns if c.endswith("_Sum")]
    nw = nw[["Muni_Name", "GI_Final_NotWeighted"] + nw_group_cols].copy()
    nw["rank_GI_NW"] = nw["GI_Final_NotWeighted"].rank(ascending=False, method="min").astype(int)
    nw_layer = obcine.merge(nw, left_on="muni_name", right_on="Muni_Name", how="left").drop(columns="Muni_Name")
    n_unmatched = int(nw_layer["GI_Final_NotWeighted"].isna().sum())
    fig03_path = GPKG / "fig03_GI_NW_municipalities.gpkg"
    nw_layer.to_file(fig03_path, driver="GPKG")
    print(f"  Saved {fig03_path} ({len(nw_layer)} municipalities, {n_unmatched} unmatched by name)")

    ahp = gpd.read_file(DATA_RAW / MUNICIPALITIES_AHP)
    ahp_group_cols = [c for c in ahp.columns if c.endswith("_Weighted")]
    ahp = ahp[["Muni_Name", "GI_AHP"] + ahp_group_cols].copy()
    ahp["rank_GI_AHP"] = ahp["GI_AHP"].rank(ascending=False, method="min").astype(int)
    ahp_layer = obcine.merge(ahp, left_on="muni_name", right_on="Muni_Name", how="left").drop(columns="Muni_Name")
    n_unmatched = int(ahp_layer["GI_AHP"].isna().sum())
    fig04_path = GPKG / "fig04_GI_AHP_municipalities.gpkg"
    ahp_layer.to_file(fig04_path, driver="GPKG")
    print(f"  Saved {fig04_path} ({len(ahp_layer)} municipalities, {n_unmatched} unmatched by name)")
    print()


def build_catchment_layer(na, dominant_series, id_col, out_path, dominant_field_name):
    """dominant_series: Village_ID -> dominant municipality name."""
    layer = na.merge(dominant_series.rename(dominant_field_name),
                      left_on="NA_MID", right_index=True, how="left")
    sizes = layer[dominant_field_name].value_counts()
    layer["catchment_size"] = layer[dominant_field_name].map(sizes)

    out_cols = ["NA_MID", "NA_UIME", dominant_field_name, "catchment_size", "geometry"]
    settlement_layer = layer[out_cols]
    settlement_layer.to_file(out_path, layer="settlements", driver="GPKG")

    dissolved = settlement_layer.dissolve(by=dominant_field_name, as_index=False)
    dissolved["catchment_size"] = dissolved[dominant_field_name].map(sizes)
    dissolved = dissolved[[dominant_field_name, "catchment_size", "geometry"]]
    dissolved.to_file(out_path, layer="dissolved", driver="GPKG")

    print(f"  Saved {out_path} (layers: settlements [{len(settlement_layer)}], "
          f"dissolved [{len(dissolved)} municipalities])")


def build_catchments():
    print("--- 3.3 Catchment layers ---")
    na = gpd.read_file(DATA_RAW / SETTLEMENTS_POLY)[["NA_MID", "NA_UIME", "geometry"]]

    ahp_sum = pd.read_csv(TABLES / "huff_AHP_summary.csv").set_index("Village_ID")["dominant_municipality"]
    build_catchment_layer(na, ahp_sum, "NA_MID", GPKG / "fig05_catchments_AHP.gpkg", "AHP_dominant_muni")

    nw_sum = pd.read_csv(TABLES / "huff_NW_summary.csv").set_index("Village_ID")["dominant_municipality"]
    build_catchment_layer(na, nw_sum, "NA_MID", GPKG / "fig06_catchments_NW.gpkg", "NW_dominant_muni")

    ml_sum = pd.read_csv(TABLES / "ml_AHP_vs_AHP_comparison.csv").set_index("Village_ID")["ml_dominant_muni"]
    build_catchment_layer(na, ml_sum, "NA_MID", GPKG / "fig10_catchments_ML.gpkg", "ml_dominant_muni")
    print()


def main():
    print("=== SPATIAL LAYERS (Task 3) ===")
    print()
    GPKG.mkdir(parents=True, exist_ok=True)

    build_study_area()
    build_gi_choropleths()
    build_catchments()

    print("Done.")


if __name__ == "__main__":
    main()
