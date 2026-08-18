"""
Task 4 (QGIS map specs): add map_class and is_ljubljana_source fields to
the three headline comparison layers (Maps A/B/C), in place, so the same
classification logic and thresholds can be styled consistently in QGIS.

12_export_outputs.py now imports build_map_class from this module and
applies it inline whenever it (re)builds Maps A/B/C, so those fields
survive a plain rerun of the pipeline without this script needing to run
afterward. This script is kept as a standalone, idempotent way to
(re)apply the fields to already-existing GPKGs without rebuilding them
from source (e.g. after manually editing a layer in QGIS).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import geopandas as gpd

from config import GPKG

# (path, disagree-destination column, gain threshold, huff-assigned column for is_ljubljana_source)
MAPS = [
    ("Map A (AHP vs NW)", GPKG / "map_AHP_vs_NW_villages.gpkg", "NW_dominant_muni", 20, "AHP_dominant_muni"),
    ("Map B (AHP vs ML)", GPKG / "map_AHP_vs_ML_villages.gpkg", "ml_dominant_muni", 40, "AHP_dominant_muni"),
    ("Map C (NW vs ML)", GPKG / "map_NW_vs_ML_villages.gpkg", "ml_dominant_muni", 40, "NW_dominant_muni"),
]
# For Map A, the "otherwise" class is the AHP destination (the centre gaining
# the settlement under AHP weighting), not the NW destination — see brief.
DESTINATION_COL_OVERRIDE = {
    "Map A (AHP vs NW)": "AHP_dominant_muni",
}


def build_map_class(gdf, destination_col, threshold):
    disagreeing = gdf["agreement"] == 0
    dest_counts = gdf.loc[disagreeing, destination_col].value_counts()
    big_centres = set(dest_counts[dest_counts >= threshold].index)

    map_class = pd.Series("agree", index=gdf.index)
    is_big = gdf[destination_col].isin(big_centres)
    map_class = map_class.mask(disagreeing & is_big, gdf[destination_col])
    map_class = map_class.mask(disagreeing & ~is_big, "other centre")
    return map_class


def main():
    print("=== MAP SYMBOLOGY FIELDS (Task 4) ===")
    print()

    for label, path, dest_col_default, threshold, huff_col in MAPS:
        dest_col = DESTINATION_COL_OVERRIDE.get(label, dest_col_default)
        print(f"{label}: {path.name}  (destination column: {dest_col}, threshold: {threshold})")
        gdf = gpd.read_file(path)

        gdf["map_class"] = build_map_class(gdf, dest_col, threshold)
        gdf["is_ljubljana_source"] = (gdf[huff_col] == "Ljubljana")

        gdf.to_file(path, driver="GPKG")

        counts = gdf["map_class"].value_counts()
        n_agree = int((gdf["map_class"] == "agree").sum())
        n_lj_source = int(gdf["is_ljubljana_source"].sum())
        print(f"  n_agree (map_class=='agree'): {n_agree}")
        print(f"  is_ljubljana_source=True (Huff-assigned to Ljubljana, via {huff_col}): {n_lj_source}")
        print(f"  map_class counts:\n{counts.to_string()}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
