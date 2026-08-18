"""
Compute Moran's I on GI values, Huff Pij values, and ML-AHP
disagreement layer, save autocorrelation results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
import esda

from config import GPKG, TABLES

AHP_VS_NW_PATH = GPKG / "map_AHP_vs_NW_villages.gpkg"
AHP_VS_ML_PATH = GPKG / "map_AHP_vs_ML_villages.gpkg"
NW_VS_ML_PATH = GPKG / "map_NW_vs_ML_villages.gpkg"
RESULTS_PATH = TABLES / "table_morans_i_results.csv"
LISA_OUTPUT_PATH = GPKG / "map_lisa_AHP_vs_NW.gpkg"

SIGNIFICANCE_LEVEL = 0.05


def load_agreement_layer(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run src/12_export_outputs.py first "
            "to build the agreement map layers."
        )
    return gpd.read_file(path)


def build_queen_weights(gdf):
    """Queen contiguity spatial weights, row-standardised."""
    w = Queen.from_dataframe(gdf, use_index=False)
    w.transform = "r"
    return w


def global_morans_i(gdf, field, label):
    w = build_queen_weights(gdf)
    y = gdf[field].astype(float).values
    mi = esda.Moran(y, w)
    print(f"  {label}: I={mi.I:.4f}  p={mi.p_sim:.4f}  z={mi.z_sim:.4f}  (n={len(y)})")
    return {
        "layer": label,
        "field": field,
        "n": len(y),
        "morans_I": mi.I,
        "expected_I": mi.EI,
        "p_value": mi.p_sim,
        "z_score": mi.z_sim,
    }, w, y


def classify_lisa(local_moran, y, significance):
    """HH / LL / HL / LH quadrant labels; 'not significant' below threshold."""
    y_mean = y.mean()
    labels = []
    for value, lag, p in zip(y, local_moran.y, local_moran.p_sim):
        if p >= significance:
            labels.append("not significant")
            continue
        if value >= y_mean and lag >= y_mean:
            labels.append("HH")
        elif value < y_mean and lag < y_mean:
            labels.append("LL")
        elif value >= y_mean and lag < y_mean:
            labels.append("HL")
        else:
            labels.append("LH")
    return labels


def main():
    print("=== MORAN'S I ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    GPKG.mkdir(parents=True, exist_ok=True)

    print("Loading map_AHP_vs_NW_villages.gpkg...")
    ahp_vs_nw = load_agreement_layer(AHP_VS_NW_PATH)
    print(f"  {len(ahp_vs_nw)} villages")

    print("Loading map_AHP_vs_ML_villages.gpkg...")
    ahp_vs_ml = load_agreement_layer(AHP_VS_ML_PATH)
    print(f"  {len(ahp_vs_ml)} villages")

    print("Loading map_NW_vs_ML_villages.gpkg...")
    nw_vs_ml = load_agreement_layer(NW_VS_ML_PATH)
    print(f"  {len(nw_vs_ml)} villages")
    print()

    print("Computing global Moran's I...")
    results = []
    result_nw, w_nw, y_nw = global_morans_i(ahp_vs_nw, "agreement", "AHP_vs_NW")
    results.append(result_nw)

    result_ml, w_ml, y_ml = global_morans_i(ahp_vs_ml, "agreement", "AHP_vs_ML")
    results.append(result_ml)

    result_nw_ml, w_nw_ml, y_nw_ml = global_morans_i(nw_vs_ml, "agreement", "NW_vs_ML")
    results.append(result_nw_ml)
    print()

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved {RESULTS_PATH}")
    print()

    print("Computing Local Moran's I (LISA) for AHP_vs_NW agreement...")
    lisa = esda.Moran_Local(y_nw, w_nw, seed=42)
    cluster_type = classify_lisa(lisa, y_nw, SIGNIFICANCE_LEVEL)

    lisa_layer = ahp_vs_nw.copy()
    lisa_layer["lisa_I"] = lisa.Is
    lisa_layer["lisa_p"] = lisa.p_sim
    lisa_layer["cluster_type"] = cluster_type

    print(f"  Cluster type counts: {pd.Series(cluster_type).value_counts().to_dict()}")

    lisa_layer.to_file(LISA_OUTPUT_PATH, driver="GPKG")
    print(f"Saved {LISA_OUTPUT_PATH}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
