"""
Rerun AHP Huff with Euclidean distances, compare dominant assignments
vs road network, save agreement table and spatial layer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from sklearn.metrics import cohen_kappa_score

from config import DATA_RAW, TABLES, GPKG, BETA, MUNICIPALITIES_AHP, VILLAGES_FILE, SETTLEMENTS_POLY

OBCINE_FILE = "obcine_poligoni.shp"
TABLE_OUTPUT_PATH = TABLES / "table_euclidean_vs_network.csv"
GPKG_OUTPUT_PATH = GPKG / "map_euclidean_vs_network_villages.gpkg"


def compute_huff(gi_values, dist_matrix):
    """attract = GI / dist**BETA, pij = attract / sum(attract) per village.

    Villages with a zero-distance municipality (village at the municipal
    seat) are assigned Pij=1 to that municipality directly.
    """
    zero_mask = dist_matrix == 0
    row_has_zero = zero_mask.any(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        attract = gi_values[np.newaxis, :] / np.power(dist_matrix, BETA)
    attract = np.where(zero_mask, 0.0, attract)
    attract = np.nan_to_num(attract, nan=0.0, posinf=0.0, neginf=0.0)

    pij = np.zeros_like(attract)
    nz_idx = ~row_has_zero
    row_sums = attract[nz_idx].sum(axis=1, keepdims=True)
    pij[nz_idx] = attract[nz_idx] / row_sums
    pij[row_has_zero] = zero_mask[row_has_zero].astype(np.float64)

    return pij


def main():
    print("=== EUCLIDEAN VS NETWORK COMPARISON ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    GPKG.mkdir(parents=True, exist_ok=True)

    network_summary_path = TABLES / "huff_AHP_summary.csv"
    if not network_summary_path.exists():
        raise FileNotFoundError(
            f"{network_summary_path} not found — run src/03_huff_ahp.py first."
        )

    print("Loading village centroids...")
    villages = gpd.read_file(DATA_RAW / VILLAGES_FILE)
    villages = villages.rename(columns={"NA_MID": "Village_ID", "NA_NA_UIME": "Village_Name"})
    print(f"  Villages: {len(villages)}")

    print("Loading municipality centroids and GI_AHP...")
    munis = gpd.read_file(DATA_RAW / MUNICIPALITIES_AHP)
    print(f"  Municipalities: {len(munis)}")
    print()

    print("Computing Euclidean distance matrix (cdist)...")
    village_xy = np.column_stack([villages.geometry.x, villages.geometry.y])
    muni_xy = np.column_stack([munis.geometry.x, munis.geometry.y])
    dist_matrix = cdist(village_xy, muni_xy, metric="euclidean")
    print(f"  Distance matrix: {dist_matrix.shape}")
    print()

    print("Computing Euclidean-distance Huff probabilities...")
    gi_values = munis["GI_AHP"].values.astype(np.float64)
    pij = compute_huff(gi_values, dist_matrix)

    dominant_idx = pij.argmax(axis=1)
    muni_names = munis["Muni_Name"].values

    euclidean = pd.DataFrame({
        "Village_ID": villages["Village_ID"].values,
        "Village_Name": villages["Village_Name"].values,
        "euclidean_dominant_muni": muni_names[dominant_idx],
        "euclidean_dominant_Pij": pij[np.arange(len(pij)), dominant_idx],
        "euclidean_dominant_dist_m": dist_matrix[np.arange(len(dist_matrix)), dominant_idx],
    })
    print(f"  Computed Euclidean Huff assignments for {len(euclidean)} villages")
    print()

    print("Loading road-network dominant assignments (huff_AHP_summary.csv)...")
    network = pd.read_csv(network_summary_path).rename(columns={
        "dominant_municipality": "network_dominant_muni",
        "dominant_Pij": "network_dominant_Pij",
        "dominant_dist_m": "network_dominant_dist_m",
    })
    print(f"  Loaded {len(network)} rows")
    print()

    print("Comparing Euclidean vs network-distance dominant assignments...")
    comparison = euclidean.merge(
        network[["Village_ID", "network_dominant_muni", "network_dominant_Pij", "network_dominant_dist_m"]],
        on="Village_ID", how="left")
    comparison["agreement"] = (
        comparison["euclidean_dominant_muni"] == comparison["network_dominant_muni"]).astype(int)

    n_agree = int(comparison["agreement"].sum())
    n_total = len(comparison)
    agreement_pct = 100.0 * n_agree / n_total
    kappa = cohen_kappa_score(
        comparison["network_dominant_muni"], comparison["euclidean_dominant_muni"])
    print(f"  Agreement: {n_agree}/{n_total} ({agreement_pct:.2f}%)  Cohen kappa={kappa:.4f}")
    print()

    print("Classifying disagreements by region (spatial join to obcine_poligoni.shp)...")
    obc = gpd.read_file(DATA_RAW / OBCINE_FILE)
    joined = villages[["Village_ID", "geometry"]].sjoin(
        obc[["SIFRA", "NAZIV", "geometry"]], how="left", predicate="within")
    comparison = comparison.merge(
        joined[["Village_ID", "SIFRA", "NAZIV"]].rename(columns={"NAZIV": "home_region"}),
        on="Village_ID", how="left")

    region_summary = comparison.groupby("home_region")["agreement"].agg(
        n_villages="count", n_agree="sum")
    region_summary["disagreement_pct"] = 100.0 * (
        1 - region_summary["n_agree"] / region_summary["n_villages"])
    region_summary = region_summary.sort_values("disagreement_pct", ascending=False)
    print("  Top 10 regions by disagreement rate:")
    print(region_summary.head(10).to_string())
    print()

    comparison.to_csv(TABLE_OUTPUT_PATH, index=False)
    print(f"Saved {TABLE_OUTPUT_PATH}")

    print("Building village-polygon spatial layer (NA.shp)...")
    na = gpd.read_file(DATA_RAW / SETTLEMENTS_POLY)
    na = na[["NA_MID", "NA_UIME", "geometry"]].copy()
    spatial = na.merge(comparison, left_on="NA_MID", right_on="Village_ID", how="left")
    spatial.to_file(GPKG_OUTPUT_PATH, driver="GPKG")
    print(f"Saved {GPKG_OUTPUT_PATH}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
