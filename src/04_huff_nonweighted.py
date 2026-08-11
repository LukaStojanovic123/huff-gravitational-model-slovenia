"""
Reuse distance matrix, swap GI to non-weighted, compute NW Huff
probabilities, save huff_NW_od_matrix.csv and huff_NW_summary.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import momepy
from scipy.spatial import cKDTree

from config import (
    DATA_RAW, DATA_PROCESSED, TABLES, BETA, CUTOFF_M,
    MUNICIPALITIES_NW, MUNICIPALITIES_PTS, VILLAGES_FILE,
)

NODED_ROADS_PATH = DATA_PROCESSED / "roads_noded.gpkg"
DIST_MATRIX_PATH = DATA_PROCESSED / "distance_matrix.npy"
DIST_VILLAGE_IDS_PATH = DATA_PROCESSED / "distance_matrix_village_ids.npy"
DIST_MUNI_IDS_PATH = DATA_PROCESSED / "distance_matrix_muni_ids.npy"
OUTPUT_PATH = TABLES / "huff_NW_summary.csv"


def load_graph():
    """Rebuild the NetworkX graph from the noded roads saved by 02_road_network.py.

    A graph object can't be persisted to GPKG directly, so 02 saves the noded
    line geometries and this rebuilds the identical graph via momepy.
    """
    if not NODED_ROADS_PATH.exists():
        raise FileNotFoundError(
            f"{NODED_ROADS_PATH} not found — run src/02_road_network.py first "
            "to build the noded road network."
        )
    noded = gpd.read_file(NODED_ROADS_PATH)
    return momepy.gdf_to_nx(noded, approach="primal", length="length_m")


def snap_to_network(points_gdf, node_coords, node_list):
    """Snap each point to its nearest graph node via cKDTree."""
    tree = cKDTree(node_coords)
    xy = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])
    _, idx = tree.query(xy)
    return [node_list[i] for i in idx]


def compute_distance_matrix(G, munis, villages):
    """Dijkstra from each municipality node outward (cutoff CUTOFF_M),
    recording distance to every village node. NaN distances (unreached
    within cutoff) are filled with the column (municipality) maximum."""
    node_list = list(G.nodes)
    node_coords = np.array(node_list, dtype=np.float64)

    muni_nodes = snap_to_network(munis, node_coords, node_list)
    village_nodes = snap_to_network(villages, node_coords, node_list)

    n_villages = len(villages)
    n_munis = len(munis)
    dist_matrix = np.full((n_villages, n_munis), np.nan, dtype=np.float64)

    village_node_to_rows = {}
    for row_idx, node in enumerate(village_nodes):
        village_node_to_rows.setdefault(node, []).append(row_idx)

    for col_idx, muni_node in enumerate(muni_nodes):
        lengths = nx.single_source_dijkstra_path_length(
            G, muni_node, cutoff=CUTOFF_M, weight="length_m")
        for node, rows in village_node_to_rows.items():
            if node in lengths:
                dist_matrix[rows, col_idx] = lengths[node]

    col_max = np.nanmax(dist_matrix, axis=0)
    global_max = np.nanmax(dist_matrix)
    col_max = np.where(np.isnan(col_max), global_max, col_max)
    nan_rows, nan_cols = np.where(np.isnan(dist_matrix))
    dist_matrix[nan_rows, nan_cols] = col_max[nan_cols]

    return dist_matrix


def load_cached_distance_matrix(village_ids, muni_ids):
    """Load data/processed/distance_matrix.npy if present and its village/muni
    order matches the current data (e.g. already computed by 03_huff_ahp.py).
    Returns None if unavailable or mismatched."""
    if not (DIST_MATRIX_PATH.exists() and DIST_VILLAGE_IDS_PATH.exists()
            and DIST_MUNI_IDS_PATH.exists()):
        return None

    dist_matrix = np.load(DIST_MATRIX_PATH)
    cached_village_ids = np.load(DIST_VILLAGE_IDS_PATH)
    cached_muni_ids = np.load(DIST_MUNI_IDS_PATH)

    if (np.array_equal(cached_village_ids, village_ids)
            and np.array_equal(cached_muni_ids, muni_ids)):
        return dist_matrix
    print("  Cached distance matrix order does not match current data — recomputing.")
    return None


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
    print("=== HUFF NON-WEIGHTED ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"Found existing {OUTPUT_PATH} — skipping full computation.")
        print("Delete this file to force recomputation from the road network.")
        existing = pd.read_csv(OUTPUT_PATH)
        print(f"  {len(existing)} rows loaded")
        print(existing.head(3).to_string(index=False))
        print()
        print("Done.")
        return

    print("No existing huff_NW_summary.csv found — running full computation.")
    print()

    print("Loading municipality centroids and GI_Final_NotWeighted...")
    munis = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)[["Muni_ID", "Muni_Name", "geometry"]].copy()
    nw = gpd.read_file(DATA_RAW / MUNICIPALITIES_NW)
    munis = munis.merge(nw[["Muni_ID", "GI_Final_NotWeighted"]], on="Muni_ID", how="left")
    print(f"  Municipalities: {len(munis)}")

    print("Loading village centroids...")
    villages = gpd.read_file(DATA_RAW / VILLAGES_FILE)
    villages = villages.rename(columns={"NA_MID": "Village_ID", "NA_NA_UIME": "Village_Name"})
    print(f"  Villages: {len(villages)}")
    print()

    print("Checking for cached distance matrix (e.g. from 03_huff_ahp.py)...")
    dist_matrix = load_cached_distance_matrix(
        villages["Village_ID"].values, munis["Muni_ID"].values)

    if dist_matrix is not None:
        print(f"  Loaded cached matrix {dist_matrix.shape}")
    else:
        print("Building road network graph...")
        G = load_graph()
        print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

        print(f"Running Dijkstra from {len(munis)} municipalities (cutoff={CUTOFF_M / 1000:.0f} km)...")
        dist_matrix = compute_distance_matrix(G, munis, villages)
        print(f"  Distance matrix: {dist_matrix.shape}")

        np.save(DIST_MATRIX_PATH, dist_matrix)
        np.save(DIST_VILLAGE_IDS_PATH, villages["Village_ID"].values)
        np.save(DIST_MUNI_IDS_PATH, munis["Muni_ID"].values)
        print(f"  Cached to {DIST_MATRIX_PATH}")
    print()

    print("Computing non-weighted Huff probabilities...")
    gi_values = munis["GI_Final_NotWeighted"].values.astype(np.float64)
    pij = compute_huff(gi_values, dist_matrix)

    dominant_idx = pij.argmax(axis=1)
    muni_names = munis["Muni_Name"].values

    summary = pd.DataFrame({
        "Village_ID": villages["Village_ID"].values,
        "Village_Name": villages["Village_Name"].values,
        "dominant_municipality": muni_names[dominant_idx],
        "dominant_Pij": pij[np.arange(len(pij)), dominant_idx],
        "dominant_dist_m": dist_matrix[np.arange(len(dist_matrix)), dominant_idx],
    })
    print(f"  Computed Huff assignments for {len(summary)} villages")
    print()

    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
