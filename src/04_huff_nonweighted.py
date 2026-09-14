"""
Step 4 of the pipeline: the non-weighted (NW) Huff model.

What this script does: the same modified Huff calculation as
03_huff_ahp.py, but using GI_Final_NotWeighted (the Gravitational Index
computed without AHP group weighting) instead of GI_AHP. This gives the
study a second, independent catchment assignment to compare the AHP-based
one against — how much difference the AHP group weighting actually makes
to which municipality each settlement is assigned to. The two scripts
share the same distance matrix (whichever one runs first computes and
caches it in data/processed/; the other reuses it) and the same Huff
formula, differing only in which GI column feeds it.

Reads: Municipalities_Points_normalized.gpkg (municipality locations),
Municipalities_All_Groups_NotWeighted_Normalized.gpkg (GI_Final_NotWeighted),
Villages_points_real.shp (settlement locations), and the road network
cached by 02_road_network.py.

Writes: huff_NW_summary.csv (one row per settlement: its assigned
municipality and the winning probability) and huff_NW_od_matrix.csv (the
full distance and probability matrix).

Runs fourth, independently of 03_huff_ahp.py — order between the two does
not matter, but both must run before anything that compares AHP against
NW results.
"""

import argparse
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
    DATA_RAW, DATA_PROCESSED, TABLES, BETA, CUTOFF_M, EPSG,
    MUNICIPALITIES_NW, MUNICIPALITIES_PTS, VILLAGES_FILE,
)
from crs_utils import ensure_crs

NODED_ROADS_PATH = DATA_PROCESSED / "roads_noded.gpkg"
DIST_MATRIX_PATH = DATA_PROCESSED / "distance_matrix.npy"
DIST_VILLAGE_IDS_PATH = DATA_PROCESSED / "distance_matrix_village_ids.npy"
DIST_MUNI_IDS_PATH = DATA_PROCESSED / "distance_matrix_muni_ids.npy"
OUTPUT_PATH = TABLES / "huff_NW_summary.csv"
OD_MATRIX_PATH = TABLES / "huff_NW_od_matrix.csv"

OUTPUT_FILES = [
    "tables/huff_NW_summary.csv",
    "tables/huff_NW_od_matrix.csv",
]


def load_graph():
    """Rebuild the road network graph saved by 02_road_network.py.

    A graph object cannot be saved directly into a GPKG file, so 02 instead
    saves the road segments and intersection points as plain geometry, and
    this function reconstructs the exact same graph from that geometry.
    """
    if not NODED_ROADS_PATH.exists():
        raise FileNotFoundError(
            f"{NODED_ROADS_PATH} not found — run src/02_road_network.py first "
            "to build the noded road network."
        )
    noded = gpd.read_file(NODED_ROADS_PATH)
    return momepy.gdf_to_nx(noded, approach="primal", length="length_m")


def snap_to_network(points_gdf, node_coords, node_list):
    """Attach each settlement or municipality point to its nearest road-network node.

    Points such as settlement centroids rarely fall exactly on a mapped
    road, so each one is matched to the closest point in the road network
    graph before any distance can be measured from it.
    """
    tree = cKDTree(node_coords)
    xy = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])
    _, idx = tree.query(xy)
    return [node_list[i] for i in idx]


def compute_distance_matrix(G, munis, villages):
    """Compute the shortest road-network distance from every municipality to every settlement.

    Runs one Dijkstra search per municipality (much faster than one per
    settlement, since there are only 212 municipalities against 6,036
    settlements), stopping each search at CUTOFF_M. A (settlement,
    municipality) pair with no path within that cutoff is left as a
    missing value here and filled in below. In practice this script reuses
    the distance matrix 03_huff_ahp.py already computed rather than
    recomputing it — see load_cached_distance_matrix below — since the
    road network and the settlement/municipality locations are identical
    between the two models; only the GI values differ.
    """
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

    # A settlement with no path to a given municipality within CUTOFF_M gets
    # that municipality's own worst (maximum) observed distance to any
    # settlement, instead of being left undefined. This makes that
    # municipality maximally unattractive to the settlement in the Huff
    # formula below without needing to special-case a missing distance —
    # the settlement can still, correctly, end up assigned elsewhere.
    col_max = np.nanmax(dist_matrix, axis=0)
    global_max = np.nanmax(dist_matrix)
    col_max = np.where(np.isnan(col_max), global_max, col_max)
    nan_rows, nan_cols = np.where(np.isnan(dist_matrix))
    dist_matrix[nan_rows, nan_cols] = col_max[nan_cols]

    return dist_matrix


def load_cached_distance_matrix(village_ids, muni_ids):
    """Reuse a previously computed distance matrix if one exists and still matches.

    Computing the full distance matrix from the road network is the slowest
    step in this script, so the result is cached to disk (typically by
    03_huff_ahp.py, since the two Huff scripts share the same distance
    matrix). The cache is only reused if the settlement and municipality ID
    lists are in exactly the same order as when it was saved — otherwise a
    stale or mismatched cache could silently attach the wrong distances to
    the wrong settlements.
    """
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
    """Apply the modified Huff formula to turn GI and distance into assignment probabilities.

    Each municipality's pull on a settlement is its GI divided by its
    distance raised to the power BETA (the standard gravity-model
    distance-decay term — pull falls off quickly with distance). Dividing
    each settlement's row of pull values by their own sum turns them into
    probabilities that sum to 1, i.e. each settlement's likelihood of being
    served by each municipality. A settlement that sits exactly at a
    municipal seat (distance 0) is assigned to that municipality with
    probability 1 directly, since the pull formula is undefined at zero
    distance (division by zero).
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Recompute even if huff_NW_summary.csv already exists.")
    args = parser.parse_args()

    print("=== HUFF NON-WEIGHTED ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not args.force:
        print(f"WARNING: reusing existing {OUTPUT_PATH} — skipping full computation.")
        print("Pass --force to recompute from the road network instead.")
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
    munis = ensure_crs(munis, EPSG, label=MUNICIPALITIES_PTS)
    nw = gpd.read_file(DATA_RAW / MUNICIPALITIES_NW)
    nw = ensure_crs(nw, EPSG, label=MUNICIPALITIES_NW)
    munis = munis.merge(nw[["Muni_ID", "GI_Final_NotWeighted"]], on="Muni_ID", how="left")
    print(f"  Municipalities: {len(munis)}")

    print("Loading village centroids...")
    villages = gpd.read_file(DATA_RAW / VILLAGES_FILE)
    villages = ensure_crs(villages, EPSG, label=VILLAGES_FILE)
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

    print("Saving full OD probability matrix...")
    od = pd.DataFrame({
        "Village_ID": villages["Village_ID"].values,
        "Village_Name": villages["Village_Name"].values,
    })
    for j, name in enumerate(muni_names):
        od[f"dist_{name}"] = dist_matrix[:, j]
        od[f"Pij_{name}"] = pij[:, j]
    od.to_csv(OD_MATRIX_PATH, index=False)
    print(f"Saved {OD_MATRIX_PATH}  ({od.shape[0]} rows x {od.shape[1]} cols)")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
