"""
For each of 86 facility types compute nearest-facility road distance
per municipality, normalise inverted 0-1, save accessibility_normalized.csv.
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

from config import DATA_RAW, DATA_PROCESSED, TABLES, REPO_ROOT, MUNICIPALITIES_PTS, CUTOFF_M, EPSG
from crs_utils import ensure_crs

NODED_ROADS_PATH = DATA_PROCESSED / "roads_noded.gpkg"
RAW_OUTPUT_PATH = TABLES / "accessibility_raw_distances.csv"
NORM_OUTPUT_PATH = TABLES / "accessibility_normalized.csv"

PRIMARY_CUTOFF_M = 80_000

# The 86 facility layers, one per line, in data/facility_layers.txt.
#
# This used to be a glob over DATA_RAW minus a hand-maintained exclude list.
# That's what let all_roads.gpkg get silently counted as an 87th facility
# type when it was dropped into DATA_RAW on 2026-08-14 (see the Stage 1/2A
# audit trail) — glob discovery makes every file anyone adds to the raw data
# directory a variable in the analysis. An explicit, version-controlled list
# means a new file in DATA_RAW does nothing until someone deliberately adds
# it here.
FACILITY_LAYERS_MANIFEST = REPO_ROOT / "data" / "facility_layers.txt"


def discover_facility_layers(data_raw):
    """Resolve the facility layers named in data/facility_layers.txt to paths
    in DATA_RAW. Raises if a listed layer or an unresolvable duplicate is found —
    this is meant to fail loudly, not silently drift."""
    names = [
        line.strip() for line in FACILITY_LAYERS_MANIFEST.read_text().splitlines()
        if line.strip()
    ]

    paths = {}
    missing = []
    for name in names:
        shp = data_raw / f"{name}.shp"
        gpkg = data_raw / f"{name}.gpkg"
        if shp.exists():
            paths[name] = shp
        elif gpkg.exists():
            paths[name] = gpkg
        else:
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} facility layer(s) listed in {FACILITY_LAYERS_MANIFEST} "
            f"not found in {data_raw}: {missing}"
        )
    return paths


def load_graph():
    """Rebuild the NetworkX graph from the noded roads saved by 02_road_network.py."""
    if not NODED_ROADS_PATH.exists():
        raise FileNotFoundError(
            f"{NODED_ROADS_PATH} not found — run src/02_road_network.py first "
            "to build the noded road network."
        )
    noded = gpd.read_file(NODED_ROADS_PATH)
    return momepy.gdf_to_nx(noded, approach="primal", length="length_m")


def snap_to_network(xy, node_coords, node_list):
    """Snap an array of (x, y) coordinates to the nearest graph node."""
    tree = cKDTree(node_coords)
    _, idx = tree.query(xy)
    return [node_list[i] for i in idx]


def load_and_snap_facilities(facility_paths, node_coords, node_list):
    """Load every facility layer and snap its features to network nodes.

    Returns {facility_name: set_of_nearest_nodes}.
    """
    facility_nodes = {}
    for name, path in facility_paths.items():
        gdf = gpd.read_file(path)
        gdf = ensure_crs(gdf, EPSG, label=path.name)
        centroids = gdf.geometry.centroid
        xy = np.column_stack([centroids.x, centroids.y])
        nodes = snap_to_network(xy, node_coords, node_list)
        facility_nodes[name] = set(nodes)
    return facility_nodes


def nearest_facility_distances(G, muni_node, facility_nodes, primary_cutoff, extended_cutoff):
    """One Dijkstra at `primary_cutoff`; only municipalities with an
    unresolved facility get a second, wider Dijkstra up to `extended_cutoff`."""
    lengths = nx.single_source_dijkstra_path_length(
        G, muni_node, cutoff=primary_cutoff, weight="length_m")

    result = {}
    missing = []
    for name, nodes in facility_nodes.items():
        reachable = [lengths[n] for n in nodes if n in lengths]
        if reachable:
            result[name] = min(reachable)
        else:
            missing.append(name)

    if missing:
        lengths_ext = nx.single_source_dijkstra_path_length(
            G, muni_node, cutoff=extended_cutoff, weight="length_m")
        for name in missing:
            reachable = [lengths_ext[n] for n in facility_nodes[name] if n in lengths_ext]
            result[name] = min(reachable) if reachable else np.nan

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Recompute even if accessibility_normalized.csv already exists.")
    args = parser.parse_args()

    print("=== ACCESSIBILITY ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)

    if NORM_OUTPUT_PATH.exists() and not args.force:
        print(f"WARNING: reusing existing {NORM_OUTPUT_PATH} — skipping computation.")
        print("Pass --force to recompute instead.")
        existing = pd.read_csv(NORM_OUTPUT_PATH)
        print(f"  {existing.shape[0]} rows, {existing.shape[1]} columns loaded")
        print()
        print("Done.")
        return

    print("Discovering facility layers in DATA_RAW...")
    facility_paths = discover_facility_layers(DATA_RAW)
    print(f"  Found {len(facility_paths)} facility layers")
    if len(facility_paths) != 86:
        print(f"  WARNING: expected 86 facility layers, found {len(facility_paths)} — "
              "check data/facility_layers.txt.")
    print()

    print("Building road network graph...")
    G = load_graph()
    node_list = list(G.nodes)
    node_coords = np.array(node_list, dtype=np.float64)
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    print("Loading municipality centroids...")
    munis = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)[["Muni_ID", "Muni_Name", "geometry"]].copy()
    munis = ensure_crs(munis, EPSG, label=MUNICIPALITIES_PTS)
    muni_xy = np.column_stack([munis.geometry.x, munis.geometry.y])
    muni_nodes = snap_to_network(muni_xy, node_coords, node_list)
    print(f"  Municipalities: {len(munis)}")
    print()

    print("Loading and snapping facility layers to the network...")
    facility_nodes = load_and_snap_facilities(facility_paths, node_coords, node_list)
    for name, nodes in facility_nodes.items():
        if not nodes:
            print(f"  WARNING: {name} has no snappable features")
    print(f"  Snapped {len(facility_nodes)} facility layers")
    print()

    print(f"Running Dijkstra per municipality (primary cutoff={PRIMARY_CUTOFF_M / 1000:.0f} km, "
          f"extended cutoff={CUTOFF_M / 1000:.0f} km)...")
    rows = []
    for i, (muni_node, muni_id, muni_name) in enumerate(
            zip(muni_nodes, munis["Muni_ID"], munis["Muni_Name"])):
        distances = nearest_facility_distances(
            G, muni_node, facility_nodes, PRIMARY_CUTOFF_M, CUTOFF_M)
        row = {"Muni_ID": muni_id, "Muni_Name": muni_name}
        row.update({f"dist_{name}": d for name, d in distances.items()})
        rows.append(row)
        if (i + 1) % 50 == 0 or (i + 1) == len(munis):
            print(f"  {i + 1}/{len(munis)} municipalities processed")

    raw = pd.DataFrame(rows)
    dist_cols = [c for c in raw.columns if c.startswith("dist_")]
    print()

    print("Filling remaining NaN distances with column maximum...")
    col_max = raw[dist_cols].max(axis=0, skipna=True)
    global_max = raw[dist_cols].max().max()
    col_max = col_max.fillna(global_max)
    raw[dist_cols] = raw[dist_cols].apply(lambda s: s.fillna(col_max[s.name]))
    print(f"  Raw distance table: {raw.shape}")

    raw.to_csv(RAW_OUTPUT_PATH, index=False)
    print(f"Saved {RAW_OUTPUT_PATH}")
    print()

    print("Applying inverted min-max normalisation...")
    norm = raw[["Muni_ID", "Muni_Name"]].copy()
    for col in dist_cols:
        name = col.replace("dist_", "")
        d = raw[col]
        dmin, dmax = d.min(), d.max()
        if dmax > dmin:
            norm[f"nacc_{name}"] = 1 - (d - dmin) / (dmax - dmin)
        else:
            norm[f"nacc_{name}"] = 1.0
    print(f"  Normalized table: {norm.shape}")

    norm.to_csv(NORM_OUTPUT_PATH, index=False)
    print(f"Saved {NORM_OUTPUT_PATH}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
