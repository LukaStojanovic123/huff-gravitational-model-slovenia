"""
Step 5 of the pipeline: how well each municipality is served by other
services (used as ML features, not by the Huff model itself).

What this script does: for each of the 86 facility types listed in
data/facility_layers.txt (hospitals, schools, post offices, and so on),
finds the nearest facility of that type to each municipality by road
distance, then rescales each facility type's distances onto a 0-1 scale
where 1 means "closest municipality to that facility type" and 0 means
"furthest." This produces one accessibility score per municipality per
facility type — 86 columns in total — used later as input features for the
Random Forest models in 06_ml_framework.py. It plays no role in the Huff
catchment assignment itself, which only uses road distance and GI.

Reads: the 86 facility layers named in data/facility_layers.txt,
Municipalities_Points_normalized.gpkg (municipality locations), and the
road network cached by 02_road_network.py.

Writes: accessibility_raw_distances.csv (raw distances, before rescaling)
and accessibility_normalized.csv (the 0-1 rescaled version the ML models
actually use).

Runs fifth. Independent of the two Huff scripts (03, 04) — needs only the
road network from 02 — but must run before 06_ml_framework.py, which reads
accessibility_normalized.csv as part of its feature table.
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

OUTPUT_FILES = [
    "tables/accessibility_raw_distances.csv",
    "tables/accessibility_normalized.csv",
]

# First, cheaper search radius for the nearest facility. Only municipalities
# still missing a facility type after this search get the slower, wider
# search up to CUTOFF_M below — most facility types are common enough that
# almost every municipality finds one well within 80 km.
PRIMARY_CUTOFF_M = 80_000

# The 86 facility layers to use, one name per line, listed explicitly in
# data/facility_layers.txt.
#
# This list used to be built automatically by scanning every file in
# DATA_RAW and guessing which ones were facility layers. That is exactly
# how a stray file (all_roads.gpkg, dropped into DATA_RAW on 2026-08-14)
# got silently counted as an 87th facility type and fed into the analysis
# for three weeks before anyone noticed — scanning the folder makes every
# file anyone ever places there a hidden input to the study. Listing the
# 86 names explicitly, and checking that list into version control, means
# a new file placed in DATA_RAW does nothing at all until a person
# deliberately adds its name here.
FACILITY_LAYERS_MANIFEST = REPO_ROOT / "data" / "facility_layers.txt"


def discover_facility_layers(data_raw):
    """Find the file for each facility layer named in data/facility_layers.txt.

    Raises an error naming exactly which layer is missing, rather than
    silently continuing with fewer than 86 facility types — a missing
    layer should stop the run, not quietly change the study's inputs.
    """
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
    """Rebuild the road network graph saved by 02_road_network.py."""
    if not NODED_ROADS_PATH.exists():
        raise FileNotFoundError(
            f"{NODED_ROADS_PATH} not found — run src/02_road_network.py first "
            "to build the noded road network."
        )
    noded = gpd.read_file(NODED_ROADS_PATH)
    return momepy.gdf_to_nx(noded, approach="primal", length="length_m")


def snap_to_network(xy, node_coords, node_list):
    """Attach a set of map coordinates to their nearest road-network nodes."""
    tree = cKDTree(node_coords)
    _, idx = tree.query(xy)
    return [node_list[i] for i in idx]


def load_and_snap_facilities(facility_paths, node_coords, node_list):
    """Load every facility layer and find each facility's nearest road-network node.

    Returns, for each facility type, the set of network nodes any facility
    of that type is closest to (a facility type usually has more than one
    location — e.g. more than one hospital — so this is a set, not a
    single node).
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
    """Find, for one municipality, its road distance to the nearest facility of each type.

    Runs a fast search out to `primary_cutoff` first. Only the facility
    types not yet found within that radius get a second, slower search out
    to `extended_cutoff` — most facility types are found in the first pass,
    so this avoids paying the cost of the wide search for every single one.
    """
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

    # A municipality with no facility of a given type within either search
    # radius gets that facility type's own worst (maximum) observed
    # distance across all municipalities, so it reads as "least accessible"
    # for that facility type rather than leaving a gap in the feature table.
    print("Filling remaining NaN distances with column maximum...")
    col_max = raw[dist_cols].max(axis=0, skipna=True)
    global_max = raw[dist_cols].max().max()
    col_max = col_max.fillna(global_max)
    raw[dist_cols] = raw[dist_cols].apply(lambda s: s.fillna(col_max[s.name]))
    print(f"  Raw distance table: {raw.shape}")

    raw.to_csv(RAW_OUTPUT_PATH, index=False)
    print(f"Saved {RAW_OUTPUT_PATH}")
    print()

    # Rescale each facility type's distance column onto a common 0-1 scale,
    # and flip the direction so that 1 means "closest" and 0 means
    # "furthest" — a higher accessibility score reads the same way a
    # higher GI or Huff probability does, which matters when these columns
    # are later used as ML features alongside those. If every municipality
    # has the exact same distance to a facility type (dmax == dmin), every
    # municipality gets the maximum score, since there is no difference
    # between them to rescale.
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
