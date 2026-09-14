"""
Step 2 of the pipeline: build the road network used for every distance
calculation in this study.

What this script does: loads the raw OpenStreetMap road layer, keeps only
road classes a car can actually drive on, splits every line at the points
where roads actually cross each other (OSM lines can overlap or cross
without sharing an endpoint, which would silently break routing), turns
the result into a graph of intersections and road segments, and keeps only
the largest connected piece of that graph. A road segment that isn't
connected to the main network has no valid driving route to anywhere, so
distances to or from it would be undefined; keeping only the largest
connected component guarantees every later distance calculation has a real
path to follow.

Reads: gis_osm_roads_free_1.shp (the raw OSM road layer, from config.py's
ROADS_FILE — see config.py for the note on why this specific file, not
the also-present all_roads.gpkg, is the correct one to use).

Writes: data/processed/roads_noded.gpkg (the cleaned, connected road
network; a cached intermediate, not a published output).

Runs second, right after the GI check. Everything from src/03 onward that
needs a road distance (the Huff models, accessibility, the ML framework)
depends on this script having been run at least once — but it does not
need to be rerun on every pipeline run, since its output is cached.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import geopandas as gpd
import networkx as nx
import momepy
from shapely.ops import unary_union
from scipy.spatial import cKDTree

from config import DATA_RAW, DATA_PROCESSED, ROADS_FILE, EPSG
from crs_utils import ensure_crs

# Writes only to data/processed/ (a gitignored cache), nothing under
# outputs/ — nothing for 19_output_manifest.py to track here.
OUTPUT_FILES = []

# The OSM road classes ("fclass") a car can actually be driven on. This
# excludes foot/cycle paths, steps, tracks and similar classes that OSM
# tags separately from the drivable road network.
FCLASS_KEEP = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "living_street",
    "unclassified", "service",
]


def _is_valid_line(geom):
    """A usable road segment: not missing, not empty, has at least two points."""
    return (geom is not None and not geom.is_empty
            and geom.geom_type == "LineString" and len(geom.coords) >= 2)


def node_topology(gdf):
    """Split every road line at the points where it crosses another road.

    Two OSM lines that physically cross on the map are not automatically
    split at that crossing point — a routing graph built directly from them
    would treat the crossing as two separate, unconnected lines passing
    through the same space rather than as a real intersection where a
    vehicle could turn. Merging all geometries with unary_union forces a
    split at every such crossing, so the graph built afterwards has a real
    node wherever roads actually meet.
    """
    merged = unary_union(gdf.geometry.tolist())
    if merged.geom_type == "LineString":
        lines = [merged]
    elif merged.geom_type == "MultiLineString":
        lines = list(merged.geoms)
    else:
        lines = [g for g in getattr(merged, "geoms", [merged]) if g.geom_type == "LineString"]

    noded = gpd.GeoDataFrame(geometry=lines, crs=gdf.crs)
    noded["length_m"] = noded.geometry.length
    return noded


def main():
    print("=== ROAD NETWORK ===")
    print()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    roads_path = DATA_RAW / ROADS_FILE
    print(f"Loading roads from {roads_path.name}...")
    roads = gpd.read_file(roads_path)
    print(f"  Loaded {len(roads)} features")
    roads = ensure_crs(roads, EPSG, label=roads_path.name)

    print(f"Filtering to drivable fclass values...")
    roads = roads[roads["fclass"].isin(FCLASS_KEEP)].copy()
    print(f"  Kept {len(roads)} features")
    if len(roads) == 0:
        raise ValueError("No roads matched the fclass filter — check ROADS_FILE contents.")

    print("Exploding multi-part geometries to LineStrings...")
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    print(f"  {len(roads)} parts after explode")

    print("Removing empty/degenerate geometries...")
    roads = roads[roads.geometry.apply(_is_valid_line)].copy()
    print(f"  {len(roads)} valid segments")

    roads["length_m"] = roads.geometry.length
    print(f"  Total length: {roads['length_m'].sum() / 1000:.1f} km")
    print()

    print("Applying unary_union noding to fix topology...")
    noded = node_topology(roads)
    print(f"  {len(noded)} segments after noding")

    print("Filtering degenerate segments after noding...")
    noded = noded[noded.geometry.apply(_is_valid_line)].reset_index(drop=True)
    noded["length_m"] = noded.geometry.length
    print(f"  {len(noded)} segments remain")
    print()

    print("Building NetworkX graph (momepy primal)...")
    G = momepy.gdf_to_nx(noded, approach="primal", length="length_m")
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    print("Extracting largest connected component...")
    components = list(nx.connected_components(G))
    largest_nodes = max(components, key=len)
    G_largest = G.subgraph(largest_nodes).copy()
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component: {G_largest.number_of_nodes()} nodes, "
          f"{G_largest.number_of_edges()} edges "
          f"({100 * len(largest_nodes) / G.number_of_nodes():.1f}% of nodes)")
    print()

    print("Building cKDTree spatial index on largest-component nodes...")
    node_list = list(G_largest.nodes)
    node_coords = np.array(node_list, dtype=np.float64)
    tree = cKDTree(node_coords)
    print(f"  Indexed {len(node_list)} nodes")
    print()

    print("Filtering to the largest connected component before saving...")
    largest_gdf = momepy.nx_to_gdf(G_largest, points=False, lines=True)
    print(f"  {len(largest_gdf)} edges in largest component (of {len(noded)} noded segments)")

    print("Saving noded roads (largest connected component only)...")
    out_path = DATA_PROCESSED / "roads_noded.gpkg"
    largest_gdf.to_file(out_path, driver="GPKG")
    print(f"  Saved {out_path}")
    print()

    print("=== SUMMARY ===")
    print(f"Input segments (post-filter):  {len(roads)}")
    print(f"Noded segments:                {len(noded)}")
    print(f"Graph nodes / edges:           {G.number_of_nodes()} / {G.number_of_edges()}")
    print(f"Connected components:          {len(components)}")
    print(f"Largest component nodes:       {G_largest.number_of_nodes()} "
          f"({100 * len(largest_nodes) / G.number_of_nodes():.1f}%)")
    print(f"Largest component edges:       {G_largest.number_of_edges()}")
    print(f"Largest component length:      "
          f"{sum(d['length_m'] for _, _, d in G_largest.edges(data=True)) / 1000:.1f} km")
    print("Done.")

    return G_largest, tree, node_list


if __name__ == "__main__":
    main()
