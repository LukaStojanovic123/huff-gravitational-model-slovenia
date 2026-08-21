"""Shared CRS handling for every script that reads a raw spatial file.

Reprojection is conditional: a GeoDataFrame is only reprojected when its
source CRS differs from the target. The source CRS is always printed so a
run's log shows exactly what was read, regardless of whether it matched.
"""


def ensure_crs(gdf, expected_epsg, label):
    """Assert/normalise gdf's CRS to expected_epsg. Reprojects only if needed.

    Raises if the source file has no CRS at all — that's a data problem to
    surface, not paper over.
    """
    if gdf.crs is None:
        raise ValueError(f"{label}: source file has no CRS defined.")

    src_epsg = gdf.crs.to_epsg()
    if src_epsg == expected_epsg:
        print(f"  [CRS] {label}: EPSG:{src_epsg} (matches target, no reprojection)")
        return gdf

    print(f"  [CRS] {label}: EPSG:{src_epsg} != target EPSG:{expected_epsg} — reprojecting")
    return gdf.to_crs(epsg=expected_epsg)
