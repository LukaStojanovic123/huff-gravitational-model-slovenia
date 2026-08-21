# Quarantine

## all_roads.gpkg

Moved out of `DATA_RAW` on 2026-08-21. Not deleted — kept here for reference,
not committed to git (205 MB; see `.gitignore`).

**Why it's here:** `all_roads.gpkg` appeared in `DATA_RAW` with filesystem
mtime 2026-08-14 10:04:39 — three days after `huff_AHP_summary.csv`,
`huff_NW_summary.csv`, and `accessibility_normalized.csv` had already been
committed to this repository. It is a different, larger OSM road extract
(496,567 raw features, EPSG:3794) than the one that actually produced the
manuscript's numbers.

`config.py`'s `ROADS_FILE` briefly pointed at it (commit `1af57e7`,
2026-08-14), which is how it entered the pipeline and produced the road
network / OD matrix / facility-count discrepancies documented in
`outputs/audit/` (`raw_data_manifest.csv`, `staleness_report.csv`,
`data_audit_report.md`).

The file that actually reproduces the manuscript's road-network numbers is
`gis_osm_roads_free_1.shp`, untouched since 2026-04-29, now copied into
`DATA_RAW` and set as `ROADS_FILE`. See `outputs/audit/` for the full
verification (254,252 filtered segments, 439,091 noded, 394,874 nodes,
390,273-node largest component — exact matches once `02_road_network.py`
was fixed to persist the largest-component filter).

Do not delete this file or restore it to `DATA_RAW` under the name
`all_roads.gpkg` — that would silently reintroduce the same contamination.
