import hashlib
import json
from pathlib import Path

# Raw data — edit this path when moving machines
DATA_RAW = Path(r"C:\Users\lstojano\Desktop\teza\HuffMethodPaper\Data")

_REPO_ROOT = Path(__file__).parent
_RAW_MANIFEST_PATH = _REPO_ROOT / "data" / "raw_manifest.json"


def _check_raw_manifest():
    """Compare every file the pipeline reads against data/raw_manifest.json
    (SHA256 + size + mtime, frozen at the end of Stage 2B). This is the
    direct fix for how all_roads.gpkg silently contaminated the analysis on
    2026-08-14: DATA_RAW drifted and nothing noticed for three weeks.

    Two distinct outcomes, handled differently on purpose:
    - The manifest is missing or unreadable: this is expected on a first
      run before Stage 2B has ever frozen one, so it only warns and returns
      — it must not block running the pipeline.
    - The manifest exists and DATA_RAW has actually drifted from it (a
      missing or changed file): this is exactly the condition the check
      exists to catch, so it raises. Printing a warning and continuing
      here is what let the all_roads.gpkg contamination run for three
      weeks undetected in the first place.
    """
    if not _RAW_MANIFEST_PATH.exists():
        print(f"WARNING: {_RAW_MANIFEST_PATH} not found — raw-data integrity not checked.")
        return
    try:
        manifest = json.loads(_RAW_MANIFEST_PATH.read_text())
    except Exception as e:
        print(f"WARNING: could not read {_RAW_MANIFEST_PATH}: {e}")
        return

    mismatches = []
    missing = []
    for fname, meta in manifest.get("files", {}).items():
        p = DATA_RAW / fname
        if not p.exists():
            missing.append(fname)
            continue
        st = p.stat()
        if st.st_size != meta["size_bytes"]:
            mismatches.append((fname, "size differs"))
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != meta["sha256"]:
            mismatches.append((fname, "SHA256 differs"))

    if missing or mismatches:
        lines = ["DATA_RAW has drifted from data/raw_manifest.json"]
        for fname in missing:
            lines.append(f"  MISSING:  {fname} (present in manifest, not found in DATA_RAW)")
        for fname, reason in mismatches:
            lines.append(f"  CHANGED:  {fname} ({reason})")
        lines.append("This is exactly how the all_roads.gpkg contamination happened — a raw "
                      "file changed silently and every downstream number drifted with it. "
                      "Investigate before trusting any output produced against the current "
                      "DATA_RAW, then refreeze data/raw_manifest.json once the drift is intentional.")
        raise RuntimeError("\n".join(lines))


_check_raw_manifest()

# Repository-relative paths
REPO_ROOT      = Path(__file__).parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
OUTPUTS        = REPO_ROOT / "outputs"
TABLES         = OUTPUTS / "tables"
FIGURES        = OUTPUTS / "figures"
GPKG           = OUTPUTS / "gpkg"
SUPPLEMENTARY  = OUTPUTS / "supplementary"

# Study constants
N_MUNICIPALITIES = 212
N_SETTLEMENTS    = 6036
# Distance-decay exponent in the Huff model's attractiveness formula
# (GI / distance**BETA) — 2 is the standard inverse-square exponent used
# in gravity-model spatial interaction, matching the original Huff (1963)
# formulation. src/07_beta_sensitivity.py checks how much the results
# change if this is varied.
BETA             = 2
EPSG             = 3794
# Maximum road-network distance (metres) a settlement can be assigned
# across when computing the Huff probabilities — comfortably larger than
# any real driving distance within Slovenia (the country's longest
# straight-line extent is well under 300 km), so this only ever excludes
# pairs that are not realistically reachable, not genuine short trips.
CUTOFF_M         = 300_000

# Key filenames in DATA_RAW
# ROADS_FILE was "all_roads.gpkg" until 2026-08-21, when it was found to be a
# contaminant: a fresher OSM extract dropped into DATA_RAW on 2026-08-14, after
# the committed huff_AHP_summary.csv / huff_NW_summary.csv / accessibility_normalized.csv
# had already been produced from gis_osm_roads_free_1.shp. Quarantined to
# data/quarantine/all_roads.gpkg — see that commit for the full audit trail.
ROADS_FILE         = "gis_osm_roads_free_1.shp"
MUNICIPALITIES_AHP = "Municipalities_All_Groups_Weighted_AHP.gpkg"
MUNICIPALITIES_NW  = "Municipalities_All_Groups_NotWeighted_Normalized.gpkg"
MUNICIPALITIES_PTS = "Municipalities_Points_normalized.gpkg"
VILLAGES_FILE      = "Villages_points_real.shp"
SETTLEMENTS_POLY   = "NA.shp"
COMMUTING_FILE      = "2023tabela.xlsx"

