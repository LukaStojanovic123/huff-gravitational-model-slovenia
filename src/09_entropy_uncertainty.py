"""
Step 9 of the pipeline: how confidently is each settlement assigned to its
dominant municipality?

What this script does: a settlement whose Huff probabilities are spread
almost evenly across many municipalities is a much less confident
assignment than one where a single municipality dominates overwhelmingly,
even though both get assigned a single "dominant" municipality elsewhere
in this pipeline. This script measures that spread using Shannon entropy
(a standard information-theory measure of how spread out a probability
distribution is) computed over each settlement's full row of Huff
probabilities, normalised to a 0-1 scale so it does not depend on the
number of municipalities. A settlement is then classified as low, medium
or high uncertainty using the thresholds below, for both the AHP and NW
Huff models.

Reads: NA.shp (settlement polygons), huff_od_matrix.csv and
huff_NW_od_matrix.csv (the full Huff probability matrices, already
computed by 03_huff_ahp.py and 04_huff_nonweighted.py).

Writes: map_entropy_AHP_villages.gpkg, map_entropy_NW_villages.gpkg,
table_entropy_summary.csv.

Runs ninth. Needs 03 and 04's outputs; independent of everything else.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd

from config import DATA_RAW, TABLES, GPKG, N_MUNICIPALITIES, SETTLEMENTS_POLY, EPSG
from crs_utils import ensure_crs

OUTPUT_FILES = [
    "gpkg/map_entropy_AHP_villages.gpkg",
    "gpkg/map_entropy_NW_villages.gpkg",
    "tables/table_entropy_summary.csv",
]

# Entropy classification bands: low uncertainty below 0.2, high uncertainty
# above 0.5, medium in between.
LOW_THRESHOLD = 0.2
HIGH_THRESHOLD = 0.5


def compute_normalized_entropy(od_path, n_municipalities):
    """Compute each settlement's normalised Shannon entropy over its Huff probability row.

    Entropy is 0 when a settlement's Huff probability is concentrated
    entirely on one municipality (a fully confident assignment) and
    approaches 1 when the probability is spread evenly across all
    municipalities (a maximally uncertain assignment). Dividing by the
    maximum possible entropy (log2 of the number of municipalities) puts
    the score on a fixed 0-1 scale, independent of how many municipalities
    there are.
    """
    od = pd.read_csv(od_path)
    pij_cols = [c for c in od.columns if c.startswith("Pij_")]
    pij = od[pij_cols].values.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(pij > 0, pij * np.log2(pij), 0.0)
    raw_entropy = -term.sum(axis=1)
    normalized = raw_entropy / np.log2(n_municipalities)

    return pd.DataFrame({
        "Village_ID": od["Village_ID"].values,
        "Village_Name": od["Village_Name"].values,
        "entropy": normalized,
    })


def classify_entropy(entropy):
    """Bucket each settlement's entropy score into low/medium/high uncertainty."""
    return np.select(
        [entropy < LOW_THRESHOLD, entropy <= HIGH_THRESHOLD],
        ["low", "medium"],
        default="high",
    )


def build_entropy_layer(entropy_df, na, field_name):
    """Join entropy values to NA.shp village polygons and classify."""
    entropy_df = entropy_df.copy()
    entropy_df[field_name] = entropy_df["entropy"]
    entropy_df["entropy_class"] = classify_entropy(entropy_df["entropy"])

    layer = na.merge(
        entropy_df[["Village_ID", field_name, "entropy_class"]],
        left_on="NA_MID", right_on="Village_ID", how="left")
    return layer[["NA_MID", "NA_UIME", field_name, "entropy_class", "geometry"]]


def main():
    print("=== ENTROPY / UNCERTAINTY ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    GPKG.mkdir(parents=True, exist_ok=True)

    ahp_od_path = TABLES / "huff_od_matrix.csv"
    nw_od_path = TABLES / "huff_NW_od_matrix.csv"

    print("Loading village polygons (NA.shp)...")
    na = gpd.read_file(DATA_RAW / SETTLEMENTS_POLY)
    na = ensure_crs(na, EPSG, label=SETTLEMENTS_POLY)
    print(f"  Villages: {len(na)}")
    print()

    print("Computing normalised Shannon entropy for AHP Huff (huff_od_matrix.csv)...")
    entropy_ahp = compute_normalized_entropy(ahp_od_path, N_MUNICIPALITIES)
    print(f"  {len(entropy_ahp)} villages  "
          f"min={entropy_ahp['entropy'].min():.4f}  max={entropy_ahp['entropy'].max():.4f}  "
          f"mean={entropy_ahp['entropy'].mean():.4f}")

    print("Computing normalised Shannon entropy for NW Huff (huff_NW_od_matrix.csv)...")
    entropy_nw = compute_normalized_entropy(nw_od_path, N_MUNICIPALITIES)
    print(f"  {len(entropy_nw)} villages  "
          f"min={entropy_nw['entropy'].min():.4f}  max={entropy_nw['entropy'].max():.4f}  "
          f"mean={entropy_nw['entropy'].mean():.4f}")
    print()

    print("Building AHP entropy layer...")
    layer_ahp = build_entropy_layer(entropy_ahp, na, "entropy_AHP")
    ahp_gpkg_path = GPKG / "map_entropy_AHP_villages.gpkg"
    layer_ahp.to_file(ahp_gpkg_path, driver="GPKG")
    print(f"  Saved {ahp_gpkg_path}")
    print(f"  Class counts: {layer_ahp['entropy_class'].value_counts().to_dict()}")

    print("Building NW entropy layer...")
    layer_nw = build_entropy_layer(entropy_nw, na, "entropy_NW")
    nw_gpkg_path = GPKG / "map_entropy_NW_villages.gpkg"
    layer_nw.to_file(nw_gpkg_path, driver="GPKG")
    print(f"  Saved {nw_gpkg_path}")
    print(f"  Class counts: {layer_nw['entropy_class'].value_counts().to_dict()}")
    print()

    print("Saving entropy summary statistics...")
    summary = pd.DataFrame([
        {
            "GI_scenario": "AHP",
            "min": entropy_ahp["entropy"].min(),
            "max": entropy_ahp["entropy"].max(),
            "mean": entropy_ahp["entropy"].mean(),
            "std": entropy_ahp["entropy"].std(),
            "n_low": int((layer_ahp["entropy_class"] == "low").sum()),
            "n_medium": int((layer_ahp["entropy_class"] == "medium").sum()),
            "n_high": int((layer_ahp["entropy_class"] == "high").sum()),
        },
        {
            "GI_scenario": "NW",
            "min": entropy_nw["entropy"].min(),
            "max": entropy_nw["entropy"].max(),
            "mean": entropy_nw["entropy"].mean(),
            "std": entropy_nw["entropy"].std(),
            "n_low": int((layer_nw["entropy_class"] == "low").sum()),
            "n_medium": int((layer_nw["entropy_class"] == "medium").sum()),
            "n_high": int((layer_nw["entropy_class"] == "high").sum()),
        },
    ])
    summary_path = TABLES / "table_entropy_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
