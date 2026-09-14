"""
Step 1 of the pipeline: check the Gravitational Index (GI) inputs.

What this script does: the two GI layers this study uses — GI_AHP (AHP group
weighting) and GI_Final_NotWeighted (no group weighting) — arrive already
computed in the raw data, one number per municipality per scenario. This
script does not calculate the GI itself. What it does is (1) independently
recompute the "rarity weight" of each of the 100 underlying indicators from
first principles and check that the result matches the reference table
(tableS3) to six decimal places, as a check that the published weighting
scheme is actually what was applied to the data, and (2) load both GI
layers and report summary statistics and the top-20 municipality rankings
under each.

Reads: Municipalities_Points_normalized.gpkg (the 100 raw indicator values
per municipality), Municipalities_All_Groups_Weighted_AHP.gpkg (GI_AHP),
Municipalities_All_Groups_NotWeighted_Normalized.gpkg (GI_Final_NotWeighted),
tableS3_individual_indicator_weights.csv (the reference rarity weights).

Writes: table_GI_summary_stats.csv, table_top20_GI_both.csv.

Runs first in the pipeline; every later script that uses GI_AHP or
GI_Final_NotWeighted reads them straight from the raw municipality layers,
not from anything this script produces.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd

from config import (
    DATA_RAW, TABLES, SUPPLEMENTARY, N_MUNICIPALITIES, EPSG,
    MUNICIPALITIES_AHP, MUNICIPALITIES_NW, MUNICIPALITIES_PTS,
)
from crs_utils import ensure_crs

OUTPUT_FILES = [
    "tables/table_GI_summary_stats.csv",
    "tables/table_top20_GI_both.csv",
]


def load_indicator_group_map(supplementary_path):
    """Map each indicator code to its thematic group, from tableS3.

    tableS3 lists 100 real indicators, but the raw indicator file
    (Municipalities_Points_normalized.gpkg) has 102 columns starting with
    "n_": n_Area_km2 is municipality area, not a service/attractiveness
    indicator, and n_Fitness_C is an exact duplicate of n_Fitness under a
    different name. Both are excluded here for that reason (the same two
    columns are excluded the same way in 06_ml_framework.py).
    """
    # No sep= argument: tableS3 is comma-delimited, like every other table
    # in this repository. A previous version of this line hardcoded
    # sep=";", left over from when tableS3 genuinely was semicolon-
    # delimited; a later fix standardised all four supplementary tables to
    # commas without updating this line, which meant the whole file was
    # read as one unsplit column and this call failed outright with
    # KeyError: ['Indicator_code'] the next time this script ran.
    ref = pd.read_csv(supplementary_path / "tableS3_individual_indicator_weights.csv")
    ref = ref.dropna(subset=["Indicator_code"])
    ref = ref[ref["Indicator_code"] != "Total"]
    return dict(zip(ref["Indicator_code"], ref["Thematic_group"])), ref


def compute_rarity_weights(pts, group_map, n_municipalities):
    """Compute each indicator's rarity weight within its thematic group.

    An indicator that only a few municipalities have (e.g. an airport) is
    more informative than one nearly every municipality has (e.g. a primary
    school), so rarer indicators get more weight. The rarity score ri is
    1 minus the share of municipalities that have a nonzero value for that
    indicator, floored at 0.01: without the floor, an indicator present in
    (almost) every municipality would get a weight of exactly zero and be
    dropped from the index entirely, even though its actual values still
    vary and still carry information. The floor keeps a small (1%) minimum
    contribution for every indicator instead. Weights are then square-rooted
    and renormalised to sum to 1 within each thematic group, so within-group
    rankings are compressed rather than dominated by the rarest indicator.
    """
    rows = []
    for code, group in group_map.items():
        if code not in pts.columns:
            continue
        nonzero = int((pts[code] != 0).sum())
        ri = max(1 - nonzero / n_municipalities, 0.01)
        rows.append({
            "Indicator_code": code,
            "Thematic_group": group,
            "Nonzero_munis": nonzero,
            "ri": ri,
        })
    weights = pd.DataFrame(rows)
    weights["sqrt_ri"] = np.sqrt(weights["ri"])
    weights["wi"] = weights.groupby("Thematic_group")["sqrt_ri"].transform(lambda s: s / s.sum())
    return weights


def main():
    print("=== GI CONSTRUCTION ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)

    print("Loading raw indicators (Municipalities_Points_normalized.gpkg)...")
    pts = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)
    pts = ensure_crs(pts, EPSG, label=MUNICIPALITIES_PTS)
    n_cols = [c for c in pts.columns if c.startswith("n_")]
    print(f"  Shape: {pts.shape}  ({len(n_cols)} n_-prefixed columns)")

    print("Loading indicator/group mapping (tableS3, 100 indicators)...")
    group_map, ref = load_indicator_group_map(SUPPLEMENTARY)
    print(f"  Mapped indicators: {len(group_map)}")
    excluded = sorted(set(n_cols) - set(group_map.keys()))
    print(f"  n_ columns not in mapping (expected: n_Area_km2, n_Fitness_C): {excluded}")
    print()

    print("Computing rarity weights within each thematic group...")
    weights = compute_rarity_weights(pts, group_map, N_MUNICIPALITIES)
    print(f"  Computed weights for {len(weights)} indicators across "
          f"{weights['Thematic_group'].nunique()} groups")

    validation = weights.merge(
        ref[["Indicator_code", "Normalised within-group weight (w_i)"]],
        on="Indicator_code", how="left")
    max_diff = (validation["wi"] - validation["Normalised within-group weight (w_i)"]).abs().max()
    print(f"  Max |computed wi - reference wi| vs tableS3: {max_diff:.6f}")
    print()
    print(weights.sort_values(["Thematic_group", "wi"], ascending=[True, False]).to_string(index=False))
    print()

    print("Loading GI_Final_NotWeighted (Municipalities_All_Groups_NotWeighted_Normalized.gpkg)...")
    nw = gpd.read_file(DATA_RAW / MUNICIPALITIES_NW)
    nw = ensure_crs(nw, EPSG, label=MUNICIPALITIES_NW)
    print(f"  Shape: {nw.shape}")

    print("Loading GI_AHP (Municipalities_All_Groups_Weighted_AHP.gpkg)...")
    ahp = gpd.read_file(DATA_RAW / MUNICIPALITIES_AHP)
    ahp = ensure_crs(ahp, EPSG, label=MUNICIPALITIES_AHP)
    print(f"  Shape: {ahp.shape}")
    print()

    print("=== SUMMARY STATISTICS ===")
    summary_stats = pd.DataFrame([
        {
            "GI_scenario": "GI_Final_NotWeighted",
            "min": nw["GI_Final_NotWeighted"].min(),
            "max": nw["GI_Final_NotWeighted"].max(),
            "mean": nw["GI_Final_NotWeighted"].mean(),
            "std": nw["GI_Final_NotWeighted"].std(),
        },
        {
            "GI_scenario": "GI_AHP",
            "min": ahp["GI_AHP"].min(),
            "max": ahp["GI_AHP"].max(),
            "mean": ahp["GI_AHP"].mean(),
            "std": ahp["GI_AHP"].std(),
        },
    ])
    print(summary_stats.to_string(index=False))
    print()

    print("=== TOP 20 MUNICIPALITIES ===")
    top20_nw = nw[["Muni_ID", "Muni_Name", "GI_Final_NotWeighted"]] \
        .sort_values("GI_Final_NotWeighted", ascending=False).head(20).reset_index(drop=True)
    top20_ahp = ahp[["Muni_ID", "Muni_Name", "GI_AHP"]] \
        .sort_values("GI_AHP", ascending=False).head(20).reset_index(drop=True)

    print("--- By GI_Final_NotWeighted ---")
    print(top20_nw.to_string(index=False))
    print()
    print("--- By GI_AHP ---")
    print(top20_ahp.to_string(index=False))
    print()

    top20_combined = pd.DataFrame({
        "rank": range(1, 21),
        "NotWeighted_Muni_Name": top20_nw["Muni_Name"].values,
        "NotWeighted_GI": top20_nw["GI_Final_NotWeighted"].values,
        "AHP_Muni_Name": top20_ahp["Muni_Name"].values,
        "AHP_GI": top20_ahp["GI_AHP"].values,
    })

    summary_stats.to_csv(TABLES / "table_GI_summary_stats.csv", index=False)
    print(f"Saved table_GI_summary_stats.csv")

    top20_combined.to_csv(TABLES / "table_top20_GI_both.csv", index=False)
    print(f"Saved table_top20_GI_both.csv")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
