"""
Load 100 indicators, min-max normalise, compute non-weighted GI and
AHP-weighted GI using rarity weights and AHP group weights.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd

from config import (
    DATA_RAW, TABLES, SUPPLEMENTARY, N_MUNICIPALITIES,
    MUNICIPALITIES_AHP, MUNICIPALITIES_NW, MUNICIPALITIES_PTS,
)


def load_indicator_group_map(supplementary_path):
    """Indicator_code -> Thematic_group, from the reference weight table.

    tableS3 lists 100 real indicators: the raw Municipalities_Points_normalized
    file has 102 n_-prefixed columns, but n_Area_km2 is not an attractiveness
    indicator and n_Fitness_C is an exact duplicate of n_Fitness, so both are
    excluded here (matching src/06_ml_framework.py's handling of n_Fitness_C).
    """
    ref = pd.read_csv(supplementary_path / "tableS3_individual_indicator_weights.csv", sep=";")
    ref = ref.dropna(subset=["Indicator_code"])
    ref = ref[ref["Indicator_code"] != "Total"]
    return dict(zip(ref["Indicator_code"], ref["Thematic_group"])), ref


def compute_rarity_weights(pts, group_map, n_municipalities):
    """ri = max(1 - nonzero_i / N, 0.01); wi = sqrt(ri) / sum(sqrt(ri)) within group."""
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
    print(f"  Shape: {nw.shape}")

    print("Loading GI_AHP (Municipalities_All_Groups_Weighted_AHP.gpkg)...")
    ahp = gpd.read_file(DATA_RAW / MUNICIPALITIES_AHP)
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
