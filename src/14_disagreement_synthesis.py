"""
Join the four settlement-level agreement layers (AHP vs NW, AHP vs ML,
NW vs ML, RF-AHP-target vs RF-NW-target) plus the AHP entropy layer,
produce a four-way disagreement synthesis layer and summary table, and
per-comparison "where do disagreeing settlements go" destination tables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd

from config import GPKG, TABLES

AHP_VS_NW_PATH = GPKG / "map_AHP_vs_NW_villages.gpkg"
AHP_VS_ML_PATH = GPKG / "map_AHP_vs_ML_villages.gpkg"
NW_VS_ML_PATH = GPKG / "map_NW_vs_ML_villages.gpkg"
MLAHP_VS_MLNW_PATH = GPKG / "map_ML_AHP_vs_ML_NW_villages.gpkg"
ENTROPY_AHP_PATH = GPKG / "map_entropy_AHP_villages.gpkg"

OUTPUT_LAYER_PATH = GPKG / "map_disagreement_count_villages.gpkg"
SYNTHESIS_TABLE_PATH = TABLES / "table_disagreement_synthesis.csv"
DESTINATIONS_TABLE_PATH = TABLES / "table_disagreement_destinations.csv"

MIN_DESTINATION_COUNT = 5


def main():
    print("=== DISAGREEMENT SYNTHESIS ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    GPKG.mkdir(parents=True, exist_ok=True)

    print("Loading agreement layers...")
    ahp_nw = gpd.read_file(AHP_VS_NW_PATH)[
        ["NA_MID", "NA_UIME", "AHP_dominant_muni", "NW_dominant_muni", "agreement", "geometry"]
    ].rename(columns={"agreement": "ag_AHP_NW"})
    ahp_ml = pd.DataFrame(gpd.read_file(AHP_VS_ML_PATH))[
        ["NA_MID", "ml_dominant_muni", "agreement"]
    ].rename(columns={"agreement": "ag_AHP_ML"})
    nw_ml = pd.DataFrame(gpd.read_file(NW_VS_ML_PATH))[
        ["NA_MID", "agreement"]
    ].rename(columns={"agreement": "ag_NW_ML"})
    mlahp_mlnw = pd.DataFrame(gpd.read_file(MLAHP_VS_MLNW_PATH))[
        ["NA_MID", "ML_AHP_dominant_muni", "ML_NW_dominant_muni", "agreement"]
    ].rename(columns={"agreement": "ag_MLAHP_MLNW"})
    entropy = pd.DataFrame(gpd.read_file(ENTROPY_AHP_PATH))[
        ["NA_MID", "entropy_AHP", "entropy_class"]
    ]
    print(f"  AHP vs NW: {len(ahp_nw)}, AHP vs ML: {len(ahp_ml)}, "
          f"NW vs ML: {len(nw_ml)}, RF-AHP vs RF-NW: {len(mlahp_mlnw)}, entropy: {len(entropy)}")

    merged = (ahp_nw.merge(ahp_ml, on="NA_MID", how="left")
                     .merge(nw_ml, on="NA_MID", how="left")
                     .merge(mlahp_mlnw, on="NA_MID", how="left")
                     .merge(entropy, on="NA_MID", how="left"))
    print(f"  Joined: {len(merged)} settlements")
    print()

    merged["n_disagree"] = (4 - (merged["ag_AHP_NW"] + merged["ag_AHP_ML"]
                                  + merged["ag_NW_ML"] + merged["ag_MLAHP_MLNW"])).astype(int)
    merged["all_four_same"] = merged["n_disagree"] == 0

    n_bucket1 = int((merged["n_disagree"] == 1).sum())
    print(f"n_disagree == 1 settlements: {n_bucket1}")
    print("  IMPORTANT: this now joins FOUR distinct classifications (AHP Huff, NW Huff, "
          "RF-on-AHP-target, RF-on-NW-target) across four of the six possible pairs "
          "(AHP-vs-NW, AHP-vs-RF(AHP), NW-vs-RF(NW), RF(AHP)-vs-RF(NW)) — see "
          "outputs/audit/ml_model_design_note.md for the full explanation of why two "
          "separately trained RF models exist. n_disagree runs 0-4, not 0-3: a settlement "
          "can, for instance, have both Huff models agree with each other and both RF "
          "models agree with each other, while the two pairs disagree across the Huff/RF "
          "boundary — that is a real, legitimate structure, not an artifact of the join.")
    print()

    out_cols = ["NA_MID", "NA_UIME", "AHP_dominant_muni", "NW_dominant_muni", "ml_dominant_muni",
                "ML_AHP_dominant_muni", "ML_NW_dominant_muni",
                "ag_AHP_NW", "ag_AHP_ML", "ag_NW_ML", "ag_MLAHP_MLNW", "n_disagree", "all_four_same",
                "entropy_AHP", "entropy_class", "geometry"]
    layer = gpd.GeoDataFrame(merged[out_cols], geometry="geometry", crs=ahp_nw.crs)
    layer.to_file(OUTPUT_LAYER_PATH, driver="GPKG")
    print(f"Saved {OUTPUT_LAYER_PATH}")
    print()

    print("=== SYNTHESIS TABLE (n_disagree 0-4) ===")
    rows = []
    for k in [0, 1, 2, 3, 4]:
        sub = merged[merged["n_disagree"] == k]
        n = len(sub)
        rows.append({
            "n_disagree": k,
            "n_settlements": n,
            "share_pct": 100 * n / len(merged),
            "mean_entropy_AHP": sub["entropy_AHP"].mean() if n else np.nan,
            "std_entropy_AHP": sub["entropy_AHP"].std() if n else np.nan,
            "pct_low_entropy": 100 * (sub["entropy_class"] == "low").sum() / n if n else np.nan,
            "pct_medium_entropy": 100 * (sub["entropy_class"] == "medium").sum() / n if n else np.nan,
            "pct_high_entropy": 100 * (sub["entropy_class"] == "high").sum() / n if n else np.nan,
        })
    synth_df = pd.DataFrame(rows)
    synth_df.to_csv(SYNTHESIS_TABLE_PATH, index=False)
    print(synth_df.to_string(index=False))
    print(f"\nSaved {SYNTHESIS_TABLE_PATH}")
    print()
    print("Previous (three-comparison) brief's expected distribution (0/1/2/3): "
          "3897 / 792 / 1303 / 44, mean entropy 0.452 / 0.578 / 0.615 / 0.649 — not directly "
          "comparable to the table above, which now has a 5th (0-4) bucket structure from "
          "the added RF-AHP-vs-RF-NW comparison.")
    print()

    n_identical = int(merged["all_four_same"].sum())
    print(f"Settlements with identical dominant centre from all four models: {n_identical}")
    print()

    print("=== DISAGREEMENT DESTINATION TABLES (>= 5 settlements) ===")
    dest_rows = []
    comparisons = [
        ("AHP_vs_NW", "AHP", "AHP_dominant_muni", "NW", "NW_dominant_muni", "ag_AHP_NW"),
        ("AHP_vs_ML", "AHP", "AHP_dominant_muni", "ML", "ml_dominant_muni", "ag_AHP_ML"),
        ("NW_vs_ML", "NW", "NW_dominant_muni", "ML", "ml_dominant_muni", "ag_NW_ML"),
        ("MLAHP_vs_MLNW", "ML(AHP-target)", "ML_AHP_dominant_muni",
         "ML(NW-target)", "ML_NW_dominant_muni", "ag_MLAHP_MLNW"),
    ]
    for comp_name, src_model, src_col, tgt_model, tgt_col, ag_col in comparisons:
        disagreeing = merged[merged[ag_col] == 0]
        counts = disagreeing.groupby([src_col, tgt_col]).size().reset_index(name="n_settlements")
        counts = counts[counts["n_settlements"] >= MIN_DESTINATION_COUNT]
        counts = counts.sort_values("n_settlements", ascending=False)
        for _, r in counts.iterrows():
            dest_rows.append({
                "comparison": comp_name, "source_model": src_model, "source_centre": r[src_col],
                "target_model": tgt_model, "target_centre": r[tgt_col],
                "n_settlements": int(r["n_settlements"]),
            })
        print(f"  {comp_name}: {len(counts)} destination pairs with >= {MIN_DESTINATION_COUNT} settlements "
              f"(of {len(disagreeing)} disagreeing settlements)")

    dest_df = pd.DataFrame(dest_rows)
    dest_df.to_csv(DESTINATIONS_TABLE_PATH, index=False)
    print(f"\nSaved {DESTINATIONS_TABLE_PATH} ({len(dest_df)} rows)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
