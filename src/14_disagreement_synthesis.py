"""
Join the three settlement-level agreement layers (AHP vs NW, AHP vs ML,
NW vs ML) plus the AHP entropy layer, produce the primary three-way
disagreement synthesis layer and summary table (n_disagree 0-3, the
paper's uncertainty measure), plus a secondary four-way version that also
folds in RF-AHP-target vs RF-NW-target, and per-comparison "where do
disagreeing settlements go" destination tables for all four comparisons.
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

# Primary (three-comparison) outputs — the paper's uncertainty measure.
OUTPUT_LAYER_PATH = GPKG / "map_disagreement_count_villages.gpkg"
SYNTHESIS_TABLE_PATH = TABLES / "table_disagreement_synthesis.csv"

# Secondary (four-comparison) outputs — folds in RF-AHP vs RF-NW as well.
OUTPUT_LAYER_4WAY_PATH = GPKG / "map_disagreement_count_4way_villages.gpkg"
SYNTHESIS_TABLE_4WAY_PATH = TABLES / "table_disagreement_synthesis_4way.csv"

DESTINATIONS_TABLE_PATH = TABLES / "table_disagreement_destinations.csv"

MIN_DESTINATION_COUNT = 5


def build_synthesis_table(merged, n_disagree_col, entropy_col, entropy_class_col, k_range):
    rows = []
    for k in k_range:
        sub = merged[merged[n_disagree_col] == k]
        n = len(sub)
        rows.append({
            "n_disagree": k,
            "n_settlements": n,
            "share_pct": 100 * n / len(merged),
            "mean_entropy_AHP": sub[entropy_col].mean() if n else np.nan,
            "std_entropy_AHP": sub[entropy_col].std() if n else np.nan,
            "pct_low_entropy": 100 * (sub[entropy_class_col] == "low").sum() / n if n else np.nan,
            "pct_medium_entropy": 100 * (sub[entropy_class_col] == "medium").sum() / n if n else np.nan,
            "pct_high_entropy": 100 * (sub[entropy_class_col] == "high").sum() / n if n else np.nan,
        })
    return pd.DataFrame(rows)


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

    # ── Primary: three-comparison n_disagree (0-3) — the paper's uncertainty measure ──
    merged["n_disagree_3way"] = (3 - (merged["ag_AHP_NW"] + merged["ag_AHP_ML"]
                                       + merged["ag_NW_ML"])).astype(int)
    merged["all_three_same"] = merged["n_disagree_3way"] == 0

    # ── Secondary: four-comparison n_disagree (0-4), adds RF(AHP) vs RF(NW) ──
    merged["n_disagree_4way"] = (4 - (merged["ag_AHP_NW"] + merged["ag_AHP_ML"]
                                       + merged["ag_NW_ML"] + merged["ag_MLAHP_MLNW"])).astype(int)
    merged["all_four_same"] = merged["n_disagree_4way"] == 0

    n_bucket1_3way = int((merged["n_disagree_3way"] == 1).sum())
    print(f"[primary, 3-way] n_disagree == 1 settlements: {n_bucket1_3way}")
    print("  IMPORTANT: n_disagree==1 is only possible because 'ml_dominant_muni' is "
          "NOT a single consistent classification across the three source layers. "
          "map_AHP_vs_ML_villages.gpkg's ml_dominant_muni comes from the RF model "
          "trained on the AHP Huff target (06_ml_framework.py Model 1), while "
          "map_NW_vs_ML_villages.gpkg's ml_dominant_muni comes from the separately "
          "trained RF model fit on the NW Huff target (Model 2) — two different "
          "models, per 12_export_outputs.py::export_agreement_maps. See "
          "outputs/audit/ml_model_design_note.md.")
    print()

    # ── Primary output layer and synthesis table (3-way) ──
    out_cols_3way = ["NA_MID", "NA_UIME", "AHP_dominant_muni", "NW_dominant_muni", "ml_dominant_muni",
                      "ag_AHP_NW", "ag_AHP_ML", "ag_NW_ML", "n_disagree_3way", "all_three_same",
                      "entropy_AHP", "entropy_class", "geometry"]
    layer_3way = gpd.GeoDataFrame(merged[out_cols_3way], geometry="geometry", crs=ahp_nw.crs)
    layer_3way = layer_3way.rename(columns={"n_disagree_3way": "n_disagree"})
    layer_3way.to_file(OUTPUT_LAYER_PATH, driver="GPKG")
    print(f"Saved {OUTPUT_LAYER_PATH} (primary, 3-way)")

    print()
    print("=== PRIMARY SYNTHESIS TABLE (3-way, n_disagree 0-3) ===")
    synth_3way = build_synthesis_table(merged, "n_disagree_3way", "entropy_AHP", "entropy_class", [0, 1, 2, 3])
    synth_3way.to_csv(SYNTHESIS_TABLE_PATH, index=False)
    print(synth_3way.to_string(index=False))
    print(f"\nSaved {SYNTHESIS_TABLE_PATH} (primary)")
    print()

    n_identical_3way = int(merged["all_three_same"].sum())
    print(f"Settlements with identical dominant centre from all three models (3-way): {n_identical_3way}")
    print()

    # ── Secondary output layer and synthesis table (4-way) ──
    out_cols_4way = ["NA_MID", "NA_UIME", "AHP_dominant_muni", "NW_dominant_muni", "ml_dominant_muni",
                      "ML_AHP_dominant_muni", "ML_NW_dominant_muni",
                      "ag_AHP_NW", "ag_AHP_ML", "ag_NW_ML", "ag_MLAHP_MLNW",
                      "n_disagree_4way", "all_four_same", "entropy_AHP", "entropy_class", "geometry"]
    layer_4way = gpd.GeoDataFrame(merged[out_cols_4way], geometry="geometry", crs=ahp_nw.crs)
    layer_4way = layer_4way.rename(columns={"n_disagree_4way": "n_disagree"})
    layer_4way.to_file(OUTPUT_LAYER_4WAY_PATH, driver="GPKG")
    print(f"Saved {OUTPUT_LAYER_4WAY_PATH} (secondary, 4-way)")

    print()
    print("=== SECONDARY SYNTHESIS TABLE (4-way, n_disagree 0-4) ===")
    synth_4way = build_synthesis_table(merged, "n_disagree_4way", "entropy_AHP", "entropy_class", [0, 1, 2, 3, 4])
    synth_4way.to_csv(SYNTHESIS_TABLE_4WAY_PATH, index=False)
    print(synth_4way.to_string(index=False))
    print(f"\nSaved {SYNTHESIS_TABLE_4WAY_PATH} (secondary)")
    print()

    n_identical_4way = int(merged["all_four_same"].sum())
    print(f"Settlements with identical dominant centre from all four models (4-way): {n_identical_4way}")
    print()

    print("=== DISAGREEMENT DESTINATION TABLES (>= 5 settlements) — all four pairwise comparisons ===")
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
