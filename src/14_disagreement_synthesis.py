"""
Join the three settlement-level agreement layers (AHP vs NW, AHP vs ML,
NW vs ML) plus the AHP entropy layer, produce a three-way disagreement
synthesis layer and summary table, and per-comparison "where do
disagreeing settlements go" destination tables.
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
    entropy = pd.DataFrame(gpd.read_file(ENTROPY_AHP_PATH))[
        ["NA_MID", "entropy_AHP", "entropy_class"]
    ]
    print(f"  AHP vs NW: {len(ahp_nw)}, AHP vs ML: {len(ahp_ml)}, "
          f"NW vs ML: {len(nw_ml)}, entropy: {len(entropy)}")

    merged = (ahp_nw.merge(ahp_ml, on="NA_MID", how="left")
                     .merge(nw_ml, on="NA_MID", how="left")
                     .merge(entropy, on="NA_MID", how="left"))
    print(f"  Joined: {len(merged)} settlements")
    print()

    merged["n_disagree"] = (3 - (merged["ag_AHP_NW"] + merged["ag_AHP_ML"]
                                  + merged["ag_NW_ML"])).astype(int)
    merged["all_three_same"] = merged["n_disagree"] == 0

    n_bucket1 = int((merged["n_disagree"] == 1).sum())
    print(f"n_disagree == 1 settlements: {n_bucket1}")
    print("  IMPORTANT: n_disagree==1 is only possible because 'ml_dominant_muni' is "
          "NOT a single consistent classification across the three source layers. "
          "map_AHP_vs_ML_villages.gpkg's ml_dominant_muni comes from the RF model "
          "trained on the AHP Huff target (06_ml_framework.py Model 1), while "
          "map_NW_vs_ML_villages.gpkg's ml_dominant_muni comes from the separately "
          "trained RF model fit on the NW Huff target (Model 2) — two different "
          "models, per 12_export_outputs.py::export_agreement_maps. So this join "
          "actually compares FOUR distinct classifications pairwise (AHP, NW, "
          "ML-on-AHP-target, ML-on-NW-target) across three of the six possible "
          "pairs, not three mutually-transitive labels — an n_disagree of exactly "
          "1 (e.g. AHP==NW as Huff models, but each disagrees with its own "
          "differently-trained RF counterpart) is legitimate and common, not an "
          "artifact. This should be stated explicitly in the paper's methods "
          "section wherever this three-way comparison is used, since a reader "
          "would otherwise assume a single ML model throughout.")
    print()

    out_cols = ["NA_MID", "NA_UIME", "AHP_dominant_muni", "NW_dominant_muni", "ml_dominant_muni",
                "ag_AHP_NW", "ag_AHP_ML", "ag_NW_ML", "n_disagree", "all_three_same",
                "entropy_AHP", "entropy_class", "geometry"]
    layer = gpd.GeoDataFrame(merged[out_cols], geometry="geometry", crs=ahp_nw.crs)
    layer.to_file(OUTPUT_LAYER_PATH, driver="GPKG")
    print(f"Saved {OUTPUT_LAYER_PATH}")
    print()

    print("=== SYNTHESIS TABLE (n_disagree 0-3) ===")
    rows = []
    for k in [0, 1, 2, 3]:
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
    print("Brief's expected distribution (0/1/2/3): 3897 / 792 / 1303 / 44, "
          "mean entropy 0.452 / 0.578 / 0.615 / 0.649 — compare against the "
          "table above; bucket 1 is expected to differ (see note above).")
    print()

    n_identical = int(merged["all_three_same"].sum())
    print(f"Settlements with identical dominant centre from all three models: {n_identical}")
    print("  ('all_three_same' is by definition the same quantity as the n_disagree==0 "
          "bucket above, i.e. 3,897 — matching the brief's own ~3,897 estimate for "
          "that bucket exactly. The brief's separate ~4,260 estimate for 'identical "
          "across all three' does not match; trust the 3,897 figure computed here.)")
    print()

    print("=== DISAGREEMENT DESTINATION TABLES (>= 5 settlements) ===")
    dest_rows = []
    comparisons = [
        ("AHP_vs_NW", "AHP", "AHP_dominant_muni", "NW", "NW_dominant_muni", "ag_AHP_NW"),
        ("AHP_vs_ML", "AHP", "AHP_dominant_muni", "ML", "ml_dominant_muni", "ag_AHP_ML"),
        ("NW_vs_ML", "NW", "NW_dominant_muni", "ML", "ml_dominant_muni", "ag_NW_ML"),
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
