"""
Step 1 of the pipeline: build the indicator rarity weights (tableS3) and
check the Gravitational Index (GI) inputs.

What this script does: the two GI layers this study uses — GI_AHP (AHP group
weighting) and GI_Final_NotWeighted (no group weighting) — arrive already
computed in the raw data, one number per municipality per scenario. This
script does not calculate the GI itself. What it does is (1) compute the
"rarity weight" of each of the 100 underlying indicators from first
principles and publish that computation as tableS3, and (2) load both GI
layers and report summary statistics and the top-20 municipality rankings
under each.

tableS3 used to be a hand-maintained file: someone ran this same
computation once, pasted the result into outputs/supplementary/, and nothing
ever regenerated it afterward. That is exactly the failure mode that let
tableS1 silently revert to a pre-correction version (see
docs/reproducibility_note.md) — a value living in outputs/ with no script
that owns it can drift from whatever actually produced it. tableS3 is now a
genuine pipeline output instead: this script reads only the part that
cannot be computed (which thematic group each indicator belongs to, and its
display name — a categorisation judgement, not a derived number, kept in
data/external/indicator_categories.csv) and computes every numeric column
itself, fresh, every run.

Reads: Municipalities_Points_normalized.gpkg (the 100 raw indicator values
per municipality), Municipalities_All_Groups_Weighted_AHP.gpkg (GI_AHP),
Municipalities_All_Groups_NotWeighted_Normalized.gpkg (GI_Final_NotWeighted),
data/external/indicator_categories.csv (the indicator name/group
categorisation — a genuine input, not something this script derives).

Writes: table_GI_summary_stats.csv, table_top20_GI_both.csv,
tableS3_individual_indicator_weights.csv.

Runs first in the pipeline; every later script that uses GI_AHP or
GI_Final_NotWeighted reads them straight from the raw municipality layers,
not from anything this script produces. Scripts that read tableS3
(11_export_outputs.py, 17_data_audit.py) rely on this script having run
first to produce it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd

from config import (
    DATA_RAW, TABLES, SUPPLEMENTARY, REPO_ROOT, N_MUNICIPALITIES, EPSG,
    MUNICIPALITIES_AHP, MUNICIPALITIES_NW, MUNICIPALITIES_PTS,
)
from crs_utils import ensure_crs

DATA_EXTERNAL = REPO_ROOT / "data" / "external"

OUTPUT_FILES = [
    "tables/table_GI_summary_stats.csv",
    "tables/table_top20_GI_both.csv",
    "supplementary/tableS3_individual_indicator_weights.csv",
]


def load_indicator_categories(data_external_path):
    """Load the indicator name/thematic-group categorisation.

    This is the one part of tableS3 that genuinely cannot be computed —
    which of the 100 indicators are grouped together, and what each one is
    called, is a categorisation judgement, not a number derived from data.
    Kept as a small, explicitly non-regenerable input in data/external/
    (the same treatment as tableS1's source citations and table1's AHP
    matrix), separate from the rarity weights this script computes below.
    """
    ref = pd.read_csv(data_external_path / "indicator_categories.csv")
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
            "Nonzero_pct": round(100 * nonzero / n_municipalities, 1),
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
    SUPPLEMENTARY.mkdir(parents=True, exist_ok=True)

    print("Loading raw indicators (Municipalities_Points_normalized.gpkg)...")
    pts = gpd.read_file(DATA_RAW / MUNICIPALITIES_PTS)
    pts = ensure_crs(pts, EPSG, label=MUNICIPALITIES_PTS)
    n_cols = [c for c in pts.columns if c.startswith("n_")]
    print(f"  Shape: {pts.shape}  ({len(n_cols)} n_-prefixed columns)")

    print("Loading indicator name/group categorisation (data/external/indicator_categories.csv)...")
    group_map, ref = load_indicator_categories(DATA_EXTERNAL)
    print(f"  Mapped indicators: {len(group_map)}")
    excluded = sorted(set(n_cols) - set(group_map.keys()))
    print(f"  n_ columns not in mapping (expected: n_Area_km2, n_Fitness_C): {excluded}")
    print()

    print("Computing rarity weights within each thematic group...")
    weights = compute_rarity_weights(pts, group_map, N_MUNICIPALITIES)
    print(f"  Computed weights for {len(weights)} indicators across "
          f"{weights['Thematic_group'].nunique()} groups")

    # Every thematic group's weights must sum to exactly 1 by construction
    # (wi is defined as sqrt_ri divided by its own group's sum) — this is a
    # self-consistency check on the computation itself, not a comparison
    # against an external reference, since tableS3 is this script's own
    # output rather than something to validate against.
    group_sums = weights.groupby("Thematic_group")["wi"].sum()
    max_sum_error = (group_sums - 1.0).abs().max()
    print(f"  Max |group weight sum - 1.0|: {max_sum_error:.2e} "
          f"({'OK' if max_sum_error < 1e-9 else 'UNEXPECTED — investigate'})")
    print()
    print(weights.sort_values(["Thematic_group", "wi"], ascending=[True, False]).to_string(index=False))
    print()

    print("Building tableS3 (indicator name, group, and computed rarity weights)...")
    table_s3 = weights.merge(ref[["Indicator_code", "Indicator_name"]], on="Indicator_code", how="left")
    table_s3 = table_s3.rename(columns={
        "ri": "Rarity_score_ri", "sqrt_ri": "Sqrt_ri", "wi": "Normalised within-group weight (w_i)",
    })
    table_s3 = table_s3[["Indicator_code", "Indicator_name", "Thematic_group", "Nonzero_munis",
                          "Nonzero_pct", "Rarity_score_ri", "Sqrt_ri",
                          "Normalised within-group weight (w_i)"]]
    table_s3.to_csv(SUPPLEMENTARY / "tableS3_individual_indicator_weights.csv", index=False)
    print(f"Saved tableS3_individual_indicator_weights.csv ({len(table_s3)} rows)")
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
