"""
Step 14 of the pipeline: how do the Random Forest models' catchment sizes
compare to the Huff models', and which feature groups drive each Random
Forest's predictions?

What this script does: two independent comparisons. First, it puts the
catchment sizes (settlement counts per municipality) from all four
models — AHP Huff, NW Huff, and the two separately-trained Random Forest
models — side by side, ranks each one, and keeps the union of every
model's top 20 municipalities (not just the AHP Huff top 20), since the
Random Forest models' size hierarchy differs enough from the Huff models'
that a plain top-20 slice from one model would hide municipalities that
rank highly under another. Second, it groups each Random Forest's 189
input features into a small number of thematic categories (distance, the
composite GI score, accessibility, individual GI indicators, and so on)
and compares how much combined importance each category carries between
the AHP-target and NW-target models.

Reads: huff_AHP_summary.csv, huff_NW_summary.csv, ml_AHP_vs_AHP_comparison.csv,
ml_NW_vs_NW_comparison.csv, ml_AHP_feature_importance.csv,
ml_NW_feature_importance.csv — all already computed by 03, 04 and 06.

Writes: table_ml_catchment_sizes.csv (plus the same content saved again as
Supplementary Table S6, tableS6_rf_catchment_sizes.csv),
table_feature_importance_comparison.csv, fig_feature_importance_comparison.png/.pdf.

Runs fourteenth. Needs 03, 04 and 06's outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import TABLES, FIGURES, SUPPLEMENTARY

CATCHMENT_TABLE_PATH = TABLES / "table_ml_catchment_sizes.csv"
FI_COMPARISON_TABLE_PATH = TABLES / "table_feature_importance_comparison.csv"
FI_COMPARISON_FIG_PATH = FIGURES / "fig_feature_importance_comparison"

OUTPUT_FILES = [
    "tables/table_ml_catchment_sizes.csv",
    "tables/table_feature_importance_comparison.csv",
    "supplementary/tableS6_rf_catchment_sizes.csv",
    "figures/fig_feature_importance_comparison.png",
    "figures/fig_feature_importance_comparison.pdf",
]

# How many top-ranked municipalities to keep, per model, before taking the
# union across all four models (see catchment_sizes below).
TOP_N = 20


def _feature_group(feature):
    """Sort one ML feature name into a thematic category for the importance comparison."""
    if feature == "dist_to_muni":
        return "Distance"
    if feature == "GI_AHP":
        return "GI_AHP"
    if feature == "GI_NW":
        return "GI_NW"
    if feature.startswith("nacc_"):
        return "Accessibility"
    if feature == "n_Area_km2":
        return "Municipality area"
    if feature.startswith("n_"):
        return "Individual GI indicators"
    return "Other"


def catchment_sizes():
    """Compare catchment sizes across all four models and save the top-20-union table."""
    print("=== RF CATCHMENT STRUCTURE ===")
    print()

    ahp_sum = pd.read_csv(TABLES / "huff_AHP_summary.csv")
    nw_sum = pd.read_csv(TABLES / "huff_NW_summary.csv")
    ml_ahp = pd.read_csv(TABLES / "ml_AHP_vs_AHP_comparison.csv")
    ml_nw = pd.read_csv(TABLES / "ml_NW_vs_NW_comparison.csv")

    ahp_sizes = ahp_sum["dominant_municipality"].value_counts()
    nw_sizes = nw_sum["dominant_municipality"].value_counts()
    rf_ahp_sizes = ml_ahp["ml_dominant_muni"].value_counts()
    rf_nw_sizes = ml_nw["ml_dominant_muni"].value_counts()

    print("RF (AHP-target model) top 10 catchments:")
    print(rf_ahp_sizes.head(10).to_string())
    print()

    all_munis = sorted(set(ahp_sizes.index) | set(nw_sizes.index)
                        | set(rf_ahp_sizes.index) | set(rf_nw_sizes.index))
    full = pd.DataFrame({"Muni_Name": all_munis}).set_index("Muni_Name")
    full["AHP_Huff_size"] = ahp_sizes
    full["NW_Huff_size"] = nw_sizes
    full["RF_AHP_target_size"] = rf_ahp_sizes
    full["RF_NW_target_size"] = rf_nw_sizes
    full = full.fillna(0).astype(int)
    for col in ["AHP_Huff_size", "NW_Huff_size", "RF_AHP_target_size", "RF_NW_target_size"]:
        full[f"rank_{col}"] = full[col].rank(ascending=False, method="min").astype(int)

    top_union = sorted(set(full["rank_AHP_Huff_size"][full["rank_AHP_Huff_size"] <= TOP_N].index)
                        | set(full["rank_NW_Huff_size"][full["rank_NW_Huff_size"] <= TOP_N].index)
                        | set(full["rank_RF_AHP_target_size"][full["rank_RF_AHP_target_size"] <= TOP_N].index))
    top_df = full.loc[top_union].sort_values("AHP_Huff_size", ascending=False).reset_index()
    top_df.to_csv(CATCHMENT_TABLE_PATH, index=False)
    print(f"Saved {CATCHMENT_TABLE_PATH} ({len(top_df)} municipalities — union of each "
          f"model's top {TOP_N} by catchment size, since the RF hierarchy differs "
          f"substantially from the Huff hierarchies and a plain AHP-top-20 slice "
          f"would hide that)")

    # Supplementary Table S6 — same DataFrame, saved a second time under the
    # manuscript-facing name, not re-read from disk. Note this is the union
    # of each model's top 20 (24 municipalities), not a table trimmed to
    # exactly 15 rows — sorting this file by RF_AHP_target_size or
    # RF_NW_target_size and taking the top 15 reproduces the manuscript's
    # stated top-15 lists for both RF models exactly.
    table_s6_path = SUPPLEMENTARY / "tableS6_rf_catchment_sizes.csv"
    SUPPLEMENTARY.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(table_s6_path, index=False)
    print(f"Saved {table_s6_path}")
    print()
    print(top_df.to_string(index=False))
    print()

    lj = full.loc["Ljubljana"]
    print(f"Ljubljana catchment size — AHP Huff: {lj['AHP_Huff_size']}, "
          f"NW Huff: {lj['NW_Huff_size']}, RF (AHP-target): {lj['RF_AHP_target_size']}, "
          f"RF (NW-target): {lj['RF_NW_target_size']}")
    print()


def feature_importance_comparison():
    """Group each Random Forest's feature importances by theme and compare the two models."""
    print("=== FEATURE IMPORTANCE COMPARISON (AHP-target vs NW-target RF) ===")
    print()

    ahp_fi = pd.read_csv(TABLES / "ml_AHP_feature_importance.csv")
    nw_fi = pd.read_csv(TABLES / "ml_NW_feature_importance.csv")

    ahp_fi["group"] = ahp_fi["feature"].apply(_feature_group)
    nw_fi["group"] = nw_fi["feature"].apply(_feature_group)

    ahp_group = ahp_fi.groupby("group")["importance"].sum()
    nw_group = nw_fi.groupby("group")["importance"].sum()
    ahp_pct = 100 * ahp_group / ahp_group.sum()
    nw_pct = 100 * nw_group / nw_group.sum()

    groups = sorted(set(ahp_pct.index) | set(nw_pct.index),
                     key=lambda g: -ahp_pct.get(g, 0))
    comparison = pd.DataFrame({
        "feature_group": groups,
        "AHP_target_pct": [ahp_pct.get(g, 0.0) for g in groups],
        "NW_target_pct": [nw_pct.get(g, 0.0) for g in groups],
    })
    comparison["difference_pct"] = comparison["AHP_target_pct"] - comparison["NW_target_pct"]
    comparison.to_csv(FI_COMPARISON_TABLE_PATH, index=False)
    print(comparison.to_string(index=False))
    print(f"\nSaved {FI_COMPARISON_TABLE_PATH}")
    print()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(groups))
    width = 0.38
    ax.bar(x - width / 2, comparison["AHP_target_pct"], width, label="AHP-target RF", color="#3b6fa0")
    ax.bar(x + width / 2, comparison["NW_target_pct"], width, label="NW-target RF", color="#c0724a")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Feature-group importance (%)")
    ax.set_title("Feature importance by group: AHP-target vs NW-target Random Forest")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FI_COMPARISON_FIG_PATH}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FI_COMPARISON_FIG_PATH}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FI_COMPARISON_FIG_PATH}.png / .pdf")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    catchment_sizes()
    feature_importance_comparison()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
