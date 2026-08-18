"""
Task 5: a clean manifest of every file in outputs/ — which script produces
it, whether it is referenced in the manuscript draft, and a one-line
description — plus a check for orphaned files (present but produced by no
current script) and expected-but-missing files (a script writes them but
they are not currently on disk).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import OUTPUTS

AUDIT_DIR = OUTPUTS / "audit"
MANIFEST_PATH = AUDIT_DIR / "output_manifest.csv"

# relative-to-outputs path -> (type, producing script, referenced_in_draft, description)
# referenced_in_draft=True marks figures/tables the brief's manuscript draft
# already names (fig01/03-08/10, table1-6, tableS1-4, the map_* comparison
# and LISA layers). New Task 2-4 deliverables the brief requests but the
# draft itself never mentions are marked False.
MANIFEST = {
    # ── audit ──────────────────────────────────────────────
    "audit/data_audit_report.md": ("audit", "13_data_audit.py", False,
        "Full data audit: raw inputs, indicators, GI, roads, OD matrix, accessibility, ML, manuscript number check"),
    "audit/raw_input_inventory.csv": ("audit", "13_data_audit.py", False,
        "Inventory (size/CRS/features/columns) of the 8 core DATA_RAW files"),
    "audit/indicator_audit.csv": ("audit", "13_data_audit.py", False,
        "Per-indicator audit: group, nonzero counts, rarity score, weights"),
    "audit/GI_full_212_municipalities.csv": ("audit", "13_data_audit.py", False,
        "GI_NW and GI_AHP for all 212 municipalities with ranks and 10 group scores each"),
    "audit/manuscript_number_check.csv": ("audit", "13_data_audit.py", False,
        "Every manuscript-draft numeric claim vs the repository-computed value"),
    "audit/output_manifest.csv": ("audit", "19_output_manifest.py", False,
        "This manifest"),

    # ── figures ────────────────────────────────────────────
    "figures/fig07_beta_sensitivity.png": ("figure", "ORPHAN — no current script", False,
        "Legacy beta-sensitivity figure; superseded by fig_beta_sensitivity.png/.pdf (07_beta_sensitivity.py)"),
    "figures/fig07_feature_importance.png": ("figure", "12_export_outputs.py", True,
        "Top-30 AHP feature importance bar chart, coloured by thematic group"),
    "figures/fig07_feature_importance.pdf": ("figure", "12_export_outputs.py", True,
        "Top-30 AHP feature importance bar chart (PDF)"),
    "figures/fig07_feature_importance_AHP.png": ("figure", "ORPHAN — no current script", False,
        "Legacy feature-importance figure; no current script writes this filename"),
    "figures/fig08_shap_bar_AHP.png": ("figure", "06_ml_framework.py (run_shap, unreachable — see note)", True,
        "SHAP bar plot, AHP model — produced by a dead-code function (run_shap) never called from main()"),
    "figures/fig08_shap_summary_AHP.png": ("figure", "06_ml_framework.py (run_shap, unreachable — see note)", True,
        "SHAP summary/beeswarm plot, AHP model — same dead-code caveat as fig08_shap_bar_AHP.png"),
    "figures/fig_beta_sensitivity.pdf": ("figure", "07_beta_sensitivity.py", True,
        "Agreement and Cohen's kappa vs beta, current beta-sensitivity figure (PDF)"),
    "figures/fig_beta_sensitivity.png": ("figure", "07_beta_sensitivity.py", True,
        "Agreement and Cohen's kappa vs beta, current beta-sensitivity figure"),
    "figures/fig_feature_importance_comparison.pdf": ("figure", "15_ml_catchment_structure.py", False,
        "Paired bar chart: AHP-target vs NW-target RF feature-group importance (Task 2.7, PDF)"),
    "figures/fig_feature_importance_comparison.png": ("figure", "15_ml_catchment_structure.py", False,
        "Paired bar chart: AHP-target vs NW-target RF feature-group importance (Task 2.7)"),
    "figures/fig_shap_dependence_AHP.pdf": ("figure", "16_shap_dependence.py", False,
        "Multi-panel SHAP dependence plot, top 5 features by mean |SHAP|, AHP model, 5,000-pair sample seed=42 (Task 2.6)"),
    "figures/fig_shap_dependence_AHP.png": ("figure", "16_shap_dependence.py", False,
        "Multi-panel SHAP dependence plot, top 5 features by mean |SHAP|, AHP model, 5,000-pair sample seed=42 (Task 2.6)"),
    "figures/fig_shap_dependence_AHP_dist_to_muni.pdf": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: dist_to_muni (rank 1, mean |SHAP|=0.005527)"),
    "figures/fig_shap_dependence_AHP_dist_to_muni.png": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: dist_to_muni (rank 1, mean |SHAP|=0.005527)"),
    "figures/fig_shap_dependence_AHP_GI_AHP.pdf": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: GI_AHP (rank 2, mean |SHAP|=0.003548)"),
    "figures/fig_shap_dependence_AHP_GI_AHP.png": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: GI_AHP (rank 2, mean |SHAP|=0.003548)"),
    "figures/fig_shap_dependence_AHP_nacc_Recycle_Bins.pdf": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: nacc_Recycle_Bins (rank 3, mean |SHAP|=0.000247)"),
    "figures/fig_shap_dependence_AHP_nacc_Recycle_Bins.png": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: nacc_Recycle_Bins (rank 3, mean |SHAP|=0.000247)"),
    "figures/fig_shap_dependence_AHP_n_NumberDr.pdf": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: n_NumberDr (rank 4, mean |SHAP|=0.000102)"),
    "figures/fig_shap_dependence_AHP_n_NumberDr.png": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: n_NumberDr (rank 4, mean |SHAP|=0.000102)"),
    "figures/fig_shap_dependence_AHP_nacc_Theaters.pdf": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: nacc_Theaters (rank 5, mean |SHAP|=0.000096)"),
    "figures/fig_shap_dependence_AHP_nacc_Theaters.png": ("figure", "16_shap_dependence.py", False,
        "Individual SHAP dependence panel: nacc_Theaters (rank 5, mean |SHAP|=0.000096)"),

    # ── gpkg ───────────────────────────────────────────────
    "gpkg/fig01_study_area.gpkg": ("gpkg", "17_spatial_layers.py", True,
        "Study area: municipalities (212 polygons) + settlements (6,036 points) layers"),
    "gpkg/fig03_GI_NW_municipalities.gpkg": ("gpkg", "17_spatial_layers.py", True,
        "GI_Final_NotWeighted choropleth layer: 212 municipality polygons, GI value, rank, 10 group scores"),
    "gpkg/fig04_GI_AHP_municipalities.gpkg": ("gpkg", "17_spatial_layers.py", True,
        "GI_AHP choropleth layer: 212 municipality polygons, GI value, rank, 10 group scores"),
    "gpkg/fig05_catchments_AHP.gpkg": ("gpkg", "17_spatial_layers.py", True,
        "AHP Huff catchments: settlements + dissolved-by-municipality layers, catchment_size field"),
    "gpkg/fig06_catchments_NW.gpkg": ("gpkg", "17_spatial_layers.py", True,
        "NW Huff catchments: settlements + dissolved-by-municipality layers, catchment_size field"),
    "gpkg/fig10_catchments_ML.gpkg": ("gpkg", "17_spatial_layers.py", True,
        "RF (AHP-target) catchments: settlements + dissolved-by-municipality layers, catchment_size field"),
    "gpkg/fig_huff_vs_commuting_municipalities.gpkg": ("gpkg", "11_commuting_comparison.py", True,
        "212 municipalities: Huff vs 2023 SURS commuting functional-centre comparison, pattern field"),
    "gpkg/map_AHP_vs_ML_villages.gpkg": ("gpkg", "12_export_outputs.py", True,
        "Map B source layer: AHP Huff vs RF, agreement + map_class + is_ljubljana_source fields (Task 4)"),
    "gpkg/map_AHP_vs_NW_villages.gpkg": ("gpkg", "12_export_outputs.py", True,
        "Map A source layer: AHP Huff vs NW Huff, agreement + map_class + is_ljubljana_source fields (Task 4)"),
    "gpkg/map_NW_vs_ML_villages.gpkg": ("gpkg", "12_export_outputs.py", True,
        "Map C source layer: NW Huff vs RF, agreement + map_class + is_ljubljana_source fields (Task 4)"),
    "gpkg/map_disagreement_count_villages.gpkg": ("gpkg", "14_disagreement_synthesis.py", False,
        "Map D source layer: n_disagree (0-3) across all three pairwise comparisons, entropy fields"),
    "gpkg/map_entropy_AHP_villages.gpkg": ("gpkg", "09_entropy_uncertainty.py", True,
        "Per-settlement normalised Shannon entropy of the AHP Huff Pij distribution, entropy_class"),
    "gpkg/map_entropy_NW_villages.gpkg": ("gpkg", "09_entropy_uncertainty.py", True,
        "Per-settlement normalised Shannon entropy of the NW Huff Pij distribution, entropy_class"),
    "gpkg/map_euclidean_vs_network_villages.gpkg": ("gpkg", "08_euclidean_comparison.py", True,
        "Per-settlement Euclidean-distance vs road-network-distance Huff dominant-assignment agreement"),
    "gpkg/map_lisa_AHP_vs_ML.gpkg": ("gpkg", "10_morans_i.py", False,
        "Local Moran's I (LISA) cluster_type for AHP-vs-ML agreement (Task 2.2)"),
    "gpkg/map_lisa_AHP_vs_NW.gpkg": ("gpkg", "10_morans_i.py", True,
        "Local Moran's I (LISA) cluster_type for AHP-vs-NW agreement"),
    "gpkg/map_lisa_NW_vs_ML.gpkg": ("gpkg", "10_morans_i.py", False,
        "Local Moran's I (LISA) cluster_type for NW-vs-ML agreement (Task 2.2)"),

    # ── supplementary ──────────────────────────────────────
    "supplementary/tableS1_indicators_sources.csv": ("supplementary", "12_export_outputs.py (consolidate_tables)", True,
        "Source (OSM/GURS/SURS/ZZZS) of each of the 100 GI indicators"),
    "supplementary/tableS2_AHP_priority_weights.csv": ("supplementary", "pre-existing (AHP pairwise matrix, not regenerated by any script)", True,
        "10x10 AHP pairwise comparison matrix and derived group priority weights, plus CR/lambda_max/CI/RI"),
    "supplementary/tableS3_individual_indicator_weights.csv": ("supplementary", "pre-existing (not regenerated by any script)", True,
        "Per-indicator group, nonzero count, rarity score, and within-group weight reference table"),
    "supplementary/tableS4_beta_sensitivity.csv": ("supplementary", "12_export_outputs.py (consolidate_tables)", True,
        "Copy of the beta-sensitivity results table"),

    # ── tables ─────────────────────────────────────────────
    "tables/accessibility_normalized.csv": ("table", "05_accessibility.py", True,
        "212 municipalities x 86 inverted min-max normalised accessibility indicators"),
    "tables/huff_AHP_summary.csv": ("table", "03_huff_ahp.py", True,
        "Per-settlement AHP Huff dominant municipality, Pij, distance"),
    "tables/huff_NW_summary.csv": ("table", "04_huff_nonweighted.py", True,
        "Per-settlement NW Huff dominant municipality, Pij, distance"),
    "tables/ml_AHP_cv_results.csv": ("table", "06_ml_framework.py --model AHP", True,
        "5-fold spatial CV R2/MAE/RMSE, AHP-target Random Forest"),
    "tables/ml_AHP_feature_importance.csv": ("table", "06_ml_framework.py --model AHP", True,
        "189-feature importance ranking, AHP-target Random Forest"),
    "tables/ml_AHP_feature_importance_original.csv": ("table", "ORPHAN — no current script", False,
        "Legacy feature-importance file; no current script writes this filename"),
    "tables/ml_AHP_vs_AHP_comparison.csv": ("table", "06_ml_framework.py --model AHP", True,
        "Per-settlement AHP-target RF dominant municipality vs AHP Huff, agreement flag"),
    "tables/ml_AHP_vs_AHP_comparison_original.csv": ("table", "ORPHAN — no current script", False,
        "Legacy comparison file; no current script writes this filename"),
    "tables/ml_NW_cv_results.csv": ("table", "06_ml_framework.py --model NW", True,
        "5-fold spatial CV R2/MAE/RMSE, NW-target Random Forest"),
    "tables/ml_NW_feature_importance.csv": ("table", "06_ml_framework.py --model NW", True,
        "189-feature importance ranking, NW-target Random Forest"),
    "tables/ml_NW_vs_NW_comparison.csv": ("table", "06_ml_framework.py --model NW", True,
        "Per-settlement NW-target RF dominant municipality vs NW Huff, agreement flag"),
    "tables/table1_AHP_group_weights.csv": ("table", "12_export_outputs.py (consolidate_tables)", True,
        "Paper Table 1: AHP group priority weights and indicator counts per group"),
    "tables/table2_top20_GI_NotWeighted.csv": ("table", "12_export_outputs.py (consolidate_tables)", True,
        "Paper Table 2: top 20 municipalities by GI_Final_NotWeighted"),
    "tables/table3_top20_GI_AHP.csv": ("table", "12_export_outputs.py (consolidate_tables)", True,
        "Paper Table 3: top 20 municipalities by GI_AHP"),
    "tables/table4_top15_catchments.csv": ("table", "12_export_outputs.py (consolidate_tables)", True,
        "Paper Table 4: top 15 catchments, AHP vs NW"),
    "tables/table5_beta_sensitivity.csv": ("table", "12_export_outputs.py (consolidate_tables)", True,
        "Paper Table 5: beta sensitivity agreement/kappa by beta"),
    "tables/table6_cv_performance.csv": ("table", "12_export_outputs.py (consolidate_tables, copy of ml_AHP_cv_results.csv)", True,
        "Paper Table 6: AHP Random Forest cross-validation performance"),
    "tables/table_beta_sensitivity.csv": ("table", "ORPHAN — no current script", False,
        "Legacy beta-sensitivity table; superseded by table_beta_sensitivity_clean.csv (07_beta_sensitivity.py)"),
    "tables/table_beta_sensitivity_clean.csv": ("table", "07_beta_sensitivity.py", True,
        "Current beta-sensitivity agreement/kappa table, beta = 1.5/2.0/2.5/3.0"),
    "tables/table_disagreement_destinations.csv": ("table", "14_disagreement_synthesis.py", False,
        "Per-comparison destination breakdown for disagreeing settlements, >=5 settlements per pair (Task 2.4)"),
    "tables/table_disagreement_synthesis.csv": ("table", "14_disagreement_synthesis.py", False,
        "Settlement counts and mean entropy by n_disagree (0-3) across all three comparisons (Task 2.3)"),
    "tables/table_entropy_summary.csv": ("table", "09_entropy_uncertainty.py", True,
        "Entropy summary statistics (min/max/mean/std, low/medium/high counts), AHP and NW"),
    "tables/table_euclidean_vs_network.csv": ("table", "08_euclidean_comparison.py", True,
        "Per-settlement Euclidean vs network-distance Huff dominant-assignment comparison"),
    "tables/table_feature_importance_comparison.csv": ("table", "15_ml_catchment_structure.py", False,
        "AHP-target vs NW-target RF feature-group importance comparison (Task 2.7)"),
    "tables/table_huff_vs_commuting.csv": ("table", "11_commuting_comparison.py", True,
        "Per-municipality Huff vs 2023 SURS commuting functional-centre comparison, pattern classification"),
    "tables/table_huff_vs_commuting_summary.csv": ("table", "11_commuting_comparison.py", True,
        "Summary: agreement/kappa and pattern counts, Huff vs commuting"),
    "tables/table_join_counts.csv": ("table", "10_morans_i.py", False,
        "Binary join-count statistic (BB/WW/BW, chi2) — robustness check for Moran's I (Task 2.2)"),
    "tables/table_lisa_summary.csv": ("table", "10_morans_i.py", False,
        "HH/LL/HL/LH/not-significant counts, LISA, all three comparisons (Task 2.2)"),
    "tables/table_ml_catchment_sizes.csv": ("table", "15_ml_catchment_structure.py", False,
        "Top-20-union catchment size and rank under AHP Huff, NW Huff, and RF (Task 2.5)"),
    "tables/table_morans_i_results.csv": ("table", "10_morans_i.py", True,
        "Global Moran's I, all three comparisons"),
    "tables/table_three_way_agreement.csv": ("table", "10_morans_i.py", False,
        "Settlement-level Cohen's kappa (212-class) + Moran's I, all three comparisons (Task 2.1)"),
}


def main():
    print("=== OUTPUT MANIFEST (Task 5) ===")
    print()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    present = sorted(
        p.relative_to(OUTPUTS).as_posix()
        for p in OUTPUTS.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    )
    print(f"Files present in outputs/: {len(present)}")

    rows = []
    unmapped = []
    for rel in present:
        if rel in MANIFEST:
            type_, script, referenced, desc = MANIFEST[rel]
        else:
            type_, script, referenced, desc = ("unknown", "UNMAPPED — update this manifest script", False, "")
            unmapped.append(rel)
        rows.append({
            "path": f"outputs/{rel}", "type": type_, "producing_script": script,
            "referenced_in_draft": referenced, "description": desc,
        })

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Saved {MANIFEST_PATH} ({len(df)} rows)")
    print()

    if unmapped:
        print(f"WARNING: {len(unmapped)} files present in outputs/ are not yet described "
              f"in this script's MANIFEST dict: {unmapped}")
        print()

    orphans = df[df["producing_script"].str.startswith("ORPHAN")]
    print(f"### Orphaned files (present, but no current script produces them): {len(orphans)}")
    print(orphans[["path", "description"]].to_string(index=False))
    print()

    dead_code = df[df["producing_script"].str.contains("unreachable", na=False)]
    print(f"### Present via dead code (function exists but is never called from main()): {len(dead_code)}")
    print(dead_code[["path", "description"]].to_string(index=False))
    print()

    expected_but_missing = [
        ("outputs/tables/table_GI_summary_stats.csv", "01_gi_construction.py",
         "01 writes this on every run; absent here, so 01 has not been (re)run in this checkout"),
        ("outputs/tables/table_top20_GI_both.csv", "01_gi_construction.py",
         "01 writes this on every run; absent here, so 01 has not been (re)run in this checkout"),
        ("outputs/tables/accessibility_raw_distances.csv", "05_accessibility.py",
         "05 only writes this in its full-computation branch, which is skipped because "
         "accessibility_normalized.csv already exists"),
    ]
    print(f"### Files a script writes that are not currently in outputs/: {len(expected_but_missing)}")
    for path, script, note in expected_but_missing:
        print(f"  - {path}  (script: {script}) — {note}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
