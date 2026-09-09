"""
Build 1.28M row ML input table, spatial 5-fold CV, Random Forest
189 features, SHAP values, save feature importance and predictions.
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings("ignore")

from config import (
    DATA_RAW, DATA_PROCESSED, TABLES, FIGURES, GPKG,
    MUNICIPALITIES_AHP, MUNICIPALITIES_NW, MUNICIPALITIES_PTS,
    N_MUNICIPALITIES, EPSG,
)
from crs_utils import ensure_crs


def build_municipality_features(munis_pts_path, acc_path, composite_path,
                                 composite_col, output_col):
    """Load and join GI indicators, accessibility, and a model-specific
    composite GI score.

    composite_col is the column read from composite_path (e.g. "GI_AHP" from
    MUNICIPALITIES_AHP, or "GI_Final_NotWeighted" from MUNICIPALITIES_NW);
    output_col is the name it's given in the returned feature table, so each
    model's own composite score is distinguishable downstream (feature
    importance CSVs, the AHP-vs-NW comparison figure) instead of both models
    training on a column literally named "GI_AHP". See
    outputs/audit/ml_model_design_note.md for why this must differ per model.
    """
    munis_pts = gpd.read_file(munis_pts_path)
    munis_pts = ensure_crs(munis_pts, EPSG, label=munis_pts_path.name)
    composite = gpd.read_file(composite_path)
    composite = ensure_crs(composite, EPSG, label=composite_path.name)
    acc = pd.read_csv(acc_path)

    # GI individual indicators — exclude duplicate n_Fitness_C
    gi_cols = [c for c in munis_pts.columns
               if c not in ["Muni_ID", "Muni_Name", "geometry", "n_Fitness_C"]]
    nacc_cols = [c for c in acc.columns if c.startswith("nacc_")]

    munis_features = pd.DataFrame(munis_pts[["Muni_ID", "Muni_Name"] + gi_cols])
    munis_features = munis_features.merge(acc[["Muni_ID"] + nacc_cols], on="Muni_ID", how="left")
    composite_slim = composite[["Muni_ID", composite_col]].rename(
        columns={composite_col: output_col})
    munis_features = munis_features.merge(composite_slim, on="Muni_ID", how="left")
    return munis_features


def build_spatial_blocks(munis_pts_path, cache_path=None, n_blocks=5, random_state=42):
    """Spatial blocks on municipality centroids via KMeans (scipy backend —
    sklearn's KMeans on this machine hits a broken MKL threadpoolctl check,
    see _check_mkl_vcomp, and crashes the process). Cached to cache_path so
    the clustering only has to run once.
    """
    if cache_path is not None and cache_path.exists():
        print(f"  Loading spatial blocks from cache: {cache_path}")
        return pd.read_csv(cache_path)

    munis_pts = gpd.read_file(munis_pts_path)
    munis_pts = ensure_crs(munis_pts, EPSG, label=munis_pts_path.name)
    coords = np.column_stack([munis_pts.geometry.x, munis_pts.geometry.y]).astype(np.float64)
    _, blocks = kmeans2(coords, k=n_blocks, seed=random_state, minit="++")
    result = pd.DataFrame({
        "Muni_ID": munis_pts["Muni_ID"].values,
        "spatial_block": blocks,
    })

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        print(f"  Cached spatial blocks: {cache_path}")

    return result


def melt_od_matrix(od_path):
    """Melt wide OD matrix to long village-municipality pairs."""
    huff_od = pd.read_csv(od_path)
    pij_cols = [c for c in huff_od.columns if c.startswith("Pij_")]
    dist_cols = [c for c in huff_od.columns if c.startswith("dist_")]

    df_pij = huff_od[["Village_ID", "Village_Name"] + pij_cols].melt(
        id_vars=["Village_ID", "Village_Name"], var_name="Muni_col", value_name="Pij")
    df_pij["Muni_Name"] = df_pij["Muni_col"].str.replace("Pij_", "")
    df_pij = df_pij.drop(columns="Muni_col")

    df_dist = huff_od[["Village_ID"] + dist_cols].melt(
        id_vars=["Village_ID"], var_name="Muni_col", value_name="dist_to_muni")
    df_dist["Muni_Name"] = df_dist["Muni_col"].str.replace("dist_", "")
    df_dist = df_dist.drop(columns="Muni_col")

    return df_pij.merge(df_dist, on=["Village_ID", "Muni_Name"], how="left")


def train_rf_spatial_cv(df_ml_cv, feature_cols, target_col="Pij", sample_frac=1.0):
    """Train Random Forest with spatial 5-fold cross-validation."""
    fold_results = []
    all_preds = {}
    feature_importances = np.zeros(len(feature_cols))

    for fold in range(5):
        t_start = time.time()
        test_df = df_ml_cv[df_ml_cv["spatial_block"] == fold]
        train_df = df_ml_cv[df_ml_cv["spatial_block"] != fold]

        if sample_frac < 1.0:
            train_df = train_df.sample(frac=sample_frac, random_state=42)

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[target_col].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[target_col].values.astype(np.float32)

        print(f"  Fold {fold+1}: train={len(X_train):,}  test={len(X_test):,}")

        rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                                    min_samples_leaf=10, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        for idx, pred in zip(test_df.index, y_pred):
            all_preds[idx] = pred

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        elapsed = time.time() - t_start

        fold_results.append({"fold": fold + 1, "r2": r2, "mae": mae,
                              "rmse": rmse, "n_test": len(X_test), "wall_time_s": elapsed})
        feature_importances += rf.feature_importances_

        print(f"    R²={r2:.4f}  MAE={mae:.6f}  RMSE={rmse:.6f}  time={elapsed:.1f}s")

        del rf, X_train, y_train, X_test, y_test
        gc.collect()

    feature_importances /= 5
    return pd.DataFrame(fold_results), all_preds, feature_importances


def annotate_wall_time_anomalies(df_results, factor=3.0):
    """Flag any fold whose wall_time_s is far outside the others' range as a
    likely wall-clock artifact (e.g. the machine sleeping mid-fold) rather
    than real compute time, so a reader of the CSV doesn't take a 65,000s
    fold at face value. Does not touch r2/mae/rmse — those are unaffected by
    how long the process was suspended.
    """
    df_results = df_results.copy()
    times = df_results["wall_time_s"]
    median_others = {
        i: times.drop(i).median() for i in df_results.index
    }
    df_results["wall_time_note"] = ""
    for i in df_results.index:
        med = median_others[i]
        if med > 0 and times[i] > factor * med:
            df_results.loc[i, "wall_time_note"] = (
                f"wall-clock artifact, not compute time (other folds median {med:.0f}s) "
                f"— almost certainly the machine sleeping/idling mid-fold, not slower training"
            )
    return df_results


def build_comparison(df_ml_cv, all_preds, huff_summary_path,
                      muni_col="dominant_municipality"):
    """Find dominant ML municipality per village and compare with Huff."""
    df_ml_cv = df_ml_cv.copy()
    df_ml_cv["Pij_predicted"] = df_ml_cv.index.map(all_preds)

    ml_dominant = df_ml_cv.loc[
        df_ml_cv.groupby("Village_ID")["Pij_predicted"].idxmax()
    ][["Village_ID", "Village_Name", "Muni_Name", "Pij_predicted"]].copy()
    ml_dominant.columns = ["Village_ID", "Village_Name",
                            "ml_dominant_muni", "ml_dominant_Pij"]

    huff_sum = pd.read_csv(huff_summary_path).rename(
        columns={muni_col: "huff_dominant_muni",
                 "dominant_Pij": "huff_dominant_Pij"})

    comparison = ml_dominant.merge(
        huff_sum[["Village_ID", "huff_dominant_muni", "huff_dominant_Pij"]],
        on="Village_ID", how="left")
    comparison["agreement"] = (
        comparison["ml_dominant_muni"] == comparison["huff_dominant_muni"]).astype(int)
    comparison["agreement_label"] = comparison["agreement"].map(
        {1: "Huff and ML agree", 0: "Huff and ML disagree"})
    return comparison


def run_shap(rf_model, X_sample, feature_cols, figures_path, prefix="AHP"):
    """Compute SHAP values on sample and save plots."""
    import shap

    print(f"  Computing SHAP values ({len(X_sample)} samples)...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                       max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(figures_path / f"fig_shap_summary_{prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                       max_display=20, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(figures_path / f"fig_shap_bar_{prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return shap_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["AHP", "NW", "both"], default="both")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()

    print("=== ML FRAMEWORK ===")
    print()

    # Paths — all repository-relative. These used to point outside the repo
    # at "Matrix and tables"; see crs_utils / Stage 2A audit trail for why
    # that was a problem (no working data availability statement is possible
    # while a script depends on a path only one machine has).
    munis_pts_path = DATA_RAW / MUNICIPALITIES_PTS
    munis_ahp_path = DATA_RAW / MUNICIPALITIES_AHP
    munis_nw_path = DATA_RAW / MUNICIPALITIES_NW
    acc_path = TABLES / "accessibility_normalized.csv"
    ahp_od_path = TABLES / "huff_od_matrix.csv"
    nw_od_path = TABLES / "huff_NW_od_matrix.csv"
    ahp_sum_path = TABLES / "huff_AHP_summary.csv"
    nw_sum_path = TABLES / "huff_NW_summary.csv"
    blocks_cache_path = DATA_PROCESSED / "spatial_blocks.csv"

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Shared setup ─────────────────────────────────────────
    # Spatial blocks depend only on municipality centroids, so they're
    # genuinely shared. Municipality *features* are NOT shared: each model
    # gets its own composite GI score (GI_AHP for Model 1, GI_Final_NotWeighted
    # -- renamed GI_NW here -- for Model 2), matching which Huff weighting
    # scheme it's actually predicting. Building one shared munis_features
    # table (as before) meant Model 2 trained on GI_AHP too, which is exactly
    # the bug this refit corrects — see outputs/audit/ml_model_design_note.md.
    print("Building spatial blocks...")
    blocks_df = build_spatial_blocks(munis_pts_path, cache_path=blocks_cache_path)
    print(f"  Blocks: {blocks_df['spatial_block'].value_counts().sort_index().to_dict()}")
    print()

    # ════════════════════════════════════════════════════════
    # MODEL 1 — AHP Huff as target
    # ════════════════════════════════════════════════════════
    if args.model in ["AHP", "both"]:
        print("=== MODEL 1: AHP Huff target ===")
        print("Building municipality features (composite: GI_AHP from MUNICIPALITIES_AHP)...")
        munis_features_ahp = build_municipality_features(
            munis_pts_path, acc_path, munis_ahp_path,
            composite_col="GI_AHP", output_col="GI_AHP")
        feature_cols_ahp = [c for c in munis_features_ahp.columns
                             if c not in ["Muni_ID", "Muni_Name"]]
        all_feature_cols_ahp = feature_cols_ahp + ["dist_to_muni"]
        print(f"  Feature table: {munis_features_ahp.shape}  "
              f"Total features: {len(all_feature_cols_ahp)}")

        print("Melting AHP OD matrix...")
        df_pairs_ahp = melt_od_matrix(ahp_od_path)
        print(f"  Pairs: {df_pairs_ahp.shape}")

        df_ml_ahp = df_pairs_ahp.merge(munis_features_ahp, on="Muni_Name", how="left")
        df_ml_ahp = df_ml_ahp.merge(blocks_df, on="Muni_ID", how="left")
        print(f"  ML table: {df_ml_ahp.shape}")
        print()

        print("Training AHP Random Forest (spatial 5-fold CV)...")
        df_results_ahp, preds_ahp, fi_ahp = train_rf_spatial_cv(
            df_ml_ahp, all_feature_cols_ahp, target_col="Pij",
            sample_frac=args.sample_frac)
        print()
        print("AHP CV Summary:")
        print(f"  Mean R²:   {df_results_ahp['r2'].mean():.4f} ± {df_results_ahp['r2'].std():.4f}")
        print(f"  Mean MAE:  {df_results_ahp['mae'].mean():.6f}")
        print(f"  Mean RMSE: {df_results_ahp['rmse'].mean():.6f}")

        # Save AHP outputs
        df_importance_ahp = pd.DataFrame({
            "feature": all_feature_cols_ahp, "importance": fi_ahp
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df_importance_ahp.to_csv(TABLES / "ml_AHP_feature_importance.csv", index=False)
        df_results_ahp = annotate_wall_time_anomalies(df_results_ahp)
        df_results_ahp.to_csv(TABLES / "ml_AHP_cv_results.csv", index=False)

        df_ml_ahp["Pij_predicted"] = df_ml_ahp.index.map(preds_ahp)
        comparison_ahp = build_comparison(df_ml_ahp, preds_ahp, ahp_sum_path)
        comparison_ahp.to_csv(TABLES / "ml_AHP_vs_AHP_comparison.csv", index=False)

        agree_ahp = comparison_ahp["agreement"].sum()
        print(f"  AHP vs ML agreement: {agree_ahp}/{len(comparison_ahp)} "
              f"({agree_ahp/len(comparison_ahp)*100:.1f}%)")
        print()

    # ════════════════════════════════════════════════════════
    # MODEL 2 — NW Huff as target
    # ════════════════════════════════════════════════════════
    if args.model in ["NW", "both"]:
        print("=== MODEL 2: NW Huff target ===")
        print("Building municipality features (composite: GI_Final_NotWeighted "
              "from MUNICIPALITIES_NW, stored as GI_NW)...")
        munis_features_nw = build_municipality_features(
            munis_pts_path, acc_path, munis_nw_path,
            composite_col="GI_Final_NotWeighted", output_col="GI_NW")
        feature_cols_nw = [c for c in munis_features_nw.columns
                            if c not in ["Muni_ID", "Muni_Name"]]
        all_feature_cols_nw = feature_cols_nw + ["dist_to_muni"]
        print(f"  Feature table: {munis_features_nw.shape}  "
              f"Total features: {len(all_feature_cols_nw)}")

        print("Melting NW OD matrix...")
        df_pairs_nw = melt_od_matrix(nw_od_path)
        print(f"  Pairs: {df_pairs_nw.shape}")

        df_ml_nw = df_pairs_nw.merge(munis_features_nw, on="Muni_Name", how="left")
        df_ml_nw = df_ml_nw.merge(blocks_df, on="Muni_ID", how="left")
        print(f"  ML table: {df_ml_nw.shape}")
        print()

        print("Training NW Random Forest (spatial 5-fold CV)...")
        df_results_nw, preds_nw, fi_nw = train_rf_spatial_cv(
            df_ml_nw, all_feature_cols_nw, target_col="Pij",
            sample_frac=args.sample_frac)
        print()
        print("NW CV Summary:")
        print(f"  Mean R²:   {df_results_nw['r2'].mean():.4f} ± {df_results_nw['r2'].std():.4f}")
        print(f"  Mean MAE:  {df_results_nw['mae'].mean():.6f}")
        print(f"  Mean RMSE: {df_results_nw['rmse'].mean():.6f}")

        # Save NW outputs
        df_importance_nw = pd.DataFrame({
            "feature": all_feature_cols_nw, "importance": fi_nw
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df_importance_nw.to_csv(TABLES / "ml_NW_feature_importance.csv", index=False)
        df_results_nw = annotate_wall_time_anomalies(df_results_nw)
        df_results_nw.to_csv(TABLES / "ml_NW_cv_results.csv", index=False)

        comparison_nw = build_comparison(df_ml_nw, preds_nw, nw_sum_path)
        comparison_nw.to_csv(TABLES / "ml_NW_vs_NW_comparison.csv", index=False)

        agree_nw = comparison_nw["agreement"].sum()
        print(f"  NW vs ML agreement: {agree_nw}/{len(comparison_nw)} "
              f"({agree_nw/len(comparison_nw)*100:.1f}%)")
        print()

    print("All ML outputs saved to outputs/tables/")
    print("Done.")


if __name__ == "__main__":
    main()
