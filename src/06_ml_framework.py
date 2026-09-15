"""
Step 6 of the pipeline: can a Random Forest learn the same catchment
pattern the Huff model produces, from the same underlying features?

What this script does: trains two separate Random Forest models — Model 1
learns to predict the AHP Huff model's settlement-to-municipality
probabilities (Pij) from municipality features (GI, accessibility) and
distance; Model 2 does the same for the non-weighted (NW) Huff model. Each
model is trained on all 1,279,632 (settlement, municipality) pairs
(6,036 settlements times 212 municipalities) using 189 features. The two
models are deliberately independent — see build_municipality_features
below for why each one must be given its own model-specific GI column, not
a shared one. Evaluation uses 5-fold spatial cross-validation: municipalities
are grouped into 5 geographic blocks (see build_spatial_blocks) and each
fold holds out one whole block, so the model is tested on municipalities it
never saw during training, not just on held-out settlement rows within
municipalities it did see. This is a check on whether the pattern the Huff
formula produces is learnable from these features at all — it is not a
validation of the Huff model's correctness, since both models are trained
to reproduce Huff's own output, not any ground truth (see
docs/ml_model_design_note.md).

Reads: Municipalities_Points_normalized.gpkg, accessibility_normalized.csv,
huff_od_matrix.csv and huff_AHP_summary.csv (Model 1), huff_NW_od_matrix.csv
and huff_NW_summary.csv (Model 2), plus the corresponding GI layers.

Writes, per model: ml_{AHP,NW}_feature_importance.csv, ml_{AHP,NW}_cv_results.csv,
ml_{AHP,NW}_vs_{AHP,NW}_comparison.csv. Also caches data/processed/spatial_blocks.csv.

Runs sixth. Needs 03, 04 (the Huff outputs it's trained to predict) and 05
(accessibility features). Its trained models are also called into directly
by 16_shap_dependence.py (via run_shap, below) rather than being reloaded
from disk, since this script does not save the fitted model objects
themselves.

Run the two models as two separate invocations (`--model AHP`, then
`--model NW`), not the default `--model both`, on a machine with limited
RAM. `--model both` still works and still frees Model 1's tables before
starting Model 2 (see main() below), but each model on its own already
holds a 1,279,632-row feature table and a 100-tree Random Forest in
memory — training two in the same process at once is exactly what
exhausted this pipeline's original 16 GB analysis machine and got the
process killed by the OS with no Python traceback. A lock file
(data/processed/06_ml_framework.lock) refuses to start a second instance
of this script while one is already running, and each model's own peak
memory use is printed at the end of its block.
"""

import argparse
import os
import sys
import threading
from datetime import datetime
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
import psutil
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

OUTPUT_FILES = [
    "tables/ml_AHP_feature_importance.csv",
    "tables/ml_AHP_cv_results.csv",
    "tables/ml_AHP_vs_AHP_comparison.csv",
    "tables/ml_NW_feature_importance.csv",
    "tables/ml_NW_cv_results.csv",
    "tables/ml_NW_vs_NW_comparison.csv",
]

# Training one model holds a 1,279,632-row feature table plus a 100-tree
# Random Forest in memory at once; training two at once (--model both on a
# machine with limited RAM) can exceed what is physically available and
# get one or both processes killed by the OS with no Python traceback at
# all — this happened in practice on a 16 GB machine. This lock file
# refuses to start a second training run while one is already in
# progress, rather than letting both compete for memory silently.
LOCK_PATH = DATA_PROCESSED / "06_ml_framework.lock"


def _parse_lock_pid(lock_text):
    """Extract the PID recorded in a lock file's text, or None if it can't be read."""
    for line in lock_text.splitlines():
        if line.startswith("pid="):
            try:
                return int(line[len("pid="):])
            except ValueError:
                return None
    return None


def _lock_owner_is_alive(pid):
    """Check whether the process that wrote the lock file is still actually running.

    A hard kill (the OS-level out-of-memory kill this lock exists to guard
    against) leaves the lock file behind, since it bypasses Python's normal
    cleanup — without this check, that permanent lock would fail every
    future run for a reason that no longer exists, and the only fix would
    be someone remembering to delete the file by hand. Matches on the PID
    still existing *and* still looking like a Python process, since the
    operating system can in principle reuse a PID number for an unrelated
    process once the original one is gone (unlikely, but cheap to guard
    against). If the process exists but its details can't be inspected
    (permissions), that is treated as "still alive" — it is safer to
    wrongly refuse to start than to wrongly clear a lock that is still
    genuinely held.
    """
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    try:
        return proc.is_running() and "python" in proc.name().lower()
    except psutil.AccessDenied:
        return True


def acquire_lock():
    """Refuse to start if another instance of this script is already running.

    If a lock file exists, its recorded PID is checked for whether that
    process is still alive (see _lock_owner_is_alive) — a lock left behind
    by a process that no longer exists is cleared automatically rather
    than blocking every future run. Only a lock whose PID is confirmed
    still running blocks a new run.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        lock_text = LOCK_PATH.read_text()
        old_pid = _parse_lock_pid(lock_text)
        if old_pid is not None and not _lock_owner_is_alive(old_pid):
            print(f"Found a stale lock at {LOCK_PATH} — process {old_pid} is no longer "
                  "running (most likely killed without a chance to clean up after "
                  "itself). Clearing it automatically.")
            print(f"  Previous lock contents:\n{lock_text}")
            LOCK_PATH.unlink()
        else:
            print(f"ERROR: {LOCK_PATH} already exists.")
            print(lock_text)
            print("This means another 06_ml_framework.py run appears to already be in "
                  "progress. Training two full models at once can exhaust this machine's "
                  "memory. If you are certain no other instance is actually running, "
                  "delete the lock file and run this script again:")
            print(f'  rm "{LOCK_PATH}"')
            sys.exit(1)
    LOCK_PATH.write_text(f"pid={os.getpid()}\nstarted={datetime.now().isoformat()}\n")


def release_lock():
    """Remove the lock file on a normal exit (including a handled error)."""
    LOCK_PATH.unlink(missing_ok=True)


class PeakMemoryTracker:
    """Track this process's peak resident memory (RSS) over a span of work.

    Samples memory in a background thread rather than at fixed points in
    the code, since the actual peak can land anywhere across loading data,
    building the feature table, or fitting any one of the five per-fold
    Random Forests, and instrumenting every one of those points by hand
    would be fragile. Used to report each model's peak memory separately
    (see main() below) so it is clear how close a single model's training
    run sits to exhausting this machine's RAM.
    """

    def __init__(self, interval_s=1.0):
        self.interval_s = interval_s
        self._process = psutil.Process()
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            rss = self._process.memory_info().rss
            self._peak_bytes = max(self._peak_bytes, rss)
            self._stop.wait(self.interval_s)

    def start(self):
        self._peak_bytes = self._process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_and_report_gb(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
        return self._peak_bytes / (1024 ** 3)


def build_municipality_features(munis_pts_path, acc_path, composite_path,
                                 composite_col, output_col):
    """Build one model's municipality feature table: GI indicators, accessibility, and its own composite GI score.

    `composite_col` names the column to read from `composite_path` — "GI_AHP"
    from the AHP municipality layer for Model 1, or "GI_Final_NotWeighted"
    from the NW municipality layer for Model 2. `output_col` is what that
    column is renamed to in the table this function returns, so each model's
    own composite score stays distinguishable in every downstream file
    (feature importance CSVs, the AHP-vs-NW comparison figure) instead of
    both models ending up with an identically-named "GI_AHP" column. Each
    model must be built from its own call to this function with its own
    composite_path — building one shared feature table and reusing it for
    both models would train Model 2 on GI_AHP instead of its own target's
    GI, which is a bug this design specifically avoids (see
    docs/ml_model_design_note.md).
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
    """Group municipalities into 5 geographic clusters for spatial cross-validation.

    Ordinary (non-spatial) cross-validation would let the model see
    settlements from a municipality in training and then get tested on
    other settlements from that same municipality — an easy test that
    would overstate how well the model generalises to places it has never
    seen. Clustering municipalities into geographic blocks by their
    location, and later holding out one whole block per fold, means each
    fold is tested on municipalities the model never saw in training,
    which is a fairer measure of genuine spatial generalisation.

    Uses scipy's clustering function (`kmeans2`) rather than scikit-learn's
    KMeans, because scikit-learn's KMeans crashes this machine's process
    outright — a known Windows-specific interaction between scikit-learn's
    internal MKL/OpenMP runtime check (`_check_mkl_vcomp`) and this
    machine's MKL installation, separate from the AVX-512 dispatch crash
    documented in docs/reproducibility_note.md. Switching to scipy's
    implementation was the practical fix. Blocks are cached to
    `cache_path`, since municipality locations never change between runs
    and reclustering every time would be wasted work.
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
    """Reshape the wide OD matrix (one row per settlement, one column per municipality) into one row per (settlement, municipality) pair.

    The Random Forest needs one training row per (settlement, municipality)
    combination, each with its own distance and its own Huff probability
    (Pij) as the value to predict — the wide OD matrix from 03/04 has to be
    unpacked into that long shape first.
    """
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
    """Train and evaluate the Random Forest, one fold per spatial block.

    Each of the 5 spatial blocks (see build_spatial_blocks) takes a turn as
    the held-out test set, with the model trained fresh on the other four
    each time — standard 5-fold cross-validation, except the folds are
    geographic blocks of municipalities rather than random rows, so a
    model can never be tested on a municipality's settlements after having
    trained on other settlements from that same municipality. The CSV's
    "fold" numbers below are 1-indexed (fold 1..5); internally this loop
    uses spatial_block 0..4 for the same five groups.
    """
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
    """For each settlement, find which municipality the Random Forest predicts most strongly, and compare it to the Huff model's own choice.

    The predicted Pij values used here always come from the fold where
    that settlement's municipality was held out (see all_preds, built in
    train_rf_spatial_cv) — every prediction is a genuine out-of-sample
    prediction, never a value the model saw during its own training.
    """
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
    """Compute SHAP feature-contribution values for a trained model and save the summary plots.

    Called from 16_shap_dependence.py, not from this script's own main() —
    this script trains the models but does not keep the fitted model
    objects around after main() finishes, so 16_shap_dependence.py imports
    this module directly and calls its own training + this function to get
    a live model to explain.
    """
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

    acquire_lock()
    try:
        _main(args)
    finally:
        release_lock()


def _main(args):
    print("=== ML FRAMEWORK ===")
    print()

    # All paths below are relative to the repository. They used to point
    # outside it, at a folder named "Matrix and tables" that existed only on
    # one machine — a script that depends on a path like that cannot be run
    # by anyone else, and breaks the data availability statement a published
    # paper needs. See docs/reproducibility_note.md for the full fix.
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
    # The 5 spatial blocks depend only on where the municipalities are, so
    # both models genuinely share the same blocks. Everything else is kept
    # separate on purpose: each model builds its own municipality feature
    # table below, with its own composite GI column (GI_AHP for Model 1,
    # GI_Final_NotWeighted — renamed GI_NW here — for Model 2), matching
    # the Huff weighting scheme it is actually being trained to predict.
    # An earlier version of this script built one shared feature table for
    # both models, which meant Model 2 was trained on GI_AHP as well as its
    # own target — see docs/ml_model_design_note.md for how that was found
    # and fixed.
    print("Building spatial blocks...")
    blocks_df = build_spatial_blocks(munis_pts_path, cache_path=blocks_cache_path)
    print(f"  Blocks: {blocks_df['spatial_block'].value_counts().sort_index().to_dict()}")
    print()

    # ════════════════════════════════════════════════════════
    # MODEL 1 — AHP Huff as target
    # ════════════════════════════════════════════════════════
    if args.model in ["AHP", "both"]:
        mem_tracker = PeakMemoryTracker()
        mem_tracker.start()
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
        print(f"  Peak memory during Model 1 (AHP): {mem_tracker.stop_and_report_gb():.2f} GB")
        print()

    if args.model == "both":
        print("Freeing Model 1's feature and training tables before starting Model 2, "
              "so the two models' large tables are never held in memory at once...")
        del munis_features_ahp, df_pairs_ahp, df_ml_ahp, preds_ahp
        gc.collect()
        print()

    # ════════════════════════════════════════════════════════
    # MODEL 2 — NW Huff as target
    # ════════════════════════════════════════════════════════
    if args.model in ["NW", "both"]:
        mem_tracker = PeakMemoryTracker()
        mem_tracker.start()
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
        print(f"  Peak memory during Model 2 (NW): {mem_tracker.stop_and_report_gb():.2f} GB")
        print()

    print("All ML outputs saved to outputs/tables/")
    print("Done.")


if __name__ == "__main__":
    main()
