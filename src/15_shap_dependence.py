"""
Step 15 of the pipeline: for each Random Forest's five most influential
features, how does the model's prediction actually change as that
feature's value changes?

What this script does: SHAP (SHapley Additive exPlanations) values
attribute each individual prediction to the features that produced it —
useful for asking not just "which features matter" (that's feature
importance, from 06_ml_framework.py) but "in which direction, and how
strongly, does a feature push the prediction." 06_ml_framework.py trains
its models but never saves the fitted model objects to disk, so this
script retrains one Random Forest per model — identical hyperparameters
and training data, purely so there is a live model to explain — computes
SHAP values on a fixed 5,000-pair sample, and plots how the prediction
moves against each of the five features with the largest average SHAP
impact. Each model uses its own composite GI feature (GI_AHP or GI_NW,
never the other one's — see docs/ml_model_design_note.md for why that
distinction matters).

Reads: Municipalities_Points_normalized.gpkg, accessibility_normalized.csv,
the AHP and NW GI layers, and huff_od_matrix.csv / huff_NW_od_matrix.csv
(already computed by 03 and 04). Imports 06_ml_framework.py directly as a
module to reuse its feature-building and OD-melting functions rather than
duplicating them.

Writes, per model (AHP and NW): fig_shap_dependence_{prefix}.png/.pdf (the
five-panel summary figure), one fig_shap_dependence_{prefix}_{feature}.png/.pdf
per top feature (file names depend on which features rank in the top 5,
so they are not fully predictable in advance — see OUTPUT_FILES below),
and fig08_shap_summary_{prefix}.png / fig08_shap_bar_{prefix}.png.

Runs fifteenth. Needs 03 and 04's Huff outputs and imports 06_ml_framework.py
directly; does not need 06's own main() to have been run first, since it
retrains its own models from scratch.

Run the two models as two separate invocations (`--model AHP`, then
`--model NW`), not the default `--model both`, on a machine with limited
RAM — training a full Random Forest for each model in the same process is
exactly what exhausted this pipeline's 16 GB analysis machine during
Stage 2.4 verification and got the process killed by the OS mid-run, the
same failure mode 06_ml_framework.py hit first (see that script's own
docstring). This script has its own lock file
(data/processed/15_shap_dependence.lock) and its own per-model peak memory
report, for the same reasons.
"""

import argparse
import gc
import os
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestRegressor

# A previous version of this script blocked matplotlib detection during
# `import shap`, working around a documented scipy.linalg/LAPACK native
# crash on this machine (shap.plots.colors calls scipy.linalg.inv() at
# import time). That crash does NOT reproduce as of this environment
# (verified directly: `scipy.linalg.inv(np.random.rand(3,3))` runs cleanly,
# and `import shap` with matplotlib already imported does not crash either —
# see docs/reproducibility_note.md for the verification). The
# workaround is removed because it had a real cost: blocking matplotlib
# detection makes `shap.summary_plot`/`shap.plots.bar` permanently think
# matplotlib isn't installed for the rest of the process, which is exactly
# what made 06_ml_framework.py::run_shap unusable and its two figures
# orphaned. If this crash resurfaces on a different machine, the fix is to
# reintroduce the blocking import *only* around `import shap` and accept
# that `run_shap`-style plotting is unavailable in that environment — do not
# silently re-add it without re-testing, since it disables real functionality.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from config import (DATA_RAW, DATA_PROCESSED, FIGURES, MUNICIPALITIES_AHP, MUNICIPALITIES_NW,
                     MUNICIPALITIES_PTS, TABLES)

SRC_DIR = Path(__file__).resolve().parent

# Like 06_ml_framework.py's own lock, and for the same reason: this script
# retrains a full Random Forest per model purely for SHAP explanation, and
# doing both models in one process (the default, for convenience) held
# enough memory at once to get the process killed by the OS on a 16 GB
# machine during this repository's own Stage 2.4 pipeline verification —
# the same failure mode 06_ml_framework.py hit first. Run `--model AHP`
# and `--model NW` as two separate invocations on a memory-constrained
# machine. This is a separate lock file from 06_ml_framework.py's own,
# since the two scripts run independently of each other.
LOCK_PATH = DATA_PROCESSED / "15_shap_dependence.lock"


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

    See the identical function in 06_ml_framework.py::_lock_owner_is_alive
    for the full reasoning — a hard kill leaves the lock file behind, so
    this lets a genuinely dead lock clear itself automatically instead of
    blocking every future run until someone remembers to delete it by hand.
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
    process is still alive — a lock left behind by a process that no
    longer exists is cleared automatically rather than blocking every
    future run. Only a lock whose PID is confirmed still running blocks a
    new run.
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
            print("This means another 15_shap_dependence.py run appears to already be in "
                  "progress. If you are certain no other instance is actually running, "
                  "delete the lock file and run this script again:")
            print(f'  rm "{LOCK_PATH}"')
            sys.exit(1)
    LOCK_PATH.write_text(f"pid={os.getpid()}\nstarted={datetime.now().isoformat()}\n")


def release_lock():
    """Remove the lock file on a normal exit (including a handled error)."""
    LOCK_PATH.unlink(missing_ok=True)

# Only the statically-named files are listed — the per-feature dependence
# panels (fig_shap_dependence_{prefix}_{feature}.png/.pdf) are named after
# whichever features happen to rank in each model's top 5 by mean |SHAP|,
# which is data-dependent and can change between reruns if the underlying
# data changes. 18_output_manifest.py treats files matching that naming
# pattern as claimed by this script even though it cannot list them by
# exact name in advance.
OUTPUT_FILES = [
    "figures/fig_shap_dependence_AHP.png",
    "figures/fig_shap_dependence_AHP.pdf",
    "figures/fig_shap_dependence_NW.png",
    "figures/fig_shap_dependence_NW.pdf",
    "figures/fig08_shap_summary_AHP.png",
    "figures/fig08_shap_bar_AHP.png",
    "figures/fig08_shap_summary_NW.png",
    "figures/fig08_shap_bar_NW.png",
]
OUTPUT_FILE_PATTERNS = [
    "figures/fig_shap_dependence_AHP_*",
    "figures/fig_shap_dependence_NW_*",
]

RF_SEED = 42
SHAP_SAMPLE_SEED = 42
SHAP_SAMPLE_SIZE = 5000
N_DEPENDENCE_FEATURES = 5


def load_module(stem):
    """Import another src/NN_name.py script as a live module, by file path.

    Needed because script filenames start with a digit ("06_ml_framework"),
    which Python's normal `import` statement cannot handle directly.
    """
    path = SRC_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod06, model_name, prefix, composite_path, composite_col, output_col, od_path):
    """Retrain one model, compute its SHAP values, and render its dependence figures.

    Run once for the AHP-target model and once for the NW-target model
    (see main() below) — everything in this function operates on a single
    model at a time.
    """
    print(f"=== SHAP DEPENDENCE ({model_name} model) ===")
    print(f"(RF seed = {RF_SEED}, SHAP sample seed = {SHAP_SAMPLE_SEED}, "
          f"sample size = {SHAP_SAMPLE_SIZE})")
    print()

    output_prefix = FIGURES / f"fig_shap_dependence_{prefix}"

    munis_pts_path = DATA_RAW / MUNICIPALITIES_PTS
    acc_path = TABLES / "accessibility_normalized.csv"

    print(f"Building municipality features (composite: {composite_col} -> {output_col})...")
    munis_features = mod06.build_municipality_features(
        munis_pts_path, acc_path, composite_path, composite_col, output_col)
    feature_cols = [c for c in munis_features.columns if c not in ["Muni_ID", "Muni_Name"]]
    all_feature_cols = feature_cols + ["dist_to_muni"]
    print(f"  Features: {len(all_feature_cols)}")

    print(f"Melting {model_name} OD matrix...")
    df_pairs = mod06.melt_od_matrix(od_path)
    df_ml = df_pairs.merge(munis_features, on="Muni_Name", how="left")
    print(f"  Training table: {df_ml.shape}")
    print()

    X = df_ml[all_feature_cols].values.astype(np.float32)
    y = df_ml["Pij"].values.astype(np.float32)

    print(f"Training one RandomForestRegressor(n_estimators=100, max_depth=15, "
          f"min_samples_leaf=10, n_jobs=-1, random_state={RF_SEED}) on the full "
          f"{len(X):,}-row {model_name} table (same hyperparameters as 06_ml_framework.py's "
          f"spatial-CV folds, but a single fit on all data — SHAP here is a "
          f"post-hoc explanation exercise, not a held-out performance claim)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=10,
                                n_jobs=-1, random_state=RF_SEED)
    rf.fit(X, y)
    print("  Done fitting.")
    print()

    print(f"Sampling {SHAP_SAMPLE_SIZE} pairs (seed={SHAP_SAMPLE_SEED}) for SHAP...")
    rng = np.random.RandomState(SHAP_SAMPLE_SEED)
    sample_idx = rng.choice(len(X), size=SHAP_SAMPLE_SIZE, replace=False)
    X_sample = X[sample_idx]
    print(f"  Sample shape: {X_sample.shape}")

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)
    print("  Done.")
    print()

    # fig08_shap_summary_AHP.png / fig08_shap_bar_AHP.png used to be produced by
    # 06_ml_framework.py::run_shap, a function defined but never called from that
    # script's main() — dead code, so no pipeline invocation actually regenerated
    # those two committed figures. Verified run_shap itself is NOT broken (a
    # synthetic-data test confirmed shap.summary_plot works fine here; only
    # shap.plots.scatter, used by the dependence plots above, hits the
    # scipy.linalg crash) — so rather than delete the figures, they're now
    # genuinely regenerated here, reusing this script's own already-fitted
    # model and SHAP values instead of retraining a third time.
    print(f"Regenerating fig08_shap_summary_{prefix}.png / fig08_shap_bar_{prefix}.png "
          "(previously dead code in 06_ml_framework.py::run_shap)...")
    mod06.run_shap(rf, X_sample, all_feature_cols, FIGURES, prefix=prefix)
    # run_shap names its own outputs fig_shap_summary_{prefix}.png / fig_shap_bar_{prefix}.png
    # (no "08"); rename to match the fig08_* filenames the manuscript references.
    import shutil as _shutil
    _shutil.move(FIGURES / f"fig_shap_summary_{prefix}.png", FIGURES / f"fig08_shap_summary_{prefix}.png")
    _shutil.move(FIGURES / f"fig_shap_bar_{prefix}.png", FIGURES / f"fig08_shap_bar_{prefix}.png")
    print(f"  Saved {FIGURES / f'fig08_shap_summary_{prefix}.png'} / fig08_shap_bar_{prefix}.png")
    print()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    top_features = [all_feature_cols[i] for i in order[:N_DEPENDENCE_FEATURES]]
    print(f"Top {N_DEPENDENCE_FEATURES} features by mean |SHAP value|:")
    for rank, i in enumerate(order[:N_DEPENDENCE_FEATURES], start=1):
        print(f"  {rank}. {all_feature_cols[i]}  (mean |SHAP| = {mean_abs_shap[i]:.6f})")
    print()

    def plot_dependence(ax, feat_idx):
        """Manual SHAP dependence plot (feature value vs. its SHAP value,
        coloured by the other top feature most correlated with this
        feature's SHAP values — the same convention shap.dependence_plot
        uses for `interaction_index="auto"`). Implemented in plain
        matplotlib because shap.plots.scatter is unreachable here (see the
        scipy.linalg note above)."""
        feat_vals = X_sample[:, feat_idx]
        sv = shap_values[:, feat_idx]

        other_idx = [i for i in order[:N_DEPENDENCE_FEATURES] if i != feat_idx]
        if other_idx:
            corrs = [abs(np.corrcoef(sv, X_sample[:, j])[0, 1]) for j in other_idx]
            interact_idx = other_idx[int(np.nanargmax(corrs))]
            interact_vals = X_sample[:, interact_idx]
            sc = ax.scatter(feat_vals, sv, c=interact_vals, cmap="coolwarm", s=14,
                             alpha=0.7, linewidths=0)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(all_feature_cols[interact_idx], fontsize=8)
        else:
            ax.scatter(feat_vals, sv, color="#3b6fa0", s=14, alpha=0.7, linewidths=0)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(all_feature_cols[feat_idx], fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)

    X_sample_df = pd.DataFrame(X_sample, columns=all_feature_cols)

    print("Rendering multi-panel dependence figure...")
    fig, axes = plt.subplots(1, N_DEPENDENCE_FEATURES, figsize=(5 * N_DEPENDENCE_FEATURES, 4.5))
    for ax, i in zip(axes, order[:N_DEPENDENCE_FEATURES]):
        plot_dependence(ax, i)
        ax.set_title(all_feature_cols[i], fontsize=10)
    fig.suptitle(f"SHAP dependence — top 5 features, {model_name} model (5,000-pair sample, seed=42)")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_prefix}.png / .pdf")
    print()

    print("Rendering individual panels...")
    for i in order[:N_DEPENDENCE_FEATURES]:
        feat = all_feature_cols[i]
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_dependence(ax, i)
        ax.set_title(feat)
        fig.tight_layout()
        safe_name = feat.replace("/", "_")
        out_path = f"{output_prefix}_{safe_name}"
        fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}.png / .pdf")

    print()
    print(f"Done ({model_name}).")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["AHP", "NW", "both"], default="both")
    args = parser.parse_args()

    acquire_lock()
    try:
        FIGURES.mkdir(parents=True, exist_ok=True)
        mod06 = load_module("06_ml_framework")

        if args.model in ["AHP", "both"]:
            mem_tracker = mod06.PeakMemoryTracker()
            mem_tracker.start()
            run(mod06, "AHP", "AHP",
                DATA_RAW / MUNICIPALITIES_AHP, "GI_AHP", "GI_AHP",
                TABLES / "huff_od_matrix.csv")
            print(f"Peak memory during AHP model: {mem_tracker.stop_and_report_gb():.2f} GB")
            print()

        if args.model == "both":
            print("Freeing AHP-model working memory before starting the NW model...")
            gc.collect()
            print()

        if args.model in ["NW", "both"]:
            mem_tracker = mod06.PeakMemoryTracker()
            mem_tracker.start()
            run(mod06, "NW", "NW",
                DATA_RAW / MUNICIPALITIES_NW, "GI_Final_NotWeighted", "GI_NW",
                TABLES / "huff_NW_od_matrix.csv")
            print(f"Peak memory during NW model: {mem_tracker.stop_and_report_gb():.2f} GB")
            print()

        print("Done.")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
