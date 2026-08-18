"""
Regenerate SHAP dependence plots for the AHP-target Random Forest.
06_ml_framework.py never persists a trained model, so this script retrains
one RF (identical hyperparameters, full training data) purely for SHAP
explanation purposes, then computes SHAP values on a fixed 5,000-pair
sample and plots dependence for the five highest mean-|SHAP| features.
"""

import sys
import importlib.util
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# shap/__init__.py eagerly imports shap.plots, which imports
# shap.plots.colors, which calls scipy.linalg.inv() at module import time
# to build its custom red/blue colormap. On this machine, scipy.linalg's
# LAPACK binding crashes with a native illegal-instruction fault on ANY
# call (verified independently of shap: `scipy.linalg.inv(np.random.rand(3,3))`
# crashes the interpreter outright, while `numpy.linalg.inv` on the same
# matrix does not — this is a scipy/LAPACK build defect on this CPU, not a
# shap bug, and it affects nothing else in this repository since no other
# script calls scipy.linalg). Blocking matplotlib detection during `import
# shap` makes shap skip `shap.plots` entirely (its own documented fallback
# path for "matplotlib not installed"), which avoids the crash and still
# leaves `shap.TreeExplainer` fully functional. Real matplotlib is imported
# separately below, after `import shap` completes, for this script's own
# (shap-independent) dependence-plot rendering.
sys.modules["matplotlib"] = None
import shap  # noqa: E402
del sys.modules["matplotlib"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_RAW, FIGURES, MUNICIPALITIES_AHP, MUNICIPALITIES_PTS, TABLES

MATRIX_TABLES = Path(r"C:\Users\lstojano\Desktop\teza\HuffMethodPaper\Data\Matrix and tables")
SRC_DIR = Path(__file__).resolve().parent

RF_SEED = 42
SHAP_SAMPLE_SEED = 42
SHAP_SAMPLE_SIZE = 5000
N_DEPENDENCE_FEATURES = 5

OUTPUT_PREFIX = FIGURES / "fig_shap_dependence_AHP"


def load_module(stem):
    path = SRC_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("=== SHAP DEPENDENCE (AHP model) ===")
    print(f"(RF seed = {RF_SEED}, SHAP sample seed = {SHAP_SAMPLE_SEED}, "
          f"sample size = {SHAP_SAMPLE_SIZE})")
    print()

    FIGURES.mkdir(parents=True, exist_ok=True)

    mod06 = load_module("06_ml_framework")

    munis_pts_path = DATA_RAW / MUNICIPALITIES_PTS
    munis_ahp_path = DATA_RAW / MUNICIPALITIES_AHP
    acc_path = MATRIX_TABLES / "accessibility_normalized.csv"
    ahp_od_path = MATRIX_TABLES / "huff_od_matrix.csv"

    print("Building municipality features...")
    munis_features = mod06.build_municipality_features(munis_pts_path, acc_path, munis_ahp_path)
    feature_cols = [c for c in munis_features.columns if c not in ["Muni_ID", "Muni_Name"]]
    all_feature_cols = feature_cols + ["dist_to_muni"]
    print(f"  Features: {len(all_feature_cols)}")

    print("Melting AHP OD matrix...")
    df_pairs = mod06.melt_od_matrix(ahp_od_path)
    df_ml = df_pairs.merge(munis_features, on="Muni_Name", how="left")
    print(f"  Training table: {df_ml.shape}")
    print()

    X = df_ml[all_feature_cols].values.astype(np.float32)
    y = df_ml["Pij"].values.astype(np.float32)

    print(f"Training one RandomForestRegressor(n_estimators=100, max_depth=15, "
          f"min_samples_leaf=10, n_jobs=-1, random_state={RF_SEED}) on the full "
          f"{len(X):,}-row AHP table (same hyperparameters as 06_ml_framework.py's "
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
    fig.suptitle("SHAP dependence — top 5 features, AHP model (5,000-pair sample, seed=42)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_PREFIX}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUTPUT_PREFIX}.png / .pdf")
    print()

    print("Rendering individual panels...")
    for i in order[:N_DEPENDENCE_FEATURES]:
        feat = all_feature_cols[i]
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_dependence(ax, i)
        ax.set_title(feat)
        fig.tight_layout()
        safe_name = feat.replace("/", "_")
        out_path = f"{OUTPUT_PREFIX}_{safe_name}"
        fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}.png / .pdf")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
