"""
Compute Moran's I, Cohen's kappa, LISA, and a binary join-count robustness
check for the three headline settlement-level agreement comparisons
(AHP vs NW, AHP vs ML, NW vs ML), save autocorrelation and three-way
agreement results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
import esda
from esda.join_counts import Join_Counts
from sklearn.metrics import cohen_kappa_score

from config import GPKG, TABLES

# comparison -> (agreement layer path, dominant-muni column for model A, for model B)
COMPARISONS = {
    "AHP_vs_NW": (GPKG / "map_AHP_vs_NW_villages.gpkg", "AHP_dominant_muni", "NW_dominant_muni"),
    "AHP_vs_ML": (GPKG / "map_AHP_vs_ML_villages.gpkg", "AHP_dominant_muni", "ml_dominant_muni"),
    "NW_vs_ML": (GPKG / "map_NW_vs_ML_villages.gpkg", "NW_dominant_muni", "ml_dominant_muni"),
}

LISA_OUTPUT_PATHS = {
    "AHP_vs_NW": GPKG / "map_lisa_AHP_vs_NW.gpkg",
    "AHP_vs_ML": GPKG / "map_lisa_AHP_vs_ML.gpkg",
    "NW_vs_ML": GPKG / "map_lisa_NW_vs_ML.gpkg",
}

RESULTS_PATH = TABLES / "table_morans_i_results.csv"
THREE_WAY_PATH = TABLES / "table_three_way_agreement.csv"
LISA_SUMMARY_PATH = TABLES / "table_lisa_summary.csv"

SIGNIFICANCE_LEVEL = 0.05
LISA_SEED = 42
JOIN_COUNT_PERMUTATIONS = 999

# esda.Moran and esda.join_counts.Join_Counts expose no `seed` kwarg (unlike
# Moran_Local, seeded explicitly below via LISA_SEED). np.random.seed() is
# set here as a best-effort attempt at reproducibility, but empirically this
# esda version's permutation draws are NOT fully pinned by the legacy global
# seed: morans_I itself (a closed-form statistic) is exactly reproducible
# run to run, but the permutation-derived z_score/p_value drift by a few
# points between runs (e.g. AHP_vs_NW z observed in the 23-24.5 range across
# repeated runs) without affecting significance conclusions (p stays 0.001
# throughout). This should be disclosed rather than silently treated as
# pinned. LISA (Moran_Local) below IS exactly reproducible via its own seed.
GLOBAL_PERMUTATION_SEED = 42


def load_agreement_layer(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run src/12_export_outputs.py first "
            "to build the agreement map layers."
        )
    return gpd.read_file(path)


def build_queen_weights(gdf):
    """Queen contiguity spatial weights, row-standardised."""
    w = Queen.from_dataframe(gdf, use_index=False)
    w.transform = "r"
    return w


def global_morans_i(gdf, field, label):
    w = build_queen_weights(gdf)
    y = gdf[field].astype(float).values
    mi = esda.Moran(y, w)
    print(f"  {label}: I={mi.I:.4f}  p={mi.p_sim:.4f}  z={mi.z_sim:.4f}  (n={len(y)})")
    return {
        "layer": label,
        "field": field,
        "n": len(y),
        "morans_I": mi.I,
        "expected_I": mi.EI,
        "p_value": mi.p_sim,
        "z_score": mi.z_sim,
    }, w, y


def compute_join_counts(y, w, label):
    """Binary join-count statistic — robustness check for Moran's I on a
    0/1 variable. BB = joins between two agreeing (1) neighbours, BW = joins
    between a 1 and a 0, WW = joins between two disagreeing (0) neighbours
    (raw count only — esda's Join_Counts only simulates BB and BW directly).
    A significant excess of BB/WW over their permutation means, alongside a
    significant chi2, corroborates spatial clustering independently of
    Moran's I's continuous-variable assumptions."""
    jc = Join_Counts(y.astype(int), w, permutations=JOIN_COUNT_PERMUTATIONS)
    print(f"  {label} join counts: BB={jc.bb} (mean {jc.mean_bb:.1f}, p={jc.p_sim_bb:.4f})  "
          f"WW={jc.ww}  BW={jc.bw} (mean {jc.mean_bw:.1f}, p={jc.p_sim_bw:.4f})  "
          f"chi2={jc.chi2:.2f} (p={jc.chi2_p:.4f})")
    return {
        "comparison": label, "total_joins": jc.J,
        "bb_observed": jc.bb, "bb_mean_expected": jc.mean_bb, "bb_p_sim": jc.p_sim_bb,
        "ww_observed": jc.ww,
        "bw_observed": jc.bw, "bw_mean_expected": jc.mean_bw, "bw_p_sim": jc.p_sim_bw,
        "chi2": jc.chi2, "chi2_p": jc.chi2_p,
    }


def classify_lisa(local_moran, y, significance):
    """HH / LL / HL / LH quadrant labels; 'not significant' below threshold."""
    y_mean = y.mean()
    labels = []
    for value, lag, p in zip(y, local_moran.y, local_moran.p_sim):
        if p >= significance:
            labels.append("not significant")
            continue
        if value >= y_mean and lag >= y_mean:
            labels.append("HH")
        elif value < y_mean and lag < y_mean:
            labels.append("LL")
        elif value >= y_mean and lag < y_mean:
            labels.append("HL")
        else:
            labels.append("LH")
    return labels


def main():
    print("=== MORAN'S I / THREE-WAY AGREEMENT / LISA ===")
    print(f"(global permutation seed = {GLOBAL_PERMUTATION_SEED}, LISA seed = {LISA_SEED}, "
          f"join-count permutations = {JOIN_COUNT_PERMUTATIONS})")
    print()
    np.random.seed(GLOBAL_PERMUTATION_SEED)

    TABLES.mkdir(parents=True, exist_ok=True)
    GPKG.mkdir(parents=True, exist_ok=True)

    layers = {}
    for name, (path, col_a, col_b) in COMPARISONS.items():
        print(f"Loading {path.name}...")
        gdf = load_agreement_layer(path)
        print(f"  {len(gdf)} villages")
        layers[name] = (gdf, col_a, col_b)
    print()

    # ── Global Moran's I ─────────────────────────────────────
    print("Computing global Moran's I...")
    moran_results = []
    weights = {}
    agreement_arrays = {}
    for name, (gdf, col_a, col_b) in layers.items():
        result, w, y = global_morans_i(gdf, "agreement", name)
        moran_results.append(result)
        weights[name] = w
        agreement_arrays[name] = y
    print()

    moran_df = pd.DataFrame(moran_results)
    moran_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved {RESULTS_PATH}")
    print()

    # ── Cohen's kappa + three-way agreement table ────────────
    print("Computing Cohen's kappa (212-class settlement-level dominant municipality)...")
    three_way_rows = []
    for name, (gdf, col_a, col_b) in layers.items():
        labels_a = gdf[col_a].astype(str).values
        labels_b = gdf[col_b].astype(str).values
        n_agree = int((labels_a == labels_b).sum())
        n_total = len(gdf)
        agreement_pct = 100.0 * n_agree / n_total
        kappa = cohen_kappa_score(labels_a, labels_b)
        moran_row = moran_df[moran_df["layer"] == name].iloc[0]
        print(f"  {name}: agreement={n_agree}/{n_total} ({agreement_pct:.2f}%)  kappa={kappa:.4f}")
        three_way_rows.append({
            "comparison": name, "n": n_total, "n_agree": n_agree,
            "agreement_pct": agreement_pct, "cohen_kappa": kappa,
            "morans_I": moran_row["morans_I"], "z_score": moran_row["z_score"],
            "p_value": moran_row["p_value"],
        })
    print()

    three_way_df = pd.DataFrame(three_way_rows)
    three_way_df.to_csv(THREE_WAY_PATH, index=False)
    print(f"Saved {THREE_WAY_PATH}")
    print()

    # ── Join counts (binary robustness check) ────────────────
    print("Computing join-count statistics (binary robustness check for Moran's I)...")
    jc_rows = []
    for name in layers:
        jc_rows.append(compute_join_counts(agreement_arrays[name], weights[name], name))
    jc_df = pd.DataFrame(jc_rows)
    jc_path = TABLES / "table_join_counts.csv"
    jc_df.to_csv(jc_path, index=False)
    print(f"Saved {jc_path}")
    print()

    # ── LISA for all three comparisons ────────────────────────
    print("Computing Local Moran's I (LISA) for all three comparisons...")
    lisa_summary_rows = []
    for name, (gdf, col_a, col_b) in layers.items():
        y = agreement_arrays[name]
        w = weights[name]
        lisa = esda.Moran_Local(y, w, seed=LISA_SEED)
        cluster_type = classify_lisa(lisa, y, SIGNIFICANCE_LEVEL)

        lisa_layer = gdf.copy()
        lisa_layer["lisa_I"] = lisa.Is
        lisa_layer["lisa_p"] = lisa.p_sim
        lisa_layer["cluster_type"] = cluster_type

        counts = pd.Series(cluster_type).value_counts().to_dict()
        print(f"  {name} cluster type counts: {counts}")

        out_path = LISA_OUTPUT_PATHS[name]
        lisa_layer.to_file(out_path, driver="GPKG")
        print(f"  Saved {out_path}")

        lisa_summary_rows.append({
            "comparison": name,
            "HH": counts.get("HH", 0), "LL": counts.get("LL", 0),
            "HL": counts.get("HL", 0), "LH": counts.get("LH", 0),
            "not_significant": counts.get("not significant", 0),
        })
    print()

    lisa_summary_df = pd.DataFrame(lisa_summary_rows)
    lisa_summary_df.to_csv(LISA_SUMMARY_PATH, index=False)
    print(f"Saved {LISA_SUMMARY_PATH}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
