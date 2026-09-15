"""
Step 13 of the pipeline: do settlements that disagree between two models
cluster together geographically, or are they scattered at random?

What this script does: for each of the four headline village-level
comparisons (AHP Huff vs NW Huff, AHP Huff vs its Random Forest, NW Huff
vs its Random Forest, and the two Random Forest models against each
other), tests whether "agreement" and "disagreement" settlements are
spatially clustered rather than randomly scattered across the map. It
does this three ways: global Moran's I (one number summarising how much
spatial clustering exists overall), Local Moran's I / LISA (which
specific settlements sit inside a significant cluster, and what kind —
see classify_lisa below), and a binary join-count statistic as an
independent cross-check on the same question using a simpler, more
easily verified method. Cohen's kappa (chance-corrected agreement) is
also computed for each comparison here, alongside the spatial statistics,
since both describe the same underlying agreement layers.

Reads: the four agreement map GPKGs built by 11_export_outputs.py.

Writes: table_morans_i_results.csv, table_three_way_agreement.csv,
table_join_counts.csv, table_lisa_summary.csv, and one map_lisa_*.gpkg
layer per comparison.

Runs after 11_export_outputs.py, not simply "script 13" in numeric
sequence for its own sake — it depends on the maps 11 builds, and running
it before 11 has actually happened once in this repository's history (see
assert_map_is_fresh below for the guard that was added because of it).
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

OUTPUT_FILES = [
    "tables/table_morans_i_results.csv",
    "tables/table_three_way_agreement.csv",
    "tables/table_join_counts.csv",
    "tables/table_lisa_summary.csv",
    "gpkg/map_lisa_AHP_vs_NW.gpkg",
    "gpkg/map_lisa_AHP_vs_ML.gpkg",
    "gpkg/map_lisa_NW_vs_ML.gpkg",
    "gpkg/map_lisa_MLAHP_vs_MLNW.gpkg",
]

# comparison -> (agreement layer path, dominant-muni column for model A, for model B)
COMPARISONS = {
    "AHP_vs_NW": (GPKG / "map_AHP_vs_NW_villages.gpkg", "AHP_dominant_muni", "NW_dominant_muni"),
    "AHP_vs_ML": (GPKG / "map_AHP_vs_ML_villages.gpkg", "AHP_dominant_muni", "ml_dominant_muni"),
    "NW_vs_ML": (GPKG / "map_NW_vs_ML_villages.gpkg", "NW_dominant_muni", "ml_dominant_muni"),
    "MLAHP_vs_MLNW": (GPKG / "map_ML_AHP_vs_ML_NW_villages.gpkg", "ML_AHP_dominant_muni", "ML_NW_dominant_muni"),
}

# Every comparison layer above is built by 11_export_outputs.py from these
# TABLES sources, so this script must always run after 11_export_outputs.py.
# The scripts are numbered so that running them in numeric order gets this
# right automatically now, but that was not always true: under an earlier
# numbering scheme this script had a lower number than the export script
# it depends on, and running scripts in plain numeric order silently
# computed these statistics from a leftover comparison map instead of the
# pipeline's current output (see docs/reproducibility_note.md and
# docs/audit-history/reconciliation.csv for how that was found). The
# scripts were renumbered specifically to prevent that, but the check
# below is kept anyway as a safety net for anyone running scripts
# individually rather than through the full sequence — it fails loudly if
# a map is older than the tables it should have been built from, rather
# than silently computing on stale input.
MAP_SOURCES = {
    GPKG / "map_AHP_vs_NW_villages.gpkg": [TABLES / "huff_AHP_summary.csv", TABLES / "huff_NW_summary.csv"],
    GPKG / "map_AHP_vs_ML_villages.gpkg": [TABLES / "huff_AHP_summary.csv", TABLES / "ml_AHP_vs_AHP_comparison.csv"],
    GPKG / "map_NW_vs_ML_villages.gpkg": [TABLES / "huff_NW_summary.csv", TABLES / "ml_NW_vs_NW_comparison.csv"],
    GPKG / "map_ML_AHP_vs_ML_NW_villages.gpkg": [TABLES / "ml_AHP_vs_AHP_comparison.csv", TABLES / "ml_NW_vs_NW_comparison.csv"],
}

LISA_OUTPUT_PATHS = {
    "AHP_vs_NW": GPKG / "map_lisa_AHP_vs_NW.gpkg",
    "AHP_vs_ML": GPKG / "map_lisa_AHP_vs_ML.gpkg",
    "NW_vs_ML": GPKG / "map_lisa_NW_vs_ML.gpkg",
    "MLAHP_vs_MLNW": GPKG / "map_lisa_MLAHP_vs_MLNW.gpkg",
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


def assert_map_is_fresh(path):
    """Stop the run if a comparison map is older than the tables it should have been built from.

    Guards against silently computing Moran's I / LISA / kappa from a
    leftover map from a previous pipeline run instead of the current one —
    see the note above MAP_SOURCES for the incident that motivated this.
    """
    sources = MAP_SOURCES.get(path, [])
    map_mtime = path.stat().st_mtime
    stale = [src for src in sources if src.exists() and src.stat().st_mtime > map_mtime]
    if stale:
        stale_list = ", ".join(f"{s.name} ({s.stat().st_mtime:.0f} > {map_mtime:.0f})" for s in stale)
        raise RuntimeError(
            f"STALE INPUT: {path.name} is older than its source(s): {stale_list}. "
            f"This means src/11_export_outputs.py has not been (re)run since those "
            f"tables last changed — run it before this script, or you will compute "
            f"Moran's I / LISA / kappa from a comparison map that does not match the "
            f"current pipeline state. (This is exactly how the AHP-vs-ML/NW-vs-ML "
            f"figures went wrong earlier in this remediation.)"
        )


def load_agreement_layer(path):
    """Load one comparison map, after checking it exists and is not stale."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run src/11_export_outputs.py first "
            "to build the agreement map layers."
        )
    assert_map_is_fresh(path)
    return gpd.read_file(path)


def build_queen_weights(gdf):
    """Define which settlements count as each other's spatial neighbours.

    Uses Queen contiguity: two settlement polygons are neighbours if they
    share so much as a single boundary point, not only a full edge (the
    stricter "Rook" contiguity rule). Queen is the more common default for
    irregular polygons like settlement boundaries, where a shared corner
    still represents genuine adjacency. The weights are row-standardised
    (each settlement's neighbour weights sum to 1), which is what Moran's I
    and LISA expect.
    """
    w = Queen.from_dataframe(gdf, use_index=False)
    w.transform = "r"
    return w


def global_morans_i(gdf, field, label):
    """Compute one global Moran's I statistic for a 0/1 agreement field across all settlements."""
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


LISA_QUAD_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}  # esda/Anselin's convention


def classify_lisa(local_moran, significance):
    """HH / LH / LL / HL quadrant labels, read directly from esda's own `q`
    attribute; 'not significant' below threshold.

    A previous version of this function rederived the quadrant by comparing
    each observation's value against `local_moran.y` as if that were the
    spatial lag. It is not — esda.Moran_Local.y is just the raw input y
    echoed back unchanged (verified directly: np.array_equal(lisa.y, y) is
    True). Comparing a value against itself makes the "opposite sides of the
    mean" condition for HL/LH mathematically impossible, so every previous
    LISA table/layer in this repository silently had HL=LH=0 for every
    comparison — not a real absence of spatial outliers, a classification
    bug. `local_moran.q` is esda's own already-correct quadrant code and is
    used directly here instead of being rederived.
    """
    labels = []
    for q, p in zip(local_moran.q, local_moran.p_sim):
        if p >= significance:
            labels.append("not significant")
        else:
            labels.append(LISA_QUAD_LABELS[q])
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
        cluster_type = classify_lisa(lisa, SIGNIFICANCE_LEVEL)

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
