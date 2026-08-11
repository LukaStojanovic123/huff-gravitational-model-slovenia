"""
Rerun Huff with beta 1.5 2.0 2.5 3.0, compute agreement and Cohen
kappa vs beta=2, save table and figure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

from config import DATA_RAW, TABLES, FIGURES, BETA, MUNICIPALITIES_AHP

MATRIX_TABLES = Path(r"C:\Users\lstojano\Desktop\teza\HuffMethodPaper\Data\Matrix and tables")

BETAS = [1.5, 2.0, 2.5, 3.0]


def load_gi_ahp(munis_ahp_path):
    """Load GI_AHP per municipality, indexed by municipality name."""
    munis_ahp = gpd.read_file(munis_ahp_path)
    return munis_ahp.set_index("Muni_Name")["GI_AHP"]


def load_distance_matrix(od_path):
    """Extract dist_ columns from the OD matrix as a float32 array.

    Returns (village_ids, village_names, muni_names, dist_array) where
    dist_array has shape (n_villages, n_munis), columns ordered to match
    muni_names.
    """
    od = pd.read_csv(od_path)
    dist_cols = [c for c in od.columns if c.startswith("dist_")]
    muni_names = [c.replace("dist_", "") for c in dist_cols]
    dist_array = od[dist_cols].values.astype(np.float32)
    return od["Village_ID"].values, od["Village_Name"].values, muni_names, dist_array


def compute_huff_dominant(gi_series, muni_names, dist_array, beta):
    """Compute Huff Pij for a given beta, return dominant muni index per village.

    Villages with a zero-distance municipality (village is the municipal seat)
    are assigned to that municipality directly, since GI / dist**beta is
    undefined at dist=0.
    """
    gi = gi_series.reindex(muni_names).values.astype(np.float32)

    zero_mask = dist_array == 0
    row_has_zero = zero_mask.any(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        attract = gi[np.newaxis, :] / np.power(dist_array, beta)
    attract = np.where(zero_mask, 0.0, attract)
    attract = np.nan_to_num(attract, nan=0.0, posinf=0.0, neginf=0.0)

    pij = np.zeros_like(attract)

    nz_idx = ~row_has_zero
    row_sums = attract[nz_idx].sum(axis=1, keepdims=True)
    pij[nz_idx] = attract[nz_idx] / row_sums

    pij[row_has_zero] = zero_mask[row_has_zero].astype(np.float32)

    dominant_idx = pij.argmax(axis=1)
    return dominant_idx


def main():
    print("=== BETA SENSITIVITY ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    munis_ahp_path = DATA_RAW / MUNICIPALITIES_AHP
    od_path = MATRIX_TABLES / "huff_od_matrix.csv"
    summary_path = MATRIX_TABLES / "huff_summary.csv"

    print("Loading GI_AHP...")
    gi_series = load_gi_ahp(munis_ahp_path)
    print(f"  Municipalities: {len(gi_series)}")

    print("Loading distance matrix...")
    village_ids, village_names, muni_names, dist_array = load_distance_matrix(od_path)
    print(f"  Distance matrix: {dist_array.shape}")

    print(f"Loading beta={BETA} baseline (huff_summary.csv)...")
    baseline = pd.read_csv(summary_path)
    baseline = baseline.set_index("Village_ID").reindex(village_ids)
    baseline_dominant_name = baseline["dominant_municipality"].values
    print()

    rows = []
    for beta in BETAS:
        print(f"Computing Huff assignments for beta={beta}...")
        dominant_idx = compute_huff_dominant(gi_series, muni_names, dist_array, beta)
        dominant_name = np.array(muni_names)[dominant_idx]

        agreement_mask = dominant_name == baseline_dominant_name
        n_agree = int(agreement_mask.sum())
        n_total = len(agreement_mask)
        agreement_pct = 100.0 * n_agree / n_total
        kappa = cohen_kappa_score(baseline_dominant_name, dominant_name)

        print(f"  Agreement vs beta={BETA}: {n_agree}/{n_total} ({agreement_pct:.2f}%)  "
              f"Cohen kappa={kappa:.4f}")

        rows.append({
            "beta": beta,
            "n_villages": n_total,
            "n_agree": n_agree,
            "agreement_pct": agreement_pct,
            "cohen_kappa": kappa,
        })
    print()

    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "table_beta_sensitivity_clean.csv", index=False)
    print(f"Saved table_beta_sensitivity_clean.csv ({len(results)} rows)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(results["beta"], results["agreement_pct"], marker="o", color="tab:blue")
    ax1.axvline(BETA, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Beta")
    ax1.set_ylabel("Agreement (%)")
    ax1.set_title(f"Dominant-municipality agreement vs beta={BETA}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(results["beta"], results["cohen_kappa"], marker="o", color="tab:orange")
    ax2.axvline(BETA, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Beta")
    ax2.set_ylabel("Cohen's kappa")
    ax2.set_title(f"Cohen's kappa vs beta={BETA}")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_beta_sensitivity.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_beta_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_beta_sensitivity.png / .pdf")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
