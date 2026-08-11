"""
Load 2023 SURS commuting matrix, derive functional centres by
dominant flow, compare with Huff assignments, classify three
disagreement patterns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import geopandas as gpd
from sklearn.metrics import cohen_kappa_score

from config import DATA_RAW, TABLES, GPKG, COMMUTING_FILE, VILLAGES_FILE

MATRIX_TABLES = Path(r"C:\Users\lstojano\Desktop\teza\HuffMethodPaper\Data\Matrix and tables")

OBCINE_FILE = "obcine_poligoni.shp"


def build_commuting_centres(commuting_path):
    """For each municipality, find self-flow vs dominant external destination,
    determine centre status, and follow dominant-flow chains to the
    ultimate functional centre."""
    xl = pd.read_excel(commuting_path)

    self_flow = {}
    dominant_external = {}
    dominant_external_flow = {}
    is_centre = {}

    for sifra, g in xl.groupby("SIF_OBC_IZV"):
        sifra = int(sifra)
        self_row = g[g["SIF_OBC_PON"] == sifra]
        s_flow = float(self_row["DV_S_23"].sum()) if len(self_row) else 0.0

        ext = g[g["SIF_OBC_PON"] != sifra]
        if len(ext) and ext["DV_S_23"].max() > 0:
            dom_row = ext.loc[ext["DV_S_23"].idxmax()]
            dom_dest = int(dom_row["SIF_OBC_PON"])
            dom_flow = float(dom_row["DV_S_23"])
        else:
            dom_dest = None
            dom_flow = 0.0

        self_flow[sifra] = s_flow
        dominant_external[sifra] = dom_dest
        dominant_external_flow[sifra] = dom_flow
        is_centre[sifra] = s_flow >= dom_flow

    max_hops = len(self_flow)
    ultimate_centre = {}
    for sifra in self_flow:
        current = sifra
        visited = set()
        for _ in range(max_hops):
            if is_centre.get(current, True) or current in visited:
                break
            visited.add(current)
            nxt = dominant_external.get(current)
            if nxt is None:
                break
            current = nxt
        ultimate_centre[sifra] = current

    return pd.DataFrame({
        "SIFRA": list(self_flow.keys()),
        "commuting_self_flow": [self_flow[s] for s in self_flow],
        "commuting_dominant_external_dest": [dominant_external[s] for s in self_flow],
        "commuting_dominant_external_flow": [dominant_external_flow[s] for s in self_flow],
        "commuting_is_centre": [is_centre[s] for s in self_flow],
        "commuting_ultimate_centre": [ultimate_centre[s] for s in self_flow],
    })


def build_huff_majority(villages_path, obcine_path, huff_summary_path):
    """Spatially join villages to municipality polygons, then find the
    majority Huff dominant municipality per home municipality."""
    vp = gpd.read_file(villages_path)
    obc = gpd.read_file(obcine_path)
    huff_sum = pd.read_csv(huff_summary_path)

    joined = vp.sjoin(obc[["SIFRA", "NAZIV", "geometry"]], how="left", predicate="within")
    joined = joined.merge(
        huff_sum[["Village_ID", "dominant_municipality"]],
        left_on="NA_MID", right_on="Village_ID", how="left")

    majority_name = joined.groupby("SIFRA")["dominant_municipality"].agg(
        lambda s: s.value_counts().idxmax())

    name_to_sifra = obc.set_index("NAZIV")["SIFRA"].to_dict()
    majority_sifra = majority_name.map(name_to_sifra)

    return pd.DataFrame({
        "SIFRA": majority_name.index,
        "huff_majority_centre_name": majority_name.values,
        "huff_majority_centre": majority_sifra.values,
    })


def classify_pattern(row):
    huff_self = row["huff_majority_centre"] == row["SIFRA"]
    commuting_self = row["commuting_is_centre"]

    if huff_self and not commuting_self:
        return "Pattern 1: Huff self, Commuting external"
    if commuting_self and not huff_self:
        return "Pattern 2: Commuting self, Huff external"
    if not huff_self and not commuting_self and row["huff_majority_centre"] != row["commuting_ultimate_centre"]:
        return "Pattern 3: both external, different centre"
    return "Agree"


def main():
    print("=== COMMUTING COMPARISON ===")
    print()

    TABLES.mkdir(parents=True, exist_ok=True)
    GPKG.mkdir(parents=True, exist_ok=True)

    commuting_path = DATA_RAW / COMMUTING_FILE
    obcine_path = DATA_RAW / OBCINE_FILE
    villages_path = DATA_RAW / VILLAGES_FILE
    huff_summary_path = MATRIX_TABLES / "huff_summary.csv"

    print("Building commuting-derived functional centres...")
    commuting = build_commuting_centres(commuting_path)
    print(f"  Municipalities: {len(commuting)}")
    print(f"  Centres by self-flow: {commuting['commuting_is_centre'].sum()}")

    print("Building Huff majority centre per municipality...")
    huff_majority = build_huff_majority(villages_path, obcine_path, huff_summary_path)
    print(f"  Municipalities: {len(huff_majority)}")
    print()

    obc = gpd.read_file(obcine_path)
    base = obc[["SIFRA", "NAZIV", "geometry"]].copy()

    result = base.merge(commuting, on="SIFRA", how="left")
    result = result.merge(huff_majority, on="SIFRA", how="left")

    result["agreement"] = (
        result["huff_majority_centre"] == result["commuting_ultimate_centre"]).astype(int)
    result["pattern"] = result.apply(classify_pattern, axis=1)

    n_agree = int(result["agreement"].sum())
    n_total = len(result)
    agreement_pct = 100.0 * n_agree / n_total
    kappa = cohen_kappa_score(
        result["commuting_ultimate_centre"].astype(int),
        result["huff_majority_centre"].astype(int))

    print(f"Agreement: {n_agree}/{n_total} ({agreement_pct:.2f}%)  Cohen kappa={kappa:.4f}")
    print(result["pattern"].value_counts().to_string())
    print()

    # ── Save outputs ─────────────────────────────────────────
    table_cols = [
        "SIFRA", "NAZIV",
        "commuting_self_flow", "commuting_dominant_external_dest",
        "commuting_dominant_external_flow", "commuting_is_centre",
        "commuting_ultimate_centre",
        "huff_majority_centre_name", "huff_majority_centre",
        "agreement", "pattern",
    ]
    result[table_cols].to_csv(TABLES / "table_huff_vs_commuting.csv", index=False)
    print("Saved table_huff_vs_commuting.csv")

    summary_rows = [{
        "n_municipalities": n_total,
        "n_agree": n_agree,
        "agreement_pct": agreement_pct,
        "cohen_kappa": kappa,
    }]
    pattern_counts = result["pattern"].value_counts().to_dict()
    summary_rows[0].update({f"n_{k.split(':')[0].replace(' ', '_')}": v
                             for k, v in pattern_counts.items()})
    pd.DataFrame(summary_rows).to_csv(TABLES / "table_huff_vs_commuting_summary.csv", index=False)
    print("Saved table_huff_vs_commuting_summary.csv")

    result.to_file(GPKG / "fig_huff_vs_commuting_municipalities.gpkg", driver="GPKG")
    print("Saved fig_huff_vs_commuting_municipalities.gpkg")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
