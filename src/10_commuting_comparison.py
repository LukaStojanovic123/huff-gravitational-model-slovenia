"""
Step 10 of the pipeline: does the Huff model's catchment structure agree
with how people actually commute?

What this script does: an independent, real-world check on the Huff
model's catchments, using Statistics Slovenia's (SURS) 2023 inter-
municipality commuting matrix instead of anything the Huff model itself
produces. For each municipality, it compares how many residents commute
to jobs within their own municipality against how many commute to any
single other municipality, and calls a municipality "self-contained" if
staying home wins. A municipality that is not self-contained is chained
forward through its own dominant external destination, repeatedly, until
it reaches a municipality that is self-contained — its real-world
"functional centre." That is then compared against the municipality most
of its own settlements are Huff-assigned to (the "Huff majority centre"),
and every municipality is classified into one of three disagreement
patterns, or "Agree" (see classify_pattern below for the exact rule).

Reads: 2023tabela.xlsx (the SURS commuting matrix, config.py's
COMMUTING_FILE), obcine_poligoni.shp (municipality polygons),
Villages_points_real.shp, and huff_AHP_summary.csv (already computed by
03_huff_ahp.py — this comparison uses the AHP Huff model only, not NW).

Writes: table_huff_vs_commuting.csv, table_huff_vs_commuting_summary.csv,
table7_commuting_comparison.csv (the manuscript-facing reshaping of the
summary above, with pattern percentages and the self-containment
breakdown by each definition individually and by both together), and
fig_huff_vs_commuting_municipalities.gpkg.

Runs tenth. Needs 03_huff_ahp.py's output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import geopandas as gpd
from sklearn.metrics import cohen_kappa_score

from config import DATA_RAW, TABLES, GPKG, COMMUTING_FILE, VILLAGES_FILE, EPSG
from crs_utils import ensure_crs

OUTPUT_FILES = [
    "tables/table_huff_vs_commuting.csv",
    "tables/table_huff_vs_commuting_summary.csv",
    "tables/table7_commuting_comparison.csv",
    "gpkg/fig_huff_vs_commuting_municipalities.gpkg",
]

OBCINE_FILE = "obcine_poligoni.shp"


def build_commuting_centres(commuting_path):
    """Derive each municipality's real-world functional centre from commuting flows.

    A municipality "is a centre" if at least as many of its residents work
    within the municipality itself as commute out to any single other
    municipality. A municipality that is not a centre is chased forward
    through its own largest outbound commuting flow, one hop at a time,
    until it reaches a municipality that is a centre — that is its
    "ultimate centre." The chase stops early (rather than looping forever)
    if it revisits a municipality it has already passed through.
    """
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
    """Find each municipality's most common Huff-assigned destination among its own settlements.

    Every settlement already has a Huff dominant municipality
    (huff_AHP_summary.csv). This groups settlements by which municipality
    they physically sit in (a spatial join to the municipality polygons)
    and takes the most frequent Huff destination within each group — the
    municipality-level analogue of the commuting comparison above, so the
    two can be compared on equal footing.
    """
    vp = gpd.read_file(villages_path)
    vp = ensure_crs(vp, EPSG, label=villages_path.name)
    obc = gpd.read_file(obcine_path)
    obc = ensure_crs(obc, EPSG, label=obcine_path.name)
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
    """Classify one municipality's disagreement type between Huff and commuting.

    Pattern 1 ("Huff self, Commuting external"): the Huff model treats this
    municipality as its own centre, but commuting data shows most of its
    workers actually leave for another municipality.
    Pattern 2 ("Commuting self, Huff external"): the reverse — residents
    are mostly self-contained by commuting, but the Huff model sends most
    of the municipality's settlements elsewhere.
    Pattern 3 ("both external, different centre"): neither source treats
    the municipality as self-contained, and they disagree on which outside
    municipality it belongs to.
    Anything else, by elimination, is "Agree."
    """
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
    huff_summary_path = TABLES / "huff_AHP_summary.csv"

    print("Building commuting-derived functional centres...")
    commuting = build_commuting_centres(commuting_path)
    print(f"  Municipalities: {len(commuting)}")
    print(f"  Centres by self-flow: {commuting['commuting_is_centre'].sum()}")

    print("Building Huff majority centre per municipality...")
    huff_majority = build_huff_majority(villages_path, obcine_path, huff_summary_path)
    print(f"  Municipalities: {len(huff_majority)}")
    print()

    obc = gpd.read_file(obcine_path)
    obc = ensure_crs(obc, EPSG, label=obcine_path.name)
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

    # Table 7: the same summary as above, reshaped for the manuscript
    # (percentages instead of raw pattern-name columns) plus the
    # self-containment breakdown by each definition individually and by
    # both together — this used to only be obtainable by re-deriving
    # huff_majority_centre == SIFRA from table_huff_vs_commuting.csv by
    # hand, since classify_pattern() only ever used it internally.
    huff_self = result["huff_majority_centre"] == result["SIFRA"]
    commuting_self = result["commuting_is_centre"].astype(bool)
    n_p1 = int((result["pattern"] == "Pattern 1: Huff self, Commuting external").sum())
    n_p2 = int((result["pattern"] == "Pattern 2: Commuting self, Huff external").sum())
    n_p3 = int((result["pattern"] == "Pattern 3: both external, different centre").sum())
    table7_rows = [
        {"category": "Agreement", "n_municipalities": n_agree, "share_pct": round(agreement_pct, 2)},
        {"category": "Pattern 1: Huff self-contained, Commuting external",
         "n_municipalities": n_p1, "share_pct": round(100 * n_p1 / n_total, 2)},
        {"category": "Pattern 2: Commuting self-contained, Huff external",
         "n_municipalities": n_p2, "share_pct": round(100 * n_p2 / n_total, 2)},
        {"category": "Pattern 3: Both external, different centre",
         "n_municipalities": n_p3, "share_pct": round(100 * n_p3 / n_total, 2)},
        {"category": "Total", "n_municipalities": n_total, "share_pct": 100.0},
        {"category": "Cohen's kappa", "n_municipalities": round(kappa, 4), "share_pct": None},
        {"category": "Commuting-based self-contained municipalities",
         "n_municipalities": int(commuting_self.sum()), "share_pct": None},
        {"category": "Huff-based self-contained municipalities",
         "n_municipalities": int(huff_self.sum()), "share_pct": None},
        {"category": "Self-contained under both definitions",
         "n_municipalities": int((huff_self & commuting_self).sum()), "share_pct": None},
    ]
    table7 = pd.DataFrame({"category": [r["category"] for r in table7_rows]})
    # Built as an explicit object Series, not through normal DataFrame
    # construction: the "Cohen's kappa" row shares this column with every
    # other row's integer municipality count, and letting pandas infer the
    # column's dtype from all the values together would upcast every
    # integer to float (146 -> 146.0) just because one row needs decimal
    # precision.
    table7["Number of municipalities"] = pd.Series(
        [r["n_municipalities"] for r in table7_rows], dtype=object)
    table7["Share (%)"] = [r["share_pct"] for r in table7_rows]
    table7.to_csv(TABLES / "table7_commuting_comparison.csv", index=False)
    print("Saved table7_commuting_comparison.csv")

    result.to_file(GPKG / "fig_huff_vs_commuting_municipalities.gpkg", driver="GPKG")
    print("Saved fig_huff_vs_commuting_municipalities.gpkg")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
