# ML model design note

Two Random Forest models are trained by `06_ml_framework.py`, not one. This note exists
because that fact isn't obvious from the comparison maps or tables alone, and a reader could
easily assume a single ML model runs throughout Section 4's comparisons.

## Two models, two targets

Both models use the same 189-feature set (101 raw GI indicators, 86 accessibility measures,
GI_AHP, distance-to-municipality) and the same hyperparameters
(`RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=10, random_state=42)`,
5-fold spatial cross-validation on KMeans municipality blocks). They differ only in what they
are trained to predict:

- **Model 1 (AHP-target)**: trained to predict `Pij` from the AHP-weighted Huff model's
  village-to-municipality probability matrix (`huff_od_matrix.csv`). Cross-validated
  R² = 0.846 ± 0.086.
- **Model 2 (NW-target)**: trained to predict `Pij` from the non-weighted Huff model's
  probability matrix (`huff_NW_od_matrix.csv`). Cross-validated R² = 0.840 ± 0.064.

Each village's out-of-fold prediction (from whichever model) is turned into a dominant
municipality the same way the Huff models are: the municipality with the highest predicted
`Pij` per village.

## Which comparison layer each model feeds

`12_export_outputs.py::export_agreement_maps` builds three headline comparison layers:

| Layer | Compares | ML model used |
|---|---|---|
| `map_AHP_vs_NW_villages.gpkg` | AHP Huff vs. NW Huff | neither — both sides are Huff outputs |
| `map_AHP_vs_ML_villages.gpkg` | AHP Huff vs. ML | **Model 1** (AHP-target) |
| `map_NW_vs_ML_villages.gpkg` | NW Huff vs. ML | **Model 2** (NW-target) |

This stage adds a fourth: `map_ML_AHP_vs_ML_NW_villages.gpkg`, comparing Model 1's and
Model 2's dominant-municipality predictions directly — the one pairing the original three
layers never covered, since it doesn't involve either Huff model.

## Why `n_disagree` counts against more than one model

`14_disagreement_synthesis.py` joins all four comparison layers on `Village_ID` and counts,
per settlement, how many of the pairwise comparisons disagree (`n_disagree`, 0 to 4). Because
the "ML" column in `map_AHP_vs_ML` and the "ML" column in `map_NW_vs_ML` come from two
*separately trained* models — not the same model's predictions reused twice — a settlement
can perfectly plausibly agree with one ML model and disagree with the other while both Huff
models agree with each other. This is not an artifact of the join; it's the direct
consequence of comparing four distinct classifications (AHP, NW, ML-on-AHP-target,
ML-on-NW-target) pairwise, with only some of the six possible pairs actually computed. A
reader assuming a single ML model throughout would find `n_disagree` values inconsistent
with a simple three-way Venn diagram — they aren't; they're consistent with four
classifications, four of six pairs compared before this stage, all six after it.

## What to write in the manuscript's Methods section

State plainly: two Random Forest models were trained, one per Huff weighting scheme, sharing
architecture and hyperparameters but not target variable. Any figure or table comparing "the
ML model" against both Huff variants is implicitly comparing against two different fitted
models, not one model evaluated twice. The fourth comparison (Model 1 vs. Model 2 directly)
answers the natural follow-up question — do the two independently trained models converge on
the same catchment structure — and its own agreement/kappa/Moran's I figures are in
`table_three_way_agreement.csv`, alongside the original three.
