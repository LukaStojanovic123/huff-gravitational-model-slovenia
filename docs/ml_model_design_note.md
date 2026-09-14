# ML model design note

Two Random Forest models are trained by `06_ml_framework.py`, not one. This note exists
because that fact isn't obvious from the comparison maps or tables alone, and a reader could
easily assume a single ML model runs throughout Section 4's comparisons.

## Corrected 2026-09-09: each model now uses its own composite GI score

**This section documents a fix, not just the current design — the previous design was wrong
and produced a model that was not the independent, expert-free counterpart the manuscript
claims it is.**

Until this correction, `build_municipality_features()` merged in exactly one composite GI
column — `GI_AHP`, from `MUNICIPALITIES_AHP` — and this single feature table was reused,
unchanged, for **both** models. `MUNICIPALITIES_NW` was imported into `06_ml_framework.py`
from the very first commit that created the file but was never actually used anywhere in it.
The result: Model 2 (NW-target) was trained to predict the *non-weighted* Huff model's `Pij`
values while being handed the *AHP-weighted* composite score as one of its 189 input
features. It never saw `GI_Final_NotWeighted` at all. Traced through the full git history of
`06_ml_framework.py` (`d7ed9db` → `3aba854` → `ac8ee65` → `19befb1` → `016cdc9`): this was
never a case of a feature being included then later removed — `GI_Final_NotWeighted` was
never a feature at any point. The bug was introduced at Model 2's creation (`3aba854`, "Model
2 NW complete"), which reused Model 1's feature-builder unchanged and even *hoisted* the
shared feature-column list to make the sharing more explicit ("this list is valid regardless
of which model(s) run") — a deliberate design choice, just the wrong one, never revisited.

**Why this mattered:** Model 2 is presented in the manuscript as an independent,
non-expert-weighted counterpart to Model 1 — the whole point of training two RF models
instead of one is to see whether a data-driven model trained on the non-weighted Huff
baseline reproduces similar structure without ever seeing the AHP panel's expert judgment.
Leaking `GI_AHP` into Model 2's inputs undermines exactly that comparison: any apparent
agreement between the two RF models could partly reflect a shared input, not independent
convergence on the same catchment structure.

**The fix:** `build_municipality_features()` now takes `composite_path`, `composite_col`,
and `output_col` parameters. Model 1 still merges `GI_AHP` from `MUNICIPALITIES_AHP` (output
column stays `GI_AHP`). Model 2 now merges `GI_Final_NotWeighted` from `MUNICIPALITIES_NW`,
stored under the output column name `GI_NW` — deliberately renamed so it's never confused
with `GI_AHP` downstream (feature importance tables, the AHP-vs-NW-target comparison figure).
`06_ml_framework.py`'s `main()` now builds two separate `munis_features`/`all_feature_cols`
tables, one per model, instead of one shared table. Feature *count* is unchanged at 189 for
both models — only which composite column occupies one of those 189 slots differs.

`MUNICIPALITIES_NW`'s composite column needed disambiguating before use: the file also
carries a near-duplicate `GI_Final_Not_Weighted` column (note the extra underscore) that is
exactly 10x the values of `GI_Final_NotWeighted` (e.g. Ljubljana: 1.0 vs. 10.0) — almost
certainly a raw pre-normalization artifact in the source data, not itself a bug in this
pipeline. Verified directly against every place this composite already appears downstream —
`GI_full_212_municipalities.csv` (`GI_Final_NotWeighted`, exact float match to 1e-17),
`table2_top20_GI_NotWeighted.csv` (Ljubljana = 1.0), `fig03_GI_NW_municipalities.gpkg`
(Ljubljana = 1.0) — all three agree unambiguously on `GI_Final_NotWeighted`, not the
underscored duplicate. `GI_Final_Not_Weighted` is otherwise unused anywhere in this
repository.

See `outputs/audit/nw_refit_comparison.csv` for the full before/after comparison across R²,
feature importance, agreement/kappa, Moran's I, disagreement synthesis, and catchment sizes.
Two things could not be refreshed this session due to unrelated native-library crashes in
this environment (`esda.Moran_Local` for LISA, `scipy.linalg.inv()`/`shap` import for SHAP) —
both reproduced in isolation on synthetic data with no repository content involved, so they
are environment fragility, not a consequence of this fix. LISA class counts and SHAP figures
on disk still reflect the pre-refit model as of this note.

## Two models, two targets

The two models share the same 189-feature *count* and the same hyperparameters
(`RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=10, random_state=42)`,
5-fold spatial cross-validation on KMeans municipality blocks). They differ in both their
target and — as of the correction above — their composite GI feature:

- **Model 1 (AHP-target)**: trained to predict `Pij` from the AHP-weighted Huff model's
  village-to-municipality probability matrix (`huff_od_matrix.csv`), using `GI_AHP` as its
  composite feature. Cross-validated R² = 0.846 ± 0.086 (unchanged by this fix — Model 1's
  inputs never changed).
- **Model 2 (NW-target)**: trained to predict `Pij` from the non-weighted Huff model's
  probability matrix (`huff_NW_od_matrix.csv`), now using `GI_NW` (`GI_Final_NotWeighted`)
  as its composite feature instead of `GI_AHP`. Cross-validated R² = 0.855 ± 0.060 — actually
  *higher* than before the fix (was 0.840 ± 0.064), consistent with the model now training on
  a composite feature that's actually correlated with the target it's predicting.

Each village's out-of-fold prediction (from whichever model) is turned into a dominant
municipality the same way the Huff models are: the municipality with the highest predicted
`Pij` per village.

## A surprising result: RF(AHP) vs RF(NW) agreement barely moved

The expectation going into this fix was that Model 1 vs. Model 2 agreement
(`map_ML_AHP_vs_ML_NW_villages.gpkg`, `table_three_way_agreement.csv`'s `MLAHP_vs_MLNW` row)
would drop once the two models no longer shared a literal input column. It didn't: 80.78%
before (4,876/6,036, κ=0.8047) vs. 80.86% after (4,881/6,036, κ=0.8053) — a +5-settlement,
+0.08-point difference, well within noise. The most likely explanation: `dist_to_muni`
dominates both models' feature importance overwhelmingly (~63-69%, see
`table_feature_importance_comparison.csv`), with the composite score a distant second
(22.7% for AHP, now 18.0% for NW). Since geography drives the dominant-municipality argmax
in both models regardless of which composite backs it up, removing the shared `GI_AHP` input
didn't meaningfully change where the two models' predictions land. The two RF models were
already converging mostly on distance, not on the leaked composite score — so the leak
inflated the *feature importance table's* apparent shared reliance on `GI_AHP`, without
being the actual driver of their agreement rate. What NW vs ML agreement (the model against
its own true target) reveals more clearly: that jumped from 75.5% to 79.0%, because Model 2
now genuinely fits the target it's supposed to.

## Which comparison layer each model feeds

`11_export_outputs.py::export_agreement_maps` builds three headline comparison layers:

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
architecture and hyperparameters but not target variable — and, as of this correction, not
their composite GI feature either. Model 1 uses `GI_AHP`, Model 2 uses `GI_Final_NotWeighted`
(the non-weighted composite), each matched to the Huff variant it predicts. Any figure or
table comparing "the ML model" against both Huff variants is implicitly comparing against two
different fitted models, not one model evaluated twice. The fourth comparison (Model 1 vs.
Model 2 directly) answers the natural follow-up question — do the two independently trained
models converge on the same catchment structure — and its own agreement/kappa/Moran's I
figures are in `table_three_way_agreement.csv`, alongside the original three. That the two
models still agree at essentially the same rate (~81%) after removing their one shared input
is itself worth a sentence: it suggests the agreement is driven by both models converging on
distance as the dominant factor, not by a shared composite score.
