# Reproducibility note

Written 2026-08-22, after Stage 2A (contamination fix) and Stage 2B (clean rerun) of the
data-drift remediation. This is meant to be adapted directly into the manuscript's data
availability statement.

## What DATA_RAW the pipeline actually reads

The pipeline reads exactly 94 files from `DATA_RAW`, all listed by name in version-controlled
repository files — nothing is discovered by globbing a directory anymore:

- **8 core files**, named as constants in `config.py` and audited in `13_data_audit.py`'s
  `CORE_FILES` list: `gis_osm_roads_free_1.shp` (road network), `Municipalities_All_Groups_
  Weighted_AHP.gpkg`, `Municipalities_All_Groups_NotWeighted_Normalized.gpkg`,
  `Municipalities_Points_normalized.gpkg`, `Villages_points_real.shp`, `NA.shp`,
  `obcine_poligoni.shp`, `2023tabela.xlsx`.
- **86 facility layers**, named explicitly in `data/facility_layers.txt` (one per line),
  resolved by `05_accessibility.py::discover_facility_layers`, which raises `FileNotFoundError`
  if any listed layer is missing rather than silently discovering a different set.

`DATA_RAW` itself contains 869 files; only these 94 are ever opened by any script. A file
dropped into `DATA_RAW` under any other name is inert until someone deliberately adds it to
`data/facility_layers.txt` or a `config.py` constant — this is the direct fix for how
`all_roads.gpkg` entered the analysis in the first place (see the Stage 1/2A history below).

## Raw inputs are now frozen and checked

`data/raw_manifest.json` records the SHA256, size, and mtime of all 94 files above, as they
stood at the end of this remediation. `config.py` — imported by every script — checks the
current `DATA_RAW` against this manifest on every run and prints a loud, impossible-to-miss
warning naming exactly which file changed or went missing if anything drifts. This is the
direct fix for how the contamination happened in the first place: `all_roads.gpkg` changed
silently and nothing noticed for three weeks. The check costs well under a second (verified
by direct measurement) and never raises — a missing manifest or unreachable `DATA_RAW`
degrades to a warning, not a crash, so it can't block anyone from running the pipeline.

## No path outside the repository

Every script in `src/` was audited for absolute paths and drive letters. The only one
remaining is `config.py`'s `DATA_RAW` constant itself, which is documented as
machine-specific by design ("edit this path when moving machines") — that is the single,
intentional seam between the repository and the raw data staging directory, not a hidden
dependency.

Three other absolute-path dependencies were found and removed during Stage 2A:

1. `06_ml_framework.py`, `07_beta_sensitivity.py`, `09_entropy_uncertainty.py`,
   `11_commuting_comparison.py`, `13_data_audit.py`, `16_shap_dependence.py` all read
   `accessibility_normalized.csv`, `huff_od_matrix.csv`, `huff_NW_od_matrix.csv`,
   `huff_summary.csv` and `huff_NW_summary.csv` from an external `Matrix and tables`
   directory that no script in the repository produced. `03_huff_ahp.py` and
   `04_huff_nonweighted.py` now also save the full OD probability matrix
   (`huff_od_matrix.csv` / `huff_NW_od_matrix.csv`) to `outputs/tables/`, and all six
   scripts above were repointed at the repository's own `outputs/tables/`.
2. `12_export_outputs.py` copied `table1`–`table4` in from a second external directory
   (`Data/tables and charts`) instead of computing them. `table2`/`table3` (top-20 GI
   ranking) and `table4` (top-15 catchment sizes) are pure derivations with no ambiguity —
   now computed directly from the municipality GI layers and the fresh Huff summaries.
   `table1` (AHP group priority weights, a pairwise-comparison judgment) and `tableS1`
   (indicator source citations) are genuine inputs, not data-derived results, so those are
   copied into `data/external/` — committed to the repository — instead of read from an
   absolute path.

`grep -rn "C:\\\\Users\|HuffMethodPaper" src/*.py` (excluding `config.py`'s documented
`DATA_RAW`) now returns nothing.

## No silent stale-output reuse

`03_huff_ahp.py`, `04_huff_nonweighted.py`, and `05_accessibility.py` are the three scripts
that skip their full computation when their output file already exists — this is exactly
the mechanism that let this repository ship manuscript-matching numbers for three weeks
without ever running against the current road network. All three now take a `--force` flag;
without it, the skip prints a `WARNING: reusing existing <file>` line naming exactly which
file it's reusing, rather than a neutral status line a reader could miss.

## Command sequence that reproduces every number in the paper

```
# One-time, from a clean checkout with DATA_RAW pointed at the raw data directory:
python src/01_gi_construction.py
python src/02_road_network.py
python src/03_huff_ahp.py --force
python src/04_huff_nonweighted.py --force
python src/05_accessibility.py --force
python src/06_ml_framework.py --model both --sample-frac 1.0
python src/07_beta_sensitivity.py
python src/08_euclidean_comparison.py
python src/09_entropy_uncertainty.py
python src/10_morans_i.py
python src/11_commuting_comparison.py
python src/12_export_outputs.py
python src/13_data_audit.py
python src/14_disagreement_synthesis.py
python src/15_ml_catchment_structure.py
python src/16_shap_dependence.py
python src/17_spatial_layers.py
python src/18_map_symbology_fields.py
python src/19_output_manifest.py
```

`--force` is only meaningful for 03/04/05 (the three with skip-if-exists behavior); it's a
no-op if passed to a script without that flag defined, so it's omitted above where it isn't
needed. `01_gi_construction.py` has always run unconditionally (no skip-if-exists guard) and
was confirmed live in this remediation: 100 indicators mapped, all 10 group counts correct,
computed rarity weights matching `tableS3`'s reference values to 1e-6 — none of it is copied
from a static file.

Random seeds are fixed and printed at the point of use: `random_state=42` for the spatial
KMeans blocks and both Random Forest models (`06_ml_framework.py`), `seed=42` for both the
global permutation test and the LISA computation in `10_morans_i.py`, `999` join-count
permutations. Re-running the sequence above on unchanged inputs reproduces every value in
this rerun's tables exactly, including the two residual discrepancies noted below (they are
properties of the input data, not of any randomness in the pipeline).

`ml_AHP_cv_results.csv` and `ml_NW_cv_results.csv` now carry a `wall_time_s` column and an
automatic `wall_time_note` flag (added in `06_ml_framework.py::annotate_wall_time_anomalies`)
for any fold whose wall-clock time is more than 3x the median of the other folds — Fold 1 of
the AHP model in this run logged 65,198s against ~900-1,100s for every other fold, which is
flagged there as a wall-clock artifact (the machine sleeping mid-fold during a long
background wait in this session), not a real compute cost. R²/MAE/RMSE are unaffected.

A separate one-off robustness check, `outputs/audit/imputation_sensitivity_check.md`, reruns
the AHP and NW Huff assignment with unreachable OD pairs given zero probability instead of
the pipeline's default column-max-distance fill. On the current network (199 missing pairs)
this changes zero settlement assignments — the pipeline default is not currently
consequential — but it is *not* a retroactive test of the manuscript's own network, which had
roughly ten times the missing-pair rate. The pipeline's default imputation method is
unchanged; this was a diagnostic only.

## Two residual discrepancies, honestly

Two things did not resolve even after fixing the contamination, and neither should be
papered over.

**A 262 km gap in total road length.** The fixed network reproduces the manuscript's road
segment count (254,252), node count (394,874 / 390,273 in the largest component), and
municipality/settlement snapping distances (39.48 m / 297.50 m and 58.33 m / 1,470.52 m)
*exactly* — to the centimeter, in the snapping case. But the largest component's total
length comes to 54,431.8 km against the manuscript's stated 54,694 km. Three targeted
diagnostics ruled out the obvious explanations: the edge-filter definition (both-endpoints
vs. any-endpoint-in-component) is mathematically identical for connected components, so
that was never viable; no round-number cutoff between 250 and 300 km reproduces the
manuscript's separate 2,107-missing-pair OD figure (closest is 2,860 at 280 km); and testing
parallel/duplicate edges between the same node pairs found 3,541 duplicate node-pairs
accounting for 244.5–432.9 km depending on which edge in each pair is assumed kept, in the
neighborhood of 262 km but not an exact match either. The cause is not identified. Given
that every other network metric reproduces exactly, this is very likely a small difference
in the specific OSM edit state or noding tolerance behind the manuscript's original network
versus `gis_osm_roads_free_1.shp` — not a defect in this pipeline's logic.

**Six villages move off Ljubljana under both AHP and NW weighting.** Ljubljana's rerun
catchment is 1,216 (AHP) and 1,016 (NW) against the manuscript's 1,222 and 1,022 — short by
exactly six villages in both, independently. Tracing them by name (Konjščica-del, Nova vas,
Vrtača, Gora, Sadni Hrib, Drešnik) shows all six shared the *identical* manuscript distance
value to Ljubljana (163,843.9 m) — the column-max fill value, meaning none of them were
actually reachable to Ljubljana within cutoff in the manuscript's network. They were
assigned to Ljubljana anyway because Ljubljana's GI still won the gravity-model argmax even
at that penalized fill distance. In the fixed, better-connected network they are genuinely
reachable at short real distances (2.9–19.2 km) and correctly lose to their true nearest
small neighbor (Zagorje ob Savi, Sežana, Brežice, Cerknica, Kočevje ×2). This is very likely
the rerun being *more* correct than the manuscript, not a regression — it is the same class
of network-connectivity artifact as the OD-matrix and accessibility gaps above, just visible
at the level of individual settlements instead of aggregate counts. Every other catchment in
the top 15 under both weighting schemes matches the manuscript exactly or within one
settlement (Brežice, off by one in both scenarios).

Both discrepancies plausibly share a root cause — the manuscript's original network was very
likely not bit-identical to `gis_osm_roads_free_1.shp`, despite the latter reproducing the
manuscript's segment count and snapping distances exactly. That tension is reported here
rather than resolved, per instruction not to adjust cutoffs or imputation logic to force a
match.
