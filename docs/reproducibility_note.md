# Reproducibility note

Written 2026-08-22, after Stage 2A (contamination fix) and Stage 2B (clean rerun) of the
data-drift remediation. This is meant to be adapted directly into the manuscript's data
availability statement.

## What DATA_RAW the pipeline actually reads

The pipeline reads exactly 94 files from `DATA_RAW`, all listed by name in version-controlled
repository files — nothing is discovered by globbing a directory anymore:

- **8 core files**, named as constants in `config.py` and audited in `17_data_audit.py`'s
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
   `10_commuting_comparison.py`, `17_data_audit.py`, `15_shap_dependence.py` all read
   `accessibility_normalized.csv`, `huff_od_matrix.csv`, `huff_NW_od_matrix.csv`,
   `huff_summary.csv` and `huff_NW_summary.csv` from an external `Matrix and tables`
   directory that no script in the repository produced. `03_huff_ahp.py` and
   `04_huff_nonweighted.py` now also save the full OD probability matrix
   (`huff_od_matrix.csv` / `huff_NW_od_matrix.csv`) to `outputs/tables/`, and all six
   scripts above were repointed at the repository's own `outputs/tables/`.
2. `11_export_outputs.py` copied `table1`–`table4` in from a second external directory
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
python src/06_ml_framework.py --model AHP --sample-frac 1.0
python src/06_ml_framework.py --model NW --sample-frac 1.0
python src/07_beta_sensitivity.py
python src/08_euclidean_comparison.py
python src/09_entropy_uncertainty.py
python src/10_commuting_comparison.py
python src/11_export_outputs.py
python src/12_morans_lisa.py
python src/13_disagreement_synthesis.py
python src/14_ml_catchment_structure.py
python src/15_shap_dependence.py --model AHP
python src/15_shap_dependence.py --model NW
python src/16_spatial_layers.py
python src/17_data_audit.py
python src/18_output_manifest.py
```

`--force` is only meaningful for 03/04/05 (the three with skip-if-exists behavior); it's a
no-op if passed to a script without that flag defined, so it's omitted above where it isn't
needed. `01_gi_construction.py` has always run unconditionally (no skip-if-exists guard) and
was confirmed live in this remediation: 100 indicators mapped, all 10 group counts correct.

`tableS3_individual_indicator_weights.csv` used to be a hand-maintained file — someone ran
the rarity-weight computation once and pasted the result into `outputs/supplementary/`, with
nothing to ever regenerate it afterward. That is the same failure mode that let tableS1
silently revert to a pre-correction version (see the BLAS/environment section above's sibling
incident, and the fix commit for the full story): a value living in `outputs/` with no script
that owns it can drift from whatever actually produced it. `01_gi_construction.py` now
computes tableS3 itself, every run, from `Municipalities_Points_normalized.gpkg` and a small
genuine input (`data/external/indicator_categories.csv`, the indicator name/group
categorisation — the one part of tableS3 that really is a judgment call, not a derived
number). `17_data_audit.py` independently re-derives the same rarity weights from the same
two inputs, using its own separately written implementation of the formula, and cross-checks
the result against what `01_gi_construction.py` actually wrote — the same "rebuild it from
scratch and compare" principle already applied to the road network, OD matrix and
accessibility scores elsewhere in that audit.

`06_ml_framework.py` and `15_shap_dependence.py` are both run above as two separate
invocations (`--model AHP` then `--model NW`), not the single `--model both` each script also
supports. Confirmed by direct incident, twice, during this remediation's own Stage 2.4
pipeline verification: `--model both` on a 16 GB machine holds two 1,279,632-row feature
tables and two 100-tree Random Forests in memory across the same process, which exhausted
available RAM and got the process killed by the operating system mid-run, with no Python
traceback at all — the failure was silent enough that it looked at first like the script had
simply hung. It happened first in `06_ml_framework.py`, and then again in
`15_shap_dependence.py` once that script's turn in the pipeline came around, since it
retrains its own pair of models independently of `06`'s and was never covered by `06`'s fix.
Both scripts now refuse to start a second instance of themselves while one is already running
(separate lock files in `data/processed/`), check the lock's recorded PID against the running
process list on every acquire and clear it automatically if that process is no longer alive —
so a kill like the one that caused this in the first place does not leave a permanent lock
behind for someone to find and delete by hand — and report each model's own peak memory at
the end of its run.

Random seeds are fixed and printed at the point of use: `random_state=42` for the spatial
KMeans blocks and both Random Forest models (`06_ml_framework.py`), `seed=42` for both the
global permutation test and the LISA computation in `12_morans_lisa.py`, `999` join-count
permutations. Re-running the sequence above on unchanged inputs reproduces every value in
this rerun's tables exactly, including the two residual discrepancies noted below (they are
properties of the input data, not of any randomness in the pipeline).

`RandomForestRegressor(..., n_jobs=-1)`'s individual predicted `Pij` values (in
`ml_AHP_vs_AHP_comparison.csv` / `ml_NW_vs_NW_comparison.csv`) are not bit-for-bit
reproducible run to run, even with `random_state=42` fixed — parallel tree averaging across
however many CPU cores are available sums each tree's contribution in a thread-completion
order that varies between runs, which can flip the last one or two digits of a float64 value
(confirmed directly: re-running produced differences no larger than 2e-17 in individual
`ml_dominant_Pij` values). This never changes which municipality wins the argmax, and every
aggregate statistic derived from these predictions (R², MAE, RMSE, agreement counts, kappa)
reproduced exactly across repeated reruns in this remediation. Only the raw per-settlement
probability column itself carries this harmless noise.

`ml_AHP_cv_results.csv` and `ml_NW_cv_results.csv` carry a `wall_time_s` column and an
automatic `wall_time_note` flag (added in `06_ml_framework.py::annotate_wall_time_anomalies`)
for any fold whose wall-clock time is more than 3x the median of the other folds — this has
caught a real wall-clock artifact before (one fold logging tens of thousands of seconds
because the machine slept mid-fold during a long background wait), not a real compute cost.
Whether any given rerun's `wall_time_note` column is empty (typical fold times around
900-1,200s each in this repository's own testing) or flags an outlier says something about
that machine's conditions during that run, not about the pipeline itself — R²/MAE/RMSE are
never affected by wall-clock artifacts either way.

A separate one-off robustness check, `docs/audit-history/imputation_sensitivity_check.md`, reruns
the AHP and NW Huff assignment with unreachable OD pairs given zero probability instead of
the pipeline's default column-max-distance fill. On the current network (199 missing pairs)
this changes zero settlement assignments — the pipeline default is not currently
consequential — but it is *not* a retroactive test of the manuscript's own network, which had
roughly ten times the missing-pair rate. The pipeline's default imputation method is
unchanged; this was a diagnostic only.

## Environment: a BLAS backend crash, and why OpenBLAS is required

Later in the remediation, `12_morans_lisa.py` and several other scripts began crashing
natively (no Python traceback) partway through a run. The cause was `libblas`/`liblapack`:
conda-forge's default is Intel's MKL build, and MKL's own CPU-dispatch logic crashes on this
machine's CPU (an i7-7700K, which has no AVX-512) the first time any code path reaches
`scipy.linalg` — even though the AVX-512 codepath's DLL (`mkl_avx512.2.dll`) is physically
present in the environment, MKL's dispatcher still fails to route around it safely. `numpy`
itself is never affected, because its wheel bundles its own self-contained OpenBLAS build and
never touches the conda environment's shared `libblas`/`liblapack` — only packages that call
`scipy.linalg` directly, or are built on top of it (`esda`'s permutation statistics,
`sklearn.cluster.KMeans`, and `shap`'s colormap initialisation), were ever exposed.

The fix is to force the OpenBLAS build variant instead of MKL, which `environment.yml` now
pins explicitly:

```
conda install "libblas=*=*openblas" "liblapack=*=*openblas"
```

`docs/audit-history/openblas_reconciliation.csv` reruns the full pipeline under the fixed
environment and compares every published number against the MKL-era run. All of them match
except Local Moran's I / LISA classification counts, which turned out not to be reproducible
under MKL at all: ten different seeds under OpenBLAS all gave the identical result
(HH=246, zero variance), while attempting to reproduce the original MKL-era number now
crashes outright on this machine. The MKL-era figure was written to disk before MKL started
failing completely, but it cannot be independently re-derived or verified — only the
OpenBLAS value should be cited. Every other statistic checked, including the join-count
statistics (same `esda` permutation family as LISA), reproduced identically across the BLAS
swap, so this is specific to Local Moran's I, not a general warning about `esda`.

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
