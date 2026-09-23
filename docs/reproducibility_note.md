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

## A broken group-name lookup in indicator_audit.csv, and whether it leaked

`17_data_audit.py`'s section 1.2 writes `outputs/audit/indicator_audit.csv`, one row per
indicator, including a column `ahp_group_weight_pct` that looks up each indicator's thematic
group in `table1_AHP_group_weights.csv` via a small translation table (`_GROUP_NAME_MAP` /
`_map_group_name`, because tableS3 spells groups "Traffic and Communications" while table1
spells the same group "Traffic & Communications"). For five of the ten groups — Judiciary,
Sports, Tourism, Trade, Traffic — this lookup produced `NaN` (and the surrounding audit
narrative reported "DO NOT MATCH") for as long as tableS3's `Thematic_group` column held
short, abbreviated forms of those five names instead of the canonical ones.

Traced by commit, fresh, rather than trusting an earlier answer given only in conversation:

- `outputs/supplementary/tableS3_individual_indicator_weights.csv` has held abbreviated
  group names for those five groups since the repository's very first commit
  (`6019cdf`, "Add existing analysis outputs, GI construction script..."), because the file
  was originally hand-maintained (pasted in once, never regenerated — see the tableS3 section
  above).
- `_GROUP_NAME_MAP` in the audit script has held the canonical full-name keys unchanged since
  the script itself was first added (`110c91f`), so the mismatch existed from the audit
  script's own inception.
- `553e70a` (2026-09-14 11:54:59, "Correct supplementary tables S1-S4...") remapped tableS3's
  `Thematic_group` column to the canonical ten names, which is what fixed the lookup.

So the broken window ran from the repository's first commit until 2026-09-14. The question
that matters is whether anything downstream of `indicator_audit.csv` ever read the broken
`ahp_group_weight_pct` column during that window. Checked directly:

```
grep -rn "indicator_audit" --include="*.py" src/ config.py
```

turns up exactly three kinds of hit: `17_data_audit.py` itself writing the file, and
`11_export_outputs.py`'s output checklist, which only checks that the file *exists* on disk
(`AUDIT / "indicator_audit.csv"` in `EXPECTED_OUTPUTS`) — it never opens or reads it. No
script anywhere in `src/` ever calls `pd.read_csv` on `indicator_audit.csv`. The only other
reference in the repository is in `docs/final_manuscript_values.md`, which cites
`indicator_audit.csv` for exactly one number — "Indicators mapped: 100" — the row count of
the file, unrelated to the `ahp_group_weight_pct` column. The group counts quoted immediately
below that line in the same document (Healthcare 9, Education 14, ...) are attributed to
`data_audit_report.md` section 1.2's *own* group-count check, which counts indicators per
group directly from tableS3's `Thematic_group` column via `value_counts()` — a separate
computation in the same section that never calls `_map_group_name` or touches table1 at all,
so it was never affected by this bug either.

**Conclusion: the bug never left `indicator_audit.csv`.** It degraded one diagnostic column
in one audit-only CSV, for the entire time this repository has existed, without ever being
read by another script or cited for any number that reached `final_manuscript_values.md`,
the README, or any table/figure/gpkg output. `17_data_audit.py`'s five other section-1.2
checks (group indicator counts, within-group weight sums, table1's own 100%-sum check, the
AHP consistency ratio, and — added in this remediation pass — a full independent
recomputation of tableS3's rarity weights) all read their inputs directly rather than through
this lookup, and all report real, non-tautological pass/fail verdicts today (see the "cannot
fail" fixes below).

This is also, on its own, a second instance of the exact failure mode documented throughout
this file: a broken check (`DO NOT MATCH` / `NaN`, printed) that nothing treated as a build
failure. `17_data_audit.py::main()` now exits non-zero if any section records a real failure
— see the top-level `record_failure` / `CHECK_FAILURES` mechanism in that script — specifically
so that a bug shaped like this one cannot again sit unnoticed in printed output for months.

## A Table 7 relabeling that was not a regression, once checked against current data

`10_commuting_comparison.py`'s `classify_pattern()` — unchanged, byte-for-byte, since the
script was first created — has always defined "Pattern 1" as Huff-self-contained /
commuting-external, and "Pattern 2" as commuting-self-contained / Huff-external. When
`table7_commuting_comparison.csv` was built (promoting a local, untracked prototype file
into a real pipeline output), an earlier hand-built version of that prototype had swapped the
English descriptions attached to each pattern's count relative to `classify_pattern()`'s own
convention. The fix wrote `table7_commuting_comparison.csv` to match `classify_pattern()`
instead of the prototype.

That fix was then itself questioned, based on a manual, by-hand verification of one
municipality done earlier in this project, before it — asserting the opposite pairing: Huff-
external/commuting-self-contained at 31 municipalities, not 22. Checked three separate ways
against the *current* `table_huff_vs_commuting.csv`, with zero reference to either
`classify_pattern()`'s labels or the prototype's: a full 2x2 cross-tab of
`huff_majority_centre == SIFRA` against `commuting_is_centre`; the `pattern` column's own
per-row consistency; and a plain-language filter with no pattern labels at all, printing the
full municipality list for each group. All three agree, exactly: Huff-self-contained /
commuting-external is 31 municipalities, commuting-self-contained / Huff-external is 22 —
matching `classify_pattern()`, not the earlier by-hand check.

**Conclusion: the by-hand verification was stale, not the code.** It most likely predates one
or more of the corrections documented elsewhere in this file that changed `huff_majority_centre`
assignments broadly (the road-network fix, the NW-target model refit) — a specific
municipality checked by hand before those landed would no longer necessarily fall in the same
group afterward. `classify_pattern()` itself was never edited during this investigation;
`table7_commuting_comparison.csv` already matched it correctly from the commit that created
it, and needed no further change.

## A latent gap in table_ml_catchment_sizes.csv's union logic

`14_ml_catchment_structure.py::catchment_sizes()` builds `table_ml_catchment_sizes.csv` by
taking the union of each model's own top `TOP_N` (20) municipalities by catchment size, across
four separate rankings — but only three of the four rank columns it computes are actually
included in that union: `rank_AHP_Huff_size`, `rank_NW_Huff_size`, and
`rank_RF_AHP_target_size`. `rank_RF_NW_target_size` is computed (it appears as a column in the
saved table) but never contributes to which municipalities get selected into the union.

This is currently harmless: every municipality in RF(NW-target)'s real top 15 already enters
the table via one of the other three rankings (confirmed while building Supplementary Table S6
from the same underlying data — see that table's own construction, which unions
`RF_AHP_target_size` and `RF_NW_target_size` top-15 directly and independently, with no such
gap). But it is a real gap in `table_ml_catchment_sizes.csv`'s own selection logic, not just an
appearance of one: if the RF(NW-target) model's catchment hierarchy were ever to diverge enough
from the other three rankings that some municipality ranked highly under
`RF_NW_target_size` alone and nowhere else, that municipality would be silently dropped from
`table_ml_catchment_sizes.csv` without any error or warning — the same "reported but not
checked" shape as several other findings in this file, just not yet manifested as a wrong
number in anything currently published. Worth fixing (add
`set(full["rank_RF_NW_target_size"][full["rank_RF_NW_target_size"] <= TOP_N].index)` to the
union in `catchment_sizes()`) if this table's selection logic is ever reused for a different
`TOP_N` cutoff or a different set of models, where the current coincidence may not hold.

## Full pipeline verified end to end, 2026-09-23

A complete fresh run, `01_gi_construction.py` through `18_output_manifest.py`, was verified
in full on 2026-09-23 (`06_ml_framework.py --model AHP` and `--model NW` were trained manually
outside the verifying session to avoid an external memory-pressure process kill; every script
from `07` onward ran inside that session, in numeric order, against the fresh `06` output).
Every value produced — beta sensitivity, Euclidean-vs-network agreement, entropy summary,
commuting comparison, the four pairwise agreement maps, Moran's I / Cohen's kappa / LISA,
three-way and four-way disagreement synthesis, RF catchment structure and feature importance,
SHAP top features for both models, and the spatial layers — matched
`docs/final_manuscript_values.md` exactly. `17_data_audit.py`'s own manuscript cross-check
(section 1.8) independently confirmed the same: every reference-value comparison read
CONFIRMED (the run also flags several rows as differing from an older, superseded manuscript
draft's prose — that is expected and unrelated to `final_manuscript_values.md`).

Step-ordering was confirmed correct, not just assumed from the run completing:
`12_morans_lisa.py`'s `assert_map_is_fresh` check — added specifically to catch a script
consuming a leftover map from a previous run instead of `11_export_outputs.py`'s current
output — passed silently, meaning `12` genuinely consumed `11`'s freshly built agreement maps
rather than a stale copy.

Two failures did occur during this verification run, and neither was a pipeline-ordering bug:

- `17_data_audit.py` failed on first attempt because
  `outputs/supplementary/tableS2_AHP_priority_weights.csv` was absent from the working tree.
  This file is the AHP pairwise-comparison matrix — maintained directly, by hand, and never
  regenerated by any script (see the Repository structure table in `README.md`); it must
  already be present in the working tree before a fresh run starts, the same way
  `data/external/indicator_categories.csv` must be. It had been deleted from disk before this
  verification began (visible as an unstaged deletion in `git status`, unrelated to the
  pipeline run itself); restoring it from git history and rerunning `17` produced the clean
  pass described above.
- `18_output_manifest.py` failed on first attempt for an unrelated, self-referential reason:
  it snapshots which files exist under `outputs/` *before* writing its own
  `output_manifest.csv`, so a first-ever run always reports that one file as missing by
  construction, then writes it anyway. Rerunning immediately afterward — with the file now
  present from the prior run's write — passed cleanly (101/101 files accounted for, 0
  orphaned, 0 ambiguous, 0 missing). Worth fixing at the source (write the manifest before
  computing the missing-files check, or exclude the manifest's own path from that check) so a
  genuinely first-ever run doesn't require two invocations.
