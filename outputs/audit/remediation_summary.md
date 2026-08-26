# Remediation summary

Written 2026-08-26, at the close of a multi-stage audit and repair of this repository's
analysis pipeline. This is the reference document for the manuscript's Methods section and
data availability statement.

## What was contaminated, and how it was found

On 2026-08-14, a new road-network file (`all_roads.gpkg`) was added to the raw data
directory, replacing the file the analysis had originally been run against. It was a larger,
differently-sourced OpenStreetMap extract — not an update of the original file, a different
one. The three tables that depend on the road network (the Huff catchment summaries and the
accessibility indicators) were never regenerated afterward, so the repository's committed
results and its raw input data quietly diverged: the numbers in the paper came from one
network, and any fresh run of the pipeline would now use another.

This was caught by a systematic audit that compared file timestamps and hashes across the
raw data directory against the pipeline's own outputs. Three independent symptoms — a
changed road-segment count, a changed accessibility facility count, and a changed
origin-destination matrix coverage rate — all traced back to the same single file swap.
Because the road-network segment count and settlement-snapping distances computed from the
*original* file matched the manuscript's stated figures exactly, the manuscript was confirmed
correct and the contaminant was identified with certainty, not just suspicion.

## Four bugs fixed

1. **Silent skip on existing output.** Three pipeline scripts recomputed nothing and printed
   no warning whenever their output file already existed on disk. This is how the
   contamination went unnoticed: the road network changed, but the scripts that depend on it
   quietly kept reusing results computed before the change. Fixed with an explicit `--force`
   flag; skipping now prints a loud warning naming the stale file being reused.
2. **Road network: largest-connected-component filter never applied.** The script that
   builds the routable road network computed the largest connected component (for a printed
   summary) but saved the *entire*, still-disconnected network to disk instead of that
   filtered component. Every downstream distance calculation was therefore run against a
   network still containing ~1,500 disconnected fragments. Fixed to persist only the largest
   connected component, which is what the original analysis evidently did — verified by
   exact agreement with the manuscript's stated network statistics after the fix.
3. **No coordinate-system check on raw spatial inputs.** No script verified or corrected the
   coordinate reference system of the files it read. This had not caused a visible problem
   yet, but it was a live risk: a CRS-safety pass added to every script that reads a spatial
   file caught one raw file genuinely in the wrong projection (a facility layer used for
   accessibility distances), which would otherwise have silently produced wrong distances the
   next time the pipeline ran end to end.
4. **Spatial cluster (LISA) misclassification.** The code that classifies each settlement
   into a spatial cluster type compared each settlement's value against itself instead of
   against its neighbours' values, a bug that made two of the four possible cluster types
   (spatial outliers) mathematically impossible to detect. This is unrelated to the road
   network contamination — it has existed since the analysis script was first written. Fixed
   to use the underlying statistics library's own classification instead of an incorrect
   manual recomputation. This does not change which settlements are statistically
   significant, only how the significant ones are labelled — confirmed directly by running
   both the old and new classification logic against identical underlying results.

## What reproduced exactly

Once the original road network file was restored and the four bugs above were fixed, the
overwhelming majority of the paper's reported numbers reproduced exactly or within rounding:
road network segment/node counts, municipality and settlement snapping distances (to the
centimetre), the AHP consistency ratio, both Gravitational Index scenarios' full descriptive
statistics and complete top-20 rankings, 13 of 15 top catchment sizes under both weighting
schemes, the single-settlement municipality count, mean/median catchment size, the AHP-vs-NW
agreement rate, the Euclidean-vs-network comparison, the beta-sensitivity kappa range, and
both Random Forest models' mean cross-validated R². A full metric-by-metric table is in
`outputs/audit/reconciliation.csv`.

## Residual discrepancies

Two things did not resolve, and are reported rather than papered over.

**A 262 km gap in total road network length**, alongside a smaller-than-expected number of
unreachable origin-destination pairs (199 in the rerun against 2,107 in the manuscript). The
manuscript's exact segment count, node count, and snapping distances all reproduced exactly
from the restored file, which rules out several obvious causes (an edge-filtering
definition, a different distance cutoff, a routing/snapping mismatch — each was tested and
ruled out directly). The most likely explanation is a small difference in the underlying OSM
edit state between when the original file and the restored file were extracted, which cannot
be resolved without the original file itself.

**Six settlements move off Ljubljana's catchment**, under both weighting schemes, at every
tested distance-decay value. Tracing them individually showed all six had shared an identical
distance value to Ljubljana in the manuscript's results — the specific value used to fill in
distances that could not be computed directly, meaning these six were not actually reachable
to Ljubljana within the network used for the original analysis. Ljubljana still won the
gravity-model calculation anyway because its attractiveness score is high enough to
outweigh even a heavily penalised distance. In the restored, better-connected network, these
six settlements are genuinely reachable at short distances and correctly assigned to their
true nearest smaller centre instead. This is very likely the corrected pipeline being *more*
accurate than the original, not a regression, and is best described in the manuscript as
such rather than silently renumbered.

One further, unrelated finding surfaced while fixing the LISA classification bug: even after
the fix, the corrected spatial-cluster counts for the AHP-versus-machine-learning comparison
do not match the manuscript's reported counts, in both magnitude and structure. This is a
genuine, currently unexplained difference in Section 4.5's cluster analysis, independent of
the road network issue.

## Current status

The repository's pipeline now runs end to end from the restored raw inputs, with every raw
file the pipeline reads named explicitly and hash-verified against a frozen manifest
(`data/raw_manifest.json`, checked automatically on every run), no script depending on a path
outside the repository, and no script silently reusing stale output. `outputs/audit/`
contains the full audit trail: `reconciliation.csv` (metric-by-metric manuscript comparison),
`reproducibility_note.md` (exact reproduction commands and raw-file inventory),
`ml_model_design_note.md` (the two-model machine learning design), and
`imputation_sensitivity_check.md` (a robustness check on the distance-imputation method). The
quarantined contaminant file and the pre-remediation output snapshot are both preserved on
disk, not deleted, for anyone who wants to audit this process independently.
