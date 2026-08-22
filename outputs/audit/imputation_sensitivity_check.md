# Imputation sensitivity check: column-max fill vs. zero probability

Requested as a robustness check after diagnosing the "Ljubljana six" (see
`reproducibility_note.md`): the pipeline's default imputation for OD pairs unreached within
the 300km cutoff fills them with the column (municipality) maximum observed distance, then
runs the Huff gravity formula as normal. Because a high-GI centre (Ljubljana, GI=1.000) can
still win the argmax even at a heavily penalized filled distance, this can assign an
unreachable village to a centre it was never actually shown to be close to. That's what
happened to six villages in the manuscript's network.

**This check does not change the pipeline default.** It's a one-off comparison, not applied
to any committed output.

## Method

On the current fixed network (390,273-node largest component), the 300km-cutoff Dijkstra
pass leaves **199 missing (village, municipality) pairs** out of 1,279,632. Two variants of
the Huff assignment were computed from the same raw distance matrix:

- **Default (column-max fill)**: missing pairs get `dist = column max`, then
  `attract = GI / dist^beta` as normal — this is what `03_huff_ahp.py` /
  `04_huff_nonweighted.py` actually do.
- **Zero probability**: missing pairs get `attract = 0` directly (excluded from the
  denominator when normalizing `Pij` for that village), rather than being assigned any
  finite filled distance.

Both AHP and NW weighting were run under both variants; each was compared for how many
villages' *dominant* (argmax) municipality changes.

## Result

**Zero settlements change assignment, under either weighting scheme, on the current
network.**

This is not surprising given the numbers: 199 missing pairs spread across 6,036 villages ×
212 municipalities means the overwhelming majority of villages have zero missing entries at
all, and among the few that do, none happens to have its argmax fall on the specific
municipality it's missing a real distance to. At this missing-pair rate, the imputation
choice is immaterial — column-max fill and zero-probability exclusion produce identical
Huff assignments.

## What this does and doesn't tell you

This is a genuine robustness result for **this pipeline's own default**, on **this fixed
network**: the choice of imputation method does not currently drive any catchment
assignment. It is a fair thing to report as a robustness check in the paper.

It is **not** a retroactive test of the manuscript's original network, which had roughly ten
times the missing-pair rate (2,107 vs. 199). The "Ljubljana six" diagnosis stands on its own
evidence (identical fill-value distances, resolved to real short distances on the fixed
network) — this check does not reproduce or contradict that finding, since the manuscript's
own raw pre-fill distance matrix is not available to rerun this same comparison against. The
honest statement is: imputation method doesn't matter *now*; it evidently did matter on
whatever network the manuscript used, at whatever (higher) missing-pair rate it had.
