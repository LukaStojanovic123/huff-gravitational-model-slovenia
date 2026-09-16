# Final manuscript values

Current pipeline state only: post-NW-refit (GI_NW composite feature fix), post-OpenBLAS
(MKL/scipy crash fix), post-full-pipeline-rerun (2026-09-09). No history, no before/after —
see `nw_refit_comparison.csv`, `openblas_reconciliation.csv`, and `ml_model_design_note.md`
for that. One number per line, source file in parentheses.

## Study area and data

- Municipalities: 212 (`Municipalities_Points_normalized.gpkg`, via `data_audit_report.md` 1.1)
- Settlements: 6,036 (`Villages_points_real.shp` / `NA.shp`, via `data_audit_report.md` 1.1)
- Indicators mapped: 100 (`indicator_audit.csv`)
- Group counts: Healthcare 9, Education 14, Traffic & Comm. 14, Trade & Business 7, Culture 12,
  Sports & Rec. 12, Tourism & Services 13, Finance 4, Judiciary & Emergency 8, Residential 7
  (`data_audit_report.md` 1.2)
- AHP group priority weights (%): Healthcare 27.16, Education 18.81, Traffic & Comm. 12.42,
  Trade & Business 12.10, Finance 8.03, Judiciary & Emergency 8.03, Culture 5.35,
  Tourism & Services 3.67, Residential 2.57, Sports & Recreation 1.88
  (`table1_AHP_group_weights.csv`)
- AHP pairwise matrix: lambda_max = 10.3225 (`data_audit_report.md` 1.2)
- AHP consistency index CI = 0.0358 (`data_audit_report.md` 1.2)
- AHP random index RI (Saaty, n=10) = 1.49 (`data_audit_report.md` 1.2)
- AHP consistency ratio CR = 0.0240 (`data_audit_report.md` 1.2)
- Facility types (accessibility): 86 (`data_audit_report.md` 1.6)
- Accessibility values computed (212 munis x 86 facility types): 18,232 (`data_audit_report.md` 1.6)

## Road network

- Filtered segments (pre-noding): 254,252 (`data_audit_report.md` 1.4)
- Total drivable length (pre-noding): 55,061.7 km (`data_audit_report.md` 1.4)
- Noded segments (largest connected component, saved file): 435,935 (`data_audit_report.md` 1.4)
- Graph nodes, largest connected component: 390,273 (`data_audit_report.md` 1.4)
- Largest component as % of pre-filter nodes (390,273 / 394,874): 98.8% (`data_audit_report.md` 1.4)
- Total nodes before largest-component filter: 394,874 (`data_audit_report.md` 1.4)
- Largest component drivable length: 54,431.8 km (`data_audit_report.md` 1.4)
- Municipality snapping distance, mean: 39.5 m (`data_audit_report.md` 1.4)
- Municipality snapping distance, max: 297.5 m (`data_audit_report.md` 1.4)
- Settlement snapping distance, mean: 58.3 m (`data_audit_report.md` 1.4)
- Settlement snapping distance, max: 1,470.5 m (`data_audit_report.md` 1.4)
- Settlements snapped > 500 m: 77 (`data_audit_report.md` 1.4)

## OD matrix

- Shape: 6,036 villages x 212 municipalities (`data_audit_report.md` 1.5)
- Valid (reached-within-300km-cutoff) pairs: 99.98% (1,279,433 / 1,279,632) (`data_audit_report.md` 1.5)
- Missing pairs (filled with column/municipality max): 199 (`data_audit_report.md` 1.5)
- Distance distribution, min: 0.0 m (`data_audit_report.md` 1.5)
- Distance distribution, max: 299,982.3 m (`data_audit_report.md` 1.5)
- Distance distribution, mean: 108,309.1 m (`data_audit_report.md` 1.5)
- Distance distribution, median: 101,996.7 m (`data_audit_report.md` 1.5)
- Settlements at zero distance (village at municipal seat): 212 (`data_audit_report.md` 1.5)

## Gravitational Index

- GI_Final_NotWeighted: min 0.0010, max 1.0000, mean 0.0437, median 0.0203, sd 0.0839,
  skew 7.8964 (`data_audit_report.md` 1.3)
- GI_AHP: min 0.0002, max 1.0000, mean 0.0388, median 0.0134, sd 0.0884, skew 7.4509
  (`data_audit_report.md` 1.3)
- Spearman rank correlation, GI_NW vs GI_AHP: rho = 0.9583, p = 3.61e-116 (`data_audit_report.md` 1.3)
- Municipalities changing rank by > 10 places between GI_NW and GI_AHP: 90 / 212 (`data_audit_report.md` 1.3)
- Top 5 by GI_Final_NotWeighted: 1 Ljubljana (1.0000), 2 Maribor (0.4675), 3 Koper (0.2561),
  4 Ptuj (0.2482), 5 Nova Gorica (0.2045) (`table_top20_GI_both.csv`)
- Top 5 by GI_AHP: 1 Ljubljana (1.0000), 2 Maribor (0.5624), 3 Ptuj (0.3197), 4 Koper (0.2465),
  5 Nova Gorica (0.2308) (`table_top20_GI_both.csv`)

## Huff catchments

- Top 15, AHP weighting: Ljubljana 1,216; Novo mesto 271; Maribor 226; Celje 218; Ptuj 210;
  Krško 167; Črnomelj 149; Brežice 126; Trebnje 126; Murska Sobota 116; Koper 113;
  Nova Gorica 110; Kranj 98; Slovenska Bistrica 85; Sevnica 85 (`huff_AHP_summary.csv`)
- Top 15, NW weighting: Ljubljana 1,016; Novo mesto 202; Maribor 185; Krško 167; Črnomelj 167;
  Ptuj 167; Celje 158; Trebnje 128; Brežice 123; Koper 110; Murska Sobota 98; Ivančna Gorica 98;
  Sevnica 92; Slovenska Bistrica 91; Šentjur 87 (`huff_NW_summary.csv`)
- Single-settlement municipalities: 32 (`huff_AHP_summary.csv`)
- Mean catchment size: 28.5 (`huff_AHP_summary.csv`)
- Median catchment size: 6.0 (`huff_AHP_summary.csv`)
- Top 5 catchments combined (AHP): 2,141 settlements, 35.5% (`huff_AHP_summary.csv`)
- Top 10 catchments combined (AHP): 2,825 settlements, 46.8% (`huff_AHP_summary.csv`)
- Ljubljana catchment size across beta sweep: 2,057 (β=1.5), 1,216 (β=2.0), 645 (β=2.5),
  364 (β=3.0) (`table_beta_sensitivity_clean.csv`)
- Maribor catchment size at β=1.5: 392 (`table_beta_sensitivity_clean.csv`)

## Random Forest — AHP-target model

- Training table: 1,279,632 rows x 189 features (`data_audit_report.md` 1.7)
- Spatial CV blocks (KMeans, k=5, seed=42): {0: 61, 1: 34, 2: 23, 3: 52, 4: 42} (`data_audit_report.md` 1.7)
- Fold 1: R² 0.8752, MAE 0.001286, RMSE 0.009212 (`ml_AHP_cv_results.csv`)
- Fold 2: R² 0.9049, MAE 0.002397, RMSE 0.012575 (`ml_AHP_cv_results.csv`)
- Fold 3: R² 0.8589, MAE 0.002616, RMSE 0.014433 (`ml_AHP_cv_results.csv`)
- Fold 4: R² 0.8952, MAE 0.001716, RMSE 0.009069 (`ml_AHP_cv_results.csv`)
- Fold 5: R² 0.6943, MAE 0.003868, RMSE 0.020473 (`ml_AHP_cv_results.csv`)
- Mean R² ± sd: 0.8457 ± 0.0865 (`data_audit_report.md` 1.7)
- Mean MAE: 0.002377 (`data_audit_report.md` 1.7)
- Mean RMSE: 0.013152 (`data_audit_report.md` 1.7)
- Feature importance by group: Distance 62.53%, GI_AHP 22.69%, Accessibility 7.19%,
  Individual GI indicators 6.84%, Municipality area 0.75% (`table_feature_importance_comparison.csv`)
- Top individual features: dist_to_muni 0.6253, GI_AHP 0.2269, n_Companies 0.0085,
  n_Area_km2 0.0075, nacc_Recycle_Bins 0.0068, n_Population 0.0056, nacc_Theaters 0.0055,
  nacc_Hostels_motels 0.0051, n_Roads_L 0.0040, n_Addresses 0.0039 (`ml_AHP_feature_importance.csv`)

## Random Forest — NW-target model

- Training table: 1,279,632 rows x 189 features (`data_audit_report.md` 1.7)
- Spatial CV blocks: identical to AHP-target, same municipality centroids (`data_audit_report.md` 1.7)
- Fold 1: R² 0.8608, MAE 0.001241, RMSE 0.009190 (`ml_NW_cv_results.csv`)
- Fold 2: R² 0.9192, MAE 0.002412, RMSE 0.011707 (`ml_NW_cv_results.csv`)
- Fold 3: R² 0.8434, MAE 0.002621, RMSE 0.015248 (`ml_NW_cv_results.csv`)
- Fold 4: R² 0.8895, MAE 0.001872, RMSE 0.009323 (`ml_NW_cv_results.csv`)
- Fold 5: R² 0.7604, MAE 0.003602, RMSE 0.017334 (`ml_NW_cv_results.csv`)
- Mean R² ± sd: 0.8547 ± 0.0600 (`data_audit_report.md` 1.7)
- Mean MAE: 0.002349 (`data_audit_report.md` 1.7)
- Mean RMSE: 0.012560 (`data_audit_report.md` 1.7)
- Feature importance by group: Distance 69.16%, GI_NW 17.99%, Accessibility 5.46%,
  Individual GI indicators 5.00%, Municipality area 2.39% (`table_feature_importance_comparison.csv`)
- Top individual features: dist_to_muni 0.6916, GI_NW 0.1799, n_Area_km2 0.0239,
  n_Addresses 0.0062, nacc_Recycle_Bins 0.0055, nacc_Theaters 0.0053, nacc_Hostels_motels 0.0027,
  n_Restaurant 0.0027, n_Thematic_S 0.0027, n_Bike_paths 0.0027 (`ml_NW_feature_importance.csv`)

## Model comparisons — agreement, kappa, Moran's I

- AHP vs NW: n_agree 5,349 / 6,036, 88.62%, kappa 0.8808, Moran's I 0.1838, z 23.68, p 0.001
  (`table_three_way_agreement.csv`)
- AHP vs ML: n_agree 4,546 / 6,036, 75.31%, kappa 0.7472, Moran's I 0.4725, z 63.71, p 0.001
  (`table_three_way_agreement.csv`)
- NW vs ML: n_agree 4,767 / 6,036, 78.98%, kappa 0.7856, Moran's I 0.4403, z 57.18, p 0.001
  (`table_three_way_agreement.csv`)
- RF(AHP-target) vs RF(NW-target): n_agree 4,881 / 6,036, 80.86%, kappa 0.8053, Moran's I 0.3608,
  z 46.20, p 0.001 (`table_three_way_agreement.csv`)
- Euclidean vs network-distance agreement: n_agree 5,341 / 6,036, 88.49%, kappa 0.8795
  (`table_euclidean_vs_network.csv`)
- Beta sweep agreement/kappa vs β=2: β=1.5 → 77.60% / 0.7560; β=2.5 → 83.20% / 0.8262;
  β=3.0 → 73.82% / 0.7321 (`table_beta_sensitivity_clean.csv`)

## LISA (Local Moran's I)

Permutations: 999 (seed=42), significance threshold p < 0.05 (`12_morans_lisa.py` / `table_lisa_summary.csv`)

- AHP vs NW: HH 246, LL 296, HL 626, LH 162, not significant 4,706 (`table_lisa_summary.csv`)
- AHP vs ML: HH 338, LL 820, HL 304, LH 87, not significant 4,487 (`table_lisa_summary.csv`)
- NW vs ML: HH 104, LL 602, HL 321, LH 82, not significant 4,927 (`table_lisa_summary.csv`)
- RF(AHP) vs RF(NW): HH 6, LL 470, HL 254, LH 98, not significant 5,208 (`table_lisa_summary.csv`)

Caveat carried forward: LISA significance counts have demonstrated run-to-run Monte Carlo
variance near the p=0.05 boundary (see `ml_model_design_note.md`) — treat as a single
representative draw, not an exactly reproducible constant.

## Uncertainty — entropy and disagreement

- Entropy, AHP weighting: min -0.0000, max 0.7569, mean 0.5049 (`table_entropy_summary.csv`)
- Entropy class counts, AHP: low 397, medium 1,892, high 3,747 (`table_entropy_summary.csv`)
- Entropy, NW weighting: min -0.0000, max 0.7931, mean 0.5265 (`table_entropy_summary.csv`)
- Entropy class counts, NW: low 383, medium 1,671, high 3,982 (`table_entropy_summary.csv`)
- Disagreement synthesis (3-way, n_disagree=0): 3,948 settlements, 65.41% (`table_disagreement_synthesis.csv`)
- Disagreement synthesis (n_disagree=1): 792 settlements, 13.12% (`table_disagreement_synthesis.csv`)
- Disagreement synthesis (n_disagree=2): 1,234 settlements, 20.44% (`table_disagreement_synthesis.csv`)
- Disagreement synthesis (n_disagree=3): 62 settlements, 1.03% (`table_disagreement_synthesis.csv`)
- Entropy x n_disagree crosstab, n_disagree=0: low 397, medium 1,646, high 1,905 (of 3,948)
  (`map_disagreement_count_villages.gpkg`)
- Entropy x n_disagree crosstab, n_disagree=1: low 0, medium 130, high 662 (of 792)
  (`map_disagreement_count_villages.gpkg`)
- Entropy x n_disagree crosstab, n_disagree=2: low 0, medium 112, high 1,122 (of 1,234)
  (`map_disagreement_count_villages.gpkg`)
- Entropy x n_disagree crosstab, n_disagree=3: low 0, medium 4, high 58 (of 62)
  (`map_disagreement_count_villages.gpkg`)

## Commuting comparison

- Agreement: 68.87% (146 / 212 municipalities) (`table_huff_vs_commuting_summary.csv`)
- Cohen's kappa: 0.6755 (`table_huff_vs_commuting_summary.csv`)
- Functional centres (Huff self-flow dominant): 101 (`table_huff_vs_commuting.csv`)
- Pattern 1 count: 31 (`table_huff_vs_commuting_summary.csv`)
- Pattern 2 count: 22 (`table_huff_vs_commuting_summary.csv`)
- Pattern 3 count: 13 (`table_huff_vs_commuting_summary.csv`)

## RF catchment structure vs Huff — top 15

- Top 15, RF(AHP-target): Novo mesto 382, Ljubljana 277, Maribor 260, Kamnik 211, Celje 200,
  Litija 177, Kranj 163, Trebnje 155, Krško 153, Črnomelj 143, Murska Sobota 142, Brežice 128,
  Kočevje 120, Ptuj 119, Ivančna Gorica 119 (`ml_AHP_vs_AHP_comparison.csv`)
- Same municipalities' Huff (AHP) sizes for comparison: Novo mesto 271, Ljubljana 1,216,
  Maribor 226, Kamnik 69, Celje 218, Litija 77, Kranj 98, Trebnje 126, Krško 167, Črnomelj 149,
  Murska Sobota 116, Brežice 126, Kočevje 71, Ptuj 210, Ivančna Gorica 74 (`huff_AHP_summary.csv`)
- Top 15, RF(NW-target): Ljubljana 270, Novo mesto 236, Ptuj 202, Kamnik 169, Maribor 168,
  Trebnje 165, Črnomelj 158, Krško 156, Ivančna Gorica 144, Kranj 139, Murska Sobota 138,
  Brežice 125, Celje 121, Šentjur 115, Koper 111 (`ml_NW_vs_NW_comparison.csv`)
- Same municipalities' Huff (NW) sizes for comparison: Ljubljana 1,016, Novo mesto 202, Ptuj 167,
  Kamnik 75, Maribor 185, Trebnje 128, Črnomelj 167, Krško 167, Ivančna Gorica 98, Kranj 81,
  Murska Sobota 98, Brežice 123, Celje 158, Šentjur 87, Koper 110 (`huff_NW_summary.csv`)

---

## Exported maps now out of date in QGIS

The 2026-09-09 full pipeline rerun regenerated every `.gpkg` layer fresh, but only some layers'
*content* actually changed relative to whatever was last loaded into a QGIS project (NW-target
model retrain, and the LISA classification/instability fix). Reload these:

**Changed by the NW-target model retrain (Model 2 now uses `GI_NW`, not `GI_AHP`):**
- `map_NW_vs_ML_villages.gpkg` — NW Huff vs ML agreement moved 75.5% → 79.0%
- `map_ML_AHP_vs_ML_NW_villages.gpkg` — RF(AHP) vs RF(NW) agreement, catchment sizes moved

**Changed by the LISA fix (classification/permutation rerun under OpenBLAS):**
- `map_lisa_AHP_vs_NW.gpkg` — HH moved 1,105 → 246 even though the underlying AHP-vs-NW
  comparison itself is unaffected by the NW retrain (see the LISA caveat above)
- `map_lisa_NW_vs_ML.gpkg` — affected by both the NW retrain and the LISA fix
- `map_lisa_MLAHP_vs_MLNW.gpkg` — affected by both the NW retrain and the LISA fix

**Changed because they're downstream of the two above:**
- `map_disagreement_count_villages.gpkg` — n_disagree buckets shifted (depends on NW_vs_ML and
  MLAHP_vs_MLNW)
- `map_disagreement_count_4way_villages.gpkg` — same

**Freshly rewritten but numerically identical to before (safe to skip, not urgent):**
- `map_lisa_AHP_vs_ML.gpkg` — LISA recomputed, landed on the same HH/LL/HL/LH counts

**Unaffected — no need to redo:**
- `fig01_study_area.gpkg`
- `fig03_GI_NW_municipalities.gpkg`
- `fig04_GI_AHP_municipalities.gpkg`
- `fig05_catchments_AHP.gpkg`
- `fig06_catchments_NW.gpkg`
- `fig10_catchments_ML.gpkg` (RF AHP-target only, Model 1 never changed)
- `fig_huff_vs_commuting_municipalities.gpkg`
- `map_AHP_vs_NW_villages.gpkg` (pure Huff-vs-Huff, no ML involved)
- `map_AHP_vs_ML_villages.gpkg` (Model 1/AHP-target unaffected by the NW retrain)
- `map_euclidean_vs_network_villages.gpkg`
- `map_entropy_AHP_villages.gpkg`
- `map_entropy_NW_villages.gpkg`
