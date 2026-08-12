# Huff Gravitational Model of Slovenia

A spatial accessibility study of Slovenia's 212 municipalities and 6,036
settlements using a Huff gravity model. Municipal attractiveness is
captured by a Gravitational Index (GI) built from 100 indicators, both
non-weighted and AHP-weighted with rarity and group weights, and combined
with road-network travel distances (via Dijkstra shortest paths over an
OSM-derived graph) to estimate settlement-to-municipality interaction
probabilities. The pipeline further evaluates model robustness through
beta sensitivity analysis, Euclidean-vs-network distance comparison,
Shannon entropy of assignment uncertainty, Moran's I spatial
autocorrelation, a machine-learning (Random Forest + SHAP) benchmark, and
validation against observed 2023 commuting flows.

## Study Area and Data

- **Study area:** Slovenia — 212 municipalities, 6,036 settlements.
- **Model parameters:** distance-decay exponent β = 2 (sensitivity tested
  at 1.5–3.0), projected CRS EPSG:3794, network cutoff distance 300 km.
- **Raw data** lives outside this repository at the path configured in
  `config.py` (`DATA_RAW`) and includes:
  - `Roads.shp` — OSM road network (motorway + primary only; see note in `02_road_network.py`)
  - `Municipalities_All_Groups_Weighted_AHP.gpkg` — AHP-weighted GI inputs
  - `Municipalities_All_Groups_NotWeighted_Normalized.gpkg` — non-weighted GI inputs
  - `Municipalities_Points_normalized.gpkg` — municipality centroids
  - `Villages_points_real.shp` — settlement points
  - `NA.shp` — settlement polygons
  - `2023tabela.xlsx` — SURS 2023 commuting matrix

## Repository Structure

```
huff-gravitational-model-slovenia/
├── README.md                          this file
├── requirements.txt                   pip dependency list
├── environment.yml                    conda environment (huff_env)
├── .gitignore                         excludes large/raw data
├── config.py                          paths and study constants
├── data/
│   ├── raw/                           (not committed) local mirror, if used
│   └── processed/                     (not committed) large intermediates
├── src/
│   ├── 01_gi_construction.py          normalise 100 indicators, build non-weighted + AHP-weighted GI
│   ├── 02_road_network.py             build routable road graph from OSM data
│   ├── 03_huff_ahp.py                 OD matrix + AHP-weighted Huff probabilities
│   ├── 04_huff_nonweighted.py         OD matrix + non-weighted Huff probabilities
│   ├── 05_accessibility.py            per-municipality accessibility to 86 facility types
│   ├── 06_ml_framework.py             Random Forest + SHAP benchmark of Huff assignments
│   ├── 07_beta_sensitivity.py         Huff model sensitivity to distance-decay exponent
│   ├── 08_euclidean_comparison.py     Euclidean vs. road-network distance comparison
│   ├── 09_entropy_uncertainty.py      Shannon entropy of Pij assignment uncertainty
│   ├── 10_morans_i.py                 spatial autocorrelation of GI, Pij, and disagreement
│   ├── 11_commuting_comparison.py     validation against observed 2023 commuting flows
│   └── 12_export_outputs.py           export paper-ready tables, figures, and GPKG layers
├── notebooks/
│   ├── 01_data_exploration.ipynb      exploratory look at raw datasets
│   ├── 02_gi_validation.ipynb         GI construction sanity checks
│   └── 03_results_visualization.ipynb interactive results exploration
└── outputs/
    ├── tables/                        paper-ready tables (committed)
    ├── figures/                       paper-ready figures (committed)
    ├── gpkg/                          spatial layers for QGIS (not committed)
    └── supplementary/                 supplementary tables (committed)
```

## How to Run

1. Create and activate the conda environment:

   ```
   conda env create -f environment.yml
   conda activate huff_env
   ```

   (Alternatively, `pip install -r requirements.txt` into an existing
   Python 3.11 environment.)

2. Edit `DATA_RAW` in `config.py` if the raw data path differs on your
   machine.

3. Run the scripts in order from the repository root:

   ```
   python src/01_gi_construction.py
   python src/02_road_network.py
   python src/03_huff_ahp.py
   python src/04_huff_nonweighted.py
   python src/05_accessibility.py
   python src/06_ml_framework.py
   python src/07_beta_sensitivity.py
   python src/08_euclidean_comparison.py
   python src/09_entropy_uncertainty.py
   python src/10_morans_i.py
   python src/11_commuting_comparison.py
   python src/12_export_outputs.py
   ```

   Each script depends on outputs from earlier scripts in the sequence
   (written to `data/processed/`), so they should be run in order on a
   first pass.

## Cached intermediates

`data/processed/distance_matrix.npy` is the key cached intermediate in the
pipeline: the village-to-municipality road-network distance matrix produced
by Dijkstra shortest paths over the noded road graph (`02_road_network.py`'s
`data/processed/roads_noded.gpkg`). It is saved alongside two companion
arrays, `distance_matrix_village_ids.npy` and `distance_matrix_muni_ids.npy`,
which record the row/column order it was computed with — a script only
reuses the cached matrix if both id arrays match its current village and
municipality data, otherwise it recomputes.

Building this matrix (212 municipalities × 6,036 villages via Dijkstra) is
the most expensive step in the pipeline. `03_huff_ahp.py` computes and
caches it if missing; `04_huff_nonweighted.py` reuses the same cached matrix
(the network distances don't depend on which GI scenario is being scored).
`07_beta_sensitivity.py` and `08_euclidean_comparison.py` also depend on
having this distance matrix available rather than recomputing it.

`03_huff_ahp.py` and `04_huff_nonweighted.py` additionally short-circuit at
the output level: if `outputs/tables/huff_AHP_summary.csv` or
`huff_NW_summary.csv` already exists, the script skips computation entirely
and just reports the existing file. Delete the relevant summary CSV (and,
if you want the distances themselves recomputed, `data/processed/distance_matrix*.npy`)
to force a fresh run.

## Outputs

| Script | Produces |
|---|---|
| 01 | Non-weighted and AHP-weighted GI per municipality |
| 02 | Largest connected component of the drivable road graph |
| 03 | `huff_od_matrix.csv`, `huff_summary.csv` (AHP-weighted) |
| 04 | `huff_NW_od_matrix.csv`, `huff_NW_summary.csv` (non-weighted) |
| 05 | `accessibility_normalized.csv` — 86-facility accessibility scores |
| 06 | Random Forest feature importances, SHAP values, predictions |
| 07 | Beta sensitivity agreement table and figure |
| 08 | Euclidean-vs-network agreement table and spatial layer |
| 09 | Settlement-level entropy/uncertainty layer |
| 10 | Moran's I results for GI, Pij, and ML–AHP disagreement |
| 11 | Commuting-vs-Huff comparison and disagreement classification |
| 12 | 6 tables, 3 supplementary tables, 9 figures (PNG/PDF), all GPKG layers |

## Results summary

| Analysis | Key result |
|---|---|
| GI_AHP top municipality | Ljubljana (1.000) |
| GI_AHP mean across 212 municipalities | 0.039 |
| AHP Huff — Ljubljana catchment | 1,222 settlements |
| AHP vs NW agreement | 88.6% (5,349/6,036 settlements) |
| Beta sensitivity range (κ) | 0.732 to 0.826 across β=1.5–3.0 |
| Random Forest R² (AHP target) | 0.845 ± 0.088 (spatial 5-fold CV) |
| Random Forest R² (NW target) | 0.844 ± 0.066 (spatial 5-fold CV) |
| AHP Huff vs ML agreement | 75.5% (4,557/6,036 settlements) |
| NW Huff vs ML agreement | 77.4% (4,672/6,036 settlements) |
| Huff vs commuting agreement | 68.9% (146/212 municipalities) |
| Distance feature importance | 62.5% |
| GI_AHP composite importance | 22.8% |

## Citation

> Placeholder — citation to be added upon publication.

## Authors and Contact

Luka Stojanovič — luka.stojanovich95@gmail.com
