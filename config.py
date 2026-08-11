from pathlib import Path

# Raw data — edit this path when moving machines
DATA_RAW = Path(r"C:\Users\lstojano\Desktop\teza\HuffMethodPaper\Data")

# Repository-relative paths
REPO_ROOT      = Path(__file__).parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
OUTPUTS        = REPO_ROOT / "outputs"
TABLES         = OUTPUTS / "tables"
FIGURES        = OUTPUTS / "figures"
GPKG           = OUTPUTS / "gpkg"
SUPPLEMENTARY  = OUTPUTS / "supplementary"

# Study constants
N_MUNICIPALITIES = 212
N_SETTLEMENTS    = 6036
BETA             = 2
EPSG             = 3794
CUTOFF_M         = 300_000

# Key filenames in DATA_RAW
ROADS_FILE         = "all_roads.gpkg"
MUNICIPALITIES_AHP = "Municipalities_All_Groups_Weighted_AHP.gpkg"
MUNICIPALITIES_NW  = "Municipalities_All_Groups_NotWeighted_Normalized.gpkg"
MUNICIPALITIES_PTS = "Municipalities_Points_normalized.gpkg"
VILLAGES_FILE      = "Villages_points_real.shp"
SETTLEMENTS_POLY   = "NA.shp"
COMMUTING_FILE      = "2023tabela.xlsx"

