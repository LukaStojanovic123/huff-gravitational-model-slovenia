"""
Rerun AHP Huff with Euclidean distances, compare dominant assignments
vs road network, save agreement table and spatial layer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED, TABLES, GPKG, BETA, MUNICIPALITIES_AHP


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
