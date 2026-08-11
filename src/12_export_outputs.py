"""
Export all paper-ready outputs — 6 tables, 3 supplementary tables,
9 figures as PNG and PDF, all GPKG layers for QGIS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED, TABLES, FIGURES, GPKG, SUPPLEMENTARY


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
