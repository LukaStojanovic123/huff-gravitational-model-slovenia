"""
Load OSM roads, filter to drivable classes, unary_union noding,
build NetworkX graph, save largest connected component.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED, ROADS_FILE


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
