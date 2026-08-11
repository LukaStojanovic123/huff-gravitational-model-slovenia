"""
Dijkstra OD matrix settlements to municipalities, compute AHP Huff
probabilities, save huff_od_matrix.csv and huff_summary.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED, TABLES, BETA, MUNICIPALITIES_AHP


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
