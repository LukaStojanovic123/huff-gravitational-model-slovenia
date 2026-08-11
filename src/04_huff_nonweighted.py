"""
Reuse distance matrix, swap GI to non-weighted, compute NW Huff
probabilities, save huff_NW_od_matrix.csv and huff_NW_summary.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED, TABLES, BETA, MUNICIPALITIES_NW


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
