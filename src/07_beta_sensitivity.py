"""
Rerun Huff with beta 1.5 2.0 2.5 3.0, compute agreement and Cohen
kappa vs beta=2, save table and figure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED, TABLES, FIGURES, BETA


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
