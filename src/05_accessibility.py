"""
For each of 86 facility types compute nearest-facility road distance
per municipality, normalise inverted 0-1, save accessibility_normalized.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED, TABLES


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
