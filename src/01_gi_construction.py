"""
Load 100 indicators, min-max normalise, compute non-weighted GI and
AHP-weighted GI using rarity weights and AHP group weights.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
