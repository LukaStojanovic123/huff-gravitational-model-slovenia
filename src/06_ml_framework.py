"""
Build 1.28M row ML input table, spatial 5-fold CV, Random Forest
189 features, SHAP values, save feature importance and predictions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED, TABLES


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
