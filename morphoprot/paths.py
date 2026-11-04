"""Data and result locations.

Set MORPHOPROT_DATA to the folder that holds the processed tonsil/, spleen/ and
liver/ inputs (see README.md, "Data layout"), and optionally MORPHOPROT_RESULTS
for outputs. Both default to folders inside the repository.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("MORPHOPROT_DATA", REPO / "data"))
RESULTS_ROOT = Path(os.environ.get("MORPHOPROT_RESULTS", REPO / "results"))
