import os
import sys

# make `ground_station`, `edge`, `tools` importable when running `pytest` from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
