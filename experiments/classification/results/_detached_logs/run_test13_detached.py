import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.classification.test13 import run_experiement

run_experiement(show_plots=False, show_plots_at_end=False)
