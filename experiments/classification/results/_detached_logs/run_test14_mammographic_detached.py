import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.classification.test14 import run_experiement

run_experiement(
    dataset_name='mammographic',
    architectures=['all_rbf'],
    protocol='paper_strict',
    n_splits=1,
    paper_seed=123,
    train_ratio=0.8,
    epochs=500,
    lr=0.01,
    lambda_ridge=1e-4,
    show_plots=False,
    show_plots_at_end=False,
)
