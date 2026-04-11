import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.classification.test16 import run_experiement

run_experiement(
    dataset_name='breastcancer',
    architectures=['all_rbf', 'half_rbf_half_linear', 'all_linear'],
    alpha_source='all_rbf',
    alpha_keep_mass=0.95,
    protocol='paper_strict',
    n_splits=1,
    paper_seed=123,
    train_ratio=0.8,
    epochs=500,
    lr=0.01,
    lambda_ridge=1e-4,
    full_vector_rbf_gamma=0.5,
    mkl_degrees=list(range(1, 11)),
    easy_mkl_lam=0.1,
    mkl_max_iter=500,
    mkl_tolerance=1e-5,
    show_plots=False,
    show_plots_at_end=False,
)
