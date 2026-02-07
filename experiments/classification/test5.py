import uuid
from typing import Callable, Optional

import torch

from data.generators.mixture_of_gaussians import generate_mixture_of_gaussians
from experiments.classification.test3 import run_experiement as run_experiement_core


def generate_gaussian_signal(
    samples_per_class: int = 300,
    noise_std: float = 0.10,
    radius: float = 1.0,
    shuffle: bool = True,
):
    """
    Build a 2-class 2D Gaussian signal dataset with controllable overlap.
    `radius` controls mean separation, `noise_std` controls covariance scale.
    """
    means = [
        torch.tensor([-radius, 0.0]),
        torch.tensor([radius, 0.0]),
    ]
    var = max(float(noise_std), 1e-6) ** 2
    cov = torch.eye(2) * var
    covariances = [cov, cov]

    X, Y = generate_mixture_of_gaussians(
        number_of_classes=2,
        samples_per_class=samples_per_class,
        dimension=2,
        list_of_means=means,
        list_of_covariances=covariances,
    )

    if shuffle:
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        Y = Y[perm]

    return X, Y


def run_experiement(
    noise_feature_counts: list = None,
    dataset_name: str = "gaussian_mixture",
    signal_generator: Callable = generate_gaussian_signal,
    signal_generator_kwargs: Optional[dict] = None,
    num_signal_features: Optional[int] = None,
    signal_expansion_mode: str = "mixed",
    signal_expansion_noise_std: float = 0.01,
    allow_signal_expansion: bool = True,
    samples_per_class: int = 300,
    two_moons_noise_std: float = 0.10,
    useless_feature_std: float = 1.0,
    epochs: int = 500,
    lr: float = 0.01,
    lambda_ridge: float = 1.0,
    full_vector_rbf_gamma: float = 0.5,
    seed: int = 42,
    run_uuid: Optional[str] = None,
    results_root: str = "experiments/classification/results",
    save_results: bool = True,
    show_plots: bool = True,
    show_plots_at_end: bool = True,
):
    """
    Test 5: same experiment protocol as test3, but default signal dataset is Gaussian mixture.

    Signature intentionally matches test3.run_experiement(...) so switching tests is easy.
    """
    if signal_generator_kwargs is None:
        signal_generator_kwargs = {
            "samples_per_class": samples_per_class,
            "noise_std": two_moons_noise_std,
            "radius": 1.0,
            "shuffle": True,
        }

    return run_experiement_core(
        noise_feature_counts=noise_feature_counts,
        dataset_name=dataset_name,
        signal_generator=signal_generator,
        signal_generator_kwargs=signal_generator_kwargs,
        num_signal_features=num_signal_features,
        signal_expansion_mode=signal_expansion_mode,
        signal_expansion_noise_std=signal_expansion_noise_std,
        allow_signal_expansion=allow_signal_expansion,
        samples_per_class=samples_per_class,
        two_moons_noise_std=two_moons_noise_std,
        useless_feature_std=useless_feature_std,
        epochs=epochs,
        lr=lr,
        lambda_ridge=lambda_ridge,
        full_vector_rbf_gamma=full_vector_rbf_gamma,
        seed=seed,
        run_uuid=run_uuid,
        results_root=results_root,
        save_results=save_results,
        show_plots=show_plots,
        show_plots_at_end=show_plots_at_end,
    )


if __name__ == "__main__":
    run_experiement(
        dataset_name="gaussian_mixture",
        signal_generator=generate_gaussian_signal,
        signal_generator_kwargs={
            "samples_per_class": 300,
            "noise_std": 0.10,
            "radius": 1.0,
            "shuffle": True,
        },
        num_signal_features=50,
        allow_signal_expansion=True,
        signal_expansion_mode="mixed",
        signal_expansion_noise_std=0.01,
        noise_feature_counts=[0, 2, 5, 10, 20, 50, 100,200],
        samples_per_class=300,
        two_moons_noise_std=0.10,
        useless_feature_std=1.0,
        epochs=500,
        lr=0.01,
        lambda_ridge=1.0,
        full_vector_rbf_gamma=0.5,
        seed=42,
        run_uuid=str(uuid.uuid4()),
        save_results=True,
        show_plots=True,
        show_plots_at_end=True,
    )
