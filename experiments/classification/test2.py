import uuid

from data.generators.two_moons import generate_two_moons
from experiments.classification.modular_experiment import run_experiement


if __name__ == "__main__":
    run_experiement(
        dataset_name="two_moons",
        data_generator=generate_two_moons,
        data_generator_kwargs={
            "samples_per_class": 400,
            "noise_std": 0.10,
            "radius": 1.0,
            "shuffle": True,
        },
        num_classes=2,
        samples_per_class=400,
        dimension=2,
        epochs=500,
        run_uuid=str(uuid.uuid4()),
        save_results=True,
        show_plots=True,
    )
