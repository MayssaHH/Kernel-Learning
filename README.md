# Kernel Learning

## Setup (uv)
0. Setup the env
   ```bash
   uv sync
   ```
1. Install the project in editable mode:
   ```bash
   uv pip install -e .
   ```

## Run an example
```bash
uv run experiments/example.py
```

## Imports
Use the package name in scripts:
```python
from kernel_learning import *
```

## Code organization

The library is split into small, focused modules:
- `kernel_learning/sub_kernels/`: Base class + individual sub-kernel implementations.
- `kernel_learning/kernel_network/`: Core `KernelNetwork` model.
- `kernel_learning/losses/`: Loss base class and concrete loss functions.
- `kernel_learning/trainers/`: Training strategies (manual/optimizer-based).

## Where to put your experiments
- `experiments/`: Each Experiment we want to do goes here in a seperate **folder** with its own scripts, results, configs, README, etc.