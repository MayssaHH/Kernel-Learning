import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASETS = [
    "breastcancer",
    "ionosphere",
    "spambase",
    "banknote",
    "haberman",
    "mammographic",
    "parkinsons",
    "wine",
    "iris",
    "heart",
]

LOGS_DIR = ROOT / "experiments" / "classification" / "results" / "_detached_logs"
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"


def ensure_runner(dataset: str) -> Path:
    runner = LOGS_DIR / f"run_test14_{dataset}_detached.py"
    if runner.exists():
        return runner

    script = f"""import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.classification.test14 import run_experiement

run_experiement(
    dataset_name='{dataset}',
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
"""
    runner.write_text(script, encoding="utf-8")
    return runner


def run_one(dataset: str) -> int:
    runner = ensure_runner(dataset)
    out_log = LOGS_DIR / f"test14_{dataset}.out.log"
    err_log = LOGS_DIR / f"test14_{dataset}.err.log"

    out_log.write_text("", encoding="utf-8")
    err_log.write_text("", encoding="utf-8")

    with out_log.open("w", encoding="utf-8") as out_f, err_log.open(
        "w", encoding="utf-8"
    ) as err_f:
        print(f"[orchestrator] starting dataset={dataset}", flush=True)
        proc = subprocess.Popen(
            [str(PYTHON_EXE), str(runner)],
            cwd=str(ROOT),
            stdout=out_f,
            stderr=err_f,
        )
        rc = proc.wait()
        print(f"[orchestrator] finished dataset={dataset} rc={rc}", flush=True)
    return rc


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"Python executable not found: {PYTHON_EXE}")

    failed = []
    for ds in DATASETS:
        rc = run_one(ds)
        if rc != 0:
            failed.append((ds, rc))

    if failed:
        print("[orchestrator] finished with failures:", flush=True)
        for ds, rc in failed:
            print(f"  - {ds}: rc={rc}", flush=True)
        sys.exit(1)

    print("[orchestrator] all datasets completed successfully.", flush=True)


if __name__ == "__main__":
    main()

