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


def run_one(dataset: str) -> int:
    out_log = LOGS_DIR / f"test15_{dataset}.out.log"
    err_log = LOGS_DIR / f"test15_{dataset}.err.log"
    out_log.write_text("", encoding="utf-8")
    err_log.write_text("", encoding="utf-8")

    cmd = [
        str(PYTHON_EXE),
        "-u",
        "-m",
        "experiments.classification.test15",
        "--dataset",
        dataset,
        "--architectures",
        "all_rbf",
        "half_rbf_half_linear",
        "all_linear",
        "--alpha-source",
        "all_rbf",
        "--alpha-keep-mass",
        "0.95",
        "--min-features-keep",
        "1",
        "--protocol",
        "paper_strict",
        "--n-splits",
        "1",
        "--paper-seed",
        "123",
        "--train-ratio",
        "0.8",
        "--epochs",
        "500",
        "--lr",
        "0.01",
        "--lambda-ridge",
        "1e-4",
    ]

    with out_log.open("w", encoding="utf-8") as out_f, err_log.open(
        "w", encoding="utf-8"
    ) as err_f:
        print(f"[test15-runner] start dataset={dataset}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=out_f,
            stderr=err_f,
        )
        rc = proc.wait()
        print(f"[test15-runner] done  dataset={dataset} rc={rc}", flush=True)
    return rc


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"Python executable not found: {PYTHON_EXE}")

    failures = []
    for ds in DATASETS:
        rc = run_one(dataset=ds)
        if rc != 0:
            failures.append((ds, rc))

    if failures:
        print("[test15-runner] finished with failures:", flush=True)
        for ds, rc in failures:
            print(f"  - {ds}: rc={rc}", flush=True)
        sys.exit(1)

    print("[test15-runner] all datasets completed successfully.", flush=True)


if __name__ == "__main__":
    main()
