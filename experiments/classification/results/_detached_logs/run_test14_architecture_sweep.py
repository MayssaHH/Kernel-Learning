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

# Requested order:
# 1) all_linear on all datasets
# 2) half_rbf_half_linear on all datasets
ARCHITECTURES = [
    "all_linear",
    "half_rbf_half_linear",
]

LOGS_DIR = ROOT / "experiments" / "classification" / "results" / "_detached_logs"
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"


def run_one(arch: str, dataset: str) -> int:
    out_log = LOGS_DIR / f"test14_{arch}_{dataset}.out.log"
    err_log = LOGS_DIR / f"test14_{arch}_{dataset}.err.log"

    out_log.write_text("", encoding="utf-8")
    err_log.write_text("", encoding="utf-8")

    cmd = [
        str(PYTHON_EXE),
        "-m",
        "experiments.classification.test14",
        "--dataset",
        dataset,
        "--architectures",
        arch,
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
        print(f"[sweep] start arch={arch} dataset={dataset}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=out_f,
            stderr=err_f,
        )
        rc = proc.wait()
        print(f"[sweep] done  arch={arch} dataset={dataset} rc={rc}", flush=True)
    return rc


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"Python executable not found: {PYTHON_EXE}")

    failures = []
    for arch in ARCHITECTURES:
        print(f"[sweep] phase start architecture={arch}", flush=True)
        for ds in DATASETS:
            rc = run_one(arch=arch, dataset=ds)
            if rc != 0:
                failures.append((arch, ds, rc))
        print(f"[sweep] phase done  architecture={arch}", flush=True)

    if failures:
        print("[sweep] completed with failures:", flush=True)
        for arch, ds, rc in failures:
            print(f"  - arch={arch} dataset={ds} rc={rc}", flush=True)
        sys.exit(1)

    print("[sweep] all architecture runs completed successfully.", flush=True)


if __name__ == "__main__":
    main()

