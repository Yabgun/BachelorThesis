import os
from pathlib import Path
import pandas as pd
import kagglehub


DATA_DIR = Path("data/covid_ct_cxr")
TARGET_CSV = DATA_DIR / "healthcare_dataset.csv"
BACKUP_CSV = DATA_DIR / "healthcare_dataset_backup.csv"

EXPECTED_COLS = [
    "Name",
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Date of Admission",
    "Doctor",
    "Hospital",
    "Insurance Provider",
    "Billing Amount",
    "Room Number",
    "Admission Type",
    "Discharge Date",
    "Medication",
    "Test Results",
]


def find_candidate_csvs(root: Path) -> list[Path]:
    candidates = []
    for p in root.rglob("*.csv"):
        try:
            head = pd.read_csv(p, nrows=1)
        except Exception:
            continue
        cols = list(head.columns)
        have = set(c.strip() for c in cols)
        need = set(EXPECTED_COLS)
        if need.issubset(have):
            candidates.append(p)
    return candidates


def load_merge_and_write(candidates: list[Path]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_CSV.exists() and not BACKUP_CSV.exists():
        TARGET_CSV.replace(BACKUP_CSV)
    frames = []
    if BACKUP_CSV.exists():
        try:
            frames.append(pd.read_csv(BACKUP_CSV))
        except Exception:
            pass
    for p in candidates:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No valid CSVs to merge from Kaggle dataset.")
    df = pd.concat(frames, ignore_index=True, copy=False)
    df = df.drop_duplicates()
    df.to_csv(TARGET_CSV, index=False)
    print(f"Wrote merged CSV to {TARGET_CSV} with {len(df):,} rows")


def main() -> None:
    path = kagglehub.dataset_download("prasad22/healthcare-dataset")
    root = Path(path)
    candidates = find_candidate_csvs(root)
    if not candidates:
        raise RuntimeError("No Kaggle CSV with expected schema found.")
    load_merge_and_write(candidates)


if __name__ == "__main__":
    main()



