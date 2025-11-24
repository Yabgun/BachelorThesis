import csv
import re
from pathlib import Path

IN_PATH = Path('data/covid_ct_cxr/healthcare_dataset.csv')
OUT_PATH = Path('data/covid_ct_cxr/healthcare_clean.csv')

# Map test result strings to numeric labels
TEST_MAP = {
    'normal': 0.0,
    'abnormal': 1.0,
    'inconclusive': 0.5,
}

# Columns we will keep (and possibly transform)
KEEP_COLS = [
    'Age', 'Gender', 'Blood Type', 'Medical Condition', 'Billing Amount', 'Test Results'
]


def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    # remove thousand separators or stray characters
    s = s.replace(',', '')
    try:
        return float(s)
    except Exception:
        return None


def to_int(x):
    f = to_float(x)
    return int(f) if f is not None else None


def norm_text(x):
    return str(x).strip()


def map_test_result(x):
    key = str(x).strip().lower()
    return TEST_MAP.get(key, None)


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f'Input not found: {IN_PATH}')

    rows = []
    with open(IN_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        pid = 1
        for r in reader:
            # Skip if essential fields missing
            age = to_int(r.get('Age'))
            bill = to_float(r.get('Billing Amount'))
            test_score = map_test_result(r.get('Test Results'))
            if age is None or bill is None or test_score is None:
                continue
            gender = norm_text(r.get('Gender'))
            btype = norm_text(r.get('Blood Type'))
            cond = norm_text(r.get('Medical Condition'))

            rows.append({
                'patient_id': pid,
                'age': age,
                'gender': gender,
                'blood_type': btype,
                'condition': cond,
                'billing_amount': bill,
                'test_results_score': test_score,
            })
            pid += 1

    if not rows:
        raise RuntimeError('No valid rows parsed from healthcare_dataset.csv')

    # Compute normalization for billing_amount
    max_bill = max(r['billing_amount'] for r in rows)
    for r in rows:
        r['billing_amount_norm'] = (r['billing_amount'] / max_bill) if max_bill > 0 else 0.0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'patient_id','age','gender','blood_type','condition',
            'billing_amount','billing_amount_norm','test_results_score'
        ])
        w.writeheader()
        w.writerows(rows)

    print(f'Wrote {len(rows)} rows to {OUT_PATH}')


if __name__ == '__main__':
    main()