import csv
import random
from pathlib import Path

HEALTH_PATH = Path('data/covid_ct_cxr/healthcare_clean.csv')
CXR_FEAT_PATH = Path('data/covid_ct_cxr/cxr_features.csv')
OUT_PATH = Path('data/covid_ct_cxr/multimodal.csv')

random.seed(42)


def read_csv(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    health_rows = read_csv(HEALTH_PATH)
    cxr_rows = read_csv(CXR_FEAT_PATH)

    if not health_rows or not cxr_rows:
        raise RuntimeError('Missing inputs for multimodal join.')

    # Match N images to N patients randomly (demo)
    N = min(len(health_rows), len(cxr_rows))
    health_sel = random.sample(health_rows, N)
    random.shuffle(cxr_rows)

    # Build output rows
    out_rows = []
    for h, c in zip(health_sel, cxr_rows[:N]):
        out_rows.append({
            'patient_id': h['patient_id'],
            'age': h['age'],
            'billing_amount': h['billing_amount'],
            'billing_amount_norm': h['billing_amount_norm'],
            'test_results_score': h['test_results_score'],
            'image_path': c['image_path'],
            'cxr_mean_intensity': c['cxr_mean_intensity'],
            'cxr_edge_density': c['cxr_edge_density'],
            'cxr_entropy': c['cxr_entropy'],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'patient_id','age','billing_amount','billing_amount_norm','test_results_score',
            'image_path','cxr_mean_intensity','cxr_edge_density','cxr_entropy'
        ])
        w.writeheader()
        w.writerows(out_rows)

    print(f'Wrote {len(out_rows)} rows to {OUT_PATH}')


if __name__ == '__main__':
    main()