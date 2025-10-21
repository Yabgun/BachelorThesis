import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from phe import paillier

MULTIMODAL_PATH = Path('data/covid_ct_cxr/multimodal.csv')
POLICY_PATH = Path('config/selective_he_policy.json')
RESULTS_CSV = Path('data/covid_ct_cxr/he_results.csv')
REPORT_JSON = Path('data/covid_ct_cxr/he_report.json')


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_policy(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def plain_score(row: Dict[str, str], weights: Dict[str, float], bias: float) -> float:
    s = 0.0
    for k, w in weights.items():
        s += to_float(row.get(k, 0.0)) * float(w)
    s += float(bias)
    return s


def he_score_paillier(row: Dict[str, str], weights: Dict[str, float], bias: float, pub, priv) -> float:
    # Encrypt selected features and accumulate weighted sum
    enc_sum = None
    for k, w in weights.items():
        v = to_float(row.get(k, 0.0))
        ct = pub.encrypt(v)
        term = ct * float(w)
        if enc_sum is None:
            enc_sum = term
        else:
            enc_sum = enc_sum + term
    enc_sum = enc_sum + float(bias)
    dec = priv.decrypt(enc_sum)
    return float(dec)


def main():
    rows = read_csv(MULTIMODAL_PATH)
    policy = read_policy(POLICY_PATH)

    weights = policy['weights']
    bias = float(policy.get('bias', 0.0))

    pub, priv = paillier.generate_paillier_keypair()

    out_rows = []
    abs_errors = []
    he_times = []
    plain_times = []

    for r in rows:
        ps_t0 = time.perf_counter()
        ps = plain_score(r, weights, bias)
        plain_times.append((time.perf_counter() - ps_t0) * 1000.0)

        he_t0 = time.perf_counter()
        hs = he_score_paillier(r, weights, bias, pub, priv)
        he_times.append((time.perf_counter() - he_t0) * 1000.0)

        abs_err = abs(ps - hs)
        abs_errors.append(abs_err)

        out_rows.append({
            'patient_id': r['patient_id'],
            'plaintext_score': ps,
            'he_decrypted_score': hs,
            'abs_error': abs_err,
            'plain_ms': plain_times[-1],
            'he_ms': he_times[-1],
        })

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['patient_id','plaintext_score','he_decrypted_score','abs_error','plain_ms','he_ms'])
        w.writeheader()
        w.writerows(out_rows)

    report = {
        'samples': len(out_rows),
        'mean_abs_error': sum(abs_errors)/len(abs_errors) if abs_errors else 0.0,
        'max_abs_error': max(abs_errors) if abs_errors else 0.0,
        'mean_he_ms': sum(he_times)/len(he_times) if he_times else 0.0,
        'mean_plain_ms': sum(plain_times)/len(plain_times) if plain_times else 0.0,
        'policy': policy,
    }
    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Wrote results to {RESULTS_CSV} and report to {REPORT_JSON}")


if __name__ == '__main__':
    main()