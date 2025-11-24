import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Require real Pyfhel. If missing, guide user to install.
try:
    from Pyfhel import Pyfhel, PyCtxt, PyPtxt
except ImportError as e:
    raise SystemExit(
        "Pyfhel is required. Install with 'pip install pyfhel' or use Python 3.11–3.12 where wheels are available on Windows. "
        "On Windows, ensure CMake, Ninja, and MSVC Build Tools are installed if building from source."
    ) from e

DATA_DIR = Path("data/covid_ct_cxr")
CONFIG_DIR = Path("config")
MULTIMODAL_PATH = DATA_DIR/"multimodal.csv"
POLICY_PATH = CONFIG_DIR/"selective_he_policy.json"
CKKS_RESULTS = DATA_DIR/"ckks_results.csv"
CKKS_REPORT = DATA_DIR/"ckks_report.json"


def load_policy(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        pol = json.load(f)
    cols = pol["encrypt_columns"]
    weights_map = pol["weights"]
    weights = [weights_map[c] for c in cols]
    bias = pol.get("bias", 0.0)
    return cols, weights, bias, pol


def run_ckks(df: pd.DataFrame, cols, weights, bias):
    HE = Pyfhel()
    # CKKS parameters: moderate size for demo; scale large enough for few ops
    HE.contextGen(scheme='CKKS', n=2**13, scale=2**30, qi_sizes=[60, 40, 40, 60])
    HE.keyGen()
    HE.relinKeyGen()  # not strictly needed for mul_plain, but good to have

    results = []
    lat_ckks = []
    lat_plain = []

    for _, row in df.iterrows():
        xvals = [float(row[c]) for c in cols]
        # plaintext score
        t0p = time.time()
        plain_score = sum(w*x for w, x in zip(weights, xvals)) + bias
        lat_plain.append((time.time()-t0p)*1000.0)

        # CKKS encrypted score: encrypt each scalar and multiply by plaintext weight, then sum
        t0 = time.time()
        ct_sum = None
        for w, x in zip(weights, xvals):
            # encodeFrac expects a buffer-compatible sequence (e.g., numpy array)
            ptxt_x = HE.encodeFrac(np.array([x], dtype=np.float64))
            ct_x = HE.encryptPtxt(ptxt_x)
            ptxt_w = HE.encodeFrac(np.array([w], dtype=np.float64))
            ct_xw = ct_x * ptxt_w  # ciphertext-plaintext multiplication
            ct_sum = ct_xw if ct_sum is None else (ct_sum + ct_xw)
        # add bias as plaintext
        ptxt_b = HE.encodeFrac(np.array([bias], dtype=np.float64))
        # ensure consistent types by encrypting bias before addition
        ct_sum = ct_sum + HE.encryptPtxt(ptxt_b)

        dec = HE.decryptFrac(ct_sum)
        ckks_score = float(dec[0])
        lat_ckks.append((time.time()-t0)*1000.0)

        results.append({
            "patient_id": row.get("patient_id", "unknown"),
            "ckks_score": ckks_score,
            "plaintext_score": plain_score,
            "abs_error": abs(ckks_score-plain_score)
        })

    return results, {
        "mean_ckks_ms": sum(lat_ckks)/len(lat_ckks) if lat_ckks else None,
        "mean_plain_ms": sum(lat_plain)/len(lat_plain) if lat_plain else None,
    }


def main():
    if not MULTIMODAL_PATH.exists():
        raise SystemExit(f"Multimodal file not found: {MULTIMODAL_PATH}")
    if not POLICY_PATH.exists():
        raise SystemExit(f"Policy file not found: {POLICY_PATH}")

    df = pd.read_csv(MULTIMODAL_PATH)
    cols, weights, bias, pol = load_policy(POLICY_PATH)

    results, perf = run_ckks(df, cols, weights, bias)

    # write results CSV
    pd.DataFrame(results).to_csv(CKKS_RESULTS, index=False)

    # write report JSON
    report = {
        "samples": len(results),
        "mean_abs_error": sum(r["abs_error"] for r in results)/len(results) if results else None,
        "max_abs_error": max(r["abs_error"] for r in results) if results else None,
        "performance_ms": perf,
        "policy": pol,
        "notes": "CKKS is approximate; small abs_error is expected and acceptable for weighted-sum use-cases"
    }
    with open(CKKS_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"CKKS results -> {CKKS_RESULTS}")
    print(f"CKKS report  -> {CKKS_REPORT}")


if __name__ == '__main__':
    main()
