import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

try:
    from Pyfhel import Pyfhel, PyCtxt, PyPtxt
    REAL = True
except ImportError as e:
    raise SystemExit(
        "Pyfhel is required. Install with 'pip install pyfhel'. On Windows, if wheel is not available for your Python version, install CMake, Ninja, and MSVC Build Tools, or use Python 3.11–3.12."
    ) from e


DATA_DIR = Path("data/covid_ct_cxr")
MM_PATH = DATA_DIR / "multimodal.csv"
POLICY_PATH = Path("config/selective_he_policy.json")
OUT_JSON = DATA_DIR / "ckks_param_optimization_multimodal.json"
OUT_MD = DATA_DIR / "ckks_param_optimization_multimodal.md"


def load_policy(path: Path) -> Tuple[List[str], List[float], float, Dict]:
    pol = json.loads(path.read_text(encoding="utf-8"))
    cols = pol["encrypt_columns"]
    weights_map = pol["weights"]
    weights = [weights_map[c] for c in cols]
    bias = float(pol.get("bias", 0.0))
    return cols, weights, bias, pol


def plain_score(row: pd.Series, cols: List[str], weights: List[float], bias: float) -> float:
    s = 0.0
    for c, w in zip(cols, weights):
        s += float(row[c]) * float(w)
    s += bias
    return float(s)


def pick_a_b_values(df: pd.DataFrame, cols: List[str], weights: List[float], bias: float) -> Tuple[float, float, Dict]:
    # Use 'test_results_score' as class proxy to choose a and b from real data
    if "test_results_score" not in df.columns:
        raise SystemExit("multimodal.csv must include column 'test_results_score'")
    mode_val = df["test_results_score"].mode(dropna=True)
    if mode_val.empty:
        mode = 0.0
    else:
        mode = float(mode_val.iloc[0])
    same = df[df["test_results_score"] == mode]
    diff = df[df["test_results_score"] != mode]
    if same.empty or diff.empty:
        # Fallback: just take first and last rows
        a_row = df.iloc[0]
        b_row = df.iloc[-1]
    else:
        a_row = same.sample(1, random_state=7).iloc[0]
        b_row = diff.sample(1, random_state=17).iloc[0]
    a_val = plain_score(a_row, cols, weights, bias)
    b_val = plain_score(b_row, cols, weights, bias)
    meta = {
        "a_test_results_score": float(a_row["test_results_score"]),
        "b_test_results_score": float(b_row["test_results_score"]),
        "a_patient_id": a_row.get("patient_id", None),
        "b_patient_id": b_row.get("patient_id", None),
        "a_row_index": int(getattr(a_row, "name", -1)),
        "b_row_index": int(getattr(b_row, "name", -1)),
    }
    return a_val, b_val, meta


def enc_sum(he: Pyfhel, values: List[float]) -> float:
    ct_sum = None
    for x in values:
        p = he.encodeFrac([x])
        c = he.encryptPtxt(p)
        ct_sum = c if ct_sum is None else (ct_sum + c)
    dec = he.decryptFrac(ct_sum)
    return float(dec[0])


def eval_params(params: Dict, a_val: float, b_val: float) -> Dict:
    he = Pyfhel()
    he.contextGen(scheme="CKKS", **params)
    he.keyGen()
    # Scenarios: good=100a, bad=99a+1b using multimodal-derived a,b
    good = [a_val] * 100
    bad = [a_val] * 99 + [b_val]
    t0 = time.time()
    sum_good = enc_sum(he, good)
    sum_bad = enc_sum(he, bad)
    elapsed = (time.time() - t0) * 1000.0
    mean_good = sum_good / 100.0
    mean_bad = sum_bad / 100.0
    exp_sum_good = 100.0 * a_val
    exp_sum_bad = 99.0 * a_val + b_val
    exp_mean_good = a_val
    exp_mean_bad = (99.0 * a_val + b_val) / 100.0
    return {
        "sum_good": sum_good,
        "sum_bad": sum_bad,
        "mean_good": mean_good,
        "mean_bad": mean_bad,
        "err_sum_good": abs(sum_good - exp_sum_good),
        "err_sum_bad": abs(sum_bad - exp_sum_bad),
        "err_mean_good": abs(mean_good - exp_mean_good),
        "err_mean_bad": abs(mean_bad - exp_mean_bad),
        "delta_mean": mean_good - mean_bad,
        "exp_delta_mean": exp_mean_good - exp_mean_bad,
        "elapsed_ms": elapsed,
    }


def optimize_on_multimodal() -> Dict:
    df = pd.read_csv(MM_PATH)
    cols, weights, bias, pol = load_policy(POLICY_PATH)
    a_val, b_val, meta = pick_a_b_values(df, cols, weights, bias)
    grid = []
    for n in [2**12, 2**13, 2**14]:
        for scale in [2**30, 2**40, 2**50]:
            qi = [60, 40, 40, 60] if n != 2**12 else [60, 40, 60]
            grid.append({"n": n, "scale": scale, "qi_sizes": qi})
    results = []
    for p in grid:
        m = eval_params(p, a_val, b_val)
        score = m["err_mean_good"] + m["err_mean_bad"]
        results.append({"params": p, "metrics": m, "score": score})
    results.sort(key=lambda x: (x["score"], x["metrics"]["elapsed_ms"]))
    best = results[0] if results else None
    return {
        "policy": pol,
        "a_value": a_val,
        "b_value": b_val,
        "a_b_meta": meta,
        "results": results,
        "best": best,
    }


def write_reports(data: Dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy/pandas types
    def _to_py(obj):
        if isinstance(obj, dict):
            return {k: _to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_py(v) for v in obj]
        # pandas/numpy scalars to python
        try:
            import numpy as _np  # type: ignore
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, (_np.ndarray,)):
                return obj.tolist()
        except Exception:
            pass
        return obj
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(_to_py(data), f, indent=2)
    lines = []
    lines.append("# CKKS Parametre Optimizasyonu (Multimodal)")
    lines.append("")
    lines.append("Senaryo: iyi=100a, kötü=99a+1b; a,b multimodal skorlarından seçildi")
    lines.append(f"a={data['a_value']:.6f}, b={data['b_value']:.6f}, delta_exp={(data['best']['metrics']['exp_delta_mean'] if data.get('best') else 0):.6f}")
    if data.get("best"):
        bp = data["best"]["params"]
        bm = data["best"]["metrics"]
        lines.append(f"En iyi: n={bp['n']}, scale={bp['scale']}, qi={bp['qi_sizes']}")
        lines.append(f"Delta mean: {bm['delta_mean']:.6f} (beklenen {bm['exp_delta_mean']:.6f})")
        lines.append(f"Hata(mean): good={bm['err_mean_good']:.3e}, bad={bm['err_mean_bad']:.3e}")
        lines.append(f"Süre(ms): {bm['elapsed_ms']:.2f}")
    lines.append("")
    lines.append("Top sonuçlar:")
    for i, r in enumerate(data.get("results", [])[:5], 1):
        p = r["params"]; m = r["metrics"]
        lines.append(f"{i}. n={p['n']} scale={p['scale']} qi={p['qi_sizes']} score={r['score']:.3e} delta={m['delta_mean']:.6f}")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    data = optimize_on_multimodal()
    write_reports(data)
    print("Multimodal senaryolarla parametre optimizasyonu tamamlandı.")


if __name__ == "__main__":
    main()


