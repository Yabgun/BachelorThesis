import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from Pyfhel import Pyfhel, PyCtxt, PyPtxt
    REAL = True
except ImportError as e:
    raise SystemExit(
        "Pyfhel is required. Install with 'pip install pyfhel'. On Windows, if wheel is not available for your Python version, install CMake, Ninja, and MSVC Build Tools, or use Python 3.11–3.12."
    ) from e


def enc_sum(he: Pyfhel, values: List[float]) -> float:
    ct_sum = None
    for x in values:
        p = he.encodeFrac([x])
        c = he.encryptPtxt(p)
        ct_sum = c if ct_sum is None else (ct_sum + c)
    dec = he.decryptFrac(ct_sum)
    return float(dec[0])


def run_scenarios(params: Dict, a: float = 1.0, b: float = 0.0) -> Dict:
    he = Pyfhel()
    he.contextGen(scheme="CKKS", **params)
    he.keyGen()
    good = [a] * 100
    bad = [a] * 99 + [b]
    t0 = time.time()
    sum_good = enc_sum(he, good)
    sum_bad = enc_sum(he, bad)
    elapsed = (time.time() - t0) * 1000.0
    mean_good = sum_good / 100.0
    mean_bad = sum_bad / 100.0
    exp_sum_good = 100.0 * a
    exp_sum_bad = 99.0 * a + b
    exp_mean_good = a
    exp_mean_bad = (99.0 * a + b) / 100.0
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


def optimize() -> Dict:
    grid = []
    for n in [2**12, 2**13, 2**14]:
        for scale in [2**30, 2**40, 2**50]:
            qi = [60, 40, 40, 60] if n != 2**12 else [60, 40, 60]
            grid.append({"n": n, "scale": scale, "qi_sizes": qi})
    results = []
    for p in grid:
        r = run_scenarios(p, a=1.0, b=0.0)
        score = r["err_mean_good"] + r["err_mean_bad"]
        results.append({"params": p, "metrics": r, "score": score})
    results.sort(key=lambda x: (x["score"], x["metrics"]["elapsed_ms"]))
    best = results[0] if results else None
    return {"results": results, "best": best}


def write_reports(out_json: Path, out_md: Path, data: Dict) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    lines = []
    lines.append("# CKKS Parametre Optimizasyonu")
    lines.append("")
    lines.append("Senaryo: iyi=100a, kötü=99a+1b; a=1.0, b=0.0")
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
        p = r["params"]
        m = r["metrics"]
        lines.append(f"{i}. n={p['n']} scale={p['scale']} qi={p['qi_sizes']} score={r['score']:.3e} delta={m['delta_mean']:.6f}")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    data = optimize()
    write_reports(
        Path("data/covid_ct_cxr/ckks_param_optimization.json"),
        Path("data/covid_ct_cxr/ckks_param_optimization.md"),
        data,
    )
    print("Parametre optimizasyonu tamamlandı.")


if __name__ == "__main__":
    main()



