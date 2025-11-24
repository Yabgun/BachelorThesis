import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

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
OUT_DIR = Path("data/test_vectors")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "ckks_advanced_test_vectors.json"
OUT_MD = OUT_DIR / "ckks_advanced_test_report.md"
OUT_CSV = OUT_DIR / "test_vector_results.csv"


@dataclass
class TestScenario:
    name: str
    description: str
    a_multiplier: int
    b_multiplier: int
    expected_diff: float
    is_edge_case: bool = False


@dataclass
class CKKSParams:
    n: int
    scale: int
    qi_sizes: List[int]
    
    def to_dict(self) -> Dict:
        return {"n": self.n, "scale": self.scale, "qi_sizes": self.qi_sizes}


class AdvancedCKKSTestGenerator:
    def __init__(self):
        self.scenarios = self._define_test_scenarios()
        self.param_grid = self._generate_param_grid()
        self.results = []
        
    def _define_test_scenarios(self) -> List[TestScenario]:
        return [
            TestScenario("İyi Senaryo", "100 iyi hasta, 0 kötü hasta", 100, 0, 0.0, False),
            TestScenario("Kötü Senaryo", "0 iyi hasta, 100 kötü hasta", 0, 100, 0.0, False),
            TestScenario("Karışık Dengeli", "50 iyi hasta, 50 kötü hasta", 50, 50, 0.0, False),
            TestScenario("Ağırlıklı İyi", "80 iyi hasta, 20 kötü hasta", 80, 20, 0.0, False),
            TestScenario("Ağırlıklı Kötü", "20 iyi hasta, 80 kötü hasta", 20, 80, 0.0, False),
            TestScenario("Marjinal İyi", "95 iyi hasta, 5 kötü hasta", 95, 5, 0.0, False),
            TestScenario("Marjinal Kötü", "5 iyi hasta, 95 kötü hasta", 5, 95, 0.0, False),
            TestScenario("Sınır Vakası 1", "99 iyi hasta, 1 kötü hasta", 99, 1, 0.0, True),
            TestScenario("Sınır Vakası 2", "1 iyi hasta, 99 kötü hasta", 1, 99, 0.0, True),
            TestScenario("Dengesiz Büyük", "150 iyi hasta, 50 kötü hasta", 150, 50, 0.0, True),
            TestScenario("Dengesiz Küçük", "10 iyi hasta, 90 kötü hasta", 10, 90, 0.0, True),
        ]
    
    def _generate_param_grid(self) -> List[CKKSParams]:
        grid = []
        
        # Temel parametreler
        n_values = [2**12, 2**13, 2**14, 2**15]
        scale_values = [2**30, 2**35, 2**40, 2**45, 2**50]
        
        # Farklı derinlikler için qi_size kombinasyonları
        qi_configs = [
            [60, 40, 60],           # Basit 3 seviye
            [60, 40, 40, 60],       # Orta 4 seviye
            [60, 40, 30, 40, 60],  # Derin 5 seviye
            [60, 35, 35, 35, 60],  # Alternatif derinlik
            [50, 40, 40, 40, 50],  # Düşük başlangıç/bitir
        ]
        
        for n in n_values:
            for scale in scale_values:
                for qi_sizes in qi_configs:
                    # n değerine göre uygun qi_size'ları filtrele
                    if n >= 2**13 or len(qi_sizes) <= 4:  # Küçük n için daha az derinlik
                        grid.append(CKKSParams(n, scale, qi_sizes))
        
        return grid
    
    def load_policy(self, path: Path) -> Tuple[List[str], List[float], float, Dict]:
        pol = json.loads(path.read_text(encoding="utf-8"))
        cols = pol["encrypt_columns"]
        weights_map = pol["weights"]
        weights = [weights_map[c] for c in cols]
        bias = float(pol.get("bias", 0.0))
        return cols, weights, bias, pol
    
    def plain_score(self, row: pd.Series, cols: List[str], weights: List[float], bias: float) -> float:
        s = 0.0
        for c, w in zip(cols, weights):
            s += float(row[c]) * float(w)
        s += bias
        return float(s)
    
    def pick_a_b_values(self, df: pd.DataFrame, cols: List[str], weights: List[float], bias: float) -> Tuple[float, float, Dict]:
        if "test_results_score" not in df.columns:
            raise ValueError("multimodal.csv must include column 'test_results_score'")
        
        # Test skorlarına göre gruplandır
        high_scores = df[df["test_results_score"] >= 0.8]
        low_scores = df[df["test_results_score"] <= 0.2]
        
        if len(high_scores) == 0 or len(low_scores) == 0:
            # Fallback: ilk ve son satırları kullan
            a_row = df.iloc[0]
            b_row = df.iloc[-1]
        else:
            # Rastgele seçim yap
            a_row = high_scores.sample(1, random_state=42).iloc[0]
            b_row = low_scores.sample(1, random_state=24).iloc[0]
        
        a_val = self.plain_score(a_row, cols, weights, bias)
        b_val = self.plain_score(b_row, cols, weights, bias)
        
        meta = {
            "a_test_results_score": float(a_row["test_results_score"]),
            "b_test_results_score": float(b_row["test_results_score"]),
            "a_patient_id": a_row.get("patient_id", None),
            "b_patient_id": b_row.get("patient_id", None),
            "a_row_index": int(a_row.name),
            "b_row_index": int(b_row.name),
        }
        return a_val, b_val, meta
    
    def enc_sum(self, he: Pyfhel, values: List[float]) -> float:
        ct_sum = None
        for x in values:
            p = he.encodeFrac([x])
            c = he.encryptPtxt(p)
            ct_sum = c if ct_sum is None else (ct_sum + c)
        dec = he.decryptFrac(ct_sum)
        return float(dec[0])
    
    def evaluate_scenario(self, he: Pyfhel, scenario: TestScenario, a_val: float, b_val: float) -> Dict[str, Any]:
        # Senaryoya göre hasta listesi oluştur
        good_patients = [a_val] * scenario.a_multiplier
        bad_patients = [b_val] * scenario.b_multiplier
        
        # Toplamları hesapla
        t0 = time.time()
        sum_good = self.enc_sum(he, good_patients) if good_patients else 0.0
        sum_bad = self.enc_sum(he, bad_patients) if bad_patients else 0.0
        elapsed = (time.time() - t0) * 1000.0
        
        # Beklenen değerleri hesapla
        exp_sum_good = sum(good_patients)
        exp_sum_bad = sum(bad_patients)
        
        # Hataları hesapla
        err_sum_good = abs(sum_good - exp_sum_good)
        err_sum_bad = abs(sum_bad - exp_sum_bad)
        
        # Ortalama değerler
        mean_good = sum_good / len(good_patients) if good_patients else 0.0
        mean_bad = sum_bad / len(bad_patients) if bad_patients else 0.0
        exp_mean_good = exp_sum_good / len(good_patients) if good_patients else 0.0
        exp_mean_bad = exp_sum_bad / len(bad_patients) if bad_patients else 0.0
        
        # Ortalama hatalar
        err_mean_good = abs(mean_good - exp_mean_good)
        err_mean_bad = abs(mean_bad - exp_mean_bad)
        
        # Delta hesaplamaları
        delta_mean = mean_good - mean_bad if good_patients and bad_patients else 0.0
        exp_delta_mean = exp_mean_good - exp_mean_bad if good_patients and bad_patients else 0.0
        
        return {
            "scenario_name": scenario.name,
            "description": scenario.description,
            "is_edge_case": scenario.is_edge_case,
            "patient_counts": {
                "good": len(good_patients),
                "bad": len(bad_patients),
                "total": len(good_patients) + len(bad_patients)
            },
            "sum_good": sum_good,
            "sum_bad": sum_bad,
            "mean_good": mean_good,
            "mean_bad": mean_bad,
            "err_sum_good": err_sum_good,
            "err_sum_bad": err_sum_bad,
            "err_mean_good": err_mean_good,
            "err_mean_bad": err_mean_bad,
            "delta_mean": delta_mean,
            "exp_delta_mean": exp_delta_mean,
            "elapsed_ms": elapsed,
            "accuracy_metrics": {
                "sum_accuracy": 1 - (err_sum_good + err_sum_bad) / (abs(exp_sum_good) + abs(exp_sum_bad) + 1e-15),
                "mean_accuracy": 1 - (err_mean_good + err_mean_bad) / (abs(exp_mean_good) + abs(exp_mean_bad) + 1e-15),
                "max_error": max(err_sum_good, err_sum_bad, err_mean_good, err_mean_bad)
            }
        }
    
    def evaluate_params(self, params: CKKSParams, a_val: float, b_val: float) -> Dict[str, Any]:
        he = Pyfhel()
        he.contextGen(scheme="CKKS", **params.to_dict())
        he.keyGen()
        
        param_results = []
        total_time = 0
        
        for scenario in self.scenarios:
            result = self.evaluate_scenario(he, scenario, a_val, b_val)
            param_results.append(result)
            total_time += result["elapsed_ms"]
        
        # Parametre bazlı metrikler
        total_accuracy = sum(r["accuracy_metrics"]["mean_accuracy"] for r in param_results) / len(param_results)
        max_error = max(r["accuracy_metrics"]["max_error"] for r in param_results)
        avg_time = total_time / len(param_results)
        
        return {
            "params": params.to_dict(),
            "scenario_results": param_results,
            "param_metrics": {
                "total_accuracy": total_accuracy,
                "max_error": max_error,
                "avg_time_ms": avg_time,
                "total_time_ms": total_time,
                "scenario_count": len(param_results)
            }
        }
    
    def run_full_evaluation(self, df: pd.DataFrame, cols: List[str], weights: List[float], bias: float) -> Dict[str, Any]:
        a_val, b_val, meta = self.pick_a_b_values(df, cols, weights, bias)
        
        print(f"Test vektörleri oluşturuluyor... a={a_val:.6f}, b={b_val:.6f}")
        print(f"Toplam {len(self.param_grid)} parametre kombinasyonu test edilecek...")
        
        all_results = []
        
        for i, params in enumerate(self.param_grid):
            if i % 10 == 0:
                print(f"Parametre test ediliyor: {i+1}/{len(self.param_grid)}")
            
            try:
                result = self.evaluate_params(params, a_val, b_val)
                all_results.append(result)
            except Exception as e:
                print(f"Parametre testi başarısız: {params.to_dict()}, Hata: {e}")
                continue
        
        # Sonuçları sırala (doğruluk + süre bazlı)
        all_results.sort(key=lambda x: (
            -x["param_metrics"]["total_accuracy"],  # Yüksek doğruluk
            x["param_metrics"]["avg_time_ms"]      # Düşük süre
        ))
        
        best_result = all_results[0] if all_results else None
        
        return {
            "metadata": {
                "test_date": datetime.now().isoformat(),
                "total_scenarios": len(self.scenarios),
                "total_params": len(self.param_grid),
                "a_value": a_val,
                "b_value": b_val,
                "a_b_meta": meta
            },
            "policy": {
                "encrypt_columns": cols,
                "weights": dict(zip(cols, weights)),
                "bias": bias
            },
            "test_scenarios": [vars(s) for s in self.scenarios],
            "all_results": all_results,
            "best_result": best_result,
            "summary": self._generate_summary(all_results)
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {}
        
        accuracies = [r["param_metrics"]["total_accuracy"] for r in results]
        times = [r["param_metrics"]["avg_time_ms"] for r in results]
        errors = [r["param_metrics"]["max_error"] for r in results]
        
        return {
            "total_tests": len(results),
            "accuracy_stats": {
                "min": min(accuracies),
                "max": max(accuracies),
                "avg": sum(accuracies) / len(accuracies),
                "median": sorted(accuracies)[len(accuracies)//2]
            },
            "time_stats": {
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "median": sorted(times)[len(times)//2]
            },
            "error_stats": {
                "min": min(errors),
                "max": max(errors),
                "avg": sum(errors) / len(errors),
                "median": sorted(errors)[len(errors)//2]
            },
            "top_5_params": [
                {
                    "rank": i+1,
                    "params": r["params"],
                    "accuracy": r["param_metrics"]["total_accuracy"],
                    "avg_time": r["param_metrics"]["avg_time_ms"],
                    "max_error": r["param_metrics"]["max_error"]
                }
                for i, r in enumerate(results[:5])
            ]
        }


def write_detailed_reports(data: Dict) -> None:
    # JSON raporu
    def _to_py(obj):
        if isinstance(obj, dict):
            return {k: _to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_py(v) for v in obj]
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(_to_py(data), f, indent=2, ensure_ascii=False)
    
    # Markdown raporu
    lines = []
    lines.append("# Gelişmiş CKKS Test Vektörleri ve Parametre Optimizasyonu")
    lines.append("")
    lines.append(f"**Test Tarihi:** {data['metadata']['test_date']}")
    lines.append(f"**Toplam Senaryo:** {data['metadata']['total_scenarios']}")
    lines.append(f"**Toplam Parametre:** {data['metadata']['total_params']}")
    lines.append("")
    
    # A ve B değerleri
    lines.append("## Test Vektörleri")
    lines.append(f"- **a değeri:** {data['metadata']['a_value']:.6f}")
    lines.append(f"- **b değeri:** {data['metadata']['b_value']:.6f}")
    lines.append(f"- **Fark (a-b):** {data['metadata']['a_value'] - data['metadata']['b_value']:.6f}")
    lines.append("")
    
    # En iyi parametre
    if data.get('best_result'):
        best = data['best_result']
        lines.append("## En İyi Parametre Kombinasyonu")
        bp = best['params']
        bm = best['param_metrics']
        lines.append(f"- **n:** {bp['n']}")
        lines.append(f"- **scale:** {bp['scale']}")
        lines.append(f"- **qi_sizes:** {bp['qi_sizes']}")
        lines.append(f"- **Toplam Doğruluk:** {bm['total_accuracy']:.6f}")
        lines.append(f"- **Maks Hata:** {bm['max_error']:.3e}")
        lines.append(f"- **Ortalama Süre:** {bm['avg_time_ms']:.2f} ms")
        lines.append("")
    
    # Özet istatistikler
    if data.get('summary'):
        summary = data['summary']
        lines.append("## Özet İstatistikler")
        lines.append(f"- **Test Sayısı:** {summary['total_tests']}")
        lines.append(f"- **Doğruluk Aralığı:** {summary['accuracy_stats']['min']:.6f} - {summary['accuracy_stats']['max']:.6f}")
        lines.append(f"- **Ortalama Doğruluk:** {summary['accuracy_stats']['avg']:.6f}")
        lines.append(f"- **Süre Aralığı:** {summary['time_stats']['min']:.2f} - {summary['time_stats']['max']:.2f} ms")
        lines.append(f"- **Ortalama Süre:** {summary['time_stats']['avg']:.2f} ms")
        lines.append("")
    
    # Test senaryoları
    lines.append("## Test Senaryoları")
    for scenario in data['test_scenarios']:
        lines.append(f"- **{scenario['name']}:** {scenario['description']}")
        if scenario['is_edge_case']:
            lines.append(f"  - *Sınır vakası*")
    lines.append("")
    
    # En iyi 5 parametre
    if data.get('summary', {}).get('top_5_params'):
        lines.append("## En İyi 5 Parametre Kombinasyonu")
        for i, param_info in enumerate(data['summary']['top_5_params']):
            lines.append(f"{i+1}. n={param_info['params']['n']}, scale={param_info['params']['scale']}, qi={param_info['params']['qi_sizes']}")
            lines.append(f"   - Doğruluk: {param_info['accuracy']:.6f}")
            lines.append(f"   - Süre: {param_info['avg_time']:.2f} ms")
            lines.append(f"   - Max Hata: {param_info['max_error']:.3e}")
        lines.append("")
    
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    # CSV raporu
    csv_data = []
    for result in data.get('all_results', []):
        for scenario in result['scenario_results']:
            csv_data.append({
                'n': result['params']['n'],
                'scale': result['params']['scale'],
                'qi_sizes': str(result['params']['qi_sizes']),
                'scenario': scenario['scenario_name'],
                'good_patients': scenario['patient_counts']['good'],
                'bad_patients': scenario['patient_counts']['bad'],
                'total_patients': scenario['patient_counts']['total'],
                'sum_good': scenario['sum_good'],
                'sum_bad': scenario['sum_bad'],
                'mean_good': scenario['mean_good'],
                'mean_bad': scenario['mean_bad'],
                'err_sum_good': scenario['err_sum_good'],
                'err_sum_bad': scenario['err_sum_bad'],
                'err_mean_good': scenario['err_mean_good'],
                'err_mean_bad': scenario['err_mean_bad'],
                'delta_mean': scenario['delta_mean'],
                'exp_delta_mean': scenario['exp_delta_mean'],
                'accuracy': scenario['accuracy_metrics']['mean_accuracy'],
                'max_error': scenario['accuracy_metrics']['max_error'],
                'elapsed_ms': scenario['elapsed_ms'],
                'is_edge_case': scenario['is_edge_case']
            })
    
    if csv_data:
        df_csv = pd.DataFrame(csv_data)
        df_csv.to_csv(OUT_CSV, index=False, encoding="utf-8")


def main():
    print("🧪 Gelişmiş CKKS Test Vektörleri Oluşturucu Başlatılıyor...")
    
    # Verileri yükle
    df = pd.read_csv(MM_PATH)
    generator = AdvancedCKKSTestGenerator()
    cols, weights, bias, pol = generator.load_policy(POLICY_PATH)
    
    print(f"📊 Veri seti yüklendi: {len(df)} hasta kaydı")
    print(f"🔐 Şifrelenecek kolonlar: {cols}")
    print(f"⚖️  Ağırlıklar: {dict(zip(cols, weights))}")
    print(f"📏 Bias: {bias}")
    
    # Testleri çalıştır
    results = generator.run_full_evaluation(df, cols, weights, bias)
    
    # Raporları yaz
    write_detailed_reports(results)
    
    print(f"\n✅ Test tamamlandı!")
    print(f"📄 JSON raporu: {OUT_JSON}")
    print(f"📄 Markdown raporu: {OUT_MD}")
    print(f"📄 CSV raporu: {OUT_CSV}")
    
    if results.get('best_result'):
        best = results['best_result']
        print(f"\n🏆 En iyi parametre:")
        print(f"   n={best['params']['n']}, scale={best['params']['scale']}, qi={best['params']['qi_sizes']}")
        print(f"   Doğruluk: {best['param_metrics']['total_accuracy']:.6f}")
        print(f"   Ortalama süre: {best['param_metrics']['avg_time_ms']:.2f} ms")


if __name__ == "__main__":
    main()