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
OUT_DIR = Path("data/comprehensive_tests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "comprehensive_scenario_analysis.json"
OUT_MD = OUT_DIR / "comprehensive_test_report.md"
OUT_CSV = OUT_DIR / "scenario_comparison.csv"


@dataclass
class WeightConfig:
    name: str
    description: str
    weights: Dict[str, float]
    bias: float
    priority: str  # "accuracy", "speed", "balanced"


@dataclass
class TestConfiguration:
    name: str
    weight_config: WeightConfig
    scenarios: List[Dict[str, Any]]
    params: Dict[str, Any]


class ComprehensiveScenarioTester:
    def __init__(self):
        self.weight_configs = self._define_weight_configurations()
        self.test_scenarios = self._define_test_scenarios()
        self.optimal_params = None  # En iyi parametreler buraya gelecek
        
    def _define_weight_configurations(self) -> List[WeightConfig]:
        return [
            WeightConfig(
                "Standart",
                "Orijinal ağırlıklandırma",
                {
                    "age": 0.02,
                    "billing_amount_norm": 5e-05,
                    "test_results_score": 1.0,
                    "cxr_mean_intensity": 0.5,
                    "cxr_edge_density": 0.1
                },
                -5.0,
                "balanced"
            ),
            WeightConfig(
                "Klinik Ağırlıklı",
                "Test sonuçlarına daha fazla ağırlık",
                {
                    "age": 0.01,
                    "billing_amount_norm": 1e-05,
                    "test_results_score": 2.0,
                    "cxr_mean_intensity": 0.3,
                    "cxr_edge_density": 0.05
                },
                -3.0,
                "accuracy"
            ),
            WeightConfig(
                "Görüntü Odaklı",
                "CXR özelliklerine daha fazla ağırlık",
                {
                    "age": 0.01,
                    "billing_amount_norm": 1e-05,
                    "test_results_score": 0.5,
                    "cxr_mean_intensity": 1.0,
                    "cxr_edge_density": 0.3
                },
                -2.0,
                "accuracy"
            ),
            WeightConfig(
                "Hızlı Değerlendirme",
                "Daha az özellik, daha hızlı işlem",
                {
                    "age": 0.05,
                    "billing_amount_norm": 0.0001,
                    "test_results_score": 1.5,
                    "cxr_mean_intensity": 0.2,
                    "cxr_edge_density": 0.05
                },
                -4.0,
                "speed"
            ),
            WeightConfig(
                "Dengeli Kapsamlı",
                "Tüm özellikleri dengeli şekilde kullan",
                {
                    "age": 0.03,
                    "billing_amount_norm": 0.0001,
                    "test_results_score": 1.2,
                    "cxr_mean_intensity": 0.6,
                    "cxr_edge_density": 0.15
                },
                -4.5,
                "balanced"
            ),
            WeightConfig(
                "Yaş ve Maliyet Odaklı",
                "Yaş ve maliyet faktörlerine ağırlık",
                {
                    "age": 0.1,
                    "billing_amount_norm": 0.001,
                    "test_results_score": 0.8,
                    "cxr_mean_intensity": 0.4,
                    "cxr_edge_density": 0.08
                },
                -6.0,
                "accuracy"
            )
        ]
    
    def _define_test_scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Mükemmel Senaryo",
                "description": "Tüm hastalar iyi (100% başarı)",
                "good_ratio": 1.0,
                "total_patients": 100,
                "noise_level": 0.0,
                "difficulty": "easy"
            },
            {
                "name": "Kötü Senaryo",
                "description": "Tüm hastalar kötü (0% başarı)",
                "good_ratio": 0.0,
                "total_patients": 100,
                "noise_level": 0.0,
                "difficulty": "hard"
            },
            {
                "name": "Dengeli Karışım",
                "description": "50-50 iyi-kötü dengesi",
                "good_ratio": 0.5,
                "total_patients": 100,
                "noise_level": 0.1,
                "difficulty": "medium"
            },
            {
                "name": "Ağırlıklı İyi",
                "description": "80% iyi, 20% kötü",
                "good_ratio": 0.8,
                "total_patients": 100,
                "noise_level": 0.05,
                "difficulty": "easy"
            },
            {
                "name": "Ağırlıklı Kötü",
                "description": "20% iyi, 80% kötü",
                "good_ratio": 0.2,
                "total_patients": 100,
                "noise_level": 0.15,
                "difficulty": "hard"
            },
            {
                "name": "Marjinal İyi",
                "description": "95% iyi, 5% kötü (sınır vakası)",
                "good_ratio": 0.95,
                "total_patients": 100,
                "noise_level": 0.02,
                "difficulty": "edge"
            },
            {
                "name": "Marjinal Kötü",
                "description": "5% iyi, 95% kötü (sınır vakası)",
                "good_ratio": 0.05,
                "total_patients": 100,
                "noise_level": 0.2,
                "difficulty": "edge"
            },
            {
                "name": "Gürültülü Dengeli",
                "description": "50-50 dengesi ama yüksek gürültü",
                "good_ratio": 0.5,
                "total_patients": 100,
                "noise_level": 0.3,
                "difficulty": "hard"
            },
            {
                "name": "Küçük Örneklem",
                "description": "Düşük hasta sayısı, dengeli",
                "good_ratio": 0.5,
                "total_patients": 20,
                "noise_level": 0.05,
                "difficulty": "medium"
            },
            {
                "name": "Büyük Örneklem",
                "description": "Yüksek hasta sayısı, dengeli",
                "good_ratio": 0.5,
                "total_patients": 500,
                "noise_level": 0.05,
                "difficulty": "medium"
            }
        ]
    
    def load_optimal_params(self) -> Dict[str, Any]:
        """En iyi parametreleri yükle (önceki optimizasyondan)"""
        opt_file = Path("data/test_vectors/ckks_advanced_test_vectors.json")
        if opt_file.exists():
            with open(opt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('best_result'):
                    return data['best_result']['params']
        
        # Varsayılan optimal parametreler
        return {
            "n": 8192,
            "scale": 1099511627776,
            "qi_sizes": [60, 40, 40, 60]
        }
    
    def calculate_plain_score(self, row: pd.Series, weights: Dict[str, float], bias: float) -> float:
        score = 0.0
        for feature, weight in weights.items():
            if feature in row:
                score += float(row[feature]) * float(weight)
        return score + bias
    
    def generate_test_patients(self, df: pd.DataFrame, scenario: Dict[str, Any], weights: Dict[str, float], bias: float) -> Tuple[List[float], List[float]]:
        """Senaryoya göe test hasta verileri oluştur"""
        good_patients = []
        bad_patients = []
        
        # Test skoruna göre hasta ayır
        high_scores = df[df["test_results_score"] >= 0.7].copy()
        low_scores = df[df["test_results_score"] <= 0.3].copy()
        
        if len(high_scores) == 0 or len(low_scores) == 0:
            # Fallback
            high_scores = df[df["test_results_score"] >= df["test_results_score"].median()].copy()
            low_scores = df[df["test_results_score"] < df["test_results_score"].median()].copy()
        
        total_patients = scenario["total_patients"]
        good_count = int(total_patients * scenario["good_ratio"])
        bad_count = total_patients - good_count
        noise_level = scenario["noise_level"]
        
        # İyi hastalar
        for _ in range(good_count):
            if len(high_scores) > 0:
                patient = high_scores.sample(1).iloc[0]
                score = self.calculate_plain_score(patient, weights, bias)
                # Gürültü ekle
                if noise_level > 0:
                    score += np.random.normal(0, noise_level * abs(score))
                good_patients.append(score)
            else:
                good_patients.append(1.0 + np.random.normal(0, 0.1))
        
        # Kötü hastalar
        for _ in range(bad_count):
            if len(low_scores) > 0:
                patient = low_scores.sample(1).iloc[0]
                score = self.calculate_plain_score(patient, weights, bias)
                # Gürültü ekle
                if noise_level > 0:
                    score += np.random.normal(0, noise_level * abs(score))
                bad_patients.append(score)
            else:
                bad_patients.append(0.0 + np.random.normal(0, 0.1))
        
        return good_patients, bad_patients
    
    def encrypt_and_sum(self, he: Pyfhel, values: List[float]) -> float:
        """Değerleri şifreleyerek topla"""
        if not values:
            return 0.0
        
        ct_sum = None
        for val in values:
            # Use numpy array for encoding
            ptxt = he.encodeFrac(np.array([val], dtype=np.float64))
            ctxt = he.encryptPtxt(ptxt)
            ct_sum = ctxt if ct_sum is None else (ct_sum + ctxt)
        
        decrypted = he.decryptFrac(ct_sum)
        return float(decrypted[0])
    
    def evaluate_configuration(self, config: TestConfiguration, df: pd.DataFrame, optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Bir konfigürasyonu değerlendir"""
        print(f"🧪 Konfigürasyon test ediliyor: {config.name}")
        
        # Pyfhel setup
        he = Pyfhel()
        he.contextGen(scheme="CKKS", **optimal_params)
        he.keyGen()
        
        config_results = []
        
        for scenario in config.scenarios:
            print(f"  📋 Senaryo: {scenario['name']}")
            
            # Test hasta verilerini oluştur
            good_patients, bad_patients = self.generate_test_patients(
                df, scenario, config.weight_config.weights, config.weight_config.bias
            )
            
            # Zamanlama ve değerlendirme
            start_time = time.time()
            
            # Şifreli toplamlar
            sum_good = self.encrypt_and_sum(he, good_patients)
            sum_bad = self.encrypt_and_sum(he, bad_patients)
            
            encryption_time = (time.time() - start_time) * 1000
            
            # Beklenen değerler
            expected_good = sum(good_patients)
            expected_bad = sum(bad_patients)
            
            # Hatalar
            error_good = abs(sum_good - expected_good)
            error_bad = abs(sum_bad - expected_bad)
            
            # Ortalamalar
            mean_good = sum_good / len(good_patients) if good_patients else 0
            mean_bad = sum_bad / len(bad_patients) if bad_patients else 0
            expected_mean_good = expected_good / len(good_patients) if good_patients else 0
            expected_mean_bad = expected_bad / len(bad_patients) if bad_patients else 0
            
            # Ortalama hatalar
            error_mean_good = abs(mean_good - expected_mean_good)
            error_mean_bad = abs(mean_bad - expected_mean_bad)
            
            # Delta hesaplamaları
            delta_mean = mean_good - mean_bad
            expected_delta = expected_mean_good - expected_mean_bad
            delta_error = abs(delta_mean - expected_delta)
            
            scenario_result = {
                "scenario_name": scenario["name"],
                "description": scenario["description"],
                "difficulty": scenario["difficulty"],
                "patient_counts": {
                    "good": len(good_patients),
                    "bad": len(bad_patients),
                    "total": len(good_patients) + len(bad_patients)
                },
                "results": {
                    "sum_good": sum_good,
                    "sum_bad": sum_bad,
                    "mean_good": mean_good,
                    "mean_bad": mean_bad,
                    "delta_mean": delta_mean
                },
                "expected_values": {
                    "sum_good": expected_good,
                    "sum_bad": expected_bad,
                    "mean_good": expected_mean_good,
                    "mean_bad": expected_mean_bad,
                    "delta_mean": expected_delta
                },
                "errors": {
                    "sum_good": error_good,
                    "sum_bad": error_bad,
                    "mean_good": error_mean_good,
                    "mean_bad": error_mean_bad,
                    "delta_mean": delta_error
                },
                "performance": {
                    "encryption_time_ms": encryption_time,
                    "total_operations": len(good_patients) + len(bad_patients)
                },
                "accuracy_metrics": {
                    "sum_accuracy": 1 - (error_good + error_bad) / (abs(expected_good) + abs(expected_bad) + 1e-15),
                    "mean_accuracy": 1 - (error_mean_good + error_mean_bad) / (abs(expected_mean_good) + abs(expected_mean_bad) + 1e-15),
                    "delta_accuracy": 1 - delta_error / (abs(expected_delta) + 1e-15),
                    "max_error": max(error_good, error_bad, error_mean_good, error_mean_bad, delta_error)
                }
            }
            
            config_results.append(scenario_result)
        
        # Konfigürasyon bazlı özet
        accuracies = [r["accuracy_metrics"]["mean_accuracy"] for r in config_results]
        times = [r["performance"]["encryption_time_ms"] for r in config_results]
        max_errors = [r["accuracy_metrics"]["max_error"] for r in config_results]
        
        return {
            "configuration_name": config.name,
            "weight_config": {
                "name": config.weight_config.name,
                "description": config.weight_config.description,
                "priority": config.weight_config.priority,
                "weights": config.weight_config.weights,
                "bias": config.weight_config.bias
            },
            "scenario_results": config_results,
            "summary": {
                "avg_accuracy": sum(accuracies) / len(accuracies),
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "avg_time_ms": sum(times) / len(times),
                "total_time_ms": sum(times),
                "avg_max_error": sum(max_errors) / len(max_errors),
                "scenario_count": len(config_results)
            }
        }
    
    def run_comprehensive_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tüm kapsamlı testleri çalıştır"""
        print("🔬 Kapsamlı senaryo testleri başlatılıyor...")
        
        # Optimal parametreleri yükle
        optimal_params = self.load_optimal_params()
        print(f"⚙️  Optimal parametreler yüklendi: n={optimal_params['n']}, scale={optimal_params['scale']}")
        
        all_results = []
        
        for i, weight_config in enumerate(self.weight_configs):
            print(f"\n📊 Ağırlık konfigürasyonu {i+1}/{len(self.weight_configs)}: {weight_config.name}")
            
            # Test konfigürasyonu oluştur
            config = TestConfiguration(
                name=f"{weight_config.name}_config",
                weight_config=weight_config,
                scenarios=self.test_scenarios,
                params=optimal_params
            )
            
            # Konfigürasyonu değerlendir
            result = self.evaluate_configuration(config, df, optimal_params)
            all_results.append(result)
        
        # Sonuçları sırala (doğruluk bazlı)
        all_results.sort(key=lambda x: -x["summary"]["avg_accuracy"])
        
        # En iyi konfigürasyonu bul
        best_config = all_results[0] if all_results else None
        
        return {
            "metadata": {
                "test_date": datetime.now().isoformat(),
                "total_configurations": len(self.weight_configs),
                "total_scenarios": len(self.test_scenarios),
                "optimal_params": optimal_params,
                "data_size": len(df)
            },
            "weight_configurations": [vars(wc) for wc in self.weight_configs],
            "test_scenarios": self.test_scenarios,
            "all_results": all_results,
            "best_configuration": best_config,
            "comparison": self._generate_comparison(all_results)
        }
    
    def _generate_comparison(self, results: List[Dict]) -> Dict[str, Any]:
        """Konfigürasyon karşılaştırması"""
        comparison = {
            "accuracy_ranking": [],
            "speed_ranking": [],
            "error_ranking": [],
            "overall_scores": []
        }
        
        for i, result in enumerate(results):
            config_name = result["configuration_name"]
            summary = result["summary"]
            
            # Doğruluk sıralaması
            comparison["accuracy_ranking"].append({
                "rank": i + 1,
                "config": config_name,
                "accuracy": summary["avg_accuracy"],
                "accuracy_range": f"{summary['min_accuracy']:.4f} - {summary['max_accuracy']:.4f}"
            })
            
            # Hız sıralaması
            comparison["speed_ranking"].append({
                "rank": 0,  # Sonradan sıralanacak
                "config": config_name,
                "avg_time": summary["avg_time_ms"],
                "total_time": summary["total_time_ms"]
            })
            
            # Hata sıralaması
            comparison["error_ranking"].append({
                "rank": 0,  # Sonradan sıralanacak
                "config": config_name,
                "avg_max_error": summary["avg_max_error"]
            })
        
        # Hız ve hata sıralamalarını yap
        comparison["speed_ranking"].sort(key=lambda x: x["avg_time"])
        for i, item in enumerate(comparison["speed_ranking"]):
            item["rank"] = i + 1
        
        comparison["error_ranking"].sort(key=lambda x: x["avg_max_error"])
        for i, item in enumerate(comparison["error_ranking"]):
            item["rank"] = i + 1
        
        return comparison


def write_comprehensive_reports(data: Dict) -> None:
    """Kapsamlı raporları yaz"""
    # JSON raporu
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    # Markdown raporu
    lines = []
    lines.append("# Kapsamlı CKKS Senaryo Test Raporu")
    lines.append("")
    lines.append(f"**Test Tarihi:** {data['metadata']['test_date']}")
    lines.append(f"**Toplam Konfigürasyon:** {data['metadata']['total_configurations']}")
    lines.append(f"**Toplam Senaryo:** {data['metadata']['total_scenarios']}")
    lines.append(f"**Veri Seti Boyutu:** {data['metadata']['data_size']} hasta")
    lines.append("")
    
    # En iyi konfigürasyon
    if data.get('best_configuration'):
        best = data['best_configuration']
        lines.append("## 🏆 En İyi Ağırlık Konfigürasyonu")
        lines.append(f"**{best['weight_config']['name']}** - {best['weight_config']['description']}")
        lines.append(f"- **Öncelik:** {best['weight_config']['priority']}")
        lines.append(f"- **Ortalama Doğruluk:** {best['summary']['avg_accuracy']:.6f}")
        lines.append(f"- **Doğruluk Aralığı:** {best['summary']['min_accuracy']:.4f} - {best['summary']['max_accuracy']:.4f}")
        lines.append(f"- **Ortalama Süre:** {best['summary']['avg_time_ms']:.2f} ms")
        lines.append(f"- **Ortalama Maks Hata:** {best['summary']['avg_max_error']:.3e}")
        lines.append("")
        
        # Ağırlık detayları
        lines.append("### Ağırlıklandırma Detayı:")
        for feature, weight in best['weight_config']['weights'].items():
            lines.append(f"- **{feature}:** {weight}")
        lines.append(f"- **Bias:** {best['weight_config']['bias']}")
        lines.append("")
    
    # Konfigürasyon karşılaştırması
    if data.get('comparison'):
        comp = data['comparison']
        lines.append("## 📊 Konfigürasyon Karşılaştırması")
        lines.append("")
        
        # Doğruluk sıralaması
        lines.append("### Doğruluk Sıralaması:")
        for item in comp['accuracy_ranking'][:5]:
            lines.append(f"{item['rank']}. **{item['config']}** - Doğruluk: {item['accuracy']:.6f}")
        lines.append("")
        
        # Hız sıralaması
        lines.append("### Hız Sıralaması:")
        for item in comp['speed_ranking'][:5]:
            lines.append(f"{item['rank']}. **{item['config']}** - Ortalama Süre: {item['avg_time']:.2f} ms")
        lines.append("")
    
    # Senaryo bazlı özet
    lines.append("## 🎯 Senaryo Performans Özeti")
    for result in data.get('all_results', [])[:3]:  # İlk 3 konfigürasyon
        lines.append(f"### {result['configuration_name']}:")
        
        # Zorluk seviyesine göre grupla
        difficulty_groups = {}
        for scenario in result['scenario_results']:
            diff = scenario['difficulty']
            if diff not in difficulty_groups:
                difficulty_groups[diff] = []
            difficulty_groups[diff].append(scenario['accuracy_metrics']['mean_accuracy'])
        
        for difficulty, accuracies in difficulty_groups.items():
            avg_acc = sum(accuracies) / len(accuracies)
            lines.append(f"- **{difficulty.title()}** senaryolar: {avg_acc:.4f} ortalama doğruluk")
        lines.append("")
    
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    # CSV karşılaştırma
    csv_data = []
    for result in data.get('all_results', []):
        summary = result['summary']
        csv_data.append({
            'configuration': result['configuration_name'],
            'weight_config_name': result['weight_config']['name'],
            'priority': result['weight_config']['priority'],
            'avg_accuracy': summary['avg_accuracy'],
            'min_accuracy': summary['min_accuracy'],
            'max_accuracy': summary['max_accuracy'],
            'avg_time_ms': summary['avg_time_ms'],
            'total_time_ms': summary['total_time_ms'],
            'avg_max_error': summary['avg_max_error'],
            'scenario_count': summary['scenario_count']
        })
    
    if csv_data:
        df_csv = pd.DataFrame(csv_data)
        df_csv.to_csv(OUT_CSV, index=False, encoding="utf-8")


def main():
    print("🔬 Kapsamlı CKKS Senaryo Testleri Başlatılıyor...")
    
    # Verileri yükle
    df = pd.read_csv(MM_PATH)
    tester = ComprehensiveScenarioTester()
    
    print(f"📊 Veri seti yüklendi: {len(df)} hasta kaydı")
    print(f"🔧 Ağırlık konfigürasyonları: {len(tester.weight_configs)}")
    print(f"🎯 Test senaryoları: {len(tester.test_scenarios)}")
    
    # Testleri çalıştır
    results = tester.run_comprehensive_tests(df)
    
    # Raporları yaz
    write_comprehensive_reports(results)
    
    print(f"\n✅ Kapsamlı testler tamamlandı!")
    print(f"📄 JSON raporu: {OUT_JSON}")
    print(f"📄 Markdown raporu: {OUT_MD}")
    print(f"📄 CSV karşılaştırma: {OUT_CSV}")
    
    if results.get('best_configuration'):
        best = results['best_configuration']
        print(f"\n🏆 En iyi konfigürasyon:")
        print(f"   {best['weight_config']['name']} - Doğruluk: {best['summary']['avg_accuracy']:.6f}")
        print(f"   Ortalama süre: {best['summary']['avg_time_ms']:.2f} ms")


if __name__ == "__main__":
    main()