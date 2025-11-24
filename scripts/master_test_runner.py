#!/usr/bin/env python3
"""
Master CKKS Test Runner
Tüm CKKS test ve optimizasyon scriptlerini sırayla çalıştırır
"""

import sys
import time
import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Proje kök dizini
PROJECT_ROOT = Path("c:/Users/MONSTER/Desktop/Tez/HEandData")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = DATA_DIR / "final_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class MasterTestRunner:
    """Master test runner for all CKKS optimization and test scripts"""
    
    def __init__(self):
        self.scripts_dir = Path(__file__).parent
        self.results_dir = self.scripts_dir.parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.execution_log = []
        self.final_report = {
            "timestamp": datetime.now().isoformat(),
            "total_scripts": 0,
            "successful_scripts": 0,
            "failed_scripts": 0,
            "scripts": []
        }
        self.test_scripts = [
            {
                "name": "Temel CKKS Optimizasyonu",
                "script": "ckks_param_optimization_multimodal.py",
                "description": "Mevcut temel optimizasyon scripti",
                "required_files": ["data/covid_ct_cxr/multimodal.csv", "config/selective_he_policy.json"],
                "output_files": ["data/covid_ct_cxr/ckks_param_optimization_multimodal.json", "data/covid_ct_cxr/ckks_param_optimization_multimodal.md"]
            },
            {
                "name": "Gelişmiş Test Vektörleri",
                "script": "advanced_ckks_test_vectors.py",
                "description": "Kapsamlı test vektörleri ve kötü senaryolar",
                "required_files": ["data/covid_ct_cxr/multimodal.csv", "config/selective_he_policy.json"],
                "output_files": ["data/test_vectors/ckks_advanced_test_vectors.json", "data/test_vectors/ckks_advanced_test_report.md", "data/test_vectors/test_vector_results.csv"]
            },
            {
                "name": "Kapsamlı Senaryo Testleri",
                "script": "comprehensive_scenario_tester.py",
                "description": "Farklı ağırlıklandırmalar ve bias değerleriyle testler",
                "required_files": ["data/covid_ct_cxr/multimodal.csv"],
                "output_files": ["data/comprehensive_tests/comprehensive_scenario_analysis.json", "data/comprehensive_tests/comprehensive_test_report.md", "data/comprehensive_tests/scenario_comparison.csv"]
            }
        ]
        
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def check_dependencies(self) -> bool:
        """Gerekli bağımlılıkları kontrol et"""
        print("🔍 Bağımlılıklar kontrol ediliyor...")
        
        try:
            import Pyfhel
            print("✅ Pyfhel yüklü")
        except ImportError:
            print("❌ Pyfhel yüklü değil. 'pip install pyfhel' ile yükleyin.")
            return False
        
        try:
            import pandas
            print("✅ Pandas yüklü")
        except ImportError:
            print("❌ Pandas yüklü değil. 'pip install pandas' ile yükleyin.")
            return False
        
        try:
            import numpy
            print("✅ NumPy yüklü")
        except ImportError:
            print("❌ NumPy yüklü değil. 'pip install numpy' ile yükleyin.")
            return False
        
        return True
    
    def check_required_files(self, script_info: Dict) -> bool:
        """Gerekli dosyaların varlığını kontrol et"""
        print(f"📁 Gerekli dosyalar kontrol ediliyor...")
        
        for file_path in script_info["required_files"]:
            full_path = PROJECT_ROOT / file_path
            if not full_path.exists():
                print(f"❌ Dosya bulunamadı: {file_path}")
                return False
            else:
                print(f"✅ Dosya mevcut: {file_path}")
        
        return True
    
    def run_script(self, script_info: Dict) -> Dict[str, Any]:
        """Tek bir script çalıştır"""
        print(f"\n🚀 {script_info['name']} başlatılıyor...")
        print(f"📝 Açıklama: {script_info['description']}")
        
        script_path = SCRIPTS_DIR / script_info["script"]
        if not script_path.exists():
            return {
                "name": script_info["name"],
                "status": "failed",
                "error": f"Script dosyası bulunamadı: {script_path}",
                "duration": 0
            }
        
        start_time = time.time()
        
        try:
            # Python scriptini çalıştır
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {script_info['name']} başarıyla tamamlandı")
                if result.stdout:
                    print(f"Çıktı:\n{result.stdout}")
                
                return {
                    "name": script_info["name"],
                    "status": "success",
                    "duration": duration,
                    "stdout": result.stdout,
                    "output_files": self._check_output_files(script_info)
                }
            else:
                print(f"❌ {script_info['name']} başarısız oldu")
                print(f"Hata kodu: {result.returncode}")
                if result.stderr:
                    print(f"Hata mesajı:\n{result.stderr}")
                
                return {
                    "name": script_info["name"],
                    "status": "failed",
                    "error": result.stderr,
                    "duration": duration,
                    "return_code": result.returncode
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {script_info['name']} çalıştırılırken hata oluştu: {e}")
            
            return {
                "name": script_info["name"],
                "status": "error",
                "error": str(e),
                "duration": duration
            }
    
    def _check_output_files(self, script_info: Dict) -> Dict[str, bool]:
        """Çıktı dosyalarının varlığını kontrol et"""
        output_status = {}
        
        for file_path in script_info.get("output_files", []):
            full_path = PROJECT_ROOT / file_path
            exists = full_path.exists()
            output_status[file_path] = exists
            
            if exists:
                size = full_path.stat().st_size
                print(f"✅ Çıktı dosyası oluşturuldu: {file_path} ({size} bytes)")
            else:
                print(f"⚠️  Çıktı dosyası bulunamadı: {file_path}")
        
        return output_status
    
    def generate_final_summary(self) -> Dict[str, Any]:
        """Tüm testlerin özetini oluştur"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["status"] == "success")
        failed_tests = sum(1 for r in self.results if r["status"] in ["failed", "error"])
        total_duration = sum(r["duration"] for r in self.results)
        
        summary = {
            "test_date": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / total_tests if total_tests > 0 else 0,
            "test_results": self.results
        }
        
        return summary
    
    def write_final_report(self, summary: Dict[str, Any]) -> None:
        """Nihai raporu yaz"""
        report_path = REPORTS_DIR / "final_test_report.json"
        md_report_path = REPORTS_DIR / "final_test_report.md"
        
        # JSON raporu
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Markdown raporu
        lines = []
        lines.append("# 🧪 CKKS Master Test Raporu")
        lines.append("")
        lines.append(f"**Test Tarihi:** {summary['test_date']}")
        lines.append(f"**Toplam Test Süresi:** {summary['total_duration_seconds']:.2f} saniye")
        lines.append("")
        
        # Özet istatistikler
        lines.append("## 📊 Özet")
        lines.append(f"- **Toplam Test:** {summary['total_tests']}")
        lines.append(f"- **Başarılı:** {summary['successful_tests']} ✅")
        lines.append(f"- **Başarısız:** {summary['failed_tests']} ❌")
        lines.append(f"- **Başarı Oranı:** {summary['success_rate']:.1%}")
        lines.append(f"- **Ortalama Süre:** {summary['average_duration_seconds']:.2f} saniye/test")
        lines.append("")
        
        # Detaylı sonuçlar
        lines.append("## 🔍 Detaylı Sonuçlar")
        for result in summary['test_results']:
            status_icon = "✅" if result['status'] == "success" else "❌"
            lines.append(f"### {status_icon} {result['name']}")
            lines.append(f"- **Durum:** {result['status']}")
            lines.append(f"- **Süre:** {result['duration']:.2f} saniye")
            
            if result['status'] == 'success':
                if 'output_files' in result:
                    lines.append("- **Çıktı Dosyaları:**")
                    for file_path, exists in result['output_files'].items():
                        status = "✅" if exists else "❌"
                        lines.append(f"  - {status} {file_path}")
            else:
                if 'error' in result:
                    lines.append(f"- **Hata:** {result['error'][:200]}...")
            lines.append("")
        
        # Öneriler
        lines.append("## 💡 Öneriler")
        if summary['success_rate'] < 1.0:
            lines.append("- Başarısız testleri kontrol edin ve gerekli düzeltmeleri yapın")
            lines.append("- Eksik bağımlılıkları yükleyin")
            lines.append("- Gerekli dosyaların varlığını kontrol edin")
        else:
            lines.append("- ✅ Tüm testler başarıyla tamamlandı!")
            lines.append("- 📊 Test çıktılarını inceleyerek en iyi parametreleri belirleyin")
            lines.append("- 🔧 Gerekirse parametre optimizasyonunu tekrar çalıştırın")
        
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        print(f"\n📄 Nihai raporlar oluşturuldu:")
        print(f"   JSON: {report_path}")
        print(f"   Markdown: {md_report_path}")
    
    def run_all_tests(self) -> bool:
        """Tüm testleri sırayla çalıştır"""
        print("🎯 CKKS Master Test Runner Başlatılıyor...")
        print("=" * 60)
        
        # Bağımlılıkları kontrol et
        if not self.check_dependencies():
            print("❌ Bağımlılık kontrolü başarısız")
            return False
        
        self.start_time = time.time()
        
        # Her test scriptini çalıştır
        for script_info in self.test_scripts:
            # Gerekli dosyaları kontrol et
            if not self.check_required_files(script_info):
                print(f"❌ {script_info['name']} için gerekli dosyalar eksik")
                self.results.append({
                    "name": script_info["name"],
                    "status": "skipped",
                    "error": "Gerekli dosyalar eksik",
                    "duration": 0
                })
                continue
            
            # Scripti çalıştır
            result = self.run_script(script_info)
            self.results.append(result)
            
            # Kısa bekleme (ardışık çalıştırma için)
            time.sleep(2)
        
        self.end_time = time.time()
        
        # Özet ve rapor oluştur
        summary = self.generate_final_summary()
        self.write_final_report(summary)
        
        # Sonuçları göster
        print("\n" + "=" * 60)
        print("📊 TEST SONUÇLARI ÖZETİ")
        print("=" * 60)
        print(f"Toplam Test: {summary['total_tests']}")
        print(f"Başarılı: {summary['successful_tests']} ✅")
        print(f"Başarısız: {summary['failed_tests']} ❌")
        print(f"Başarı Oranı: {summary['success_rate']:.1%}")
        print(f"Toplam Süre: {summary['total_duration_seconds']:.2f} saniye")
        print("=" * 60)
        
        return summary['success_rate'] > 0.8  # %80 başarı oranı hedefi


def main():
    runner = MasterTestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n🎉 Tüm testler başarıyla tamamlandı!")
        sys.exit(0)
    else:
        print("\n⚠️  Bazı testler başarısız oldu. Detaylar için raporları kontrol edin.")
        sys.exit(1)


if __name__ == "__main__":
    main()