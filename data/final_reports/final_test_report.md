# 🧪 CKKS Master Test Raporu

**Test Tarihi:** 2025-11-11T02:01:09.860381
**Toplam Test Süresi:** 75.45 saniye

## 📊 Özet
- **Toplam Test:** 3
- **Başarılı:** 2 ✅
- **Başarısız:** 1 ❌
- **Başarı Oranı:** 66.7%
- **Ortalama Süre:** 25.15 saniye/test

## 🔍 Detaylı Sonuçlar
### ❌ Temel CKKS Optimizasyonu
- **Durum:** failed
- **Süre:** 0.65 saniye
- **Hata:** c:\Users\MONSTER\Desktop\Tez\HEandData\scripts\ckks_param_optimization_multimodal.py:83: UserWarning: <Pyfhel Warning> qi_sizes [60, 40, 60] do not support rescaling for scale 1073741824.0.
  he.conte...

### ✅ Gelişmiş Test Vektörleri
- **Durum:** success
- **Süre:** 30.97 saniye
- **Çıktı Dosyaları:**
  - ✅ data/test_vectors/ckks_advanced_test_vectors.json
  - ✅ data/test_vectors/ckks_advanced_test_report.md
  - ❌ data/test_vectors/test_vector_results.csv

### ✅ Kapsamlı Senaryo Testleri
- **Durum:** success
- **Süre:** 43.84 saniye
- **Çıktı Dosyaları:**
  - ✅ data/comprehensive_tests/comprehensive_scenario_analysis.json
  - ✅ data/comprehensive_tests/comprehensive_test_report.md
  - ✅ data/comprehensive_tests/scenario_comparison.csv

## 💡 Öneriler
- Başarısız testleri kontrol edin ve gerekli düzeltmeleri yapın
- Eksik bağımlılıkları yükleyin
- Gerekli dosyaların varlığını kontrol edin