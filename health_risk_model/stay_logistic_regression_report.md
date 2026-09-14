# Stay Sınıflandırma Model Raporu

- Oluşturulma zamanı: 2025-12-23 05:08:53
- Model tipi: `logistic_regression`
- Veri kaynağı: `C:\Users\MONSTER\Desktop\Tez\HEandData\health_risk_model\StayData\multimodal_train.csv`

## Özet

- Amaç: Hastanın `Stay` (yatış süresi aralığı) sınıfını tahmin etmek
- Test doğruluğu (accuracy): `0.8202`
- Macro-F1: `0.4751`
- Weighted-F1: `0.8011`
- 5-fold CV mean accuracy: `0.8143` (±`0.0061`)

## Veri ve Bölme

- Train örnek sayısı: `4115`
- Test örnek sayısı: `1029`
- Feature sayısı: `28`
- `test_size`: `0.2`
- `random_state`: `42`

## Özellik Hazırlama (Feature Engineering)

- Encoding modu: `target_encoding`
- Yaş (`Age`) aralığı `age_numeric` sayısal değerine çevrilir
- `Severity of Illness` sayısallaştırılır (`Minor/Moderate/Extreme` → `1/2/3`)
- Türetilen bazı etkileşim/yardımcı özellikler: `age_deposit_interaction`, `risk_severity_synergy`, `risk_high`, `risk_medium`, `disease_risk_boost`, `age_money_ratio`
- `target_encoding` kullanıldığında bazı kategorik sütunlar tek sayısal sütuna indirgenir (feature sayısını azaltır)

## Model Yapısı

- `C`: `3000.0`
- `solver`: `lbfgs`
- `max_iter`: `5000`
- `class_weight`: `None`
- Karar kuralı: `argmax` (alpha: `0.0`)

## Performans Detayı

- Eğitim doğruluğu (train accuracy): `0.8248`
- Eğitim süresi: `3.5624` saniye
- Tahmin süresi: `0.0009` saniye
- Testte hiç tahmin edilmeyen sınıf indeksleri: `[]`

## HE (Seçmeli Homomorfik Şifreleme) Uyumu

- Logistic Regression; lineer karar fonksiyonu nedeniyle HE senaryolarında en uygun modellerden biridir.
- Feature sayısının düşük tutulması (ör. `target_encoding`) şifreli hesaplama maliyetini azaltır.

## Nasıl Çalıştırılır

- Python içinde:
  - `from health_risk_model.core_model import run_multimodal_core_model`
  - `results = run_multimodal_core_model()`
- Script olarak:
  - `python scripts/ml_classification_model.py`

## Katsayı Bilgisi

- `coef_l2_norm`: `999.5630`
- `coef_max_abs`: `352.0705`
- `intercept_max_abs`: `521.1665`
