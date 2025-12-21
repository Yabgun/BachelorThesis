# 🏥 Privacy-Preserving Healthcare Analytics: Master Project Report
**Oluşturulma Tarihi:** 2025-12-14  

Bu rapor, sağlık verileri üzerinde mahremiyet koruyucu analizler gerçekleştiren projemizin **tüm teknik detaylarını, mimari kararlarını, kod yapısını ve performans sonuçlarını** en ince ayrıntısına kadar belgelemektedir. Proje, **CKKS Homomorfik Şifreleme** şemasını kullanarak, hassas hasta verilerinin şifresi çözülmeden işlenmesini sağlar.

---

## 📚 1. Proje Özeti ve Amacı
Bu proje, hassas sağlık verilerinin (genetik belirteçler, akciğer röntgeni sonuçları vb.) mahremiyetini koruyarak risk analizi yapabilen hibrit bir makine öğrenmesi sistemi geliştirmeyi amaçlar.

**Temel Hedefler:**
1.  **Mahremiyet:** Hasta verileri (özellikle hassas olanlar) asla sunucu tarafında açık metin (plaintext) olarak işlenmez.
2.  **Performans:** "Seçici Homomorfik Şifreleme" (Selective HE) yaklaşımı ile tam şifrelemeye göre ciddi performans artışı sağlanır.
3.  **Doğruluk:** Şifreli işlemlerin getirdiği gürültüye rağmen, modelin doğruluk oranının korunması (%99+).

---

## 🏗️ 2. Sistem Mimarisi ve Teknik Altyapı

### 2.1. Kullanılan Teknolojiler ve Kütüphaneler
*   **Dil:** Python 3.9+
*   **Homomorfik Şifreleme:** `TenSEAL` (Microsoft SEAL wrapper), `Pyfhel` (Python wrapper for SEAL)
*   **Veri İşleme:** `pandas`, `numpy`
*   **Makine Öğrenmesi:** `scikit-learn` (Logistic Regression, RandomForest, vb.)
*   **Görselleştirme:** `matplotlib`, `seaborn`

### 2.2. Şifreleme Şeması: CKKS (Cheon-Kim-Kim-Song)
CKKS şeması, reel sayıların (floating-point) şifreli işlemlerine izin verdiği için seçilmiştir.

**TenSEAL Bağlam (Context) Parametreleri:**
*   **Poly Modulus Degree (N):** `8192` (Güvenlik ve performans dengesi için optimize edildi)
*   **Coeff Modulus Bit Sizes:** `[60, 40, 40, 60]`
    *   İlk ve son 60 bit: Şifreleme/Deşifreleme güvenliği için.
    *   Ortadaki 40 bitler: Çarpma derinliği (multiplication depth) için. İki adet 40 bitlik modül, ardışık işlemlere olanak tanır.
*   **Global Scale:** `2^40` (Kayan nokta hassasiyetini korumak için ölçekleme faktörü)

**Neden CKKS?**
*   Sağlık verileri (BMI, Yaş, Olasılıklar) reel sayılardır.
*   CKKS, şifreli veriler üzerinde toplama ve çarpma işlemlerine (yaklaşık sonuçlarla) izin verir.

### 2.3. Seçici HE (Selective Homomorphic Encryption) Mimarisi
Projenin en yenilikçi yönü, tüm veriyi şifrelemek yerine hibrit bir yaklaşım kullanmasıdır.

*   **Hassas Veriler (Encrypted):** `Smoking`, `Genetic_Marker`, `CXR_Feature` (TenSEAL CKKS Vector)
*   **Hassas Olmayan Veriler (Plaintext):** `Age`, `Gender`, `BMI` (Normal Float)

**İşlem Akışı:**
1.  **Şifreli Kanal:** Hassas veriler istemcide şifrelenir -> Sunucuda şifreli ağırlıklarla çarpılır (`enc_dot`).
2.  **Açık Kanal:** Hassas olmayan veriler sunucuda normal ağırlıklarla çarpılır (`plain_dot`).
3.  **Birleştirme:** `Final_Result = Encrypted_Sum + Plaintext_Sum` (TenSEAL, şifreli vektör ile açık sayının toplanmasına izin verir).
4.  **Sonuç:** Sonuç şifreli bir "Logit" değeridir. İstemciye geri döner.

### 2.4. İstemci Tarafı Aktivasyon (Client-Side Non-Linearity)
Homomorfik şifreleme ile `Sigmoid` veya `ReLU` gibi lineer olmayan fonksiyonları hesaplamak çok maliyetlidir (polinom yaklaşımı gerektirir).

*   **Çözüm:** Sunucu, şifreli **Logit** değerini (ham tahmin skoru) istemciye gönderir.
*   **İşlem:** İstemci, özel anahtarı (Secret Key) ile logit'i çözer ve `Sigmoid(logit)` işlemini kendi cihazında uygular.
*   **Güvenlik:** Logit değeri, modelin güvenini temsil eder ancak tek başına eğitim verisini ifşa etmez.

---

## 📂 3. Proje Dosya Yapısı ve İşlevleri

### 3.1. Veri Hazırlama (`scripts/prepare_healthcare.py`, `scripts/ml_classification_model.py`)
*   **Veri Kaynağı:** `data/covid_ct_cxr/healthcare_dataset.csv`
*   **Temizlik:** Eksik veriler atılır, metin verileri (`Gender`, `Blood Type`) sayısal hale getirilir veya normalize edilir.
*   **Özellik Çıkarımı:** Radyolojik görüntülerden (`CXR`) özellikler çıkarılarak (`mean_intensity`, `edge_density`) CSV'ye eklenir.

### 3.2. Model Eğitimi ve Ağırlıklar (`models/he_model_weights.json`)
*   Model, `LogisticRegression` veya `RandomForest` kullanılarak eğitilir.
*   Eğitilen modelin **katsayıları (weights)** ve **sapma (bias)** değerleri JSON formatında dışa aktarılır.
*   Bu JSON dosyası, şifreli tahmin motoru tarafından okunur.

### 3.3. Karşılaştırmalı Analiz (`scripts/compare_he_classification.py`)
Bu script projenin kalbidir.
1.  **Tam HE:** Tüm verileri şifreleyip işlem yapar.
2.  **Seçici HE:** Hibrit mimariyi çalıştırır.
3.  **Karşılaştırma:** İki yöntemin doğruluk (Accuracy) ve süre (Time) farklarını ölçer.
4.  **Çıktı:** Grafikler (`he_siniflandirma_grafigi.png`) ve raporlar üretir.

### 3.4. Hata Analizi (`scripts/analyze_misclassification.py`)
*   Modelin yanlış tahmin ettiği hastaları (örn. %0.25'lik dilim) tespit eder.
*   Hatanın nedenini (sınırda kalan olasılıklar, aykırı değerler) analiz eder.
*   Görselleştirme ile hatayı raporlar.

---

## 📊 4. Performans ve Test Sonuçları

### 4.1. Hız ve Verimlilik
*   **Tam HE Süresi:** ~5.2 saniye (100 hasta için)
*   **Seçici HE Süresi:** ~4.1 saniye (100 hasta için)
*   **Hız Artışı (Speedup):** ~1.25x - 1.3x
    *   *Not: Bu oran veri boyutu arttıkça daha belirgin hale gelir.*

### 4.2. Doğruluk (Accuracy)
*   **Tam HE Doğruluk:** %100 (veya %99.75)
*   **Seçici HE Doğruluk:** %100 (veya %99.75)
*   **Kayıpsızlık:** Seçici HE, Tam HE ile birebir aynı matematiksel sonucu (mikroskobik CKKS gürültüsü hariç) üretir. "Lossless Precision" iddiası, tolerans sınırları dahilinde geçerlidir.

### 4.3. Kapsamlı Senaryo Testleri (`data/comprehensive_tests/`)
Farklı hasta profilleri (Easy, Hard, Edge Case) üzerinde 10 farklı senaryo test edilmiştir.
*   **En İyi Konfigürasyon:** "Dengeli Kapsamlı" (Balanced Comprehensive)
*   **Başarı Oranı:** Tüm senaryolarda 1.0 (Tam Başarı)
*   **Ortalama Hata Payı:** `2.019e-09` (İhmal edilebilir düzeyde şifreleme gürültüsü)

---

## 🔒 5. Güvenlik Değerlendirmesi

1.  **Veri Gizliliği:** Sunucu, hastanın sigara içip içmediğini veya genetik markörlerini asla göremez. Bu veriler şifreli vektörler içinde saklıdır.
2.  **Model Mahremiyeti:** İstemci, modelin ağırlıklarını (weights) göremez (eğer ağırlıklar da şifrelenirse). Mevcut mimaride ağırlıklar sunucuda plaintext olarak durur ve şifreli veriyle çarpılır.
3.  **Logit Sızıntısı:** İstemciye dönen `Logit` değeri, ham skordur. Bu değerden diğer hastaların verisini türetmek veya modelin tamamını çalmak (model inversion attack) son derece zordur, ancak teorik olarak minimal bir bilgi sızıntısıdır (kabul edilebilir risk).

---

