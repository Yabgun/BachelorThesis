# Stroke (İnme) Risk Analizi Modeli - Teknik Rapor

Bu rapor, `MLLesson` klasörü altında geliştirilen makine öğrenmesi projesinin teknik detaylarını, kullanılan metodolojiyi, veri işleme adımlarını ve model performansını belgelemektedir.

## 1. Proje Özeti ve Amaç
Bu projenin temel amacı, hastaların demografik bilgileri ve sağlık geçmişi verilerini kullanarak inme (stroke) geçirme riskini tahmin eden bir makine öğrenmesi modeli geliştirmektir. İnme tespiti, hayati önem taşıyan bir medikal teşhis problemi olduğu için, modelin **duyarlılığı (recall)** ön planda tutulmuştur; yani gerçek vakaları kaçırmamak (False Negative'i en aza indirmek) amaçlanmıştır.

## 2. Veri Seti ve Özellikler
Model, `healthcare-dataset-stroke-data.csv` dosyasındaki verilerle eğitilmiştir.

### 2.1. Girdi Değişkenleri (Features)
Veri seti aşağıdaki öznitelikleri içermektedir:
*   **Demografik:** `gender` (Cinsiyet), `age` (Yaş), `ever_married` (Evlilik durumu), `Residence_type` (Yaşam alanı tipi).
*   **Sağlık Durumu:** `hypertension` (Hipertansiyon), `heart_disease` (Kalp hastalığı).
*   **Ölçümler:** `avg_glucose_level` (Ortalama glikoz seviyesi), `bmi` (Vücut kitle indeksi).
*   **Yaşam Tarzı:** `work_type` (Çalışma şekli), `smoking_status` (Sigara kullanım durumu).

### 2.2. Hedef Değişken (Target)
*   **`stroke`**: 0 (İnme yok) veya 1 (İnme var).

## 3. Teknik Altyapı ve Kütüphaneler
Proje **Python** programlama dili kullanılarak geliştirilmiştir. Temel kütüphaneler:
*   **Scikit-learn:** Model eğitimi, ön işleme ve değerlendirme.
*   **Pandas & Numpy:** Veri manipülasyonu ve matris işlemleri.
*   **Joblib:** Modelin kaydedilmesi ve tekrar yüklenmesi.

## 4. Veri Ön İşleme (Preprocessing) ve Özellik Mühendisliği
Ham veriler modele verilmeden önce aşağıdaki işlemlerden geçirilmiştir:

### 4.1. Veri Temizleme ve Doldurma
*   **Eksik Veriler (BMI):** `bmi` sütunundaki eksik değerler (N/A), sayısal eksik değer işleme adımında medyan ile doldurulacak şekilde `NaN` olarak ele alınmıştır.
*   **Sayısal Eksikler (Genel):** Tüm sayısal sütunlarda eksik değerler **medyan** ile doldurulmuş ve ayrıca eksik olup olmadığını belirten gösterge sütunları eklenmiştir (`SimpleImputer(add_indicator=True)`).
*   **Eksik Kategorik Veriler:** Kategorik değişkenlerdeki eksiklikler "Missing" etiketi ile doldurulmuştur.

### 4.2. Özellik Dönüşümleri
*   **Sayısal Değişkenler:** Standartlaştırma işlemi `StandardScaler(with_mean=False)` ile uygulanmıştır. Bu seçim, One-Hot Encoding sonrası sparse matrise uyumluluk için merkezleme (mean çıkarma) yapılmadığı; ölçekleme adımının uygulandığı anlamına gelir.
*   **Kategorik Değişkenler:** `OneHotEncoder` kullanılarak sayısal vektörlere dönüştürülmüştür.

### 4.3. Özellik Mühendisliği (Feature Engineering)
Modelin karmaşık ilişkileri öğrenebilmesi için "Basic" modunda ek özellikler türetilmiştir:
*   **Karesel Terimler:** `age_squared`, `bmi_squared`.
*   **Logaritmik Dönüşüm:** `glucose_log1p` (Glikoz dağılımını düzeltmek için).
*   **Etkileşim Terimleri:**
    *   `age_x_glucose`
    *   `age_x_hypertension`
    *   `age_x_heart_disease`
    *   `hypertension_x_heart_disease`

## 5. Modelleme Stratejisi

### 5.1. Kullanılan Algoritma
Mevcut `metrics.json` çıktısına göre, en son eğitilen model **Logistic Regression (`logreg`)** algoritmasıdır. Bu algoritma, özellikle medikal risk skorlamasında yorumlanabilirliği ve olasılık tabanlı çıktısı nedeniyle tercih edilmiştir.

### 5.2. Dengesiz Veri (Imbalanced Data) Yönetimi
İnme vakaları toplumda az görüldüğü için veri seti dengesizdir. Bunu yönetmek için iki strateji uygulanmıştır:
1.  **Bootstrap Oversampling:** Eğitim verisindeki inme vakaları (sınıf 1), yapay olarak çoğaltılarak (resampling) modelin bu sınıfı daha iyi öğrenmesi sağlanmıştır (`target_pos_ratio` ile kontrol edilir).
2.  **Class Weights:** Lojistik Regresyon modeli `class_weight='balanced'` parametresi ile çalıştırılarak azınlık sınıfına verilen hata cezası artırılmıştır.

### 5.3. Eşik Değeri Ayarlama (Threshold Tuning)
Standart 0.5 eşik değeri yerine, medikal öncelikler gözetilerek **Dynamic Thresholding** uygulanmıştır.
*   **Strateji:** `precision_at_recall` (Belirli bir duyarlılık seviyesinde en iyi kesinliği bulma).
*   **Hedef:** Doğrulama (Validation) setinde **%80 Recall (Duyarlılık)** hedeflenmiştir.
*   **Seçilen Eşik:** `0.58` (Bu değerin üzerindeki olasılıklar "İnme Riski Var" olarak sınıflandırılır).

## 6. Model Performansı ve Sonuçlar
Test seti üzerindeki güncel performans metrikleri (`artifacts/stroke_classification_metrics.json` dosyasından):

*   **Accuracy (Doğruluk):** ~%78.8
*   **Balanced Accuracy:** ~%77.5
*   **Recall (Duyarlılık):** ~%76.0
    *   *Yorum:* Model, gerçek inme vakalarının %76'sını başarıyla tespit edebilmektedir. Bu, medikal bir tarama modeli için kabul edilebilir bir başlangıç seviyesidir.
*   **Precision (Kesinlik):** ~%15.6
    *   *Yorum:* Yüksek duyarlılık hedefi nedeniyle, model "yalancı pozitif" (False Positive) üretmeye meyillidir. Yani riski olmayan bazı kişilere de risk uyarısı verebilir. Bu, hastalığı kaçırmaktan daha güvenli bir yaklaşımdır.
*   **ROC AUC:** ~0.837
*   **PR AUC (Average Precision):** ~0.201
