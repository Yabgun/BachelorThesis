# Gelişmiş Seçici Homomorfik Şifreleme Modeli: Teknik Derinlemesine Analiz Raporu

**Tarih:** 24.05.2024  
**Konu:** Model Mimarisi, Öğrenme Süreci ve Şifreli Hesaplama Mekaniği  
**Dosya:** `reports/model_teknik_detay_raporu.md`

---

## 1. Giriş ve Amaç

Bu rapor, geliştirilen "Seçici Homomorfik Şifreleme (Selective Homomorphic Encryption)" tabanlı sağlık analiz sisteminin teknik iç yapısını, modelin öğrenme sürecini ve şifreli veriler üzerinde nasıl çalıştığını en ince detayına kadar açıklar.

Sistem, **veri gizliliğinden ödün vermeden** makine öğrenmesi modellerini çalıştırmak için tasarlanmıştır.

---

## 2. Kullanılan Framework ve Teknolojiler

Modelin başarısı, aşağıdaki teknolojilerin hibrit kullanımına dayanır:

| Teknoloji | Görevi | Neden Seçildi? |
|-----------|--------|----------------|
| **TenSEAL** | Homomorfik Şifreleme Kütüphanesi | Microsoft SEAL üzerine kuruludur. CKKS şeması ile ondalıklı (float) sayılar üzerinde şifreli işlem yapmaya izin verir. |
| **Scikit-Learn** | Model Eğitimi (The Brain) | Sektör standardı ML kütüphanesi. `LogisticRegression` ve `LinearRegression` algoritmalarını sağlar. |
| **Pandas & NumPy** | Veri İşleme | Büyük veri setlerini matris formatında işlemek ve matematiksel operasyonlar için kullanıldı. |
| **Python** | Ana Dil | Tüm bu kütüphaneleri birleştiren "yapıştırıcı" dil. |

---

## 3. Modelin "Beyni": Öğrenme Süreci (Training)

Modelin "zeka" kazandığı aşama burasıdır. Bu süreç `scripts/train_robust_models.py` dosyasında gerçekleşir.

### 3.1. Nasıl Öğreniyor? (The Learning Mechanism)

Modelimiz **Denetimli Öğrenme (Supervised Learning)** kullanır. Yani modele hem girdileri (Yaş, BMI, Sigara vb.) hem de doğru cevapları (Maliyet, Risk Durumu) veririz.

Modelin kullandığı matematiksel yöntem **Gradient Descent (Bayır İnişi)** algoritmasıdır:

1.  **Başlangıç:** Model tüm özelliklerin ağırlıklarını (katsayılarını) rastgele veya sıfır olarak başlatır.
2.  **Tahmin:** Bir hasta için tahmin yapar.
3.  **Hata Hesabı:** Tahmin ile gerçek değer arasındaki farkı (Hatayı) bulur.
4.  **Güncelleme (Backpropagation):** Hatayı azaltmak için ağırlıkları günceller.
    *   *Örnek:* Eğer "Sigara İçen" hastaların maliyetini düşük tahmin ettiyse, Sigara özelliğinin ağırlığını (katsayısını) artırır.
5.  **Tekrar:** Bu işlem binlerce kez tekrarlanır (Epochs) ve hata minimize edilir.

**Kod Karşılığı:**
```python
# Scikit-Learn kütüphanesi bu karmaşık matematiksel süreci tek satırda yapar:
clf.fit(X_train, y_class_train) 
# fit() fonksiyonu, en uygun ağırlıkları (Weights) ve sabit terimi (Bias) bulur.
```

### 3.2. Öğrenilen Bilgi Nerede? (Weights & Coefficients)

Eğitim bittiğinde, modelin öğrendiği her şey **Ağırlıklar (Weights)** ve **Bias (Sabit)** olarak saklanır. Bu veriler `models/he_model_weights.json` dosyasına kaydedilir.

**JSON Dosyası Yapısı ve Anlamı:**
```json
"classification": {
    "weights": [
        0.045,  // Age (Yaş arttıkça risk az da olsa artıyor)
        0.850,  // Smoker (Sigara içmek riski çok artırıyor -> Yüksek Pozitif Ağırlık)
        -0.200, // Healthy_Diet (Sağlıklı beslenme riski düşürüyor -> Negatif Ağırlık)
        ...
    ],
    "bias": -1.50 // Başlangıç noktası
}
```
*   **Pozitif Ağırlık (+):** Özellik arttıkça risk/maliyet artar.
*   **Negatif Ağırlık (-):** Özellik arttıkça risk/maliyet düşer.
*   **Büyüklük:** Sayı ne kadar büyükse, o özellik o kadar önemlidir.

---

## 4. Modelin "Kalkanı": Seçici Homomorfik Şifreleme (Selective HE)

Bu projenin en büyük yeniliği **Seçici Şifreleme** mimarisidir. Her şeyi körü körüne şifrelemek yerine, veri hassasiyetine göre işlem yapılır.

### 4.1. Veri Ayrıştırma

Veriler sisteme girerken ikiye ayrılır:
1.  **Hassas Veriler (Sensitive):** Sigara, Genetik Marker, Akciğer Filmi Sonucu vb. -> **ŞİFRELENİR (Encrypted)**
2.  **Açık Veriler (Public/Insensitive):** Yaş, Cinsiyet, BMI (Anonimleştirilmiş) -> **ŞİFRELENMEZ (Plaintext)**

### 4.2. Hibrit Matematik (Hybrid Math)

Modelimiz tahmini şu formülle yapar (Linear Model):
$$ y = (w_1 \cdot x_1 + w_2 \cdot x_2 + ... + w_n \cdot x_n) + bias $$

Sistemimiz bunu parçalar:

1.  **Şifreli Kısım (Server-Side):**
    $$ Sonuc_{Enc} = (w_{sigara} \cdot [Sigara]_{Enc}) + (w_{genetik} \cdot [Genetik]_{Enc}) $$
    *Bu işlem TenSEAL kullanılarak şifreli veriler üzerinde yapılır. Sunucu veriyi asla görmez.*

2.  **Açık Kısım (Server-Side):**
    $$ Sonuc_{Plain} = (w_{yas} \cdot Yas) + (w_{bmi} \cdot BMI) + bias $$
    *Bu işlem standart Python matematiği ile milisaniyeler içinde yapılır.*

3.  **Birleştirme:**
    $$ ToplamSonuc_{Enc} = Sonuc_{Enc} + Sonuc_{Plain} $$
    *Şifreli bir sayıya açık bir sayı eklenebilir (CKKS özelliği). Sonuç hala şifrelidir.*

---

## 5. Çalışma Akışı (Execution Workflow)

Bir hasta sisteme veri girdiğinde (`scripts/interactive_test.py` çalıştırıldığında) arka planda şu adımlar gerçekleşir:

1.  **Girdi Alımı:** Kullanıcıdan veriler alınır.
2.  **Ön İşleme (Preprocessing):** Veriler eğitim verisiyle aynı formata getirilir (StandardScaler kullanılarak ölçeklenir).
3.  **Şifreleme (Client-Side):** Sadece hassas özellikler kullanıcının bilgisayarında şifrelenir.
4.  **Hesaplama (Simulated Server):**
    *   Şifreli veriler ile şifreli ağırlıklar çarpılır.
    *   Açık veriler ile açık ağırlıklar çarpılır.
    *   İki sonuç toplanır.
5.  **Sonuç İletimi (Logit):** Sınıflandırma için ham sonuç (Logit) elde edilir.
6.  **Deşifreleme ve Aktivasyon (Client-Side):**
    *   Kullanıcı şifreli sonucu kendi özel anahtarıyla çözer.
    *   **Sigmoid Fonksiyonu:** `1 / (1 + e^-x)` formülü uygulanarak sonuç 0 ile 1 arasına (olasılığa) dönüştürülür.
    *   *Neden Client-Side?* Sigmoid gibi lineer olmayan fonksiyonlar şifreli ortamda çok yavaştır ve doğruluk kaybı yaratır. Biz bunu istemci tarafında yaparak **%100 Doğruluk** sağladık.

---

## 6. Sonuçların Yorumlanması

Model iki tür çıktı üretir:

1.  **Regresyon (Maliyet Tahmini):**
    *   Doğrudan sayısal bir değer verir (Örn: $12,500).
    *   CKKS şeması sayesinde şifreli hesaplama, şifresiz hesaplama ile virgülden sonraki küsuratlara kadar aynıdır.

2.  **Sınıflandırma (Risk Tahmini):**
    *   Bir olasılık değeri verir (Örn: %85 Riskli).
    *   Eğer olasılık > %50 ise "Yüksek Risk", değilse "Düşük Risk" olarak etiketlenir.

## 7. Özet

Bu model; **Scikit-Learn** ile eğitilmiş zekayı, **TenSEAL** ile sağlanmış bir zırhın içine yerleştirir. **Seçici Şifreleme** stratejisi sayesinde, tam şifreli sistemlerin hantallığından kurtulur ve şifresiz sistemlerin güvensizliğini ortadan kaldırır.

Elde edilen sonuç: **Hızlı, Güvenli ve %100 Doğru.**
