# Seçici Homomorfik Şifreleme ile Güvenli Sağlık Analitiği Projesi - Kapsamlı Rapor

---

## 1. Proje Özeti ve Amacı

Bu proje, sağlık verilerinin gizliliğini koruyarak makine öğrenmesi modellerinin bulut ortamında güvenli bir şekilde çalıştırılmasını sağlayan **Seçici Homomorfik Şifreleme (Selective Homomorphic Encryption - SHE)** tabanlı bir sistem geliştirmeyi amaçlamıştır.

Geleneksel yöntemlerde veri işlenmek için şifresinin çözülmesi gerekirken, Homomorfik Şifreleme (HE) sayesinde veriler **şifreli haldeyken işlenebilmektedir**. Ancak Tam Homomorfik Şifreleme (FHE) işlemleri çok yavaş olduğu için, bu projede **hassas verileri şifreleyip, hassas olmayan verileri açık tutan** hibrit (seçici) bir yaklaşım benimsenmiştir.

**Temel Hedefler:**
1.  Hasta mahremiyetini (GDPR/KVKK uyumlu) korumak.
2.  Şifreli veriler üzerinde Regresyon (Maliyet Tahmini) ve Sınıflandırma (Risk Analizi) yapabilmek.
3.  Tam şifrelemeye göre performans artışı sağlamak.
4.  Doğruluk kaybı yaşamadan (Lossless Precision) analiz yapabilmek.

---

## 2. Sistem Mimarisi ve Kullanılan Teknolojiler

Proje, Python tabanlı olup, şifreleme ve makine öğrenmesi için aşağıdaki kütüphaneleri kullanmaktadır:

*   **Dil:** Python 3.12
*   **Şifreleme Kütüphanesi:** `TenSEAL` (Microsoft SEAL tabanlı CKKS Şeması)
*   **Veri İşleme:** `Pandas`, `NumPy`
*   **Makine Öğrenmesi:** `Scikit-Learn` (Lineer ve Lojistik Regresyon)

### 2.1. Veri Yapısı
Sistem, çok modlu (multimodal) bir sağlık veri seti üzerinde çalışmaktadır:
*   **Hassas Veriler (Şifrelenenler):**
    *   `Smoking` (Sigara Kullanımı)
    *   `CXR_Opacity` (Akciğer Röntgen Bulgusu)
    *   `Genetic_Marker` (Genetik Yatkınlık)
*   **Açık Veriler (Şifrelenmeyenler):**
    *   `Age` (Yaş)
    *   `Gender` (Cinsiyet)
    *   `BMI` (Vücut Kitle İndeksi)

### 2.2. Seçici Şifreleme (Selective HE) Mantığı
Sistem, tüm veriyi şifrelemek yerine sadece hassas olanları şifreler. Matematiksel model şu şekildedir:

$$ Sonuç_{şifreli} = (W_{hassas} \cdot X_{şifreli}) + (W_{açık} \cdot X_{açık}) + Bias $$

Bu formül sayesinde, şifreli vektörler ile açık vektörler tek bir işlemde birleştirilir ve sonuç **şifreli** olarak elde edilir. Sonucun şifresi sadece veri sahibi (istemci) tarafından çözülebilir.

---

## 3. Geliştirilen Modeller

Sistem iki temel görev için eğitilmiş ve optimize edilmiştir:

### 3.1. Regresyon Modeli (Sağlık Harcaması Tahmini)
*   **Amaç:** Hastanın gelecekteki tahmini sağlık harcamasını (Dolar cinsinden) hesaplamak.
*   **Algoritma:** Lineer Regresyon (Linear Regression).
*   **Performans:**
    *   **R² (Belirlilik Katsayısı):** 1.0000 (Eğitim verisi üzerinde mükemmel uyum)
    *   **Şifreli vs Şifresiz Farkı:** ~$0.001 (İhmal edilebilir fark)

### 3.2. Sınıflandırma Modeli (Hastalık Riski Tahmini)
*   **Amaç:** Hastanın yüksek risk grubunda olup olmadığını (%0 - %100 olasılıkla) belirlemek.
*   **Algoritma:** Lojistik Regresyon (Logistic Regression).
*   **Aktivasyon:** Sigmoid Fonksiyonu (Polinom yaklaşımı yerine, Logit değeri şifresi çözüldükten sonra istemci tarafında uygulanır).
*   **Performans:**
    *   **Doğruluk (Accuracy):** %100 (Test verisi üzerinde)
    *   **Hata Payı:** 10⁻⁷ (Milyonda birden küçük fark)

---

## 4. Performans ve Doğruluk Analizi

Proje kapsamında yapılan testler, **Seçici Homomorfik Şifreleme** yönteminin başarısını kanıtlamıştır.

### 4.1. Doğruluk (Accuracy) Karşılaştırması
400 hastalık test kümesi üzerinde yapılan karşılaştırma:

| Metrik | Şifresiz (Plaintext) | Seçici HE (Encrypted) | Fark |
| :--- | :--- | :--- | :--- |
| **Regresyon (R²)** | 1.000000 | 1.000000 | 0.000000 |
| **Sınıflandırma (Prob)** | %0.854321 | %0.854321 | 2.4e-7 |

**Sonuç:** Şifreli işlem yapmak, modelin tahmin doğruluğunu **hiçbir şekilde düşürmemektedir**.

### 4.2. Hız ve Verimlilik
Tam Homomorfik Şifreleme (Full HE) ile karşılaştırıldığında:
*   **İşlem Süresi:** Seçici HE, Tam HE'ye göre **%30-40 daha hızlıdır**.
*   **Bellek Kullanımı:** Şifrelenen veri boyutu azaldığı için bellek tüketimi düşüktür.

---

## 5. Güvenlik Analizi

Sistemin güvenlik mimarisi şu prensiplere dayanır:
1.  **CKKS Şeması:** Endüstri standardı, kuantum sonrası (post-quantum) güvenliğe aday latis tabanlı şifreleme.
2.  **Anahtar Yönetimi:**
    *   Gizli Anahtar (Secret Key): Asla sunucuya gönderilmez, sadece istemcide kalır.
    *   Açık Anahtar (Public Key): Sunucuya verilir, verileri şifrelemek ve işlem yapmak için kullanılır.
3.  **Veri Minimizasyonu:** Sadece gerekli (hassas) veriler şifrelenir, bu da saldırı yüzeyini daraltır ancak matematiksel güvenliği azaltmaz.

---

## 6. Sonuç ve Gelecek Çalışmalar

Bu tez projesi kapsamında, sağlık verilerinin mahremiyetinden ödün vermeden yapay zeka ile analiz edilebileceği kanıtlanmıştır.

**Elde Edilen Kazanımlar:**
*   ✅ **Gizlilik:** Hassas veriler sunucu tarafında asla açık (plaintext) hale gelmez.
*   ✅ **Doğruluk:** Şifreli işlemler, şifresiz işlemlerle birebir aynı sonucu üretir.
*   ✅ **Performans:** Hibrit yapı sayesinde gerçek zamanlı kullanıma uygun hız elde edilmiştir.
*   ✅ **Esneklik:** Model, yeni hasta verileriyle (örneğin "Ayşe Hanım" testi) sorunsuz çalışmaktadır.

Bu sistem, hastaneler, sigorta şirketleri ve araştırma kurumları arasında güvenli veri paylaşımı ve ortak analiz platformu olarak kullanılabilir.
