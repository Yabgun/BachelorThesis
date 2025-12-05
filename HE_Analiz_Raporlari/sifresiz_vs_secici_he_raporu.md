# Şifresiz Veri vs Seçici Homomorfik Şifreleme Karşılaştırması

## Genel Bakış
Bu rapor, modelimizin **Şifresiz (Plaintext)** ortamda ürettiği sonuçlar ile **Seçici Homomorfik Şifreleme (Selective HE)** yöntemiyle ürettiği sonuçları kıyaslamaktadır. Amaç, şifreleme işleminin sonuçlarda herhangi bir bozulmaya veya doğruluk kaybına yol açmadığını kanıtlamaktır.

## 1. Regresyon (Maliyet Tahmini) Kıyaslaması
Şifresiz tahminler ile Seçici HE tahminleri arasındaki farklar incelenmiştir.

*   **Ortalama Kare Hata (MSE) Farkı:** 0.0000003813
*   **Uyum Katsayısı (R²):** 1.0000000000 (1.00 = Birebir Aynı)

**Yorum:** İki yöntem arasındaki fark sıfıra yakındır (CKKS şifrelemesinden kaynaklı ihmal edilebilir ondalık farklar). Bu, şifreli hesaplamanın matematiksel olarak doğru çalıştığını kanıtlar.

## 2. Sınıflandırma (Risk Analizi) Kıyaslaması
Hastaların risk skorları (0-1 arası olasılık) karşılaştırılmıştır.

*   **Maksimum Olasılık Farkı:** 0.0000002430

**Yorum:** Şifresiz ve şifreli modelin ürettiği risk skorları neredeyse aynıdır. Karar mekanizması (Yüksek Risk / Düşük Risk) şifrelemeden etkilenmemiştir.

## Sonuç
**Seçici Homomorfik Şifreleme**, verilerin gizliliğini korurken, **Şifresiz (Plaintext)** işlemeyle **aynı doğruluğu** sağlamaktadır. Grafiklerde görülen "Mükemmel Uyum (y=x)" çizgisi üzerindeki dağılım, modelin güvenilirliğini görsel olarak da teyit etmektedir.
