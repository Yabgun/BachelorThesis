# Hatalı Sınıflandırma Analizi

## Özet
Toplam **400** hasta içerisinden **1** adet hatalı sınıflandırma tespit edilmiştir.
Doğruluk Oranı: **%99.75**

## Hatalı Vaka Detayı (ID: 189)
Bu hasta gerçekte **Düşük Risk (0)** grubundadır, ancak model **Yüksek Risk (1)** tahmini yapmıştır.

### Neden Hata Yapıldı?
Model bu hasta için **%54.23** risk hesaplamıştır.
Karar sınırı %50 olduğu için, bu değer sınıra çok yakın olabilir veya hastanın bazı özellikleri (örn. yaşı, genetik skoru) modelin kafasını karıştırmış olabilir.

### Hastanın Verileri:
| Özellik | Değer | Model Ağırlığı (Etkisi) |
| :--- | :--- | :--- |
| **Age** | -0.8580 | 5.2706 (Katkı: -4.5223) |
| **Gender** | 1.0279 | 0.0754 (Katkı: 0.0775) |
| **BMI** | 0.0947 | 1.9404 (Katkı: 0.1837) |
| **Smoking** | -0.6304 | 4.9850 (Katkı: -3.1425) |
| **CXR_Opacity** | -0.3433 | 3.7328 (Katkı: -1.2816) |
| **Genetic_Marker** | -0.0970 | 2.1794 (Katkı: -0.2115) |

**Toplam Logit Skoru:** 0.1697 (Bias: 9.0664 dahil)
**Sigmoid Sonrası Olasılık:** 0.542331
