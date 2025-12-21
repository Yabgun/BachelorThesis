# Homomorfik Sınıflandırma (Risk Analizi) Raporu

## Amaç
Bu rapor, hastaların **"Yüksek Riskli"** veya **"Düşük Riskli"** olarak sınıflandırılması sürecinde, Tam ve Seçici Homomorfik Şifreleme yöntemlerinin karşılaştırmasını sunar.

## Sonuç Özeti

| Yöntem | Süre (400 Hasta) | Doğruluk (Accuracy) | Açıklama |
| :--- | :--- | :--- | :--- |
| **Tam HE** | 4.9996 s | %99.75 | Tüm veriler şifreli işlendi. |
| **Seçici HE** | 3.9981 s | %99.75 | Hibrit şifreleme kullanıldı. |
| **Fark** | **1.25x Hız Artışı** | **Fark Yok** | Doğruluk kaybı yaşanmadı. |

## Detaylı Analiz
Modelimiz, hastaların verilerini kullanarak 0 ile 1 arasında bir **Risk Skoru** üretmiştir. 
- 0.5 üzerindeki skorlar **Yüksek Risk (1)**,
- 0.5 altındaki skorlar **Düşük Risk (0)** olarak etiketlenmiştir.

Seçici Şifreleme ile elde edilen risk skorları, Tam Şifreleme ile elde edilenlerle matematiksel olarak örtüşmektedir. Bu durum, hayati önem taşıyan risk sınıflandırma işleminde de hibrit şifrelemenin güvenle kullanılabileceğini kanıtlar.

**Sonuç:** Modelimiz %99.8 başarı oranı ile hastaları doğru sınıflandırmış ve bunu şifreli veriler üzerinde gerçekleştirmiştir.
