# Sağlık Verilerinde Homomorfik Şifreleme ve Zafiyet Analizi

Pamukkale Üniversitesi Bilgisayar Mühendisliği lisans bitirme tezi ve ilgili TÜBİTAK 2209-A projesinin kod ve yazım deposu.

- **Öğrenci:** Buğra Tutumlu
- **Danışman:** Dr. Öğr. Üyesi Alper Uğur

## Konu

ROI-seçici homomorfik şifreleme, tıbbi görüntünün yalnızca ilgi bölgesini (ROI) şifreleyip görüntünün geri kalanını açık bırakır ve bunu "izin verilen sızıntı" olarak kabul eder (örn. Π_ROI, ePrint 2026/103; Encrypt What Matters, arXiv 2609.09357).

Bu tez, söz konusu izin verilen sızıntının gerçek göğüs röntgeni (COVID-QU-Ex) ve beyin MR (Cheng) görüntülerinde teşhisi büyük ölçüde ele verdiğini sistematik olarak gösterir; gizlilik şartı konduğunda seçici şifrelemenin hız avantajını yitirdiğini ölçer ve sızıntıyı yeniden kullanılabilir bir değerlendirme boru hattıyla belgeler.

## Depo yapısı

| Klasör | İçerik |
|---|---|
| `roi-leakage/` | Deney kodu: Π_ROI yeniden üretimi, meta veri ve bağlam saldırıları, kök neden analizi, savunmalar; sonuç tabloları ve şekilleri `results/` altında |
| `tez-yazim/2ASILTEZ/` | Tezin LaTeX kaynağı (PAÜ şablonu) |

## Ortam (kod)

- Python 3.13, PyTorch (CUDA), TenSEAL 0.3.16. Bağımlılıklar: `roi-leakage/requirements.txt`.
- Deneyler proje kökünden `python -m experiments.<ad>` ile çalışır; adım listesi `roi-leakage/README.md` içindedir.
- Ham veriler ve büyük ara çıktılar (sanal ortam, günlükler, model ağırlıkları) depoya dahil değildir.

## Tezi derleme

`tez-yazim/2ASILTEZ/` içinde MiKTeX ile:

```
pdflatex -interaction=nonstopmode tez.tex
bibtex tez
pdflatex -interaction=nonstopmode tez.tex
pdflatex -interaction=nonstopmode tez.tex
```

## Durum

Bu, devam eden bir araştırmadır. Kişisel çalışma notları ve belgeler depoya dahil edilmemiştir.
