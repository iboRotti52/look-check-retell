# Look E3 — inceleme ölçümleri

Üreten: `scripts/look_e3_review_table.py`. Kaynaklar: `results/look_e3_review.json`,
`look_e3_drift_probe.json`, `look_e3_live_generation.json`, `look_benchmark_A_validation_look4.json`.

Başlıklar (katman ve başlık 1’den başlar): L9H8, L10H8, L10H9, L9H3.

| Yeniden üretim | İddia | Değişen bölge | En büyük harita farkı | Eski tutarsız excess |
|---|---:|---:|---:|---:|
| A_B_look4 | 6856 | 0 | 0.000000000 | 2262 |
| A_validation_look4 | 2272 | 0 | 0.000000000 | 769 |

Canlı sabit taslak deneyi: 4 B görüntüsü, 9 iddia; en büyük harita/kutu farkı 0.000000000/0.000000000.
Gerçek taslak üretimi: 2 B görüntüsü, 5 iddia; taslak eşitliği True, kutu eşitliği True, en büyük harita farkı 0.000000000.

| Bağımsız validation yeniden ölçümü | Bölge | Pointing |
|---|---:|---:|
| all | 80.39 | 90.97 |
| person | 76.48 | 91.72 |
| thing | 85.09 | 91.95 |
| stuff | 78.26 | 86.23 |

| Eşlenmiş yeniden hesap hattı (bölge) | S3 | Seçili başlıklar | Fark |
|---|---:|---:|---:|
| B | 74.16 | 78.23 | +4.07 |
| validation (önceden raporlanmış) | 75.15 | 79.22 | +4.07 |

Aşağıdaki küçük m ölçümleri sonradan yapılan **B tanısıdır**, yeni seçim değildir.

| Başlık sayısı | B bölge |
|---|---:|
| 1 | 69.29 |
| 2 | 77.34 |
| 3 | 77.54 |
| 4 | 78.23 |

| Tek başlık (B) | person pointing | thing pointing | stuff pointing |
|---|---:|---:|---:|
| L9H8 | 71.26 | 90.96 | 88.20 |
| L10H8 | 96.48 | 71.68 | 66.46 |
| L10H9 | 67.56 | 89.89 | 84.60 |
| L9H3 | 86.99 | 59.68 | 72.05 |

| Kişi grubu (B, GT kutu adedi) | İddia | Bölge |
|---|---:|---:|
| multiple_person_boxes | 1099 | 78.80 |
| single_person_box | 1145 | 81.14 |

SAM–skor kutusu proxy’si (B, GT’li kişi): 215/2244 eşleşmede IoU < 0.5; medyan IoU 0.8751. Bu, SAM bölgesinin yeniden detektörle skorlanması değildir.

Sayısal sapma tanısı: ilk 16 B görüntüsündeki 37 iddia.

| Koşul | İddia başına en büyük yama hatasının ortalaması | En büyük hata |
|---|---:|---:|
| exclude_sep | 0.008182 | 0.038372 |
| include_sep | 0.005256 | 0.026078 |
| half_attention | 0.008181 | 0.038366 |
| half_means | 0.008173 | 0.038357 |
| half_subtract | 0.008184 | 0.038390 |
| half_model | 0.008184 | 0.038422 |

Sonuç: SEP dahil etmek örnekte farkı azaltıyor; CPU yarım duyarlık deneyi arşivi yeniden üretmiyor.
Tarihsel GPU hattındaki sapmanın kök nedeni henüz saptanmadı. V4 kendi yeniden hesap hattında birebir üretildi.

Önceki 95% aralıklar tekil, görüntü bootstrap aralıklarıdır; aileler/türler için eşzamanlı güvence vermez.
Seçim sonrası B ÇD sayıları tüm A→B seçicisinin iç içe dış-kat performansı değildir.
