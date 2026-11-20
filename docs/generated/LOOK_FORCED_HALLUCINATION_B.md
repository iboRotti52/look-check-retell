# Look L1 — zorlanmış halüsinasyon testi (B)

Üreten: `scripts/47_look_forced_hallucination.py`. Protokol: `docs/LOOK_L1_PROTOKOL.md`. GA: görüntü düzeyi bootstrap %95. *predict* = kelimeyi üreten adım (v4/LCR); *read* = kelimeyi okuyan adım.
Bundle `c7156af83dcca3b4`, Look kuralı `{'min_patches': 0, 'alpha': 0.5}`, seed 415, yokluk marjı 0.0. Atılan örnekler: {'misidentified_not_reextracted': 12, 'present_no_certified_class': 48, 'present_not_reextracted': 1, 'invented_not_reextracted': 3}.

## Yanlış tanımlanmış (n = 300)

B3 (yanlış kelimenin detektör kutusu) top-1 isabet: 22.3 [17.7, 27.0]

| okuma | top-1 | top-3 | top-1 − B3 | yeniden kullanma − yeniden hesaplama | top-1 değişti % | aday yok |
|---|---|---|---|---|---|---|
| predict | 81.7 [77.3, 86.0] | 91.0 [87.7, 94.0] | 59.3 [52.7, 65.7] | 0.3 [0.0, 1.0] | 0.3 | 0 |
| read | 36.3 [30.7, 41.7] | 63.3 [58.0, 68.7] | 14.0 [7.3, 20.7] | 45.7 [39.3, 52.0] | 30.0 | 0 |

## Uydurulmuş (n = 300, konum kontrolü olan 251; yokluk ve varlık yalnız S4 detektör proxy'si)

Eklenen her isim başlığın sonunda ("and a ..."). Ana ölçüm bu yüzden aynı ekle eklenmiş, referansta anılan ve detektörün bulduğu bir nesneye karşı; taslaktaki doğru claim'lere karşı olan konum farkını da içerir.

### Uydurulmuş vs aynı konumda eklenmiş gerçek nesne (konum kontrollü, ana ölçüm)

251 görüntü; pozitif: 251 eklenmiş gerçek nesne, negatif: 251 bozulmuş claim. Hücrede (n pozitif/n negatif) tanımlı değerler; her iki tarafta 10'dan az varsa –.

| sinyal | AUROC predict | AUROC read |
|---|---|---|
| strength | 0.443 [0.407, 0.480] (251/251) | 0.544 [0.510, 0.579] (251/251) |
| top1_concentration | 0.489 [0.459, 0.518] (251/251) | 0.571 [0.528, 0.612] (251/251) |
| neg_entropy | 0.455 [0.422, 0.488] (251/251) | 0.544 [0.498, 0.590] (251/251) |
| null_margin | 0.496 [0.474, 0.519] (251/251) | 0.535 [0.492, 0.578] (251/251) |

### Uydurulmuş vs taslaktaki doğru claim (konum karışık)

244 görüntü; pozitif: 362 referans destekli gerçek claim, negatif: 244 bozulmuş claim. Hücrede (n pozitif/n negatif) tanımlı değerler; her iki tarafta 10'dan az varsa –.

| sinyal | AUROC predict | AUROC read |
|---|---|---|
| strength | 0.500 [0.455, 0.544] (362/244) | 0.625 [0.591, 0.655] (362/244) |
| top1_concentration | 0.822 [0.787, 0.853] (362/244) | 0.705 [0.669, 0.740] (362/244) |
| neg_entropy | 0.635 [0.591, 0.679] (362/244) | 0.773 [0.740, 0.805] (362/244) |
| null_margin | 0.686 [0.649, 0.723] (348/244) | 0.411 [0.363, 0.460] (348/244) |

### Yanlış tanımlanmış vs doğru (betimsel)

300 görüntü; pozitif: 454 referans destekli gerçek claim, negatif: 300 bozulmuş claim. Hücrede (n pozitif/n negatif) tanımlı değerler; her iki tarafta 10'dan az varsa –.

| sinyal | AUROC predict | AUROC read |
|---|---|---|
| strength | 0.514 [0.495, 0.534] (454/300) | 0.724 [0.697, 0.751] (454/300) |
| top1_concentration | 0.491 [0.471, 0.511] (454/300) | 0.599 [0.568, 0.631] (454/300) |
| neg_entropy | 0.462 [0.440, 0.482] (454/300) | 0.621 [0.595, 0.647] (454/300) |
| null_margin | 0.465 [0.443, 0.486] (439/289) | 0.431 [0.397, 0.466] (439/289) |

