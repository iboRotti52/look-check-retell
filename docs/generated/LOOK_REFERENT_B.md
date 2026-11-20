# Look L1 — referans sinyalleri (B, otomatik proxy)

Üreten: `scripts/46_look_referent.py`. Protokol: `docs/LOOK_L1_PROTOKOL.md`. Etiket (`in_refs`) otomatik referans proxy'sidir, insan etiketi değil. Harita: kelimeyi üreten adım (E3 önbelleği); sıralama: `region_evidence.candidates`, S4 Look kuralı.

Skorlanan claim: 6856; Look adayı olmayan (top-1 yok): 0. Hücre başına en az 25 pozitif ve negatif yoksa AUROC verilmez (–).

## Genel (thing + stuff, kişi hariç)

| Sinyal | AUROC | n destekli | n desteksiz |
|---|---|---|---|
| strength | 0.537 | 3494 | 1100 |
| top1_concentration | 0.588 | 3494 | 1100 |
| neg_entropy | 0.413 | 3494 | 1100 |
| null_margin | 0.500 | 3411 | 1093 |
| top1_area | 0.583 | 3494 | 1100 |

`top1_area` yalnız boyutun ne kadar ayırdığını gösterir; bir sinyal bunu geçmiyorsa boyutu izliyor olabilir.

## Look top-1 kutu alanına göre üçte birlikler

Sınırlar: 0.1218, 0.4628

| Sinyal | küçük | orta | büyük |
|---|---|---|---|
| strength | 0.544 (1047/485) | 0.611 (1208/323) | 0.556 (1239/292) |
| top1_concentration | 0.523 (1047/485) | 0.596 (1208/323) | 0.564 (1239/292) |
| neg_entropy | 0.456 (1047/485) | 0.467 (1208/323) | 0.430 (1239/292) |
| null_margin | 0.535 (1033/482) | 0.570 (1166/321) | 0.540 (1212/290) |
| top1_area | 0.527 (1047/485) | 0.549 (1208/323) | 0.475 (1239/292) |

Parantez içi: (n destekli / n desteksiz).

## Neden F30kE kutu alanı kullanılmıyor

F30kE kutuları referans öbeklerine bağlı: destekli 3138/3494, desteksiz 1/1100 claim'in kutusu var.

Bu dosya yalnız B üzerindedir; validation/test okunmaz (`docs/LOOK_L1_PROTOKOL.md` §5).
