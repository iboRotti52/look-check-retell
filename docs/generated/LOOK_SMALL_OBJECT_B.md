# Look kapsaması: küçük nesne sayımı düzeltildi (B_fit)

Üreten: `scripts/49_look_small_object.py`. Küçük kutu: kısa kenar < 2 BLIP yaması. Yeni tanım: isabet küçük kutuya gelmeli; eski tanım (script 37'nin ilk sayımı) sınıfın herhangi bir kutusunu sayıyordu.

Görüntü: 2365.

| Look kuralı | n | top-3 | küçük n | küçük top-3 (yeni) | küçük top-3 (eski) |
|---|---|---|---|---|---|
| with_filter | 2470 | 90.04 | 162 | 4.94 | 39.51 |
| without_filter | 2470 | 93.28 | 162 | 55.56 | 83.33 |
