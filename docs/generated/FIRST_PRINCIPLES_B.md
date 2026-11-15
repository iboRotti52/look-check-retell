# İlk ilkeler tanısı (yalnız B)

Üreten: `scripts/review_first_principles_B.py`. Validation/test okunmadı.
Etiket otomatik referans proxy'si; görsel denetim Claude'un gözlemi, insan etiketi değil.

Bölgesi olan kişi dışı claim: 4594; proxy-yanlış: 1100.

## 1. Yanlış claim'de doğru eylem: yerine koymak mı, silmek mi?
Referanslarda claim'le aynı ikame grubundan başka bir sınıf var mı (varsa yerine koymak mümkün):

| grup | oran |
|---|---|
| all_wrong | 0.506 |
| thing | 0.452 |
| stuff | 0.683 |
| clothing | 0.500 |

## 2. Alternatif önerisinin kalitesi (C2 R2; proxy-yanlış claim'ler)

| alternatif kümesi | referansta | aynı grup | ikame mümkünken referansta |
|---|---|---|---|
| A1 | 0.290 | 0.556 | 0.386 |
| A2 | 0.255 | 0.653 | 0.366 |
| A3 | 0.177 | 1.000 | 0.350 |

En sık A1 alternatifleri (proxy-yanlış): dress 41, uniform 40, suit 34, bikini 32, court 28, costume 27, couch 26, dog 23, street 22, bed 18

## 3. Nesne bölgede değil ama görüntüde mi? (ham OWLv2 olasılığı, Look v4 bölgesi)

| eşik | yanlış: görüntüde bulunur | yanlış: bölgede bulunur | doğru: bölgede yok, görüntüde var | doğru: hiçbir yerde |
|---|---|---|---|---|
| 0.1 | 0.825 | 0.556 | 0.161 | 0.065 |
| 0.3 | 0.422 | 0.279 | 0.130 | 0.254 |

Görüntünün hiçbir yerinde bulunmayan (<0.1) proxy-yanlış: 192; bunlarda ikame mümkün: 0.401.

'camera' claim'i: 42; 'at/to/from the camera' kalıbı: 34; kalıp ve proxy-yanlış: 21.

## 4. Claude görsel denetimi (proxy-yanlış claim'ler; küçük örnek, insan etiketi değil)

| örnek | n | doğru | yanlış | belirsiz | nesne iddiası değil |
|---|---|---|---|---|---|
| detector_confident_seed0 | 9 | 8 | 1 | 0 | 0 |
| detector_absent_seed0 | 3 | 1 | 1 | 1 | 0 |
| random_seed7 | 12 | 6 | 0 | 4 | 2 |

Görseller: `results/claude_audit_B_detector_confident.png`, `results/claude_audit_B_random.png`.
