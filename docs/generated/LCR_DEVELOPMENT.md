# Birleşik LCR — yalnız B geliştirme sonuçları

Üreten: `scripts/37_unify_lcr.py`.
Validation/test okunmadı. Etiket otomatik referans proxy'si. B sonuçları yeni bağımsız başarı iddiası değildir.

## Look (B_fit insan kutuları; kişi dışı)

| min patch | alpha | n | ilk kutu | ilk üçten biri | küçük GT n | küçük GT ilk üç |
|---|---|---:|---:|---:|---:|---:|
| 2 | 0.5 | 2470 | 0.805 | 0.900 | 162 | 0.395 |
| 2 | 1.0 | 2470 | 0.528 | 0.776 | 162 | 0.358 |
| 0 | 0.5 | 2470 | 0.830 | 0.933 | 162 | 0.833 |
| 0 | 1.0 | 2470 | 0.389 | 0.645 | 162 | 0.716 |

## Check (B_fit görüntü ÇD; seçilmiş Look koşullu)

| birleştirme | alternatif | J | contr. kesinlik | yanlış yakalama |
|---|---|---:|---:|---:|
| weighted | all | 0.497 | 0.497 | 0.275 |
| weighted | group | 0.425 | 0.481 | 0.174 |
| max | all | 0.492 | 0.492 | 0.302 |
| max | group | 0.433 | 0.495 | 0.216 |

Seçim: Look `{'min_patches': 0, 'alpha': 0.5}`; Check `max/all`.
Bundle: `4c0481274067b130`. B_fit görüntü: 2365; B_cal: 635.
Son ölçek ve eşikler yalnız B_cal'dan. Retell B_fit üzerinde eğitilir; sabit epoch kullanır.

## Alternatif proxy'si

Yalnız referans desteklemeyen ve contradicted olan claim'lerde alternatif referansta mı?

| bölüm | n | referansta alternatif | oran |
|---|---:|---:|---:|
| fit | 295 | 90 | 0.3050847457627119 |
| cal | 89 | 29 | 0.3258426966292135 |

B_cal eşik fit verisidir, bağımsız test değildir. İnsan kutuları yalnız Look seçim/ölçümünde kullanıldı.
