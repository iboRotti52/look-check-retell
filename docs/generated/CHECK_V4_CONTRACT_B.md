# Check v4 → Retell: B tanısı

Üreten: `scripts/review_check_v4_contract.py`. Yalnız B; seçim yapılmadı.
Etiket: otomatik referans proxy'si. Referansta geçmek, doğru ikame olduğunu kanıtlamaz.

| Grup | n | supported | contradicted | uncertain | alternatif referansta |
|---|---:|---:|---:|---:|---:|
| all | 4594 | 2869 | 669 | 1056 | 23.2% |
| reference_unsupported | 1100 | 452 | 321 | 327 | 29.0% |
| contradicted | 669 | 0 | 669 | 0 | 31.8% |
| contradicted_reference_unsupported | 321 | 0 | 321 | 0 | 33.0% |
| stuff_reference_unsupported | 259 | 93 | 3 | 163 | 35.9% |

Alternatif adı değiştiği halde eski ham s_alt kalan kayıt: **2394**.

Mevcut `pipeline.finalise` ile karar geçişleri (yukarıdaki JSON'da kaydedilen eşiklerle):

| v4 kararı | finalise sonrası | n |
|---|---|---:|
| contradicted | contradicted | 292 |
| contradicted | supported | 135 |
| contradicted | uncertain | 242 |
| supported | contradicted | 70 |
| supported | supported | 2391 |
| supported | uncertain | 408 |
| uncertain | contradicted | 268 |
| uncertain | supported | 346 |
| uncertain | uncertain | 442 |

Bu sayılar mevcut tüketici yolunun tanısıdır; yeni bir sistem performansı değildir.
