# Check v2 — otomatik inceleme ölçümleri

Üreten: `scripts/review_check_v2.py`; kaynak: `results/check_v2_review.json`.

Validation; yalnızca thing/stuff; otomatik referans proxy'si. Eşleştirilmiş görüntü bootstrap'ı, %95 yüzdelik aralık. Modeller ve eşikler sabit; eğitim/seçim belirsizliği ve çoklu karşılaştırma düzeltmesi dahil değil.

| Ölçüm | Tahmin | %95 GA |
|---|---:|---:|
| auc:prior | 0.7632 | [0.7368, 0.7899] |
| auc:prior+det | 0.8104 | [0.7881, 0.8321] |
| auc:prior+det+cap | 0.8286 | [0.8065, 0.8496] |
| delta_auc:prior+det-prior | 0.0472 | [0.0312, 0.0642] |
| delta_auc:prior+det+cap-prior+det | 0.0182 | [0.0075, 0.0300] |
| delta_auc:prior+det+look-prior+det | -0.0004 | [-0.0023, 0.0017] |
| delta_auc:prior+region-prior+detector_region | 0.0039 | [-0.0041, 0.0123] |
| delta_contradicted_precision:v2-round1 | 0.1040 | [0.0579, 0.1494] |
| delta_recall_wrong:v2-round1 | 0.2188 | [0.1500, 0.2853] |
| delta_supported_precision:v2-round1 | 0.1002 | [0.0773, 0.1231] |

## B'de iç içe seçim tanısı

Sabit özellik ailesi. Lambda ve eşikler dışarıda tutulan görüntü katını görmez; önceki özellik ailesi seçiminin geçmişini ortadan kaldırmaz.

| Ölçüm | Değer |
|---|---:|
| auroc | 0.824322 |
| log_loss | 0.416294 |
| ece | 0.012052 |
| contradicted_precision | 0.498356 |
| recall_wrong | 0.689091 |
| supported_precision | 0.950048 |

## Doğrulamalar

- Olasılık yeniden üretimi, en büyük mutlak fark: 6.71e-13.
- B: 6856 claim, 0 konum/kelime uyumsuzluğu, 352 çok-parçalı claim.
- validation: 2272 claim, 0 konum/kelime uyumsuzluğu, 100 çok-parçalı claim.
- Gradyan sonlu fark kontrolünde en büyük hata: 6.34e-08.

## Alternatif önerisinin referans proxy'si

| Küme | Claim | Öneri var | Öneri referansta | Oran |
|---|---:|---:|---:|---:|
| all_reference_unsupported | 352 | 352 | 113 | 32.102% |
| contradicted_reference_unsupported | 238 | 238 | 75 | 31.513% |

## Eşit skorları gruplayan AP (yanlış), validation

| Özellikler | AP |
|---|---:|
| prior | 0.426421 |
| det (no class prior) | 0.444359 |
| cap (no class prior) | 0.385715 |
| prior+region | 0.494385 |
| prior+image | 0.473300 |
| prior+det | 0.496352 |
| prior+cap | 0.520200 |
| prior+det+look | 0.495057 |
| prior+det+cap | 0.554924 |
| det+cap (no class prior) | 0.479073 |
| all | 0.551658 |
