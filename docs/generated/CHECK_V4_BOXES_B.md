# Check v4: üç kutunun ayrı tanısı (yalnız B)

Üreten: `scripts/review_check_v4_boxes_B.py`. Yeni seçim veya validation/test erişimi yok.
Referans desteği otomatik proxy'dir. Kutu kararları ortalama için fit edilen OOF eşiklerini kullanır;
yeni bir kutu-başına yöntemin kalibrasyonu veya performans iddiası değildir.

| Tanı | n |
|---|---:|
| claims_with_region | 4594 |
| boxes_give_both_support_and_contradiction | 2221 |
| weighted_not_supported_but_one_box_above_support | 1014 |
| same_reference_supported | 654 |
| same_reference_unsupported | 360 |

Referansta desteklenen fakat v4'ün desteklemediği frizbi iddiaları:

| B görüntüsü | v4 | ilk 3 kutuda ham frisbee skorları | görüntüde en yüksek | boyut filtresini geçer mi | ilk 3'te mi |
|---|---|---|---:|---|---|
| train_2921 | contradicted | 0.008, 0.002, 0.004 | 0.965 | hayır | hayır |
| train_3467 | contradicted | 0.002, 0.003, 0.000 | 0.696 | hayır | hayır |
| train_3693 | contradicted | 0.007, 0.007, 0.011 | 0.893 | hayır | hayır |
| train_3719 | contradicted | 0.010, 0.009, 0.012 | 0.884 | hayır | hayır |
| train_3875 | uncertain | 0.007, 0.009, 0.001 | 0.928 | hayır | hayır |
| train_4066 | contradicted | 0.004, 0.001, 0.001 | 0.541 | hayır | hayır |
| train_5206 | contradicted | 0.011, 0.001, 0.002 | 0.894 | hayır | hayır |
| train_5240 | uncertain | 0.002, 0.003, 0.002 | 0.933 | hayır | hayır |

İlk dört örnek görsel olarak denetlendi. Yeşil kutu frizbiye oturuyor;
diğer dört örnekte bu tablo yalnız skor ve filtre tanısıdır.
