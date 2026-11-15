# Check düzeltmesi — B (otomatik tablo)

Üreten: `scripts/43_check_fix.py`. Protokol: `docs/CHECK_DUZELTME_PROTOKOL.md`. Yalnız B, dış-kat (5 görüntü katı). FCR: referansta doğrulanmış claim'lerin contradicted olma oranı (temiz pozitif = referansta anılma, otomatik proxy). Swap: sentetik yanlış etiket.

Bundle `6f68b7dd7ed0892e`; kişi dışı claim 4594; swap 2401.

| sistem | FCR | FCR giysi | FCR diğer thing | FCR stuff | swap yakalama | yüksek-taban yakalama | giysi yakalama | gerçek sınıfla replace | ikame kesinliği | proxy J | proxy contr. kesinlik |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 beş-yollu z | 0.085 | 0.320 | 0.043 | 0.016 | 0.559 | 0.587 | 0.591 | 0.286 | 0.689 | 0.464 | 0.491 |
| S3 α=0.02 | 0.010 | 0.011 | 0.007 | 0.014 | 0.140 | 0.110 | 0.126 | 0.057 | 0.868 | 0.096 | 0.514 |
| S3 α=0.03 | 0.015 | 0.017 | 0.012 | 0.018 | 0.165 | 0.134 | 0.150 | 0.066 | 0.850 | 0.117 | 0.469 |
| S3 α=0.05 | 0.026 | 0.028 | 0.022 | 0.033 | 0.209 | 0.175 | 0.205 | 0.081 | 0.852 | 0.130 | 0.387 |
| S3 α=0.075 | 0.041 | 0.046 | 0.035 | 0.046 | 0.261 | 0.231 | 0.259 | 0.096 | 0.852 | 0.157 | 0.382 |
| S3 α=0.1 | 0.057 | 0.060 | 0.049 | 0.070 | 0.305 | 0.271 | 0.304 | 0.109 | 0.864 | 0.187 | 0.375 |
| S3 α=0.15 | 0.083 | 0.085 | 0.073 | 0.099 | 0.388 | 0.367 | 0.415 | 0.130 | 0.889 | 0.263 | 0.389 |
| S3 α=0.2 | 0.120 | 0.123 | 0.113 | 0.127 | 0.452 | 0.463 | 0.508 | 0.138 | 0.904 | 0.309 | 0.372 |
| S3 α=0.25 | 0.156 | 0.153 | 0.151 | 0.165 | 0.499 | 0.536 | 0.574 | 0.145 | 0.908 | 0.340 | 0.361 |

- Seçilen α* = 0.075 (FCR ≤ 0.05 hedefi tutuldu: True).
- Eşleşmiş FCR (S2'ninkini aşmayan en büyük α = 0.15): S3 yakalama 0.388 vs S2 0.559 → ölçüt (a) False.
- Ölçüt (b) α*'da FCR S2'den düşük: True. **Varsayılan olur: False.**
- Son bundle çekimser sınıflar (21): airplane, bone, camel, camera, cat, cone, elephant, fish, goat, kite, ladder, net, newspaper, raft, scooter, sheep, stroller, suit, towel, trampoline, unicycle.
- α*'da replace ikamesi referansta: 0.231; remove'da referansta aynı karışma kümesinden sınıf yok: 0.916.

## Eylemler (dış-kat)

| sistem | keep | keep_elsewhere | replace | remove | skip |
|---|---|---|---|---|---|
| S2 | 2872 | 164 | 324 | 257 | 977 |
| S3 α*=0.075 | 4066 | 253 | 26 | 202 | 47 |

## Görsel denetlenen claim'ler (Claude gözlemi, insan etiketi değil)

| claim | Claude | proxy | S2 | S2 ikame | S3 | S3 ikame |
|---|---|---|---|---|---|---|
| train_4594:shorts | correct | wrong | keep |  | keep |  |
| train_3690:ball | correct | wrong | keep |  | keep |  |
| train_2949:surfboard | wrong | wrong | keep |  | keep |  |
| train_1587:shorts | correct | wrong | keep |  | keep |  |
| train_0928:pole | correct | wrong | keep |  | keep |  |
| train_1391:field | correct | wrong | keep |  | keep |  |
| train_0429:newspaper | correct | wrong | keep |  | keep |  |
| train_0165:house | correct | wrong | keep |  | keep |  |
| train_0107:bat | correct | wrong | keep |  | keep |  |
| train_5707:mountain | wrong | wrong | skip |  | keep |  |
| train_3334:sweater | correct | wrong | replace | suit | keep_elsewhere |  |
| train_3254:street | unclear | wrong | skip |  | remove |  |
| train_0295:wall | correct | wrong | replace | stairs | keep_elsewhere |  |
| train_1792:camera | non_visual | wrong | skip |  | skip |  |
| train_5021:book | unclear | wrong | keep_elsewhere |  | keep_elsewhere |  |
| train_5416:dress | unclear | wrong | replace | scarf | keep |  |
| train_4171:sky | correct | wrong | keep |  | keep |  |
| train_5299:mud | correct | wrong | skip |  | keep |  |
| train_4690:vest | correct | wrong | keep |  | keep |  |
| train_3803:dress | unclear | wrong | skip |  | keep |  |
| train_1281:shorts | correct | wrong | keep |  | keep |  |
| train_3561:vest | unclear | wrong | replace | scarf | keep |  |
| train_5639:shirt | correct | wrong | replace | uniform | keep |  |
| train_1714:camera | non_visual | wrong | skip |  | skip |  |
