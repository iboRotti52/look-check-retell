# Look E3 — otomatik tablo

Üreten: `scripts/30_look_e3_table.py`. Kaynak: results/look_e3*.json.

Başlıklar (iki eksen de 1-tabanlı): L9H8, L10H8, L10H9, L9H3.

Aile/tür GA95 aralıkları tekildir; eşzamanlı güvence vermez. E3 uyarlama sınırları ve canlı v4: `docs/LOOK_E3_INCELEME.md`.

B v3 = 75.59, B yeniden hesaplanan S3 = 74.16; validation v3 = 76.10 (pointing 88.30).

| Aile | seçilen | B ÇD kazancı | val bölge | val pointing | person | thing | stuff | val GA95 vs v3 |
|---|---|---|---|---|---|---|---|---|
| A1_G2_layers | layer10 | +0.22 | 76.32 | 85.96 | 70.43 | 81.66 | 77.90 | [-1.13, +1.59] |
| A1_G3_labelfree_heads | top4 | +2.64 | 79.22 | 91.31 | 73.72 | 85.09 | 78.26 | [+1.66, +4.52] |
| A1_G4_supervised_heads | top4 | +1.10 | 78.38 | 89.42 | 71.48 | 85.36 | 78.26 | [+0.89, +3.66] |
| A0_sink_all_positions | S3_all | -1.06 | 75.26 | 87.24 | 67.41 | 81.79 | 78.99 | [-1.50, -0.11] |
| A2_phrase | A2_phrase | -0.24 | 76.32 | 88.75 | 67.81 | 83.38 | 80.43 | [-0.79, +1.20] |
| A3_ratio | A3_ratio | -16.53 | 57.27 | 88.86 | 44.81 | 66.36 | 66.67 | [-21.09, -16.63] |
| A4_chefer | A4_chefer | -2.40 | 75.04 | 92.26 | 67.67 | 82.32 | 75.36 | [-2.66, +0.50] |
| A5_salience_dropout | A5_q5 | -3.40 | 73.15 | 80.89 | 65.97 | 80.61 | 72.46 | [-4.42, -1.47] |
| A6_ensemble | S3+S4_contrast:prob | +0.35 | 76.55 | 89.47 | 68.07 | 84.83 | 77.17 | [-0.56, +1.44] |

## B aşaması (SAM)

| Tür | B taban | B ÇD | kazanç | kural |
|---|---|---|---|---|
| person | 75.27 | 79.99 | +4.72 | B2_a0.25 |
| thing | 81.15 | 74.16 | -6.98 | base |
| stuff | 78.01 | 77.02 | -0.99 | base |

## Look v4, validation

| | hepsi | person | thing | stuff |
|---|---|---|---|---|
| v3 | 76.10 | 67.15 | 83.77 | 79.71 |
| A sonrası | 79.22 | 73.72 | 85.09 | 78.26 |
| v4 | 80.39 | 76.48 | 85.09 | 78.26 |
| SAM üst sınır | 83.34 | 82.79 | 85.49 | 78.99 |

| v4 − v3 | fark (puan) | GA95 |
|---|---|---|
| all | +4.29 | [+2.46, +6.09] |
| person | +9.33 | [+5.32, +13.10] |
| thing | +1.32 | [-0.14, +2.91] |
| stuff | -1.45 | [-4.74, +1.73] |
