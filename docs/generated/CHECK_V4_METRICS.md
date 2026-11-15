# Check v4 — otomatik tablo

Üreten: `scripts/36_check_v4_table.py`. Kaynak: results/check_v4_selection.json, results/check_v4.json.
Etiket otomatik referans proxy'si (insan etiketi değil); değerlendirme kişi dışı claim'ler.
Bileşen kodları: `docs/CHECK_V4_PROTOKOL.md` §1; eşik politikası §5 (değişiklik 1).

Seçilen konfigürasyon: **Q1 C2 A1 R2 G0 E0** (varsayılan Q1 C1 A1 R1 G0 E0; en iyi ÇD J 0.473).

## B, görüntüye göre 5 katlı ÇD (seçim burada; J'ye göre ilk 20 + varsayılan)

| Konfigürasyon | değişen | J | contr. kesinlik | yanlış yakalama | sup. kesinlik | supported oranı | contradicted oranı |
|---|---|---|---|---|---|---|---|
| **Q1 C2 A1 R2 G0 E0** | 2 | 0.473 | 0.480 | 0.292 | 0.842 | 0.625 | 0.146 |
| Q1 C1 A1 R2 G1 E0 | 2 | 0.452 | 0.546 | 0.113 | 0.887 | 0.532 | 0.049 |
| Q2 C1 A1 R2 G1 E0 | 3 | 0.451 | 0.555 | 0.111 | 0.886 | 0.536 | 0.048 |
| Q2 C2 A1 R2 G0 E0 | 3 | 0.450 | 0.467 | 0.260 | 0.844 | 0.617 | 0.133 |
| Q2 C2 A2 R2 G0 E0 | 4 | 0.450 | 0.466 | 0.262 | 0.842 | 0.634 | 0.135 |
| Q2 C1 A1 R1 G1 E0 | 2 | 0.443 | 0.489 | 0.175 | 0.876 | 0.513 | 0.086 |
| Q1 C2 A2 R2 G0 E0 | 3 | 0.442 | 0.469 | 0.267 | 0.842 | 0.603 | 0.136 |
| Q1 C1 A2 R2 G1 E0 | 3 | 0.442 | 0.480 | 0.098 | 0.903 | 0.483 | 0.049 |
| Q2 C1 A2 R2 G1 E0 | 4 | 0.439 | 0.510 | 0.092 | 0.899 | 0.492 | 0.043 |
| Q2 C1 A2 R2 G0 E0 | 3 | 0.434 | 0.449 | 0.148 | 0.874 | 0.549 | 0.079 |
| Q1 C1 A1 R1 G1 E0 | 1 | 0.434 | 0.488 | 0.170 | 0.874 | 0.512 | 0.083 |
| Q2 C2 A1 R2 G0 E1 | 4 | 0.431 | 0.472 | 0.242 | 0.848 | 0.571 | 0.123 |
| Q2 C1 A1 R1 G1 E1 | 3 | 0.430 | 0.494 | 0.149 | 0.885 | 0.482 | 0.072 |
| Q1 C1 A2 R2 G0 E0 | 2 | 0.430 | 0.458 | 0.149 | 0.870 | 0.562 | 0.078 |
| Q1 C2 A1 R2 G0 E1 | 3 | 0.430 | 0.473 | 0.250 | 0.845 | 0.578 | 0.126 |
| Q2 C2 A2 R2 G0 E1 | 5 | 0.429 | 0.465 | 0.230 | 0.846 | 0.601 | 0.118 |
| Q2 C1 A1 R2 G1 E1 | 4 | 0.429 | 0.500 | 0.081 | 0.904 | 0.475 | 0.039 |
| Q1 C2 A1 R1 G1 E0 | 2 | 0.426 | 0.499 | 0.165 | 0.862 | 0.564 | 0.079 |
| Q1 C1 A1 R1 G1 E1 | 2 | 0.425 | 0.504 | 0.156 | 0.882 | 0.478 | 0.074 |
| Q1 C1 A2 R2 G1 E1 | 4 | 0.424 | 0.497 | 0.080 | 0.899 | 0.487 | 0.039 |
| Q1 C1 A1 R1 G0 E0 | 0 | 0.341 | 0.308 | 0.004 | 0.895 | 0.459 | 0.003 |

### Tek bileşen etkisi (B ÇD; varsayılandan yalnız bir bileşen değişik)

| Değişiklik | J | contr. kesinlik | yanlış yakalama | sup. kesinlik | supported oranı | contradicted oranı |
|---|---|---|---|---|---|---|
| Q1 C1 A1 R1 G0 E1 | 0.327 | – | 0.000 | 0.897 | 0.436 | 0.000 |
| Q1 C1 A1 R1 G0 E2 | 0.302 | – | 0.000 | 0.905 | 0.381 | 0.000 |
| Q1 C1 A1 R1 G1 E0 | 0.434 | 0.488 | 0.170 | 0.874 | 0.512 | 0.083 |
| Q1 C1 A1 R2 G0 E0 | 0.389 | 0.410 | 0.045 | 0.889 | 0.515 | 0.027 |
| Q1 C1 A2 R1 G0 E0 | 0.331 | 0.370 | 0.031 | 0.874 | 0.510 | 0.020 |
| Q1 C1 A3 R1 G0 E0 | 0.360 | 0.407 | 0.030 | 0.887 | 0.493 | 0.018 |
| Q1 C2 A1 R1 G0 E0 | 0.312 | 0.350 | 0.006 | 0.862 | 0.557 | 0.004 |
| Q2 C1 A1 R1 G0 E0 | 0.348 | 0.308 | 0.004 | 0.893 | 0.476 | 0.003 |

## Validation (bir kez, eşikler tüm B'den)

Kalibre bölge skorunun AUROC'u (sürekli ölçek, seçilen konfigürasyon): 0.715

| Sistem | hedef | grup | J | contr. kesinlik | yanlış yakalama | sup. kesinlik | supported oranı | contradicted oranı |
|---|---|---|---|---|---|---|---|---|
| original_live | 0.5 | all | 0.481 | 0.440 | 0.219 | 0.861 | 0.655 | 0.116 |
| original_live | 0.5 | thing | 0.678 | 0.483 | 0.242 | 0.925 | 0.731 | 0.084 |
| original_live | 0.5 | clothing | 0.260 | 0.516 | 0.239 | 0.667 | 0.498 | 0.197 |
| original_live | 0.5 | stuff | 0.289 | 0.294 | 0.160 | 0.854 | 0.649 | 0.110 |
| original_tuned | 0.5 | all | 0.410 | – | 0.000 | 0.919 | 0.478 | 0.000 |
| original_tuned | 0.5 | thing | 0.584 | – | 0.000 | 0.967 | 0.602 | 0.000 |
| original_tuned | 0.5 | clothing | 0.171 | – | 0.000 | 0.764 | 0.222 | 0.000 |
| original_tuned | 0.5 | stuff | 0.226 | – | 0.000 | 0.876 | 0.468 | 0.000 |
| original_tuned | 0.6 | all | 0.410 | – | 0.000 | 0.919 | 0.478 | 0.000 |
| original_tuned | 0.6 | thing | 0.584 | – | 0.000 | 0.967 | 0.602 | 0.000 |
| original_tuned | 0.6 | clothing | 0.171 | – | 0.000 | 0.764 | 0.222 | 0.000 |
| original_tuned | 0.6 | stuff | 0.226 | – | 0.000 | 0.876 | 0.468 | 0.000 |
| original_tuned | 0.7 | all | 0.410 | – | 0.000 | 0.919 | 0.478 | 0.000 |
| original_tuned | 0.7 | thing | 0.584 | – | 0.000 | 0.967 | 0.602 | 0.000 |
| original_tuned | 0.7 | clothing | 0.171 | – | 0.000 | 0.764 | 0.222 | 0.000 |
| original_tuned | 0.7 | stuff | 0.226 | – | 0.000 | 0.876 | 0.468 | 0.000 |
| v4 | 0.5 | all | 0.553 | 0.523 | 0.287 | 0.859 | 0.665 | 0.128 |
| v4 | 0.5 | thing | 0.702 | 0.551 | 0.358 | 0.899 | 0.835 | 0.109 |
| v4 | 0.5 | clothing | 0.268 | 0.504 | 0.420 | 0.678 | 0.363 | 0.354 |
| v4 | 0.5 | stuff | 0.199 | – | 0.000 | 0.850 | 0.616 | 0.000 |
| v4 | 0.6 | all | 0.358 | – | 0.000 | 0.873 | 0.598 | 0.000 |
| v4 | 0.6 | thing | 0.388 | – | 0.000 | 0.904 | 0.756 | 0.000 |
| v4 | 0.6 | clothing | 0.151 | – | 0.000 | 0.740 | 0.225 | 0.000 |
| v4 | 0.6 | stuff | 0.199 | – | 0.000 | 0.850 | 0.616 | 0.000 |
| v4 | 0.7 | all | 0.358 | – | 0.000 | 0.873 | 0.598 | 0.000 |
| v4 | 0.7 | thing | 0.388 | – | 0.000 | 0.904 | 0.756 | 0.000 |
| v4 | 0.7 | clothing | 0.151 | – | 0.000 | 0.740 | 0.225 | 0.000 |
| v4 | 0.7 | stuff | 0.199 | – | 0.000 | 0.850 | 0.616 | 0.000 |
| ref_check_v2 | 0.5 | all | 0.959 | 0.506 | 0.682 | 0.945 | 0.482 | 0.315 |
| ref_check_v2 | 0.5 | thing | 1.274 | 0.544 | 0.767 | 0.973 | 0.630 | 0.237 |
| ref_check_v2 | 0.5 | clothing | 0.368 | 0.514 | 0.826 | 0.926 | 0.083 | 0.683 |
| ref_check_v2 | 0.5 | stuff | 0.550 | 0.410 | 0.362 | 0.895 | 0.532 | 0.179 |
| ref_check_v3 | 0.5 | all | 1.131 | 0.484 | 0.841 | 0.944 | 0.569 | 0.407 |
| ref_check_v3 | 0.5 | thing | 1.370 | 0.500 | 0.867 | 0.968 | 0.697 | 0.291 |
| ref_check_v3 | 0.5 | clothing | 0.618 | 0.529 | 0.913 | 0.915 | 0.218 | 0.732 |
| ref_check_v3 | 0.5 | stuff | 0.858 | 0.398 | 0.702 | 0.909 | 0.616 | 0.358 |

| Fark (hedef 0.5, hepsi) | metrik | fark | GA95 (görüntü bootstrap, 2.000) |
|---|---|---|---|
| v4_minus_original_live | contradicted_precision | 0.083 | [0.007, 0.157] |
| v4_minus_original_live | recall_wrong | 0.068 | [0.015, 0.120] |
| v4_minus_original_live | J | 0.072 | [-0.021, 0.159] |
| v4_minus_original_tuned | contradicted_precision | – | [–, –] |
| v4_minus_original_tuned | recall_wrong | 0.287 | [0.243, 0.332] |
| v4_minus_original_tuned | J | 0.143 | [0.059, 0.229] |

Eşikler (τs, τc, τalt; C2'de z ölçeği, C1'de olasılık): v4@0.5: thing (0.167, 0.058, 0.167), stuff (0.697, -inf, 0.697); v4@0.6: thing (0.525, -inf, 0.525), stuff (0.697, -inf, 0.697); v4@0.7: thing (0.525, -inf, 0.525), stuff (0.697, -inf, 0.697); original_tuned@0.5: thing (0.397, -inf, 0.397), stuff (0.380, -inf, 0.380); original_tuned@0.6: thing (0.397, -inf, 0.397), stuff (0.380, -inf, 0.380); original_tuned@0.7: thing (0.397, -inf, 0.397), stuff (0.380, -inf, 0.380)
