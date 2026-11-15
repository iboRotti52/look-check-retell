# Check S4: sonuç (2026-11-15, Claude) — yeni varsayılan Check

- **Protokol** (koşudan önce): `docs/CHECK_S4_PROTOKOL.md`. Kaynak: Fable'ın ikinci incelemesi
  (`docs/FABLE_INCELEME_2.md`, I1 + I2 + I3 + I5).
- **Sayılar:** `docs/generated/CHECK_S4_B.md` (`scripts/45_check_s4.py`, `results/check_s4_B.json`).
- **Kapsam:** yalnız B, 5 görüntü katı, dış-kat. Validation ve test okunmadı.
- **Uyarı:** bileşenler Fable tarafından aynı B katlarında keşfedildi. Bu koşu, proje kodundaki uygulamayı ve önceden
  konmuş ölçütleri doğruluyor; bağımsız kanıt değil. Bağımsız doğrulama validation'da, bir kez.
- **Etiketler:** FCR ve swap otomatik proxy / sentetik test. İnsan etiketi yok.

## Ne değişti
- **Kod:** `lcr/region_evidence.py` içinde `decision="rival_np"`. Yeni parçalar: `rival_matrix`, `fit_rival_np`,
  `substitute_centred`, `CONFUSION_CLEAN`, `set_matrix`. `lcr/claims.py` içinde `non_visual(..., compounds=True)`.
  `lcr/pipeline.py` bu bayrağı bundle'dan okuyor.
- **Mantık aynı:** yer seçimi Look'ta, karar donmuş OWLv2'de.
  - Claim'in kanıtı, aynı kutuda detektörün tercih ettiği dışlayan bir ad varsa düşürülüyor (rakip-farkında kanıt).
  - Her sınıfın eşiği kendi skor dağılımından, tek bir λ ile seçiliyor (maliyete duyarlı eşik).
  - İkame Look'un ilk kutusundan, merkezlenmiş skorla okunuyor.
- **Varsayılan bundle:** `results/lcr_check4_bundle.json` (`c7156af83dcca3b4`, λ = 2.5). Kayıtlar
  `data/cache/pipeline/A_B_lcr4.json`. Önceki bundle'lar aynen çalışıyor: S2 `lcr_check_bundle.json`, S3
  `lcr_check3_bundle.json`, Codex'in `lcr_bundle.json`.
- **Doğrulama:** `39_verify_lcr.py --live`. İki kod yolu (vektörel uydurma ve `RegionChecker.record`) her claim'de
  aynı eylemi verdi. Tüm testler geçti (`tests/test_check_fix.py` S4 testleri dahil).

## Sonuç (B, dış-kat)
| sistem | thing-FCR | giysi FCR | temiz dışlayan yakalama | aynı-küme giysi | ikame kesinliği |
|---|---|---|---|---|---|
| S2 (eski varsayılan) | 0.117 | 0.320 | 0.552 | 0.626 | 0.689 |
| S3 (ön kayıtlı) | 0.038 | 0.046 | 0.242 | 0.276 | 0.852 |
| **S4 (λ = 2.5)** | **0.030** | **0.041** | **0.808** | 0.484 | **0.912** |

- **Ölçütler:** üçü de tuttu (temiz yakalama ≥ S2, giysi FCR ≤ 0.05, ikame kesinliği ≥ S2) → **S4 varsayılan.**
- **Ablation'lar:**
  - Rakip-farkında kanıt olmadan (S4a): aynı FCR'de temiz yakalama 0.629 (S4: 0.808), giysi FCR 0.055.
  - S3'ün ikame sıralamasıyla: ikame kesinliği 0.859 (merkezli: 0.912).
  - Yani iki bileşen de kendi payını getiriyor.
- **Bedeli:**
  - stuff FCR 0.054 (S2 stuff'ı çoğunlukla skip ettiği için 0.016).
  - Aynı-küme giyside yakalama 0.484. S2'nin 0.626'sı, doğru giysi claim'lerinin %32'sini bozma pahasına geliyor.
- **Proxy contradicted kesinliği:** 0.524 (S2 0.491). Betimsel; proxy-yanlışların çoğu aslında doğru.

## Hâlâ açık
- **Gerçek hata oranı:** insan etiketi olmadan doğru FCR hedefi bilinmiyor.
- **L1:** Look'un yanlış kelimede ölçümü. Swap testinde bölge doğru kelimeden geliyor; bu, ikame sonucunu Look lehine
  iyimser yapıyor.
- **Giysi türü:** detektörle sınırlı. İsteğe bağlı yol: OWLv2-large (günlükte not edildi).
- **Retell (kapalı):** `remove` → kelimeyi yasakla; `replace` (kesinlik ~0.91) → ikameyi öner;
  `keep_elsewhere` → dokunma.
