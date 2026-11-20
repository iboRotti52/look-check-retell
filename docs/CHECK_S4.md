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

## Düzeltme (2026-11-19): gösterilen kanıt kutusu kararın kutusu

Astra (GPT-6.2, İbrahim'in incelemesi) B kayıtlarında iki gösterim hatası buldu; sayılar onun B sayımıdır, burada
yeniden üretilmedi. **Hiçbir karar değişmez**: verdict/action/substitute, τ, λ ve bundle (`c7156af83dcca3b4`) aynı.
- **keep kutusu.** keep kararı rakip-farkında skorla (v_c) veriliyor, ama kanıt olarak gösterilen kutu ham claim
  logit'iyle seçiliyordu. 4.024 keep'in 29'unda gösterilen kutunun v_c'si τ_c altındaydı (ör. train_0505 slide:
  τ −3,172, gösterilen kutu −4,305, kararı veren kutu −2,644). Artık kutu v_c ile seçiliyor; s_claim tam o kutunun
  v_c'si.
- **replace skoru.** İkame Look top-1 kutusunda okunuyor ve o kutu gösteriliyor, ama `s_alt` aday kutular üzerinden
  birleştirilmiş skoru taşıyordu (52 ikamenin 6'sında başka bir kutuya ait). Artık `s_alt` gösterilen kutunun
  skoru; birleştirilmiş değer `s_alt_pooled` alanında. `s_substitute_best` değişmedi. Retell kanıtı aday başına
  `regions` jetonlarını kullandığı için Retell girdisi değişmez.

Kod: `lcr/region_evidence.py` (`RegionChecker.record`). Testler `tests/test_check_fix.py` (eski kodla ikisi de
başarısız olur). `scripts/39_verify_lcr.py` artık S4 kayıtlarında bu sözleşmeyi denetler (keep: kutunun v_c'si
= s_claim ≥ τ_c; replace: kutu top-1, s_alt o kutunun skoru) — **eski `A_B_lcr4.json` bu denetimden geçmez.**
Kayıtları yenilemek için veri sahibi: `python scripts/48_rewrite_lcr4_records.py` (kuru koşu, raporu
`results/lcr4_witness_rewrite.json`), sonra `--overwrite` (eski dosya `data/cache/pipeline/archive/pre_witness_fix/`'e),
sonra `python scripts/39_verify_lcr.py --live`. Script 48 yeniden kalibre etmez ve tek bir karar alanı değişirse
yazmayı reddeder.

Ayrıca `scripts/42_check_complete.py` (S2) kendi dosyası yerine varsayılan `re.BUNDLE_PATH`'e (S4'e) bakıyor ve
oraya yazıyordu; artık `FIVE_WAY_BUNDLE_PATH`. Mevcut ayarları etkilemiyordu, deneyleri yeniden kurmayı bozuyordu.

## Bilinen kalibrasyon hatası (2026-11-19, düzeltilmedi — karar Check sahibinin)

Canlı Check "at the camera" kalıbını ve birleşik kelimeleri ("hot dog", "home plate", "toy car", "dirt bike")
`claims.non_visual` ile atlıyor; ama `scripts/45_check_s4.py` kalibrasyonda referans anmalarını
`claims.reference_classes` ile sayıyor, yani "a man looks at the camera" diyen bir referans kamera için pozitif sayılıyor
(kodda doğrulandı, satır 129–134). Astra aynı filtreyi referanslara uygulayınca kamera eşiğinin −6,08'den −1,79'a
çıktığını, B'deki 4.594 claim'in yalnız 2'sinde contradicted kararının değiştiğini buldu (onun sayımı; burada yeniden
üretilmedi). Düzeltme τ'yu değiştirir, yani yeni bir kalibrasyon, yeni bundle kimliği ve yeni `check_s4_B.json`
metrikleri demektir; dondurulmuş S4 bu yüzden olduğu gibi bırakıldı. Düzeltilirse: anma, referans cümlesi
`claims.non_visual(ref, c, compounds=True)` değilse sayılmalı (hem `refs` hem `cnt`).

Ayrıca rapor (`report/sections/3_design.tex`, Check bölümü) hâlâ ilk birleşik Check'in z ölçeği ve üç yollu eşik
kuralını anlatıyor; S4'ün (rakip-farkında kanıt, maliyete duyarlı τ_c, top-1 ikamesi) metne geçmesi bekliyor.

### data-v1 kullananlar için tek seferlik geçiş

`data/cache/pipeline/A_B_lcr4.json` git'te değil; GitHub release `data-v1` (`scripts/fetch_data.sh`) eski kayıtları
içeriyor. Bu dalı birleştirdikten sonra `data-v1` indiren herkes bir kez şunu koşmalı:
`python scripts/48_rewrite_lcr4_records.py --overwrite` ve `python scripts/39_verify_lcr.py`. Unutulursa 39'un yeni
kanıt sözleşmesi eski dosyayı yakalar. Kalıcı çözüm: düzeltilmiş önbellekle yeni bir sürümlü release (`data-v2`) ve
`fetch_data.sh`'de ona geçmek (ekip kararı). Yeniden yazmayı yapan kişi `results/lcr4_witness_rewrite.json` ve
`results/lcr4_contract.json`'u bu PR'a eklemeli; değişen kayıtların ve doğrulamanın kanıtı onlar.
