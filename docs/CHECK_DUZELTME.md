# Check düzeltmesi: sonuç ve yorum (2026-11-15, Claude)

- **Protokol** (koşudan önce): `docs/CHECK_DUZELTME_PROTOKOL.md`. Kaynak: Fable incelemesi C1a+C2+C3+C4+C7.
- **Sayılar:** `docs/generated/CHECK_FIX_B.md` (`scripts/43_check_fix.py`, `results/check_fix_B.json`).
  Sonradan ablation: `docs/generated/CHECK_FIX_ABLATION_B.md` (`scripts/44_check_fix_ablation.py`).
- **Kapsam:** yalnız B, 5 görüntü katı, dış-kat. Validation/test okunmadı.
- **Etiketler:** FCR = referansta doğrulanmış claim'lerin contradicted olma oranı (temiz pozitif, otomatik proxy).
  Swap = sentetik yanlış etiket (Fable'ın kurgusu, 2401 claim). İnsan etiketi yok.

## Ne yapıldı
- `lcr/region_evidence.py`: yeni karar sürümü `decision="positive_anchored"` (S3). Eski bundle'lar aynen çalışır.
  - Ham OWLv2 logit; sınıf eşiği τ_c = temiz pozitiflerin α-kantili (nadir sınıfta türe doğru büzülür).
  - keep / keep_elsewhere / (çekimser sınıf → skip) / contradicted = görüntünün hiçbir yerinde yok.
  - replace yalnız birbirini dışlayan **karışma kümesinde** (`CONFUSION`), Look top-1 kutusunda, ikame kendi
    τ'sunu ve claim'i geçerse. Boş `sa_any` koşulu kaldırıldı. Ham ölçekte alternatif kendi τ'sunu geçmeli
    (`evidence.py` bunu artık claim'in τ'su ile değil kayıttaki kapıyla okuyor).
- Bundle `results/lcr_check3_bundle.json` (`6f68b7dd7ed0892e`, α* = 0.075, 21 çekimser sınıf), kayıtlar
  `data/cache/pipeline/A_B_lcr3.json`. `39_verify_lcr.py --mode check3` geçti; iki kod yolu (vektörel uydurma ve
  `RegionChecker.record`) her claim'de aynı eylemi verdi. Testler: `tests/test_check_fix.py`; tüm takım 94 geçti.

## Sonuç: ön kayıtlı ölçüt tutmadı → varsayılan değişmedi
| sistem | FCR | FCR giysi | FCR diğer thing | swap yakalama |
|---|---|---|---|---|
| S2 (varsayılan, beş-yollu z) | 0.085 | 0.320 | 0.043 | 0.559 |
| S3 α*=0.075 (ön kayıtlı) | 0.041 | 0.046 | 0.035 | 0.261 |
| S3 α=0.15 (S2'nin FCR'sine eşleşmiş) | 0.083 | 0.085 | 0.073 | 0.388 |

- Ölçüt (b) tuttu: α*'da FCR yarıya indi, giysi FCR'si 0.32 → 0.05.
- Ölçüt (a) tutmadı: eşleşmiş FCR'de S3 swap'ların 0.388'ini, S2 0.559'unu yakalıyor. Protokol gereği
  **varsayılan S2 kaldı** (`lcr_check_bundle.json`); S3 ablation olarak duruyor.

## Neden (sonradan tanı, seçim için kullanılmadı)
1. **Suçlu çekimserlik kuralı (C4), özellikle "az pozitif" kısmı.** Swap etiketleri çoğunlukla nadir sınıflara
   düşüyor (captioner alternatifleri, aynı gruptan rastgele sınıf); n < 10 pozitifli sınıflar çekimser olduğu için
   bunlar skip ediliyor (α*'da swap'ların %16'sı). AUROC kuralını kaldırmak neredeyse hiçbir şey değiştirmiyor;
   çekimserliği tamamen kaldırmak Fable'ın ön ölçümünü yeniden üretiyor (α=0.10: FCR 0.059, yakalama 0.481).
2. **Çekimserlik olmadan, eşleşmiş FCR'de** (α=0.15, FCR 0.086): yakalama 0.585 (S2 0.559), ikame kesinliği
   0.88 (S2 0.69), gerçek sınıfla replace 0.265 (S2 0.286). Yani toplamda S2'ye eşit ya da biraz iyi.
3. **Ama kazanç gruplara eşit dağılmıyor:**
   - *Giysi:* asıl iyileşme burada. S2 giysi swap'larının 0.59'unu, doğru giysi claim'lerinin 0.32'sini
     çelişkili sayarak yakalıyor (ayırt etme yok denecek kadar az). S3 aynı yakalamayı (0.60, α=0.25) yarı FCR ile
     (0.16) veriyor.
   - *Giysi dışı thing:* z ölçeği kötü değil. S2: FCR 0.043, yakalama 0.547; S3 FCR 0.050'de 0.541. Fable'ın "z,
     sınıf önselinden iyi değil" bulgusu havuzlanmış AUROC için geçerli; giysi dışında çalışma noktasında fark yok.
   - *Yüksek-taban swap'lar* (yanlış etiket daha yaygın bir sınıf): S3 0.39, S2 0.59. Yaygın sınıfların pozitif
     dağılımının alt kuyruğu uzun (referansta anılıp detektörün zayıf gördüğü durumlar), bu yüzden düşük kantil
     eşiği yaygın bir yanlış etiketi geçiriyor. z ölçeği yaygın sınıfta katı olduğu için onları yakalıyor, bedelini
     de doğru claim'leri çelişkili sayarak ödüyor.
4. **Proxy contradicted kesinliği** S3'te 0.39, S2'de 0.49. Ama proxy-yanlışların çoğunun aslında doğru olduğu
   (görsel denetim) düşünülürse bu sayı iki sistemi ayırmak için güvenilir değil.

## Yorum
- Fable'ın ana teşhisi doğrulandı: **giysi** için z ölçeği yanlış ve pozitif-çapalı eşik bunu düzeltiyor.
- Ama "S3 her yerde daha iyi" doğru değil. Takas gerçek: S3 doğru claim'i daha az bozuyor (Retell için kritik),
  S2 yaygın sınıfa kayan yanlışları daha çok yakalıyor. Hangisinin daha önemli olduğu gerçek hata dağılımına bağlı
  ve onu yalnız insan etiketi söyler.
- Çekimserlik kuralının n < 10 kısmı, nadir sınıfları korumak isterken sentetik testteki yanlışların büyük kısmını
  görünmez yapıyor. Gerçek captioner hatalarının ne kadarı nadir sınıflara gidiyor, bilinmiyor.

## Öneri
- Varsayılan protokol gereği **S2**. S3'ü (ya da çekimsersiz S3'ü) sonradan seçmek, aynı swap/FCR sayılarına
  bakarak seçim yapmak olur.
- **İnsan pilotu** iki sistemin ayrıştığı claim'leri kapsamalı: S2 contradicted / S3 keep (özellikle giysi),
  S3 contradicted / S2 keep, S3 replace. Pilot sonucu, giysi için S3 kuralının ve çekimserliğin kalıp kalmayacağına
  karar verir. Kayıtlar hazır: `A_B_lcr2.json` (S2) ve `A_B_lcr3.json` (S3).
- Retell açısından: S3'ün düşük FCR'si "yanlışsa ucuz olmalı" ilkesiyle uyumlu. Retell kapalı döngüde
  (yeniden Look+Check, yalnız kabul edilirse değiştir) S2'nin yüksek FCR'sinin zararı da sınırlı.
