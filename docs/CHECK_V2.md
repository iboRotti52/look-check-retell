# Check v2: tasarım, sonuçlar, açık noktalar (2026-11-14, Claude)

**Bağımsız inceleme tamamlandı:** `docs/CHECK_V2_INCELEME.md` ve script'ten üretilen
`docs/generated/CHECK_V2_REVIEW_METRICS.md`. Ana olasılıklar/AUROC yeniden üretildi; seçim protokolü,
proxy yorumu ve alternatif önerisinin sınırları bu incelemede ayrıntılandırılıyor. Aşağıdaki temel tablolar
ilk koşunun sonuçlarını gösterir; AP'nin eşit-skor düzeltmesi inceleme tablosundadır.

Kaynak bölümler: `docs/LITERATUR_TARAMASI.md` §4, 6, 7, 8, 8a. Kayıtlar Look v2 (`*_look2`) üzerine kurulu.
Tüm etiketler **otomatik referans proxy'si** ("bir referans bu sınıfı anıyor"), insan etiketi değil.
Metrikler thing/stuff claim'leri üzerinde. Validation'da yanlış claim taban oranı %23.4.

## 1. Ne değişti, neden

Round-1 Check, OWLv2'nin ham olasılığına **tüm sınıflar için tek eşik** koyuyordu ve contradicted kararını iki farklı
sorgunun skorunu karşılaştırarak veriyordu. OWLv2 model kartına göre bu skorlar karşılaştırılamaz. Ayrıca
Check sadece detektöre bakıyordu. Oysa nesne halüsinasyonu tespitinde captioner'ın kendi belirsizliği, konum
ve eş-geçiş de güçlü sinyaller (LURE ICLR'24; Petryk+ WACV'24; MetaToken arXiv 2405.19186; HaloProbe ICML'26).

Check v2 = **öğrenilmiş, kalibre bir doğrulayıcı**. Paylaşılan ağırlıklı, sınıf başına büzülmüş sabit terimli bir
lojistik regresyon (`lcr/calibrate.py: FeatureLogistic`). Çıktısı P(claim doğru). Karar bu olasılık üzerinden,
seçici sınıflandırma mantığıyla veriliyor.

**C0, çapraz uydurma.** B'deki her kayıt, B'nin görüntüye göre 5 katından kendi katı dışındakilerde uydurulmuş
modelle skorlanıyor. Validation ise B'nin tamamında uydurulmuş modelle. Eşikler B'nin katlar-dışı skorlarında
seçiliyor. Her B kaydının katsayı uyumu kendi görüntüsünü dışarıda bırakıyor; global lambda ve eşik seçimi
ise tüm B etiketlerini kullanıyor. Tam iç içe seçim değildir (inceleme §2). Bu, Retell'in de B'de
eğitileceği düşünülerek seçildi: sınıf başına ~170 parametre için B'nin %20'si yetmezdi.

## 2. Adım adım sonuçlar (validation)

### 2a. Kalibrasyon tek başına (`scripts/20_check.py`, `results/check_v2.json`)
| | AUROC | ECE |
|---|---|---|
| Round 1 (ham skor) | 0.740 | 0.418 |
| Sınıf başına yüzdelik | 0.634 | – |
| Sınıf başına lojistik (sabit terim) | 0.786 | 0.03 |
| Sınıf başına lojistik (sabit + eğim) | 0.785 | 0.028 |

- Yüzdelik kötü, çünkü sınıfın taban oranını siliyor.
- Alternatif seçim yöntemi (detektör ilk 6 + captioner, kalibre ya da ham, aynı tür, sadece captioner) ve
  kural (a) ile (b) arasındaki fark ihmal edilebilir.
- **J'yi maksimize etmek yanlış hedef:** J-optimal nokta claim'lerin %49'una "contradicted" diyor, kesinlik
  0.40. Retell'e verilse doğru nesnelerin yarısı silinirdi. Bu yüzden karar noktası **kesinlik hedefiyle** seçiliyor.

### 2b. Öğrenilmiş doğrulayıcı, sinyal grupları (`scripts/22_check_verifier.py`, `results/check_verifier.json`)
Gruplar:
- **prior:** sınıf sabit terimi + A-referanslarından eş-geçiş PMI'ı + draft'taki claim sayısı. Görüntü yok.
- **det:** OWLv2 bölge skoru + objectness + claim'in görüntüdeki en yüksek skoru (s_image).
- **look:** excess, entropi, attention–detektör IoU.
- **cap:** captioner log p, entropi, marj, görsel kazanç (görüntülü − ortalama-görüntülü log p), göreli konum.
  `scripts/21_caption_signals.py`, 9.128 claim'in hepsi eşleşti.

Bölge = Look v2 attention bölgesi. Karar noktası: B'de contradicted kesinliği ≥ 0.5, supported kesinliği ≥ 0.95.

| Sinyaller | AUROC | AUROC (F30kE proxy) | AP (yanlış) | ECE | contradicted | c-kesinlik | yanlışları yakalama | supported | s-kesinlik |
|---|---|---|---|---|---|---|---|---|---|
| prior (görüntüsüz) | 0.763 | 0.704 | 0.427 | 0.031 | 0.124 | 0.489 | 0.259 | 0.337 | 0.959 |
| det (prior'sız) | 0.755 | 0.766 | 0.444 | 0.096 | 0.049 | 0.575 | 0.119 | 0.267 | 0.973 |
| cap (prior'sız) | 0.704 | 0.670 | 0.386 | 0.052 | 0.083 | 0.456 | 0.162 | 0.265 | 0.960 |
| prior + bölge | 0.801 | 0.747 | 0.494 | 0.030 | 0.206 | 0.495 | 0.435 | 0.387 | 0.966 |
| prior + görüntü düzeyi | 0.799 | 0.753 | 0.473 | 0.030 | 0.251 | 0.491 | 0.526 | 0.426 | 0.956 |
| prior + det | 0.810 | 0.765 | 0.496 | 0.035 | 0.254 | 0.510 | 0.554 | 0.414 | 0.960 |
| prior + det + look | 0.810 | 0.763 | 0.495 | 0.033 | 0.264 | 0.506 | 0.571 | 0.407 | 0.961 |
| **prior + det + cap (seçilen)** | **0.829** | 0.779 | **0.555** | 0.018 | 0.311 | 0.509 | **0.676** | 0.470 | 0.952 |
| hepsi | 0.829 | 0.777 | 0.552 | 0.031 | 0.319 | 0.508 | 0.693 | 0.474 | 0.954 |
| *round 1 (ham, script-11 eşikleri)* | 0.740 | – | – | 0.418 | 0.265 | 0.405 | 0.457 | – | 0.852 |

Mevcut iki sistemin farklı eşik/kapsam noktaları karşılaştırıldığında: contradicted kesinliği 0.405'ten 0.509'a, yanlışları yakalama 0.457'den 0.676'ya,
supported kesinliği 0.852'den 0.952'ye çıktı. Daha yüksek kesinlik hedefleri (0.6 ve 0.7) JSON'da: 0.6'da
validation c-kesinlik 0.592, yakalama 0.401.

**Mevcut keşif tercihi:** "prior + det + cap". Validation'da "Hepsi" ile yakın AUROC'u veriyor; look grubunun ek katkısı gösterilemedi (0.810 → 0.810),
sadelik için dışarıda. `*_check2.json` kayıtları bununla yazıldı. Yalnızca görsel varyant ("prior + det") de
`*_check2vis.json` olarak yazıldı. Retell'de "kanıtın görsel kısmı mı işe yarıyor?" ablation'ı için.
Bu tercih kodda B ölçütüyle otomatik seçilmiyor; B'de "Hepsi" daha iyi. Önceden sabitlenmiş sadelik kuralı
olmadığından B-only özellik seçimi iddiası doğrulanamıyor. `check2vis` sınıf/draft önceliği de içeriyor.

### 2c. Sorgu şablonları (`scripts/23_owl_queries.py`, `results/owl_queries.json`)
Tek şablon ("a photo of a {}"), dilbilgisi düzeltilmiş şablon ve 7 şablonluk ensemble (ViLD tarzı "in the scene"
dahil): kalibre bölge AUROC'u B'de 0.782 / 0.782 / 0.782, validation'da 0.784 / 0.785 / 0.785. **Etki yok.**
Mevcut sorgular korundu. Önbellekteki sorguların birebir yeniden üretildiği doğrulandı (fark < 1e-6).

## 3. Bulgular (rapora ve hipotezlere etkisi)

1. **Sınıf + PMI + claim sayısı öncelik modeli neredeyse detektör kadar güçlü** (0.763'e karşı 0.755). "Hangi sınıflar genelde
   yanlış söyleniyor" bilgisi bir dil önceliği, görsel kanıt değil. Raporda Check'in gücü bundan ayrıştırılmalı.
   Görüntünün katkısı: prior → prior+det = +0.047 AUROC. Captioner güveninin katkısı: +0.019.
2. **H6 (attention mı detektör mü):** Look bölgesi (prior+bölge 0.801), detektörün kendi bölgesi (0.797) ve
   bölgesiz görüntü düzeyi skor (0.799) neredeyse aynı. Varlık doğrulamak için "captioner'ın baktığı yer" ek
   bilginin ek katkısı bu ölçümde gösterilemedi. Farkın bootstrap aralığı sıfırı içeriyor;
   bu, H6 için keşif bulgusudur, yöntemlerin eşdeğerliğinin kanıtı değildir (proxy etiketleriyle).
3. **Look'un kendi sinyal grubu** (excess, entropi, attention–detektör uyumu), bu özellikler ve doğrusal model altında ek katkı göstermedi.
4. **Kesinlik tavanı:** yanlış claim'leri yakalarken kesinlik 0.5–0.6 civarında kalıyor. *Düzeltme
   (2026-11-14):* Burada önce proxy gürültüsünün ölçülen kesinliği düşürdüğünü yazmıştım. Yön tersi.
   "Görüntüde var ama referansta yok" hatası (ör. boardwalk/pier, stump/log, seats/chair, kimsenin anmadığı
   t-shirt) bayraklı claim'i doğru tespit gibi sayar. Yani ölçülen kesinlik büyük ihtimalle **olduğundan yüksek**,
   ölçülen yakalama ise olduğundan düşük. Referansta geçen ama bayraklanan claim'ler (yanlış alarmlar) ise
   güvenilir biçimde gerçekten doğru. **İnsan pilot etiketlemesi gerekli.** Gerçek kesinliği ancak o söyler.
5. **Hata nerede: giyim** (`scripts/24_check_errors.py`, `results/check_errors.json`). Giyim claim'lerinde proxy
   yanlış oranı %42, çünkü referanslar kıyafetleri nadiren anıyor. Doğrulayıcı giyim claim'lerinin %68'ini
   bayraklıyor. Kesinlik 0.50, taban orana göre kazanç sadece 1.19×. Yani giyimde neredeyse yalnızca sınıf
   önceliğiyle karar veriyor. Diğer nesnelerde kesinlik 0.55, kazanç 3.29×. Zeminde (stuff) kesinlik 0.42,
   yakalama 0.35. Giyim hem proxy'nin en güvenilmez olduğu hem de detektörün en zayıf olduğu yer.
   Seçenekler (ablation ile): giyimi Retell'in silme kapsamından çıkarmak ya da ayrı ele almak; kişi kırpımında
   giyim sorgusu; Retell'e sert karar yerine sürekli `p_correct` vermek.

6. **Look v4 ile yeniden koşu (2026-11-14):** `*_look4` kayıtlarıyla seçilen doğrulayıcı val AUROC 0.826
   (look2 ile 0.829), prior+bölge 0.800 (0.801). Look'un doğru kutu oranı +4.3 puan arttı ama varlık
   doğrulaması değişmedi. Bulgu 2 ile tutarlı. `results/check_verifier_look4.json`, `*_look4_check2.json`.

## 4. Kod değişiklikleri
- Yeni: `lcr/calibrate.py` (katlar, yüzdelik, sınıf lojistiği, FeatureLogistic, AUROC eşitlik düzeltmeli, ECE,
  karar kalitesi), `scripts/20_check.py`, `21_caption_signals.py`, `22_check_verifier.py`, `23_owl_queries.py`,
  `tests/test_calibrate.py` (5 test).
- `lcr/pipeline.py: finalise`: `check == "v2"` kayıtlarında kararı yeniden hesaplamıyor.
- `lcr/evidence.py`: ρ > 0 ve v2 kaydı birlikteyse `NotImplementedError` (aşağıdaki açık nokta 1).
- Veri: `data/cache/pipeline/*_capsig.json`, `*_look2_check2.json`, `*_look2_check2vis.json`,
  `data/cache/owl/class_queries_{single,grammatical,ensemble}.npy`.
- Codex inceleme düzeltmesi: script 20 artık `*_check2cal.json` yazar; script 22'nin öğrenilmiş kayıtlarıyla
  çakışmaz. İki scriptte de mevcut pipeline dosyasını değiştirmek için `--overwrite` gerekir.

## 5. Açık noktalar (Retell'e geçmeden önce ya da Retell'le birlikte)
1. **Sentetik bozma v2 ile tanımlı değil.** `relabel()` bozulmuş claim'e round-1 eşik kuralıyla karar veriyor.
   Gerçek claim'ler ise doğrulayıcı kararı taşıyor. Retell bu farkı kısayol olarak öğrenebilirdi, bu yüzden
   engellendi. Doğru yol: bozulmuş kelimeyi draft'a yerleştirip captioner sinyallerini (21) ve detektör
   özelliklerini o kelime için yeniden hesaplamak, sonra doğrulayıcıyla skorlamak (çevrimdışı, B için tek seferlik).
2. Retell kanıt token'ı hâlâ ham `s_claim`/`s_alt` kullanıyor. `p_correct` (kalibre) skaler olarak eklenmeli
   (`lcr/retell.py: record_features`, N_SCALARS).
3. `s_image` (H6 karıştırıcısı) artık doğrulayıcının içinde. "ours" ile B3'ü karşılaştırırken bölge dışı bilgi
   iki tarafta da var. Bulgu 2 bunu zaten ölçüyor; Retell ablation'ında "prior+region" doğrulayıcısı da denenebilir.
4. İnsan pilot etiketlemesi (`annotation/pilot/`): proxy kesinliğini ölçmek için.
5. Alternatif önerisi: contradicted claim'ler için captioner adayları arasından ham detektör skoru en yüksek olan.
   Bu sıralama hâlâ sorgular-arası kalibrasyon sorununu taşıyor. Önceki captioner ilk-aday ölçümü bu seçicinin
   başarısı değildir; gerçekten yazılmış alternatiflerin sayımları `results/check_v2_review.json` ve otomatik
   inceleme tablosundadır. Retell'in "düşür" ile "değiştir" kararını vermesi gerekecek.
