# Look v2 — plan ve gerekçeler

Durum: **Look v2 donduruldu; ek iyileştirmeler inceleniyor** (2026-11-14). Sonuçlar §7'de;
Codex incelemesi §7.5 ve `docs/LOOK_V2_INCELEME.md` içinde.

## 1. Problem

Look adımı, captioner bir nesne kelimesi (claim) ürettiğinde görüntünün **hangi bölgesine dayandığını** bulmalı;
Check o bölgeyi detektörle doğrular. 1. turda Look = BLIP decoder cross-attention'ının son 4 katman × 12 başlık
ortalaması + en yüksek "excess mass"lı OWLv2 kutusu. Ölçümler (validation, otomatik):

- Seçilen bölgedeki claim skoru, referansta geçen/geçmeyen iddiaları **AUROC 0.51** ile ayırıyor (şans).
  Detektörün kendi kutusu 0.74.
- Check verdict'inin bilgi değeri J = 0.01 (şans), detektör bölgesiyle 0.31.
- Sonuç: Retell'e giden kanıt gürültü → kapılar kapalı kalıyor.

Bu bulgu literatürle tutarlı: captioner attention'ının insan kutularıyla uyumu zayıf
(Liu+ AAAI'17 — Flickr30k Entities ile ölçmüşler; Zhou+ CVPR'20 — POS-SCAN), ham attention üst katmanlarda
açıklama olarak güvenilmez (Abnar & Zuidema ACL'20), ve görsel "attention sink" token'ları attention'ın büyük
kısmını metinden bağımsız topluyor (Darcet+ ICLR'24; Kang+ ICLR'25).

## 2. Tasarım ilkeleri (bilimsel gerekçe)

1. **Look, captioner'ın kendi kararını açıklamalı.** Soru "nesne nerede?" değil, "captioner bu kelimeyi
   söylerken neye dayandı?". Bu yüzden harita kaynağı captioner'ın iç sinyalleridir (attention, gradyan).
   Dış modeller (BLIP-ITM, OWLv2) yalnızca **referans/üst sınır** olarak ölçülür.
2. **Look, iddia edilen sınıfın detektör skorunu kullanmamalı.** Kullanırsa bölge iddiayı destekleyecek
   şekilde seçilir ve doğrulama döngüsel olur (bu zaten B3 kontrolü). Sınıftan bağımsız bilgi
   (ör. "burada herhangi bir nesne var mı" = objectness) kullanılabilir.
3. **Literatürdeki bileşenler birleştirilebilir, kendi eklemelerimiz olabilir; her bileşen ablation ile ölçülür.**
   Hiçbir bileşen "iyi görünüyor" diye eklenmez; B'de ölçülen katkısı yoksa çıkarılır.
4. **Ayar B'de, rapor validation'da, test asla.** Tüm hiperparametreler (katman, başlık, σ, ağırlıklar)
   B yarısında (Flickr8k train'in ikinci yarısı, 3.000 görüntü) seçilir. Validation'da her konfigürasyon bir kez
   raporlanır. Test split'inin F30kE kutularına bakılmaz.

## 3. Bileşenler

### 3.1 Harita kaynakları (her iddia için, captioner kendi draft'ında teacher-forced)
| Kod | Kaynak | Literatür | Neden |
|---|---|---|---|
| S1 | Ham cross-attention A_{l,h} | — (1. tur) | taban çizgisi |
| S2 | Grad-CAM on attention: ReLU(∂log p(w)/∂A ⊙ A) | ALBEF (Li+ NeurIPS'21), BLIP; Chefer+ ICCV'21 | sadece kelimenin olasılığını gerçekten artıran attention'ı tutar |
| S3 | Sink/kontrast düzeltme: A(p) − ortalama_t A(t), ya da A(p)/ortalama | Darcet+ ICLR'24; Kang+ ICLR'25; Zhang+ ICLR'25 ("MLLMs know where to look": genel attention'a oranlama) | kelimeden bağımsız bileşeni çıkarır |
| S4 | **Kontrastif gradyan** (bizim ekleme): ∂[log p(w) − log p(w_alt)]/∂A, w_alt = captioner'ın 2. aday ismi | kontrastif açıklama: Jacovi+ EMNLP'21; Zhang+ ECCV'16 (contrastive attention) | "neden *dog* dedi de *cat* demedi?" — ayırt edici kanıtı vurgular; tam da yanlış nesne halüsinasyonunun sorusu |
| S5 | Relevance propagation (encoder-decoder kuralları) | Chefer+ ICCV'21; rollout: Abnar & Zuidema ACL'20 | katmanlar arası karışımı hesaba katar (öncelik düşük) |
| Ref | BLIP-ITM Grad-CAM (ayrı checkpoint) | ALBEF, BLIP, PnP-OVSS (Luo+ CVPR'24) | dış referans; captioner'ın kararını açıklamaz |
| Ref | OWLv2 kutusu (B3), rastgele kutu | — | üst / alt referans |

### 3.2 Katman / başlık birleştirme
- G1: son k katman × tüm başlıklar ortalaması (1. tur).
- G2: katman bazında tek tek ölç (ALBEF: grounding belirli bir katmanda uzmanlaşıyor).
- G3: **localization heads** — görüntüye çok attention veren ve uzamsal olarak yoğun (düşük entropi)
  başlıkları seç (Kang+ CVPR'25: birkaç başlık yetiyor). Ölçüt etiketsiz; B'de hesaplanır.
- G4: **B'de denetimli başlık seçimi** (bizim ekleme): F30kE kutularıyla B'de pointing doğruluğu en yüksek m başlık.
  Etiketsiz G3 ile karşılaştırılır.

### 3.3 Harita son işleme
- P1: Gauss yumuşatma (Kang+ CVPR'25).
- P2: Salience DropOut — en yüksek yamaları maskeleyip yeniden geçir, nesnenin geri kalanını yakala (PnP-OVSS).
  Grad-CAM'in "en ayırt edici parçaya daralma" sorununa çözüm. (Ek forward geçiş gerektirir; öncelik orta.)

### 3.4 Topluluk (bizim ekleme)
- E1: seçilen kaynakların sıra-normalize (rank) haritalarının ortalaması / geometrik ortalaması; ağırlıklar B'de.
  Gerekçe: attention "taşırır", Grad-CAM "daraltır" (PnP-OVSS gözlemi) → hatalar kısmen bağımsız.

### 3.5 Haritadan bölgeye
- R1: en yüksek excess mass'lı OWLv2 kutusu (1. tur).
- R2: **CAM-box** — haritayı τ·max'ta eşikle, en büyük bağlı bileşenin kutusu (Zhou+ CVPR'16 CAM), sonra
  en yüksek IoU'lu OWLv2 kutusu.
- R3: **objectness ağırlıklı kütle** (bizim ekleme): excess mass × sınıftan bağımsız nesnellik
  (OWLv2'nin tüm sözlük sınıfları üzerindeki max skoru). İlke 2'ye uyar: iddia edilen sınıfa özel skor kullanılmaz.
  Gerekçe: attention'ın arka plan dilimlerine düşen kutuları elemek.
- R4: yumuşak bölge (bizim ekleme): tek kutu yerine ağırlıklı ilk m kutu; Check skorları ağırlıklı toplar.

## 4. Ölçüt (benchmark) — insan çizimi kutular

Kaynak: Flickr30k Entities (Plummer+ ICCV'15). Flickr8k'nin 8.000 görüntüsünün tamamını kapsıyor,
boyutlar birebir aynı, F30kE cümleleri bizim Flickr8k referanslarımızla aynı (2026-11-13'te doğrulandı).

Hedef üretimi: bir görüntüdeki draft iddiası, sınıf c → o görüntünün 5 cümlesindeki F30kE öbeklerinden
bizim claim çıkarıcımızla c'ye eşlenenlerin **tüm kutularının birleşimi**. F30kE'de karşılığı olmayan iddialar
lokalizasyon metriklerinden çıkarılır (sayısı ayrıca raporlanır — bunlar çoğunlukla referansta olmayan iddialar).

Metrikler:
- **Pointing game**: haritanın maksimumu bir GT kutusunun içinde mi? (Zhang+ ECCV'16). Şans = kutu alanı payı.
- **Enerji-tabanlı pointing**: harita kütlesinin GT kutularına düşen oranı (Score-CAM, Wang+ CVPRW'20).
- **Bölge doğruluğu**: seçilen OWLv2 kutusunun herhangi bir GT kutusuyla IoU ≥ 0.5 olması
  (F30kE zayıf denetimli grounding'deki standart ölçü).
- **Aşağı akış**: Check verdict'inin bilgi değeri J ve claim skoru AUROC (referans proxy'si).
- Hepsi sınıf türüne göre (person / thing / stuff) ayrıca.

## 5. Protokol

1. Colab'da bir geçiş: B ve validation'daki her draft iddiası için tüm katman × başlıklarda A, ∂log p(w)/∂A,
   ∂log p(w_alt)/∂A kaydedilir (yalnızca Colab diskinde, fp16).
2. Tüm varyantlar bu kayıtlardan GPU'suz hesaplanır.
3. **Aşamalı (ileri) seçim B'de**: önce kaynak + birleştirme (pointing), sonra son işleme, sonra topluluk,
   sonra haritadan bölgeye (bölge doğruluğu). Her aşamada sadece bir şey değişir.
4. Seçilen yol validation'da bir kez raporlanır → ablation tablosu (her satır bir bileşen ekler).
5. Seçilen Look kodda dondurulur; pipeline kayıtları yeniden üretilir; sonra Check'e geçilir.

## 6. Riskler
- Hiçbir captioner-içi yöntem detektörün kendi kutusuna yaklaşamayabilir. Bu da bir bulgudur (H6'yı doğrudan
  besler). O durumda seçenekler: (a) Look'u "captioner'ın baktığı yer" olarak tutup Check'te detektörle
  karşılaştırmak (attention–detector uyuşmazlığının kendisi bir halüsinasyon sinyali olabilir), (b) B3'ü ana
  sistem yapmak ve Look'u analiz bileşeni olarak raporlamak. Karar ölçümden sonra ekip olarak verilir.
- Grad-CAM tek bir parçaya daralabilir → P2 ve E1 bunun için.
- F30kE öbek → sınıf eşlemesi hatalı olabilir → eşleme örnekleri elle kontrol edilir, eşleme oranı raporlanır.

## 7. Sonuçlar

### 7.1 Taban çizgisi: 1. tur Look, insan kutularıyla (validation, 2026-11-13)
`python scripts/13_look_benchmark.py --pipe A_validation` → `results/look_benchmark_A_validation.json`.
Harita: ham cross-attention, son 4 katman × tüm başlıklar; bölge: en yüksek excess mass. Decoder A.
2.272 draft iddiasından 1.795'inin (%79.0) F30kE'de kutusu var.

| Tür | n | Pointing | Şans | Enerji | Bölge (attention) | Bölge (detektör, B3) | Rastgele kutu | Oracle |
|---|---|---|---|---|---|---|---|---|
| hepsi | 1795 | 23.9 | 37.1 | 35.8 | 12.9 | 82.7 | 4.7 | 95.3 |
| person | 761 | 20.6 | 37.7 | 33.8 | 11.3 | 85.9 | 6.8 | 95.5 |
| thing | 758 | 16.0 | 28.3 | 28.9 | 12.5 | 83.8 | 3.7 | 95.4 |
| stuff | 276 | 54.7 | 59.9 | 60.0 | 18.5 | 71.0 | 1.8 | 94.2 |

(Bölge = seçilen kutunun bir insan kutusuyla IoU ≥ 0.5 olma oranı, %. Oracle = OWLv2 adaylarından en az biri ≥ 0.5.)

Yorum: ham attention'ın tepe noktası **şansın altında** (sink yamaları tepeyi nesneden uzağa çekiyor), enerji
şans düzeyinde. OWLv2 aday kutuları nesneyi neredeyse her zaman içeriyor (%95), yani sorun aday kutularda değil,
**haritada ve seçimde**. Örnek figür: `report/figures/look_benchmark_examples_A_validation.pdf` (yeşil = insan
kutusu, kırmızı kesikli = attention'ın seçtiği kutu); eşlemenin doğru çalıştığı gözle kontrol edildi.

### 7.2 Sink düzeltmesi (S3), insan kutularıyla (validation, 2026-11-13)
`python scripts/13_look_benchmark.py --pipe A_validation_sink` → `results/look_benchmark_A_validation_sink.json`.
Harita: A(p) − caption ortalaması, sıfırda kırpılmış; son 4 katman × tüm başlıklar; bölge: en yüksek excess mass.

| Tür | n | Pointing | Şans | Enerji | Bölge (attention) | Bölge (detektör, B3) | Rastgele kutu | Oracle |
|---|---|---|---|---|---|---|---|---|
| hepsi | 1795 | **88.3** | 37.1 | **73.2** | **55.7** | 82.7 | 4.7 | 95.3 |
| person | 761 | 89.2 | 37.7 | 68.6 | 52.7 | 85.9 | 6.8 | 95.5 |
| thing | 758 | 88.3 | 28.3 | 77.1 | 64.1 | 83.8 | 3.7 | 95.4 |
| stuff | 276 | 85.9 | 59.9 | 75.1 | 40.9 | 71.0 | 1.8 | 94.2 |

**Sağlamlık kapıları** (`scripts/14_look_sanity.py`, `results/look_sanity_*.json`):
- Yönelim: transpoze harita ile pointing 88.3 → 51.9, dikey ters çevrilmiş ile → 54.4. Izgaradan görüntüye
  eşleme doğru.
- Kelimeye özgüllük (takas testi, 1.443 claim): claim'in kendi haritası %87.5; aynı görüntüdeki **başka sınıftan
  bir claim'in** haritası aynı kutularda %45.4 (şans %35.8). Enerji 72.0'a karşı 44.2. Harita sadece "göze
  çarpan nesneyi" değil, söylenen kelimenin nesnesini gösteriyor.
- Ham haritada aynı test: 23.7'ye karşı 23.1, yani ham attention kelimeye özgü değil.

Aşağı akış (referans proxy'si, thing/stuff, validation; `scripts/12_look_variants.py --attn sink`): claim skoru
AUROC'u ham attention bölgesiyle 0.51 iken sink + argmax bölgesiyle 0.648, sink + top-10 ile 0.724. Detektör kutusu
0.744. B'de kalibre edilmiş J: sink-attention 0.284, detektör 0.414 (`results/thresholds_proxy_*.json`).

Yorum: **harita sorunu büyük ölçüde çözüldü.** Captioner nereye baktığını biliyor, ama ham attention sink
yamalarıyla kirlenmiş. Bu Darcet+ ICLR'24 ve Kang+ ICLR'25 ile tutarlı. Darboğaz artık **haritadan kutuya
geçiş**: harita tepesi %88 doğru nesnede, ama seçilen kutu yalnızca %55.7'sinde IoU≥0.5 (detektör %82.7,
oracle %95.3). Stuff (%40.9) en zayıf; kutu sınırı belirsiz sınıflar için IoU zaten sert bir ölçü.

### 7.3 Haritadan bölgeye (R1–R3), seçim B'de, rapor validation'da (2026-11-14)
`python scripts/16_look_regions.py` → `results/look_regions_A_validation_sink.json`. 72 konfigürasyon:
excess (β), ratio (kütle / alan^α), objectness^γ çarpanı (R3), tepe kısıtı, CAM-box (τ, R2). Hiçbiri iddia
edilen sınıfın detektör skorunu kullanmıyor. Seçim ölçütü: B'de bölge doğruluğu (IoU≥0.5, 5.383 claim).

| Konfigürasyon (en iyisi B'de) | B | val | val mIoU | person | thing | stuff | AUROC (proxy) |
|---|---|---|---|---|---|---|---|
| R1 (1. tur): excess mass | 55.7 | 55.7 | 0.585 | 52.7 | 64.1 | 40.9 | 0.648 |
| **ratio α=0.5 + objectness γ=0.5 (seçilen)** | **74.6** | **75.3** | **0.702** | 66.4 | 83.8 | 76.4 | 0.740 |
| ratio + objectness + tepe | 72.5 | 74.1 | 0.695 | 67.7 | 79.7 | 76.4 | 0.725 |
| excess + objectness | 70.4 | 70.3 | 0.673 | 64.5 | 74.9 | 73.2 | 0.746 |
| CAM-box + objectness | 63.1 | 63.5 | 0.618 | 58.5 | 75.6 | 44.2 | 0.684 |
| ratio (objectness'sız) | 56.6 | 56.7 | 0.594 | 40.3 | 67.3 | 72.8 | 0.693 |
| CAM-box (objectness'sız) | 42.6 | 43.7 | 0.493 | 38.5 | 52.8 | 33.3 | 0.615 |
| *referans: detektör kutusu (B3)* | 82.9 | 82.7 | | 85.9 | 83.8 | 71.0 | 0.744 |
| *referans: oracle* | 94.9 | 95.3 | | | | | |

AUROC = seçilen kutudaki claim skoru ile "referansta geçiyor" ayrımı, **tüm** thing/stuff claim'leri üzerinde
(F30kE kutusu olan alt kümede neredeyse hiç negatif yok: 1.032'ye karşı 2. İlk sürüm bu yüzden şişik 0.887
gösterdi, düzeltildi).

Yorum:
- En büyük katkı **objectness** (R3, bizim ekleme): ratio 56.6 → 74.6. Attention nesneyi gösteriyor, objectness
  "nesne gibi duran" kutuyu seçtiriyor. İkisi tamamlayıcı.
- B ile validation neredeyse aynı (74.6 / 75.3), yani aşırı uyum yok.
- thing'de detektörle eşit (83.8), stuff'ta detektörün üstünde (76.4 vs 71.0). person'da geride (66.4 vs 85.9);
  muhtemelen çok kişili görüntülerde attention kişilerden birine odaklanıyor, F30kE ise hepsini işaretliyor.
- Check'e giden sinyal (AUROC) artık detektörün kendi kutusuyla aynı (0.740 vs 0.744). B'de kalibre edilmiş
  J: **0.461** (sink-v1 0.284, detektör 0.414; referans proxy'si).

### 7.4 Dondurulan Look (v2)
Sink düzeltmeli harita (son 4 katman × tüm başlıklar) + `look.select_region` (kütle / alan^0.5 × objectness^0.5,
`config.REGION_ALPHA/GAMMA`). `lcr/pipeline.py` artık bunu kullanıyor (kayıtlarda `look="v2"`). Kayıtlar GPU'suz
yeniden yazıldı: `scripts/17_freeze_look.py` → `A_B_look2.json`, `A_validation_look2.json` (claim'lerin ~%49'unda
bölge değişti). Script 06 ve 11 için `--attn look2`.

Açık kalanlar (Look'u yeniden açmayı gerektirmez, sonra denenebilir):
- Harita kaynakları (S2 Grad-CAM, S4 kontrastif, baş seçimi) ve 5a'daki zaman adımı / alt-kelime / öbek ablation'ları.
  Colab çıkarım geçişi gerekir. Kazanç alanı: person bölgesi (66 vs 86).
- R4 (yumuşak bölge) Check'in bir parçası olarak denenecek.
- Excess-mass kapısı v2 kayıtlarında kapalı. Check'te yeniden tanımlanacak.

### 7.5 Codex incelemesi: katkı ayrımı ve person tanısı (B, 2026-11-14)

`scripts/review_look_v2.py` → `results/look_v2_review_B.json`, `results/look_v2_person_review_B.png`.
Yeni konfigürasyon seçilmedi. B'de 5.383 kutulu claim: v2 %74.64; yalnızca objectness %35.95;
tekdüze harita + v2 kuralı %39.61; aynı alpha ile objectness'sız %50.77. Aynı 4.991 örnekte kendi
haritası %74.65, başka sınıfın haritası %22.12 doğru kutu seçiyor. Attention'ın ek katkısı belirgin.

Person: 2.244 claim, %68.27 doğru kutu. GT kutusu 1 / 2 / 3+ gruplarında %72.14 / %72.80 / %57.38.
712 yanlış seçimin 569'unda tepe noktası doğru GT içinde; 206 küçük parça, 201 fazla geniş kapsayıcı,
266 diğer örtüşme, 39 örtüşmesiz seçim (geometriyle tanımlanan otomatik hata grupları).
§7.3'teki "attention kişilerden birine odaklanıyor" açıklaması tek başına yeterli değil: ölçüm herhangi
bir doğru kişiyi kabul ediyor. Kutu ölçeği daha güçlü bir açıklama; örnekler görsel olarak incelendi.

Kayıt sorunu: freeze script'i `rec_att` değiştirirken `iou_att_det`, `regions_by_k`, `excess_by_k` eski
kalıyor. Ana insan-kutusu doğruluğu etkilenmiyor; sonraki analizden önce sürüm/metadata düzeltilmeli.
Literatür uyarlamaları, ek ölçüm sınırlamaları ve öncelikler: `docs/LOOK_V2_INCELEME.md`.

### 7.6 Ölçüm ve kayıt tutarlılığı düzeltmesi (2026-11-14)

`results/look_consistency_fix.json` ve `scripts/verify_look_artifacts.py`: B/validation'daki 9.128
claim'in v2 kutuları ve Check skorları aynı; `iou_att_det`, `regions_by_k["4"]`, `excess_by_k["4"]`
tutarlı. k=1/12 eski sonuçları `legacy_look` altında. Yeni çıktılar açık v2 adları ve sürüm metadata'sı
taşıyor; script 04 mevcut çıktıyı açık `--overwrite` olmadan değiştirmiyor.

§7.1/7.2'deki enerji/şans sayıları ilk maskenin tarihsel ölçümleri. Geometrik birleşim düzeltmesiyle
validation sink enerji %73.1512 → %73.2064, şans %37.1340 → %37.1728. Pointing %88.3008 ve v2 bölge
%75.2646 aynı. AUROC artık eşit skorları doğru işliyor; mevcut ölçümlerde sayısal fark çıkmadı.
Yöntem seçimi değişmedi; yeni validation taraması yapılmadı. Script 13/14/16 yalnızca aynı yöntemlerin
düzeltilmiş ölçümünü tekrar üretti. 43 test ve harita kapıları geçti. Güncel tablolar `look_report.py`
ile rapora aktarılıyor. Sıradaki iyileştirme, B'de sınırlı kişi kutusu / yumuşatma deneyi.

### 7.7 E1 — sabit tür/σ ızgarası (2026-11-14)

`results/look_e1.json`: tür başına 36 konfigürasyon; ≥1 B puanı kabul kuralıyla person σ=1 ve
stuff γ=.25 kabul, diğer parametreler v2. Validation bölge %76.1003, v2 %75.2646; fark GA95
[+0.1614,+1.5514] puan (2.000 görüntü bootstrap). Person +0.7884 (GA sıfırı içeriyor),
thing 0, stuff +3.2609 puan. Yumuşatma haritaları 14 kapılarını geçti. E2 bitmeden Look kapanmadı.

### 7.8 E3 — literatürde denenmemiş yöntemler, Look v4 (2026-11-14, Claude)

Protokol koşmadan önce yazıldı: `docs/LOOK_E3_PROTOKOL.md`. Kod: `lcr/look_e3.py`, `lcr/look_e3_eval.py`,
scripts 25–29. Sonuçlar: `results/look_e3_selection.json`, `look_e3_stageA.json`, `look_e3_boxes_selection.json`,
`look_e3.json`. Seçim B'de, aile başına görüntüye göre 5 katlı ÇD; kabul ≥ 1 puan ÇD kazancı; validation bir kez.

**A aşaması, harita kaynağı** (kutu kuralı v3). B'de ÇD kazancı, v3'e (%75.59) göre; validation doğru bölge,
v3 = %76.10; GA95 v3'e karşı eşleştirilmiş görüntü bootstrap'ı:

| Aile | B ÇD kazancı | val bölge | val pointing | val GA95 (puan) |
|---|---|---|---|---|
| **A1-G3 etiketsiz başlık seçimi (4 başlık: L9H8, L10H8, L10H9, L9H3)** | **+2.64** | **79.22** | **91.31** | [+1.66, +4.52] |
| A1-G4 denetimli başlık seçimi (4 başlık) | +1.10 | 78.38 | 89.42 | [+0.89, +3.66] |
| A6 topluluk (S3 + kontrastif, olasılık ort.) | +0.35 | 76.55 | 89.47 | [−0.56, +1.44] |
| A1-G2 tek katman (L10) | +0.22 | 76.32 | 85.96 | [−1.13, +1.59] |
| A2 isim öbeği | −0.24 | 76.32 | 88.75 | [−0.79, +1.20] |
| A0 sink ortalamasına [SEP] dahil | −1.06 | 75.26 | 87.24 | [−1.50, −0.11] |
| A4 Chefer relevance | −2.40 | 75.04 | 92.26 | [−2.66, +0.50] |
| A5 Salience DropOut | −3.40 | 73.15 | 80.89 | [−4.42, −1.47] |
| A3 oran tipi sink | −16.53 | 57.27 | 88.86 | [−21.09, −16.63] |

(Tablo `scripts/30_look_e3_table.py` ile üretilen `docs/generated/LOOK_E3_METRICS.md`'den.)
- Etiketsiz başlık skoru (özgüllük / (1 + bağlı-bileşen entropisi)) başlıkların B pointing doğruluğuyla
  0.81 korelasyonlu. Sıralama etiketsiz; m ve aile seçimi B insan kutularıyla yapılıyor.
  Bu, Kang+ CVPR'25'ten esinlenen BLIP uyarlamasıdır; özgün seçicinin birebir tekrarı değildir. Izgaranın en küçüğü m=4 her katta seçildi; daha az başlık
  ilk protokolde denenmedi; Codex incelemesinde yalnızca B tanısı olarak ölçüldü, m=4 korundu.
- Harita kapıları (B / val): pointing 92.0 / 91.3, transpoze 53.6 / 52.3, takas kendi 91.8 / 91.2,
  başka sınıf 51.7 / 48.7. Geçti.
- Chefer relevance pointing'de en iyi (92.26) ama kutu seçiminde kötü: tepe noktası doğru, kütle dağılımı
  kutu skoruna uymuyor. Oran tipi sink dağınık harita veriyor (kütle tüm görüntüye yayılıyor).
- Arşiv/yeniden hesap sapması: CPU fp32 ile yeniden hesaplanan v2 haritası B'de v3 kuralıyla %74.16, arşiv
  %75.59. Kabul edilen başlık haritası yeniden hesap hattında olduğu için kazanç bu dezavantaja rağmen geliyor.
  Sapmanın kaynağı çözülmedi. Codex SEP ve ayrı CPU fp16 decoder dahil tanılarla bu varsayımların
  arşivi bütünüyle yeniden üretmediğini gördü; bkz. `docs/LOOK_E3_INCELEME.md`.

**B aşaması, kutu (A'nın haritası üstünde, türe göre):** tepe noktası SAM'e (Kirillov+ ICCV'23) nokta istemi.
| Tür | B taban | B ÇD | kazanç | kabul | val (A sonrası → son) |
|---|---|---|---|---|---|
| person | 75.27 | 79.99 | +4.72 | **SAM, 3 maskenin çevreleyen kutularından kütle/kutu alanı^0.25 en yüksek** | 73.72 → **76.48** |
| thing | 81.15 | 74.16 | −6.98 | hayır | 85.09 |
| stuff | 78.01 | 77.02 | −0.99 | hayır | 78.26 |

SAM'in kendi IoU tahminiyle seçilen maske (B1) kötü (val kişi %24): en küçük parçayı (yüz, gömlek) seçiyor.
Düşük α büyük maskeyi, yani bütün kişiyi tercih ettiriyor. SAM maskelerinin herhangi biri doğru olsa
üst sınır val kişi %82.8.

**Look v4 (validation, insan kutuları):** doğru bölge **%80.39** (v3 %76.10), fark **+4.29 puan, GA95 [+2.46, +6.09]**.
Tür bazında: person **76.48** (v3 67.15; +9.33 [+5.32, +13.10]), thing 85.09 (+1.32 [−0.14, +2.91]),
stuff 78.26 (−1.45 [−4.74, +1.73]). Detektör kutusu (B3) referansı: hepsi 82.7, person 85.9, thing 83.8,
stuff 71.0. Look artık thing ve stuff'ta detektörün üstünde, kişide 9 puan geride (önce 19).
Dondurma: `scripts/29_freeze_look4.py` → `*_look4.json`; kayıtlardan yeniden ölçüm %80.39'u birebir veriyor.
Kişi kutusu SAM kutusu; detektör skorları ona en çok örtüşen OWLv2 önerisinden.

**Check'e etkisi** (`scripts/22_check_verifier.py --select A_B_look4 --report A_validation_look4
--results check_verifier_look4.json`): seçilen doğrulayıcı (prior+det+cap) val AUROC 0.829 → 0.826,
prior+bölge 0.801 → 0.800. **Daha iyi yerelleştirme varlık doğrulamasını değiştirmiyor**; Check v2 §3 bulgu 2
(bölge ≈ görüntü düzeyi skor) ile tutarlı. Kayıtlar: `*_look4_check2.json`, `*_look4_check2vis.json`.

**Codex review tamamlandı:** `docs/LOOK_E3_INCELEME.md`; sayısal ek `docs/generated/LOOK_E3_REVIEW_METRICS.md`.
Canlı `pipeline.run` ve script 04 artık ortak `lcr/look_v4.py` koduyla v4 üretir; script 29 ile kutu/harita
birebir eşitliği doğrulandı. SAM yardımcı excess alanları ve skor geometrisi provenance'ı düzeltildi.
Başlıklar yukarıda iki eksende de 1'den sayılır; düz indeksler değişmedi. Seçim ÇD'si bütün A→B seçicisinin
tam iç içe tahmini değildir; alt aile/tür GA'ları eşzamanlı güvence vermez. E3 Salience DropOut özgün
PnP-OVSS'nin özellik düzeyinde sınırlı bir uyarlamasıdır. Arşiv sapması açık bir sınırlamadır.
Yeni Look seçimi yapılmadı; test değerlendirmesi veya Retell eğitimi başlatılmadı.

## Kaynaklar (hepsi varlığı doğrulandı, 2026-11-13)
- Liu, Mao, Sha, Yuille. Attention Correctness in Neural Image Captioning. AAAI 2017.
- Zhou ve ark. More Grounded Image Captioning by Distilling Image-Text Matching Model. CVPR 2020.
- Li ve ark. Align before Fuse (ALBEF). NeurIPS 2021.
- Chefer, Gur, Wolf. Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers. ICCV 2021.
- Abnar, Zuidema. Quantifying Attention Flow in Transformers. ACL 2020.
- Darcet ve ark. Vision Transformers Need Registers. ICLR 2024.
- Kang ve ark. See What You Are Told: Visual Attention Sink in Large Multimodal Models. ICLR 2025.
- Kang ve ark. Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding. CVPR 2025.
- Zhang, Khayatkhoei, Chhikara, Ilievski. MLLMs Know Where to Look. ICLR 2025.
- Luo ve ark. Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf VLMs (PnP-OVSS). CVPR 2024.
- Jacovi ve ark. Contrastive Explanations for Model Interpretability. EMNLP 2021.
- Zhang ve ark. Top-down Neural Attention by Excitation Backprop. ECCV 2016.
- Wang ve ark. Score-CAM. CVPR Workshops 2020.
- Zhou ve ark. Learning Deep Features for Discriminative Localization (CAM). CVPR 2016.
- Plummer ve ark. Flickr30k Entities. ICCV 2015.
- Kirillov ve ark. Segment Anything. ICCV 2023, s. 3992–4003 (Crossref ile doğrulandı, 2026-11-14).
