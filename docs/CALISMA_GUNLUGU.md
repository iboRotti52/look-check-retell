# Çalışma Günlüğü — Look–Check–Retell

Bu dosya projenin hafızasıdır: ne yaptık, neden yaptık, şu an neredeyiz, sırada ne var.
Her oturumun (Claude, Codex veya ekipten biri) sonunda güncellenir. En yeni kayıt en üstte değil,
**"Günlük"** bölümünün sonundadır; "Şu anki durum" ve "Sıradaki adımlar" her zaman güncel tutulur.

Ayrıntılı geçmiş: `docs/handoff/transcript_main.md` (ilk büyük oturumun dökümü).

---

## Şu anki durum (2026-11-15)

**YENİ VARSAYILAN CHECK: S4** (`docs/CHECK_S4.md`, protokol `docs/CHECK_S4_PROTOKOL.md`, tablo
`docs/generated/CHECK_S4_B.md`). Rakip-farkında OWLv2 kanıtı, maliyete duyarlı sınıf eşikleri (tek λ), Look top-1
kutusunda merkezli ikame, temizlenmiş karışma kümeleri ve birleşik kelime atlama. B dış-kat, λ = 2.5: thing-FCR 0.030
(S2 0.117), giysi FCR 0.041 (S2 0.320), temiz dışlayan swap yakalama 0.808 (S2 0.552), ikame kesinliği 0.912 (S2 0.689).
Ön kayıtlı üç ölçüt tuttu. Bundle `results/lcr_check4_bundle.json` (`c7156af83dcca3b4`), kayıtlar `A_B_lcr4.json`,
`39 --live` geçti. **Uyarı:** bileşenler Fable tarafından aynı B katlarında keşfedildi; bağımsız doğrulama validation'da.
S2 (`lcr_check_bundle.json`) ve S3 kıyas satırı olarak duruyor.

**Fable'ın Check düzeltmesi denendi; ön kayıtlı ölçüt tutmadı, varsayılan değişmedi:** `docs/CHECK_DUZELTME.md`
(protokol `docs/CHECK_DUZELTME_PROTOKOL.md`, tablolar `docs/generated/CHECK_FIX_B.md` ve sonradan ablation
`CHECK_FIX_ABLATION_B.md`). Pozitif-çapalı Check (S3: ham OWLv2 logit, sınıf eşiği temiz pozitiflerden, çekimser
sınıflar, karışma kümesi ikamesi, FCR ≤ %5) FCR'yi 0.085 → 0.041, giysi FCR'sini 0.32 → 0.05 indirdi. Ama eşleşmiş
FCR'de sentetik swap'ların 0.388'ini yakaladı (S2: 0.559). Sonradan tanı: suçlu "az pozitif" çekimserliği (swap'lar
nadir sınıflara düşüyor); çekimsersiz S3 eşleşmiş FCR'de S2'ye eşit/biraz iyi, giyside açıkça iyi, yaygın sınıfa
kayan yanlışlarda (yüksek-taban) kötü. Takas gerçek; hangisinin varsayılan olacağına insan pilotu karar vermeli.
Varsayılan hâlâ S2 `results/lcr_check_bundle.json`; S3 `results/lcr_check3_bundle.json`, kayıtlar `A_B_lcr3.json`.

**Fable ikinci inceleme (Check nasıl iyileşir):** `docs/FABLE_INCELEME_2.md`, betikler `scripts/review_fable2_B/`.
Swap testinin %52'si birbirini dışlamayan etiket (çoğu görüntüde gerçekten var), %31'i kolay köpek swap'ı — başlık sayısı
yanıltıcı (Claude bu iki oranı ve köpek eşik dağılımını yeniden hesaplayıp doğruladı). S3 tek α yüzünden bütçeyi
kolay sınıflara harcıyor; claim tek başına skorlanıyor. Önerilen S4 (maliyete duyarlı sınıf eşikleri + rakip-farkında
kanıt + Look top-1'de merkezli ikame + karışma kümesi/vocab temizliği): thing-FCR ≤ 0.05'te temiz dışlayan swap'larda
yakalama 0.70 → 0.87, giyside 0.28 → 0.58, ikame kesinliği 0.89 (Look olmadan 0.71). Giysi detektörle sınırlı
(AUROC shirt 0.71, sweater 0.53); CPU kırpma pilotu kazanç vermedi → GPU kırpma pilotu bu haliyle gereksiz.
Hepsi keşif sayısı (aynı B katları); S4 ön kayıtla koşulmalı.

**Fable bağımsız yöntem incelemesi (yalnız B):** `docs/FABLE_INCELEME.md`. En önemli bulgu (Claude yeniden hesaplayıp
doğruladı): Check'in sınıf başına z ölçeği giysi gibi yaygın sınıflarda yanlış normalizasyon; referansla doğrulanmış
giyim claim'lerinin %32.5'i contradicted. Look tavana yakın, geometri doğru. Düzeltme denemesi yukarıda.


**Retell kaynak araştırması (Codex):** `docs/handoff/RETELL_ARASTIRMA_NOTLARI.md`. Kullanıcının isteği
yalnız ilgili çalışmalar bulmak ve düşünmek; yol haritası veya uygulama hazırlamak değil. Güncel Retell
kodunun kanıt arayüzü ve literatürün getirdiği fikirler/sınırlar incelendi. Yeni yöntem seçimi veya eğitim yok.

**Check tamamlandı (mantık içinde; ölçüm insan pilotunu bekliyor):** `docs/CHECK_TAMAMLAMA.md`, protokol
`docs/CHECK_TAMAMLAMA_PROTOKOL.md`, tablo `docs/generated/CHECK_COMPLETE_B.md`. Codex'in ortak `RegionChecker`'ına beş
eylem eklendi (keep / keep_elsewhere / replace / remove / skip): görüntü düzeyi veto, contradicted'ın replace/remove
ayrımı, "camera" kalıbı, tüm B'de çapraz ayar. Yeni varsayılan bundle `results/lcr_check_bundle.json`
(`15fdb954e7de71c6`), kayıtlar `A_B_lcr2.json`; Codex'in bundle'ı tarihsel. Proxy J veto yüzünden düştü (önceden
yazıldığı gibi). Clothing'de `replace` güvenilmez (sweater→suit gibi). Doğrulama `--live` dahil geçti. Validation okunmadı.

## Şu anki durum (2026-11-14, tarihsel)

**Claude ilk ilkeler değerlendirmesi ve Codex LCR incelemesi:** `docs/ILK_ILKELER_TASARIM.md`
(tanı `docs/generated/FIRST_PRINCIPLES_B.md`, yalnız B). Codex'in küçük kutu filtresi düzeltmesi ve v4→Retell
sözleşme hatası tespiti doğru. Asıl darboğaz etiket: proxy-yanlış claim'lerin Claude görsel denetiminde çoğu
doğru çıktı (rastgele 12'den 6 doğru, 2 "kamera" kalıbı, 4 belirsiz; insan etiketi değil). Yanlışların yarısında
referansta ikame yok (silme gerekir). Öneri: insan pilotu önce; Check'e görüntü düzeyi veto + `remove` sonucu;
Retell R0 = aynı captioner ile kısıtlı yeniden üretim (yasakla / yumuşak ikame / hatadan sonrasını yeniden üret).

**Güncel tek yöntem `docs/LCR.md`: önce Look, sonra Check. Retell bekliyor.** Kullanıcı ağır model
hesaplamasına gelince Claude'a devredecek; Codex GPU işi veya Retell eğitimi başlatmadı. Colab şu an
kullanılmıyor. Hazır devir: `docs/handoff/CLAUDE_LOOK_CHECK_GPU.md`.

Look'un seçili dört attention başlığı/haritası korunuyor. B_fit insan kutularında küçük kutu filtresi
kaldırıldı, alan kuvveti 0.5 kaldı. İlk üç adayın kapsaması ve küçük nesne sonucu otomatik tabloda:
`docs/generated/LCR_DEVELOPMENT.md`. Sınıfın detektör skoru Look seçimine girmez; kişi SAM tercihi korunur.

Check aynı üç adayı ayrı skorlar; sınıf başına max ile tek claim kararı oluşturur. Eşik/ölçek yalnız
B_cal'da fit edilir; dil önceliği, VQA/ITM, captioner tutarlılığı yok. Ortak hesaplama
`lcr/region_evidence.py`; canlı `pipeline.run` ve script 04 varsayılan olarak `lcr` üretir.
Kimlikli ayar `results/lcr_bundle.json`, B kaydı `data/cache/pipeline/A_B_lcr.json`.

B kayıtlarının skor/kutu/alternatif uyumu ve iki sabit taslağın yeniden hesaplanan canlı eşitliği
`results/lcr_contract.json` ile doğrulandı. Eski v4 kararlarının eski ham kuralla yeniden yazılması
engellendi. Yeni kanıt arayüzü hazırlığı var; ana LCR ile script 06 Retell eğitimini özellikle reddeder.

**Check güvenilirlik kapısı tamamen kapanmadı.** Alternatif nesnenin görünmesi doğru ikame olduğunu
kanıtlamıyor; referans desteği otomatik proxy. Sıradaki ağır hesap: aynı donmuş OWLv2'yi sabit attention
kırpımlarında yeniden çalıştırma kıyası. Sabit B pilotu, CUDA zorunlu çıkarıcı ve Colab defteri hazır;
GPU sonucu henüz yok. Eski validation sonuçları yeni yolun başarısı gibi sunulmaz; test açılmadı.

Önceki durum/adımlar kayıpsız olarak `docs/handoff/BIRLESTIRME_ONCESI_DURUM.md` içinde saklandı.
Aşağıdaki gece notları ve eski deney sayıları tarihseldir.

### 2026-11-13 gece notları (tarihsel)

Neden Look'tan başladık: 1. tur deneyler, zincirin ilk halkası Look bozuk olduğu için Check ve Retell'in de çalışmadığını gösterdi
(sayılar aşağıda). Ekip kararı: önce Look'u ölçülebilir şekilde düzelt, sonra Check, sonra Retell,
hipotez testleri en sonda.

Elimizde olanlar:
- Tüm pipeline kodu (`lcr/`, `scripts/`), 33 birim testi geçiyor.
- Colab'da 1. tur sonuçları: fine-tune, draft'lar, Look/Check kayıtları, Retell + B0+/B2/B3 kontrolleri.
- **Flickr30k Entities kutuları** `data/flickr30k_entities/` (Annotations/ + Sentences/). Flickr8k'nin 8.000
  görüntüsünün **tamamı** kapsanıyor, görüntü boyutları birebir aynı, ve F30kE cümleleri bizim Flickr8k
  referans caption'larımızla aynı. Yani isim öbeği → insan çizimi kutu eşlemesi doğrudan bizim veriye uyuyor.
  **Yalnızca Look'u ölçmek için kullanılır.** Test split'inin kutularına geliştirme sırasında bakılmaz.

İnsan kutularıyla ilk ölçüm (validation, 1. tur ham attention, `scripts/13_look_benchmark.py`):
pointing %23.9 (şans %37.1), attention'ın seçtiği kutu doğru %12.9; detektör kutusu %82.7; OWLv2 adayları
nesneyi %95.3 oranında içeriyor. → Sorun haritada ve seçimde, aday kutularda değil. Ayrıntı: `docs/LOOK_V2_PLAN.md` §7.

Colab durumu (oturum kapanırken, Colab saati 20:37 UTC):
- `colab/run_sink.sh`: `drafts_val_sink`, `drafts_B_sink`, `ours_sink`, `attn_variants` **bitti**; tam metrikler
  (`colab_full_metrics.py`, SPICE yavaş) hâlâ çalışıyordu; bitince `/content/lcr_results.zip` oluşacak.
- `/content/lcr_look.zip` (16 MB) hazır: sink pipeline dosyaları + haritalar (validation ve B), `A_B.json`,
  `results/*.json`, loglar, `report/generated`, `report/figures`. (İndirildi ve ölçüldü, 2026-11-13.)
  İndirilince ayrı bir klasöre açılıp ilgili dosyalar projeye kopyalanacak, sonra
  `python scripts/13_look_benchmark.py --pipe A_validation_sink` çalıştırılacak.
- `ours_sink` (Retell, sink attention, 1. tur ayarları): masked-evidence hassasiyeti %10.9, false %0.0.
- `run_fix.sh` öldürülmeden önce yalnızca iki eşik kalibrasyonunu (script 11) çalıştırabildi
  (`results/thresholds_proxy_att_sink.json`, `results/thresholds_proxy_det.json`); hiçbir Retell eğitimi başlamadı.
- `ours_rho0` koşusu **düştü**: Colab'daki eski `run_all.sh` `--attn` bayrağı vermiyor, yeni `06_train_retell.py`
  varsayılanı `sink`. Yeniden koşmak için: `python scripts/06_train_retell.py --mode ours --rho 0 --tag rho0 --attn raw`.
- `colab/run_fix.sh` (Retell 2. tur: kalibre eşik + patch dropout + eşleşik kontroller) **iptal edildi** —
  Look düzelmeden Retell ayarlamak sırayı atlamak olur. Script duruyor ama Look v2 bitmeden çalıştırılmamalı.

---

## Sıradaki adımlar (sırayla)

15 Kasım Retell araştırması bu sıraya uygulama işi eklemedi; araştırma notu bir yol haritası değildir.

0. **İnsan pilotu:** ~300–400 B claim, tabakalı. S4 (`A_B_lcr4.json`) contradicted'ların hepsi, S2 contradicted / S4
   keep (özellikle giysi), replace'ler, keep_elsewhere ve taban oranı için rastgele keep örneği. Sonuç: gerçek hata
   oranı → FCR hedefi (λ), S4 kararlarının gerçek kesinliği.
   Araç `annotation/pilot/annotate.html`. Seçim pilotun etiketiyle yapılır; swap/FCR'ye bakarak yeniden seçim yok.
0b. **(İsteğe bağlı, GPU) Daha büyük detektör: OWLv2-large.** Şu an `google/owlv2-base-patch16-ensemble` kullanılıyor
   (`lcr/config.py`). Giysi türünde (shirt/jacket/sweater) ayırt etme detektörle sınırlı (aynı-sınıf AUROC shirt 0.71,
   sweater 0.53; kırpıp yeniden bakmak kazandırmadı, `docs/FABLE_INCELEME_2.md`). Aynı ailenin large sürümü mantığın
   içinde kalır (yine donmuş OWLv2) ama öneri önbelleğinin Colab'da yeniden çıkarılmasını ve Check'in yeniden
   kalibrasyonunu gerektirir. Kazanç belirsiz; önce küçük bir B pilotunda giysi aynı-sınıf AUROC'u kıyaslanmalı.
1. **Claude'a GPU Check pilotu devri:** `docs/handoff/CLAUDE_LOOK_CHECK_GPU.md`,
   `output/lcr_check_gpu_pilot.zip`, `colab/LOOK_CHECK_GPU_PILOT.ipynb`. M4'te model çalıştırma;
   aynı OWLv2 için Colab CUDA kullan. Crop geometrileri ve pilot sabit; referans etiketi pakette yok.
2. **Önce Look/Check:** GPU sonucu geldiğinde aynı B claim'lerinde eski öneri özellikleri ile crop içi
   kanıtı karşılaştır. Yeni skor dağılımını eski eşiklere doğrudan koyma; B protokolü/kalibrasyon gerekir.
   İkame kalitesi, yanlış alarm, belirsizlik ve insan pilotu açıklarını ayrı yaz.
3. **Retell'e geçme.** Kullanıcının sonraki yönlendirmesine kadar eğitim engeli yerinde kalır.
4. **Validation/test:** validation yeni yöntemin seçimi için tekrar kullanılmaz. Test yalnız önceden
   sabitlenmiş final değerlendirmede bir kez. Eski sonuçlar tarihsel olarak korunur.

---

## Önemli sayılar (1. tur, tarihsel; validation, automatic proxy)

Captioner:
- BLIP zero-shot CIDEr 0.669 → A'da fine-tune (decoder) 0.979 (1. epoch en iyisi). B0 = bu checkpoint.
- 1000 görüntü, 2.272 draft iddiası (2.27/görüntü). Referans desteği %84.2 (person %99.1, thing %75.2, stuff %79.7).

Look (1. tur, ham attention, son 4 katman × tüm başlıklar ortalaması, en yüksek "excess mass" kutusu):
- Attention bölgesinin claim skoru, referansta geçen/geçmeyen thing/stuff iddiaları ayırmada **AUROC 0.51**
  (şans). Detektörün kendi kutusu (B3) 0.74; tüm görüntü max skoru 0.74. (`scripts/12_look_variants.py`)
- Attention'ın en güçlü k kutusu içinde en iyi eşleşme: top-3 0.58, top-10 0.64 — yine detektörün altında.
- Sink düzeltmesi (yerel, zero-shot decoder, 200 görüntü, hedef = detektör kutusu — proxy):
  pointing %35.0 → %73.4 (şans %40.2). A decoder'ı ve 1000 görüntüyle tekrar Colab'da koşuyor.

Check (1. tur, geçici eşikler τs=0.2, τc=0.05):
- Verdict: supported %9.5, contradicted %30.3, uncertain %60.2.
- Bilgi değeri J (referans proxy'sine göre): ham attention bölgesi **0.01**, detektör bölgesi 0.31.
- B1 (contradicted kelimeyi alternatifle değiştir): 24 düzeltme / 409 bozma → eşikler kullanılamaz.

Retell (1. tur):
- Kapılar neredeyse kapalı kaldı: attn ≈ 0.001–0.002, ff ≈ 0.02. Yanlış kanıtla çıktı gerçek kanıtla **aynı**.
- Eğitimde verdict dağılımı: contradicted %31, uncertain %60, supported %8 → kanıt büyük ölçüde gürültü.
- Teşhis: (1) Look kötü bölge seçiyor → kanıt gürültü; (2) decoder görüntüyü zaten tam görüyor → kanıt gereksiz.

Güç analizi: H1 için bağımsız tasarımda p0=0.08'de 7.469 iddia gerekir; eşleştirilmiş (paired) tasarımda
1.000 iddia ≈ %80 güç. Draft hata oranı %8'in çok altındaysa H1 zayıf kalır.

---

## Kararlar ve gerekçeleri (kısa kayıt)

| Tarih | Karar | Gerekçe |
|---|---|---|
| 11-13 | Yalnızca BLIP decoder'ı fine-tune, encoder donuk | Mac/Colab bütçesi; görüntü özellikleri önbelleğe alınabiliyor |
| 11-13 | GPU işleri Colab A100'de | Mac MPS'te fine-tune 14 GB+ kullanıp swap'a düştü |
| 11-13 | Sink-düzeltmeli attention | Ham BLIP attention'ı kelimeden bağımsız birkaç arka plan yamasına yığılıyor (Darcet+ ICLR'24; Kang+ ICLR'25) |
| 11-13 | Eşikler B yarısında, referans proxy'sine göre ara kalibrasyon (script 11, Youden J) | Geçici eşikler her şeyi "contradicted" yapıyordu; insan etiketleri gelene kadar ara çözüm |
| 11-13 | Retell 2. turu iptal, önce Look | Kanıt gürültüyken Retell'i ayarlamak anlamsız; zincir sırayla düzeltilecek |
| 11-13 | Look'u F30kE insan kutularıyla ölç | Detektör kutusu da bir tahmin; F30kE tüm Flickr8k görüntülerini kapsıyor |
| 11-13 | Look v2: literatürdeki yöntemler birleştirilecek + kendi eklemelerimiz, hepsi ablation ile | Ekip kararı; tek yönteme kilitlenmeden, her bileşenin katkısı ölçülerek |
| 11-13 | Her işlemin bir "kapısı" var (ara aşamayı bağımsız bir kaynağa karşı ölçen test); kapı geçilmeden sonraki işleme geçilmez | Round-1 hatası uçtan uca metrikle saklandı; bkz. `docs/LITERATUR_TARAMASI.md` §0 |

---

## Bu oturumda eklenen/değişen dosyalar (2026-11-13)

- `lcr/look.py`: `attention_map(..., sink=True)` sink düzeltmesi.
- `lcr/pipeline.py`: `sink` seçeneği; `thresholds()` artık `LCR_THRESHOLDS` ortam değişkeniyle başka eşik dosyası okuyabiliyor.
- `lcr/captioner.py`: `decoder_forward(..., img_mask=None)` (patch dropout için).
- `scripts/06_train_retell.py`: `--attn <ek>` (raw | sink | sink_top3 …), `--img-drop`, `--img-keep`, kanıt hassasiyeti ölçümü (`sensitivity`).
- `scripts/08_analyse.py`: Check kalite tablosu (bölge kaynağı × eşik), B1 varyantları, Retell iki tur tablosu.
- `scripts/09_attention_variants.py`: ham vs sink attention (proxy hedef = detektör kutusu).
- `scripts/10_score_labels.py`: insan etiketlerinden HICR, κ, eşik kalibrasyonu.
- `scripts/11_proxy_thresholds.py`: B'de referans proxy'sine göre ara eşik kalibrasyonu.
- `scripts/12_look_variants.py`: kayıtlı haritalardan GPU'suz Look varyantları ve AUROC.
- `colab/run_all.sh`, `colab/run_sink.sh`, `colab/run_fix.sh` (sonuncusu şimdilik kullanılmayacak).
- `report/`: `main.tex`, bölüm 1–5, figürler; `4_experiments.tex` içinde doldurulacak `RETELLPARAGRAPH` yer tutucusu var.
- `AGENTS.md`, `CLAUDE.md`, bu dosya, `docs/LOOK_V2_PLAN.md`.
- `data/flickr30k_entities/` (indirildi; 29 MB zip, sadece ölçüm için).

---

## Günlük

### 2026-11-14 — Check v4: özgün mantığa sadık Check (Claude)
- Kullanıcı: "check yapan herhangi bir şeyle değiştirme; asıl mantıkta nasıl en iyi olur onu araştır." v3 belgeleri
  "yalnız karşılaştırma" diye işaretlendi; Look/Check kodu özgün tasarıma göre denetlendi (Look sadık).
- Yeni: `docs/CHECK_V4_PROTOKOL.md` (önceden yazıldı; §5 eşik politikası değişikliği seçimden önce),
  `scripts/35_check_v4.py`, `scripts/36_check_v4_table.py`, `tests/test_check_v4.py`, `docs/CHECK_V4.md`,
  `docs/generated/CHECK_V4_METRICS.md`, `docs/handoff/CODEX_CHECK_V4_REVIEW.md`.
- Sonuçlar: `results/check_v4_selection.json` (B kilit, Q1 C2 A1 R2 G0 E0), `results/check_v4.json` (validation bir
  kez), kayıtlar `data/cache/pipeline/A_{B,validation}_look4_check4.json`. Loglar `results/logs/check_v4_*.log`.
- Güncellenen: README (35/36), AGENTS.md, LITERATUR §8c, bu dosya. Testler 74 geçti.

### 2026-11-14 — Check v3 (Claude)
- Yeni: `scripts/31_check_vlm_signals.py`, `32_check_consistency.py`, `33_check_v3.py`, `34_check_v3_table.py`,
  `colab/run_check3.sh` (+ `run_check3_cons.sh` yeniden deneme), `tests/test_check_v3.py`, `docs/CHECK_V3.md`,
  `docs/generated/CHECK_V3_METRICS.md`, `docs/handoff/CODEX_CHECK_V3_REVIEW.md`.
- Sonuçlar: `results/check_v3_selection.json` (B kilit), `results/check_v3.json` (validation), kayıtlar
  `data/cache/pipeline/A_{B,validation}_look4_{vlm,cons,check3}.json`.
- Yerelde 4 ağır iş Mac'i dondurdu; iş Colab T4'e taşındı (A100 gereksiz). Cons `--bs 64` T4'te bellek taşırdı → 16.
- Güncellenen: README, AGENTS.md, LITERATUR §8b + doğrulanmış kaynaklar, `report/refs.bib` (SelfCheckGPT, POPE doi).

### 2026-11-13
- Proje önerisi kabul edildi; progress report istendi (IEEE, sayfa sınırı yok).
- Kod tabanı ve rapor iskeleti kuruldu (Fable 5.2, sonra Opus). Mac'te fine-tune bellek yüzünden durduruldu → Colab A100.
- Colab 1. tur: fine-tune, draft/Look/Check, Retell + kontroller. Sonuçlar yukarıda.
- Teşhis: Retell kanıtı yok sayıyor; kök neden Look (attention bölgesi şans düzeyinde).
- Literatür taraması (Look için): ALBEF/BLIP Grad-CAM, PnP-OVSS, attention sink (Darcet+; Kang+ ICLR'25),
  localization heads (Kang+ CVPR'25), "MLLMs know where to look" (ICLR'25), Chefer+ ICCV'21,
  captioner attention'ı zayıf (Liu+ AAAI'17; Zhou+ CVPR'20). Ayrıntı ve plan: `docs/LOOK_V2_PLAN.md`.
- Flickr30k Entities indirildi; Flickr8k'nin tamamını kapsadığı ve boyutların eşleştiği doğrulandı.
- Colab'da Retell 2. turu iptal edildi; sink aşaması tamamlanmak üzere bırakıldı.
- `AGENTS.md`, `CLAUDE.md`, bu günlük ve `docs/LOOK_V2_PLAN.md` eklendi (ekip Codex ile de devam edecek).
- `lcr/grounding.py` (F30kE okuyucu, sınıf eşleme, kapsama maskesi) + `tests/test_grounding.py` (3 test) +
  `scripts/13_look_benchmark.py` (pointing, enerji, bölge IoU, rastgele/oracle referanslar, örnek figür).
- Taban çizgisi ölçüldü (yukarıda). Sıradaki: sink haritalarını ölç, sonra Look v2 çıkarım geçişi.
- **İşlem bazlı literatür taraması** yazıldı: `docs/LITERATUR_TARAMASI.md`. 17 işlem (veri bölme, decoding,
  claim çıkarma, proxy, Look, detektör skoru, karar kuralı, eşik, bozma, kanıt girişi, eğitim, yeniden üretim,
  metrikler, hallucination metrikleri, insan değerlendirmesi, istatistik, müdahaleler). Her biri için şu an /
  literatür / çıkarım / kapı, ~45 doğrulanmış kaynak. Kod değişmedi; yeni koşu yok.
  Önemli bulgular: OWLv2 sorgular arası skor karşılaştırması (Check kuralı), eşik kalibrasyonu ve Retell
  eğitiminin ikisi de B'yi kullanıyor (alt küme ayırma önerisi), referans hedefi kanıtı kullanmaya doğrudan
  teşvik vermiyor (kontrastif yardımcı kayıp önerisi). Neden önceden fark edilmediği: §0.
- `lcr_look.zip` indirildi (kullanıcı onayıyla), `data/cache/pipeline` ve `results/` içine açıldı.
  Eski `results/` ve `report/generated/` scratchpad'e yedeklendi.
- **Sink haritaları insan kutularıyla ölçüldü (validation):** pointing **%88.3** (ham 23.9, şans 37.1), enerji 73.2,
  bölge IoU≥0.5 %55.7 (detektör 82.7, oracle 95.3). Kapılar (`scripts/14_look_sanity.py`): transpoze harita
  51.9'a düşüyor (yönelim doğru). Takas testinde kendi haritası 87.5, başka claim'in haritası 45.4 (harita kelimeye
  özgü). Sonuç: harita büyük ölçüde düzeldi; **darboğaz artık haritadan kutuya geçiş**. Ayrıntı: LOOK_V2_PLAN §7.2.
- Kullanıcının sorusu üzerine ("attention'a bakma işlemi de var mı, bir şey atladık mı") taramaya iki işlem
  eklendi: 5a attention'ı okuma (zaman adımı, alt-kelime, öbek, koordinat, özgüllük) ve 5b alternatif aday
  üretimi. 5b ölçüldü (`scripts/15_alt_quality.py`): yanlış claim'lerde 1. alternatif %36.6, ilk 3'ten biri
  %58.2 oranında referansta geçiyor.
- B yarısının OWLv2 önbelleği (`data/cache/owl/train.npz`) yerelde yok. Haritadan kutuya seçimini B'de yapmak
  için Colab'dan indirilmesi gerekiyor. (2026-11-14: indirildi, aşağıda.)

### 2026-11-14
- Kullanıcı tüm izinleri verdi ("sor ma"). Proje işleri için onay sorulmadan ilerleniyor (güvenlik yasakları hariç).
- B'nin OWLv2 önbelleği indirildi (`data/cache/owl/train.npz`, 623 MB, Colab'dan).
- `scripts/16_look_regions.py`: 72 haritadan-bölgeye konfigürasyonu, B'de seçim → **ratio α=0.5 + objectness γ=0.5**.
  Validation'da bölge %75.3 (R1 55.7, detektör 82.7, oracle 95.3). İlk sürümdeki AUROC şişikti (F30kE alt kümesinde
  negatif yok), tüm claim'lerle düzeltildi: 0.740 vs detektör 0.744.
- Look v2 donduruldu: `lcr/look.py` (`region_score`, `select_region`), `lcr/config.py` (`REGION_ALPHA/GAMMA`),
  `lcr/pipeline.py` (yeni kural, `look="v2"`, v2 kayıtlarında excess kapısı yok), `tests/test_look.py` (+1 test, 33 geçiyor).
  `scripts/17_freeze_look.py` ile `A_B_look2.json` ve `A_validation_look2.json` yazıldı. `scripts/11 --attn look2`:
  B'de J 0.461 (`results/thresholds_proxy_att_look2.json`).
- Kullanıcının sorusu üzerine Check ve Retell de kod satır satır okunarak gizli kararlara ayrıldı
  (LITERATUR_TARAMASI 8a ve 12a). Önemli bulgular:
  - `s_image` (detektörün tüm-görüntü skoru) "ours" kanıtına da giriyor, H6'da karıştırıcı.
  - Gate α'larına weight decay uygulanıyor, kapalı gate'lerin ek bir nedeni olabilir.
  - Retell epoch'u validation CIDEr'e göre seçiliyor (seçim ve rapor aynı split'te, düzeltilecek).
  - Check'te alternatif, detektör skoruna göre sınıflar arası karşılaştırmayla seçiliyor.
- Kullanıcı kararı: Look bir süre **Codex ile** ilerletilecek, sonra Check. Görev tanımı yazıldı:
  `docs/handoff/CODEX_LOOK_GOREVI.md`. İçeriği: T1 person hata analizi ve türe özel kutu kuralı, T2 harita son
  işleme, T3 Grad-CAM, kontrastif ve 5a ablation'ları için yeni çıkarım, T4 başlık seçimi, T5 seçim ve dondurma.
  AGENTS.md bu dosyaya yönlendiriyor.
- Fine-tune edilmiş decoder `data/cache/ckpt/decoder_A.pt` (552 MB) Colab'dan indirildi ve yerelde yüklendiği
  doğrulandı. Yeni attention/gradyan çıkarımı artık yerelde (CPU) de yapılabilir.

### 2026-11-14 — Codex Look v2 incelemesi
- Kullanıcının güncel isteği doğrultusunda Look tasarımı, uygulama, ölçüm scriptleri ve birincil literatür incelendi.
- `scripts/review_look_v2.py` eklendi ve yalnızca B'de çalıştırıldı. Çıktılar:
  `results/look_v2_review_B.json`, `results/look_v2_person_review_B.png`,
  `results/look_v2_person_review_examples_B.json`. Dondurulmuş seçimler kutulu örneklerde birebir yeniden üretildi.
- Attention katkısı ayrıştırıldı: tam yöntem %74.64, objectness-only %35.95, uniform-v2 %39.61;
  aynı örneklerde kendi/başka sınıf haritasıyla kutu doğruluğu %74.65 / %22.12.
- T1'in tanısal kısmı tamamlandı: kişi hatalarında küçük parça ve fazla geniş kutular birlikte önemli.
  Türe özel kural seçimi, yeni harita çıkarımı ve v3 dondurma bu incelemede yapılmadı.
- Freeze yardımcı alanlarının eski kalması, script 04 çıktı sürümü riski, coverage-mask birleşim yaklaşımı,
  AUROC eşit skor sıralaması ve raporun eski yöntem tanımı kaydedildi. Üretim kodu değiştirilmedi.
- Araştırma notu: `docs/LOOK_V2_INCELEME.md`; LOOK_V2_PLAN §7.5 ve güncel durum/öncelikler güncellendi.
  Başlık seçiminin BLIP'e doğrudan taşınamayacağı ve post-hoc sink haritasının nedensel kanıt olmadığı belirtildi.
- Mevcut CPU testleri: 33 geçti. Yeni validation koşusu, test verisi erişimi ve Check/Retell eğitimi yok.

### 2026-11-14 — Look tutarlılık ve ölçüm düzeltmeleri
- Kullanıcı gerekli düzeltmeleri istedi. `lcr/pipeline.py`, script 04 ve 17: harita/kutu sürüm metadata'sı,
  v1/v2 için ayrı çıktı adları, üstüne yazma koruması ve limitli koşulara ayrı ad. Eski Colab akışları
  v1'i açıkça seçiyor. Corruption sırasında `look` etiketi korunuyor; v2'ye yanlış v1 kapısı uygulanmıyor.
- Script 17 ile B/validation kayıtları yenilendi; k=4 kutu/excess ve attention–detector IoU birlikte
  hesaplandı. Haritası olmayan k=1/12 eski verileri `legacy_look` içinde korundu.
- `grounding.coverage_mask`: kutuların geometrik birleşimini tam hesaplıyor. Script 16 ortak, eşit
  skorlara yarım puan veren AUROC'u kullanıyor; uygun kutu yoksa sahte bir indeks seçilmiyor.
- Script 13/14/16 ve B incelemesi tekrar çalıştırıldı; yeni yöntem/ayar eklenmedi. Validation kutu
  doğruluğu %75.2646 ve pointing %88.3008 aynı. Enerji %73.1512 → %73.2064; şans %37.1340 → %37.1728.
  Mevcut sonuçlarda AUROC değişmedi; kullanılan hesap genel eşit skor durumları için düzeltildi.
- `scripts/verify_look_artifacts.py` ve `results/look_consistency_fix.json`: 6.856 B + 2.272 validation
  claim'inde sıfır kutu/skor değişimi ve sıfır eski yardımcı alan doğrulandı. B'de seçilmiş kural aynı.
  Harita yönelim/özgüllük kapıları geçiyor. Önceki çıktılar `archive/pre_consistency_fix` altında korundu.
- `scripts/look_report.py`: sonuç JSON'larından insan-kutusu tabloları ve sayı makroları. Rapor tasarımı,
  deneyler ve sonuç anlatısı güncellendi; eski Check/Retell sonuçları yeni Look'a mal edilmedi.
- Script 08 tam COCO dosyası yokken gerçek kayıtlı BLEU/CIDEr'i kullanıyor ve eksik metrikleri açıkça
  eksik gösteriyor. Derlemeyi engelleyen 23 BibTeX satır içi yorumunun sözdizimi düzeltildi.
- Regresyonlar: kutu birleşimi, eşit skorlar, idempotent kayıt göçü, çıktı koruması ve look etiketi;
  CPU'da toplam 43 test geçti. Test split'i okunmadı; yeni eğitim veya Look v3 dondurma yapılmadı.
- Rapor ve Look v2 şeması derlendi; on sayfanın tamamı görsel olarak kontrol edildi. Taşan iki eski
  tablo düzeltildi; son derlemede taşma ve çözümlenmemiş referans uyarısı yok. Teslim PDF'i:
  `output/pdf/CS415_Look_Check_Retell_Progress_Report.pdf` (kaynak derleme: `report/main.pdf`).

- Claude, Codex'in incelemesini ve düzeltmelerini doğruladı: 43 test, v2 kutuları değişmemiş, validation %75.3.
  Codex'e yön dosyası yazıldı: `docs/handoff/CODEX_SONRAKI_YON.md` (E1, E2, Look'u kapatma, Check paketi C0–C5).

### 2026-11-14 — E1 tamamlandı
- `scripts/18_look_e1.py`: yön dosyasındaki sabit 36 konfigürasyon/tür, seçim yalnızca B'de.
  Önceki γ=2/peak içeren keşif E1'e alınmadı; ızgara genişletilmedi.
- B'de ≥1 puan kuralı: person (α=.5, γ=.5, σ=1), stuff (α=.5, γ=.25, σ=0) kabul; thing v2 kaldı.
- Validation: %75.2646 → %76.1003; görüntü düzeyinde eşleştirilmiş bootstrap fark GA95
  [+0.1614, +1.5514] puan. Person farkı +0.7884 puan (GA sıfırı içeriyor); stuff +3.2609; thing 0.
- `results/look_e1.json` seçim, tam grid, sonuç, 2.000 bootstrap ve B/validation harita kapılarını içeriyor.
  Yönelim ve özgüllük kontrolleri geçti. Person yumuşatmasının enerji kaybı da sonuç dosyasında raporlu.
- Değişen kaynaklar: `lcr/look_eval.py`, script 18; F30kE fotoğraf ID okuyucusu görüntü baytları yerine
  yalnızca parquet yol sütununu okuyarak deneylerin bellek yükünü azaltıyor.
- E2: B'den sabit tohumla 500 görüntü, belirlenen kaynaklar ve özellik düzeyi maskeleme sırada.

- **Check v2 (Claude).** Kullanıcı: "check'i önce sen yap, sonra Codex review'lasın"; "C'lerle sınırlı kalma".
  - `lcr/calibrate.py` ve scriptler 20–23 eklendi, `tests/test_calibrate.py` (+5 test, toplam 48 geçiyor).
  - Kalibrasyon: AUROC 0.740 → 0.786, ECE 0.418 → 0.03. J-optimal nokta yozlaşık (claim'lerin %49'u contradicted),
    karar noktası kesinlik hedefiyle seçiliyor.
  - Öğrenilmiş doğrulayıcı (MetaToken/HaloProbe tarzı): prior + det + cap, validation AUROC 0.829.
  - Bulgular: sınıf önceliği tek başına 0.763 (görüntüsüz). Look bölgesi ≈ detektör bölgesi ≈ görüntü düzeyi
    (0.801 / 0.797 / 0.799), H6 ön sonucu. Look sinyalleri katkı vermiyor. Sorgu şablonlarının etkisi yok.
  - `finalise` v2 kararını koruyor. ρ bozması v2 ile engellendi (yöntem Retell'de tanımlanacak).
  - Codex review görevi: `docs/handoff/CODEX_CHECK_REVIEW.md`.
- Kullanıcı geri bildirimi: Codex'e verilen görev dosyaları kısıtlayıcı olmamalı. `CODEX_CHECK_REVIEW.md` açık uçlu
  hale getirildi; `CODEX_SONRAKI_YON.md`'deki ifadeler öneriye çevrildi.

### 2026-11-14 — Check v2 bağımsız incelemesi (Codex)

- Kullanıcı: “check kısmını da geliştirdik, incele araştır ve düşün”. Kapsam Check incelemesi ve somut
  doğruluk düzeltmeleri; yeni Look seçimi ve Retell eğitimi yapılmadı.
- `scripts/review_check_v2.py`: mevcut özellik modellerini bağımsız yeniden hesaplar; ana kayıtların
  olasılıkları sayısal hassasiyet içinde aynı. Kelime/konum/göreli konum uyumsuzluğu yok; gradyanlar sonlu
  farkla doğrulandı. Test split'i ve pilot anahtarı okunmadı.
- `results/check_v2_review.json`: 2.000 görüntü düzeyinde eşleştirilmiş bootstrap, tüm lambda kayıpları,
  alternatif sayımları, düzeltilmiş AP ve B'de iç içe lambda/eşik tanısı. Detektör ve captioner katkılarının
  aralıkları pozitif; Look sinyal grubu ve attention/detektör bölge farkları sıfırı içeriyor.
- İç içe tanıda tüm dış katlar mevcut lambda çiftini seçti; OOF AUROC değişmedi. Ortak B eşiğinin kendi
  etiketini dolaylı görmesi ve özellik ailesi seçiminin validation'a dayanması raporlandı. Ana eğitim
  kayıtlarının protokolü değiştirilmedi; Retell öncesinde ele alınmalı.
- `scripts/20_check.py`: iki kayıt türünün yazımında attention güncellemelerinin kaybı düzeltildi.
  Kalibrasyon çıktıları artık `_check2cal`; script 22'nin öğrenilmiş kayıtlarıyla çakışmıyor. Her ikisinde
  mevcut pipeline dosyasını değiştirmek için açık `--overwrite` var.
- `lcr/calibrate.py`/script 22: eksik değer doldurma eğitim katına taşındı; AP eşit skorları grupluyor;
  kararlı sigmoid kullanılıyor. Script 21 kelime/konum değişimini açık hata ile yakalıyor.
- Tam script 20 geçici dizinde çalıştırıldı; iki kayıt türünün bütün claim'lerde güncellendiği ve mevcut
  öğrenilmiş dosyaların değişmediği doğrulandı: `results/check_v2_writer_review.json`.
- Üç regresyon testi eklendi; tam CPU koşusunda 54 test geçti (diğer Look deney testleriyle birlikte).
- `docs/CHECK_V2_INCELEME.md` eleştirel değerlendirme ve doğrulanmış kaynaklar;
  `docs/generated/CHECK_V2_REVIEW_METRICS.md` script üretimi sayılar. CHECK_V2/README/günlük güncellendi.
  İnsan pilotu, sürekli olasılığın Retell kanıtına aktarılması, model saklama ve sentetik kayıtların aynı
  doğrulayıcıyla yeniden skorlanması açık işler. Eski rapor PDF'i veya öğrenilmiş kayıtlar yenilenmedi.
- Check hata analizi (`scripts/24_check_errors.py`): contradicted kesinliği giyimde 0.50 (taban oran 0.42, kazanç 1.19×),
  diğer nesnelerde 0.55 (kazanç 3.29×), zeminde 0.42. CHECK_V2.md §3.4 düzeltildi: proxy gürültüsü ölçülen kesinliği
  muhtemelen yükseltiyor, düşürmüyor. İnsan pilotu gerekli.

### 2026-11-14 — E2 tamamlandı
- `scripts/19_look_e2.py` / `lcr/look_sources.py`: B'den tohum 415 ile aynı 500 görüntü, 1.142 claim
  (918'i insan kutulu). S1, S3-prefix, Grad-CAM, kontrastif, okuma adımı ve alt-kelime haritaları çıkarıldı.
- Sabit v2 kutu kuralında arşiv S3 bölge %73.6383; en iyi yeni kaynak S4 %73.2026. Hiçbir kaynak
  ≥2 puan kabul eşiğini geçmedi; E2 bileşeni kabul edilmedi. Başlık seçimi veya yeni grid açılmadı.
- S1 ham kontrolü fp16 arşive uyuyor (maks. mutlak fark 0.00005890). Yeniden hesaplanan caption S3
  arşivle tam uyuşmuyor (maks. fark 0.169888); arşiv baseline sabit tutuldu ve bu sınırlama sonuçta açık.
  Kabul kararı sabit arşiv baseline ile verildi: hiçbir aday ≥2 puan şartını geçmedi.
- E2 maskelemesinde transformers'ın görsel attention mask argümanını iç modüle yanlış aktardığı
  yerel kaynak kodundan görüldü. Yalnızca E2 decoder'ına ağırlıkları değiştirmeyen yönlendirme hook'u
  eklendi; gerçek attention olasılıklarında maskeli yerlerin sıfır olduğu regresyon testi geçti.
  İlk geçersiz sıfır-müdahale sonucu `results/archive/invalid_e2_mask_route` altında tutuldu.
- Aynı 500 görüntüde düzeltilen özellik düzeyi müdahale: ortalama log-p düşüşü Look 1.6852,
  eşit token sayılı rastgele maske 0.9813. Eşleştirilmiş fark 0.7039, görüntü-bootstrap GA95
  [0.6168,0.7904]. Özellik düzeyi kapı geçti; piksel düzeyi nedensellik iddiası yok.
- `results/look_e2.json`, protokol, kaynak haritaları ve checkpoint ilerlemesi kaydedildi.
  Bundan sonra yalnızca Look kapanışı: E1 bileşenleriyle v3, tek ablation ve rapor güncellemesi.

### 2026-11-14 — Look E3 / v4 (Claude)
- Protokol koşmadan önce: `docs/LOOK_E3_PROTOKOL.md`. Kod: `lcr/look_e3.py`, `lcr/look_e3_eval.py`, scripts 25–30,
  `tests/test_look_e3.py` (+4 test; toplam 59 geçiyor). Değişen: `scripts/22_check_verifier.py`'ye `--results`.
- A aşaması (B'de 5 katlı ÇD): etiketsiz başlık seçimi +2.64 puan → kabul; denetimli başlık +1.10, topluluk +0.35,
  katman +0.22, öbek −0.24, Chefer −2.40, Salience DropOut −3.40, oran −16.53.
- B aşaması: kişi için SAM (3 maskeden kütle/alan^0.25) ÇD +4.72 → kabul; thing/stuff reddedildi.
- Validation bir kez: v4 %80.39 (v3 %76.10), +4.29 [+2.46, +6.09]; kişi +9.33, thing +1.32, stuff −1.45 (GA sıfırı içeriyor).
- `scripts/29_freeze_look4.py` → `*_look4.json`; kayıtlardan yeniden ölçüm birebir. Check v4 ile: AUROC 0.826.
- SAM atfı Crossref'le doğrulandı, `refs.bib`'e eklendi. Tablolar: `docs/generated/LOOK_E3_METRICS.md`.
- Bilinen sınırlamalar: başlık ızgarasının en küçüğü (m=4) seçildi; arşiv/yeniden hesap sapması tam çözülmedi.


### 2026-11-14 — Codex Look E3/v4 incelemesi ve canlı taşıma
- AGENTS, E3 devir notu, protokol, günlük, tasarım ve ilgili literatür okundu; Kang/Chefer/PnP-OVSS birincil
  kaynaklarıyla uyarlamalar karşılaştırıldı. Kapanış: `docs/LOOK_E3_INCELEME.md`.
- `lcr/look_v4.py` eklendi; `pipeline.py`, `config.py`, script 04 ve script 29 canlı/dondurulmuş v4 için
  ortaklaştırıldı. Çıktı koruması, proxy skor geometrisi, başlık indeksleri, SAM dikdörtgen skoru açıklandı.
- SAM sonrası 3.031 yardımcı excess tutarsızlığı ve sentetik yeniden etiketlemede SAM kutusunun kaybolması
  düzeltildi. Reddedilen rank topluluğunda eşit değerlere farklı sıra verilmesi düzeltildi; tarihsel
  seçim/validation dosyaları yeniden yazılmadı. Seçili v4 bu rank işlevini kullanmıyor.
- `scripts/review_look_e3.py`, `review_look_e3_drift.py`, `look_e3_review_table.py` ve B harita galerisi eklendi.
  Tüm 9.128 claim'de harita/kutu/skor farkı sıfır; dört B taslağında yeni decoder+SAM ve iki gerçek
  taslak üretiminde birebir eşitlik. Bağımsız benchmark validation bölge %80.39'u yeniden verdi.
- Küçük m, tek başlık/tür ve çok GT kişi kutusu tanıları yalnızca B'de. Yeni seçim yapılmadı.
  Seçim ÇD'sinin transdüktif/aşamalı sınırı, çoğul aralıkların eşzamanlı olmadığı ve SAM–Check eşleme
  proxy'si açıklandı. Chefer özellik düzeyi açıklama; E3 DropOut özgün PnP-OVSS deneyi değildir.
- Sayısal arşiv sapması SEP/attention/ortalama/çıkarma yuvarlaması ve ayrı CPU float16 decoder ile
  araştırıldı; kesin kök neden çözülmedi. Mevcut v4 çıkarım hattının tekrarlanabilirliği doğrulandı.
- Ana v4 ve türetilmiş Check kayıtlarının eski sürümleri `data/cache/pipeline/archive/pre_e3_review/`
  altında korundu. Yeni yardımcı metadata yazıldı; bütün eski Check kayıt değerleri (p_correct,
  verdict, alternatif ve eşik dahil) birebir korundu. Sayısal kanıt `results/look_e3_review.json`.
- Doğrulama: 64 CPU testi geçti. Test split okunmadı; yeni Check/Retell eğitimi yok.

### 2026-11-14 — Codex: üç Look kutusu ve frizbi sorusunun hedefli incelemesi
- Kullanıcı ilk üç kutunun ayrı kontrol edilmesini ve frizbi varken düşük skorun nedenini sordu.
- `scripts/review_check_v4_boxes_B.py` mevcut B OOF kararlarını yeniden üretti; ilk üç kutunun ayrı
  skor/kararlarını, güçlü adayın dışarıda kalmasını ve ortalamada seyrelmesini tanısal olarak ölçtü.
  Sonuç `results/check_v4_boxes_B.json`, otomatik tablo `docs/generated/CHECK_V4_BOXES_B.md`.
- Referansta desteklenen ama v4'ün desteklemediği frizbi örneklerinin en güçlü OWLv2 adayları küçük
  kutu filtresinden eleniyor. İlk dört örnek görsel denetlendi; yeşil kutular frizbiye oturuyor.
  Galeri `results/check_v4_frisbee_B.png`, tek örnek `results/check_v4_frisbee_example_B.png`.
- `scripts/review_check_v4_contract.py` B'de alternatiflerin referans desteğini ve mevcut Retell
  finalise yolunun v4 kararlarını yeniden yazdığını ölçtü. Eski s_alt ile yeni alternatif adı
  uyuşmazlığı saptandı. Çıktılar `results/check_v4_contract_B.json` ve otomatik tablo.
- `docs/CHECK_V4_ON_INCELEME.md`, README ve günlüğün durum/adım bölümleri güncellendi.
  Yalnız tanı kodu ve belgeler eklendi; canlı Look/Check/Retell, kilitler ve ana kayıtlar değişmedi.
- Doğrulama: B'nin saklı v4 kararları OOF eşiklerini yeniden fit ederek birebir doğrulandı;
  üç kutunun yeniden hesaplanan ağırlıklı claim skoru saklı v4 skoruyla eşleşti.
  Validation/test açılmadı; yeni eğitim/seçim yok. Tam Check v4 incelemesi ve canlı taşıma açık.


### 2026-11-14 — Codex: özgün mantıkta tek Look/Check yolu ve GPU sınırına devir
- Kullanıcı attention verilen yerlerde aynı detektörle nesne kanıtı arama mantığını koruyarak tekleştirme
  istedi; ardından kapsamı açıkça Look → Check ile sınırladı. Retell'e geçilmedi; GPU işini Claude yapacak.
- Önceden B protokolü yazıldı (`docs/LCR_BIRLESTIRME_PROTOKOL.md`). Look minimum kenar/alan kuvveti
  kıyası B_fit insan kutularında; Check max/ağırlıklı ve alternatif kıyası B_fit görüntü ÇD'de yapıldı.
  B_cal son ölçek/eşik için ayrıldı. Sonuçlar `results/lcr_development.json` ve otomatik ekte.
- `lcr/region_evidence.py`, scriptler 37/38/39 ve `tests/test_unified_lcr.py` eklendi. Canlı ve kayıtlı
  hesaplama ortaklaştırıldı; eski skorla yeni alternatif eşleştirilmesi ve kararın yeniden yazılması
  düzeltildi. Kaynak bundle attention/decoder/sorgu kimliklerini taşır. Kaynak bilgisi seçimden sonra
  metadata düzeltmesi olarak eklendi; skor/kararlar değişmedi.
- Kutu başına kanıt ayrı kalır; bir yerdeki destek başka yerdeki düşük skorla çelişki sayılmaz. Görüntü
  maksimumu ana kanıt token'ına girmez. Mevcut Retell arayüzündeki hazırlıklar eğitim başlatmaz;
  ana lcr girdisiyle eğitim kullanıcı sınırı nedeniyle açık hatayla engellenir.
- B'nin tüm kayıtlarında skorlar doğrudan OWL özellik/sorgu hesabıyla kontrol edildi; sabit iki B taslağının
  yeni attention'ıyla canlı yol aynı bölge/kararı verdi. İlk tam CPU koşusunda 83 test geçti; test görselleri
  validation yerine B'ye taşındı. Validation/test verisi açılmadı, yeni eğitim yapılmadı.
- Aynı OWLv2'yi dondurulmuş attention crop'larında yeniden çalıştırmak için 40/41, sabit pilot paketi ve
  Colab defteri hazırlandı. Girdi denetimi yapıldı; yeni OWLv2/GPU hesabı başlatılmadı.
- `docs/LCR.md` güncel tek giriş noktası oldu; AGENTS, README, literatür ve günlük güncellendi.
  Claude devir notu hazır. Eski durum/adımlar ayrı tarihsel notta korundu; eski v4/validation dosyaları
  ezilmedi. Check ikame/proxy güvenilirliği açık; bölge kanıtının üst sınırı iddia edilmez.

- Son kontroller: kaynak kimliği eklendikten sonra B kayıt uyumu tekrar geçti; ilgili 20 CPU testi geçti.
  Tam önceki CPU koşusu 83 testti (bu iki sayı toplanmaz). Retell eğitim engeli model yüklemeden test edildi.
  Pilot paketinin arşiv bütünlüğü, sorgu/bundle kimliği ve dry-run kontrolü geçti. Raporun Look/Check
  tasarımı ve B geliştirme bölümü güncellendi; LaTeX/BibTeX derlemesi başarıyla tamamlandı.

### 2026-11-14 — Claude: Codex LCR incelemesi ve ilk ilkeler tasarımı
- Codex'in `region_evidence.py`, 37–39, `LCR_BIRLESTIRME_PROTOKOL.md`, `CHECK_V4_ON_INCELEME.md` incelendi; kod
  değiştirilmedi. Look'un p−1 adımını kullandığı doğrulandı.
- Yeni: `scripts/review_first_principles_B.py` → `results/first_principles_B.json`,
  `docs/generated/FIRST_PRINCIPLES_B.md`; görsel denetim `results/claude_visual_audit_B.json` ve iki galeri
  (`results/claude_audit_B_*.png`); tasarım/inceleme `docs/ILK_ILKELER_TASARIM.md`.
- Validation/test okunmadı; yeni seçim veya eğitim yok.

### 2026-11-15 — Claude: Check tamamlama
- Kullanıcı: "Check'i tamamla". Önceden yazılmış protokol; konfigürasyon araması yok; yalnız B.
- Değişen: `lcr/region_evidence.py` (decide5, veto'lu eşik uydurma, beş eylem, varsayılan bundle yolu),
  `lcr/claims.py` (`non_visual`), `lcr/pipeline.py` (canlı yol idiom bayrağı), `scripts/37_unify_lcr.py` (tarihsel
  yol), `scripts/39_verify_lcr.py` (`--historical`, eylem eşitliği).
- Yeni: `scripts/42_check_complete.py`, `tests/test_check_complete.py`, `docs/CHECK_TAMAMLAMA{,_PROTOKOL}.md`,
  `docs/generated/CHECK_COMPLETE_B.md`, `results/check_complete_B.json`, `results/lcr_check_bundle.json`,
  `results/lcr2_contract.json`, `data/cache/pipeline/A_B_lcr2.json`.


### 2026-11-15 — Codex: Retell kaynakları ve düşünce notları
- Kullanıcı Retell hakkında araştırma istedi; ardından kapsamı “yol haritası hazırlama, ilgili şeyler bul ve
  düşün” diye netleştirdi. Yalnız araştırma belgesi yazıldı: `docs/handoff/RETELL_ARASTIRMA_NOTLARI.md`.
- HALC, Woodpecker, VFC, LURE, Show–Edit–Tell, Volcano, HELPD, constrained beam, VCD, Flamingo/SmallCap,
  grounding karşı bulgusu ve daha yeni VASparse/CausalLens birincil kaynakları incelendi. Kaynak bulguları
  ile projeye ilişkin çıkarımlar ayrıldı; uygulama sırası, protokol veya yöntem seçimi oluşturulmadı.
- Güncel kodda `action/substitute/pos` bilgisinin eski kanıt kodlamasına taşınmadığı; gate decay ve sabit
  epoch düzenlemelerinin zaten mevcut olduğu; son B kayıtları ile OOF değerlendirmelerin farklı olduğu
  not edildi. Transformers 5.23.0 generation kaynakları model yüklemeden okundu.
- Değişen dosyalar yalnız bu günlük ve araştırma notu. Yeni sonuç/metrik, kod değişikliği, model çıkarımı,
  eğitim veya test koşusu yok; validation/test verisi açılmadı.

### 2026-11-15 — Claude: Codex Retell araştırma notunun incelemesi
- `docs/handoff/RETELL_ARASTIRMA_NOTLARI.md` okundu; kod gözlemleri doğrulandı (eylemler Retell token'ına girmiyor,
  gate_ff, force_words_ids uzak koda yönleniyor, generate prefix almıyor), CausalLens atfı CVF'de doğrulandı.
- R0 tasarımına etkiler `docs/ILK_ILKELER_TASARIM.md` §5'e yazıldı. Kod değişmedi.

### 2026-11-15 — Fable yöntem incelemesi
- Kullanıcı isteğiyle Fable (Claude Fable 5.2) sistemin doğruluğunu ve mantık içinde iyileştirmeyi inceledi.
  Rapor `docs/FABLE_INCELEME.md`; betikler `scripts/review_fable_B/` (özellik dosyası `data/cache/review_fable_B_feat.pkl`).
- Claude V1'i (z ölçeği) bağımsız olarak yeniden hesapladı; sayılar birebir tuttu. Kod değişmedi.

### 2026-11-15 — Claude: Fable'ın Check düzeltmesi (pozitif-çapalı Check)
- Kullanıcı "Evet, düzelt" dedi. Önce protokol yazıldı (`docs/CHECK_DUZELTME_PROTOKOL.md`; karar kuralı koşudan önce
  eşleşmiş-FCR kıyasına çevrildi). Kod: `lcr/region_evidence.py` (`decision="positive_anchored"`, `CONFUSION`,
  `decide_pa`, `substitute`, `fit_positive_anchored`), `lcr/evidence.py` (ham ölçekte alternatif kapısı),
  `scripts/43_check_fix.py`, `scripts/44_check_fix_ablation.py` (sonradan), `scripts/39_verify_lcr.py` (`--mode`),
  `tests/test_check_fix.py`. Tüm testler 94 geçti; `39 --mode check3` ve varsayılan doğrulama geçti.
- Sonuç: ölçüt (b) tuttu (FCR yarıya, giysi 0.32 → 0.05), (a) tutmadı (eşleşmiş FCR'de yakalama 0.388 vs 0.559) →
  varsayılan S2 kaldı. Sonradan ablation: neden "az pozitif" çekimserliği; çekimsersiz S3 toplamda S2 düzeyinde,
  giyside iyi, yüksek-taban swap'larda kötü. Ayrıntı ve öneri `docs/CHECK_DUZELTME.md`. Validation okunmadı.

### 2026-11-15 — Fable ikinci inceleme (Check'in tavanı ve iyileştirme)
- Kullanıcı isteğiyle Fable "ikisi de yetersiz, nasıl düzelir" sorusunu inceledi; rapor `docs/FABLE_INCELEME_2.md`.
  Proje dosyası değişmedi; betikler `scripts/review_fable2_B/`, önbellekler `data/cache/review_fable2_B_*`.
- Claude doğruladı: swap'ların %51.6'sı aynı karışma kümesinde değil, %30.9'u köpek; köpek pozitif %5 kantili −0.18,
  anılmayan %99 kantili −1.25 (Fable'ın sayılarıyla aynı).

### 2026-11-15 — Claude: Check S4 (yeni varsayılan)
- Kullanıcı "Yes, run S4" dedi. Önce protokol (`docs/CHECK_S4_PROTOKOL.md`), sonra kod: `lcr/region_evidence.py`
  (`rival_np`, `rival_matrix`, `fit_rival_np`, `substitute_centred`, `CONFUSION_CLEAN`, `set_matrix`, varsayılan
  `BUNDLE_PATH` → `lcr_check4_bundle.json`), `lcr/claims.py` (`non_visual(..., compounds=True)`), `lcr/pipeline.py`
  (bayrağı bundle'dan okur), `scripts/45_check_s4.py`, `scripts/39_verify_lcr.py` (`--mode check4`),
  `tests/test_check_fix.py` (S4 testleri). Tüm testler 98 geçti; `39 --live` sapma 0.
- Sonuç (B dış-kat): ölçütlerin üçü tuttu → S4 varsayılan. Ablation: rakip-farkında kanıt olmadan temiz yakalama
  0.629, S3 ikame sıralamasıyla kesinlik 0.859. Bedel: stuff FCR 0.054, aynı-küme giyside yakalama 0.484.
- İlk koşuda S2 "geri kazanım" metriği payda/pay uyumsuzluğundan 1.11 çıktı; düzeltilip yeniden koşuldu (S2 0.685),
  S4 sayıları değişmedi. Validation okunmadı.

### 2026-11-15 — Claude: GitHub'a devir
- Proje git deposu yapıldı ve gizli repoya yüklendi: `github.com/iboRotti52/look-check-retell` (`main`).
  `data/` (~16 GB: görüntüler, model/OWLv2 önbelleği, checkpoint, pipeline kayıtları), `.venv`, `tmp/`, eski zip'ler
  ve LaTeX ara dosyaları `.gitignore` ile dışarıda.
- Yeni: `docs/handoff/YOLCULUK.md` (baştan sona hikâye ve devir), `docs/handoff/sohbetler/` (11 Claude oturumunun
  okunur dökümü; araç çıktıları çıkarıldı, e-posta/ev dizini maskelendi). Fable betiklerindeki mutlak yollar
  göreliye çevrildi; günlük dosyalarında ve `transcript_main.md`'de ev dizini maskelendi.
- Commit yazarı GitHub noreply adresi; kişisel e-posta repoda yok.
- Aynı gün: kullanıcı isteğiyle repo **public** yapıldı. `data/cache` (~9 GB) GitHub release `data-v1` olarak yüklendi
  (tar + bölünmüş BLIP özellikleri, SHA256SUMS); `scripts/fetch_data.sh` indirip birleştirir. Flickr8k'den türetilen
  6 inceleme PNG'si ve `annotation/pilot/annotate.html` lisans nedeniyle public repodan çıkarıldı (yerelde duruyor,
  script 07 ile yeniden üretilir). Git geçmişi tek temiz commit'e indirildi. Ham Flickr8k görüntüleri ve test split'i
  release'te yok.


### 2026-11-20 — Claude: Retell kanıt kodlayıcısı Check'in kararını görüyor (`retell/encoder-actions`)
- Sorun (ChatGPT incelemesi, kodda doğrulandı): Retell her kutu için yalnız claim, kutunun alternatifi, bölge, skorlar ve
  karar (verdict) alıyordu; S4'ün `remove` ve `replace` kararları ikisi de `contradicted` olarak geliyor, seçilen ikame
  ancak bir kutunun alternatifi tesadüfen aynıysa ulaşıyordu.
- `lcr/retell.py`: her jetona S4 eylemi (`ACTION_IDS`) ve seçilen ikame (BLIP kelime gömmesiyle, claim gibi) eklendi;
  ikisi de kapılar gibi sıfırla başlıyor. Eylemsiz kayıtlar (tarihsel, B2 etiketleri) birebir eskisi gibi kodlanıyor; eski
  checkpoint'ler `load_pathway_state` ile yeni alanlar sıfırda yükleniyor. Jetonlar adlı alanlı (`EvidenceToken`).
  `lcr/evidence.py`: eylem ve ikame her kutu jetonuna kopyalanıyor; H4 yanlış kanıtı artık ikameyi de değiştiriyor
  (yoksa gerçek ikame yanlış-kanıt koşusuna sızardı). `scripts/06_train_retell.py` checkpoint'i bu yükleyiciyle açıyor.
- Eğitim engeli yerinde; hiçbir şey eğitilmedi. Testler `tests/test_retell_actions.py` (CPU). `tests/test_retell.py`
  (BLIP + Flickr8k gerekir) burada koşulamadı; sıfır-başlangıç özelliği kapılardan geldiği için etkilenmez.
- Bileşen sahibi İbrahim Billurcu (öneri: evidence encoder); PR'da onun incelemesi istenmeli.
