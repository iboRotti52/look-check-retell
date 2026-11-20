# Look–Check–Retell (CS415)

> **Devralan ekip için:** önce `docs/handoff/YOLCULUK.md` (hikâye + sıradaki adımlar), sonra `AGENTS.md` ve `docs/CALISMA_GUNLUGU.md`.

## Kurulum ve veri (büyük dosyalar repoda değil)
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/00_download.py      # Flickr8k görüntüleri + NLTK verisi (resmî kaynaklardan)
bash scripts/fetch_data.sh                   # data/cache: GitHub release `data-v1` (~9 GB)
.venv/bin/python -m pytest -q                # hepsi CPU'da
```
- `data/cache` içeriği: OWLv2 öneri önbelleği, BLIP görüntü özellikleri, fine-tune edilmiş `decoder_A.pt`, Look
  haritaları (`look_e3`), B/validation pipeline kayıtları (`A_B_lcr4.json` = güncel Check S4).
- HF modelleri (BLIP, OWLv2, SAM) ilk kullanımda `data/hf_cache`'e iner.
- Flickr30k Entities (yalnız Look ölçümü için) `data/flickr30k_entities/` altına yazarlarının dağıtımından konur.
- Flickr8k'den türetilen inceleme görüntüleri ve etiketleme aracı (`annotation/pilot/annotate.html`) veri seti
  lisansı nedeniyle repoda yok; `scripts/07_make_annotation.py` ile yeniden üretilir.


Proje kodu, deney scriptleri ve progress report kaynağı.

## Güncel ana yol — Look → Check

**Tek giriş noktası: [`docs/LCR.md`](docs/LCR.md).** Kullanıcı kararıyla önce Look, sonra Check;
Retell eğitimi bekliyor. Güncel kayıt `data/cache/pipeline/A_B_lcr.json`, ayarlar `results/lcr_bundle.json`.
Küçük nesne filtresi B insan kutularıyla ölçülerek kaldırıldı. Attention'ın seçtiği üç aday ayrı skorlanır;
bir adaydaki güçlü destek ortalamada kaybolmaz. Yeni bir doğrulayıcı eklenmedi.

Ölçümler: [`docs/generated/LCR_DEVELOPMENT.md`](docs/generated/LCR_DEVELOPMENT.md).
Sıradaki ağır hesap için Claude brifi: [`docs/handoff/CLAUDE_LOOK_CHECK_GPU.md`](docs/handoff/CLAUDE_LOOK_CHECK_GPU.md).
Hazır Colab defteri `colab/LOOK_CHECK_GPU_PILOT.ipynb`, girdi paketi `output/lcr_check_gpu_pilot.zip`.
GPU işi başlatılmadı. Aşağıdaki v1–v4 ve Retell çalıştırma anlatımları tarihsel deneylerdir.

**Look L1 (yalnız ölçüm, 2026-11-18):** dondurulmuş Look v4'ün neden öncesi taahhüt ettiği "uygun bölge yok"
çekimserliğini B'de ölçmeye başlayan paralel bir çalışma; [`docs/LOOK_L1_PROTOKOL.md`](docs/LOOK_L1_PROTOKOL.md).
Script 46/47 ve `lcr/look_referent.py` hazır ama henüz çalıştırılmadı (`data/cache` + proje `.venv`'i gerekir);
hiçbir seçilen bölgeyi veya yukarıdaki ana yolu değiştirmez.

## Klasörler
| Klasör | İçerik |
|---|---|
| `lcr/` | Kütüphane: veri, claim çıkarma, Look (attention→bölge), Check (OWLv2), Retell (gated pathway), metrikler |
| `scripts/` | Numaralı deney scriptleri (aşağıda) |
| `colab/` | Colab'da GPU aşamaları: `run_all.sh`, `run_sink.sh`; Check v3 sinyalleri `run_check3.sh` (T4 yeterli) |
| `tests/` | CPU'da çalışan birim testleri (`.venv/bin/python -m pytest -q`) |
| `results/` | Scriptlerin ürettiği küçük JSON sonuçlar ve loglar |
| `report/` | IEEE LaTeX rapor (`main.tex`), `generated/` (scriptlerin yazdığı tablolar/sayılar), `figures/` |
| `annotation/pilot/` | 100 draft'lık kör etiketleme sayfası (`annotate.html`) — `key.json`'u etiketleme bitene kadar açmayın |
| `data/` | Flickr8k ve önbellekler (büyük, rapora girmez) |

## Scriptler
| Script | Ne yapar | Nerede |
|---|---|---|
| `00_download.py` | Flickr8k + NLTK verisi | her yer |
| `01_data_stats.py` | Vocabulary/claim istatistikleri | CPU |
| `02_cache_features.py` | BLIP ve OWLv2 özelliklerini önbelleğe alır | GPU |
| `03_finetune_decoder.py` | BLIP decoder'ını A yarısında fine-tune eder | GPU |
| `04_draft_look_check.py` | Draft → Look → Check, kayıtları yazar (`--attn sink` varsayılan) | GPU |
| `06_train_retell.py` | Retell / B0+ / B2 / B3 eğitimi + validation değerlendirmesi | GPU |
| `07_make_annotation.py` | Kör etiketleme sayfası üretir; `--pilot lcr`: S4/S2 kararlarından tabakalı B insan pilotu, ikame sorusu ve referans kutusu etiketten sonra (`docs/PILOT_PROTOKOL.md`) | CPU |
| `08_analyse.py` | Colab çıktılarından raporun tablolarını, figürlerini, sayılarını üretir | CPU |
| `09_attention_variants.py` | Ham vs sink-düzeltilmiş attention karşılaştırması | GPU/CPU |
| `10_score_labels.py` | İnsan etiketlerinden HICR, κ ve eşik kalibrasyonu; pilot anahtarında grup başına S4 kesinliği, ikame doğruluğu, silme zararı ve referans kutuları | CPU |
| `11_proxy_thresholds.py` | B yarısında referans proxy'sine göre ara eşik kalibrasyonu | CPU |
| `12_look_variants.py` | Kayıtlı haritalardan Look varyantları + AUROC (GPU'suz) | CPU |
| `13_look_benchmark.py` | Look'u Flickr30k Entities insan kutularıyla ölçer (pointing, enerji, bölge IoU) | CPU |
| `14_look_sanity.py` | Look harita kapıları: yönelim (transpoze) ve kelimeye özgüllük (takas testi) | CPU |
| `15_alt_quality.py` | Captioner alternatiflerinin kalitesi (yanlış claim'lerde alternatif referansta mı) | CPU |
| `16_look_regions.py` | Haritadan bölgeye varyantları: B'de seçim, validation'da rapor | CPU |
| `17_freeze_look.py` | Kayıtları dondurulmuş Look v2 bölge kuralıyla GPU'suz yeniden yazar (`*_look2.json`) | CPU |
| `20_check.py` | Check v2 adım 1: sınıf başına kalibrasyon × alternatif × karar kuralı ızgarası (B-OOF seçim) | CPU |
| `21_caption_signals.py` | Captioner sinyalleri (log p, entropi, marj, görsel kazanç, konum) her claim için | CPU |
| `22_check_verifier.py` | Check v2: öğrenilmiş kalibre doğrulayıcı, sinyal grubu ablation'ı, `*_check2.json` yazar | CPU |
| `23_owl_queries.py` | OWLv2 sorgu şablonu: tek / dilbilgisel / ensemble | CPU |
| `review_check_v2.py` | Check bağımsız yeniden üretimi, kelime konumları/gradyanlar, iç içe B tanısı ve görüntü bootstrap aralıkları | CPU |
| `24_check_errors.py` | Check v2 hata analizi: giyim / nesne / zemin gruplarında contradicted kesinliği ve yakalama → `results/check_errors.json` |
| `25_look_e3_extract.py` | Look E3: başlık başına sink haritaları + aday haritalar (öbek, oran, Grad-CAM, kontrastif, Chefer, Salience DropOut) | CPU |
| `26_look_e3_select.py` | Look E3 A aşaması: harita aileleri, B'de 5 katlı ÇD seçim; `--report` validation | CPU |
| `27_look_e3_sam.py` | Look E3 B aşaması girdisi: harita tepe noktasıyla SAM maskeleri | MPS/CPU |
| `28_look_e3_boxes.py` | Look E3 B aşaması: tür başına SAM kutu kuralı seçimi; `--report` validation | CPU |
| `29_freeze_look4.py` | Look v4 kayıtları (`*_look4.json`), captioner sinyallerini kopyalar | CPU |
| `30_look_e3_table.py` | E3 sonuç tablosu → `docs/generated/LOOK_E3_METRICS.md` | CPU |
| `31_check_vlm_signals.py` | Check v3: bağımsız BLIP VQA (evet/hayır) ve ITM sinyalleri → `*_vlm.json` | GPU (Colab T4) |
| `32_check_consistency.py` | Check v3: decoder_A'dan 10 örnek + 5 beam caption, sınıf geçme oranları → `*_cons.json` | GPU (Colab T4) |
| `33_check_v3.py` | Check v3 (yalnız karşılaştırma): B'de tam iç içe ÇD ile aile/model seçimi (kilit); `--report` validation bir kez, `*_check3.json` | CPU |
| `34_check_v3_table.py` | Check v3 sonuç tablosu → `docs/generated/CHECK_V3_METRICS.md` | CPU |
| `35_check_v4.py` | Tarihsel Check v4: özgün bölge kuralı içinde 144 konfigürasyon, B'de ÇD seçimi (kilit); `--report` validation bir kez, `*_check4.json` | CPU |
| `36_check_v4_table.py` | Check v4 sonuç tablosu → `docs/generated/CHECK_V4_METRICS.md` | CPU |
| `37_unify_lcr.py` | (Tarihsel, Codex) İlk birleşik Look/Check: B'de katkı kıyası, B_cal ölçek/eşikleri → `lcr_bundle.json` | CPU, kayıtlı özellikler |
| `38_lcr_table.py` | Kayıtlı B sonuçlarını rapor makrolarına çevirir | CPU |
| `39_verify_lcr.py` | B kutu–skor–alternatif/karar/eylem uyumu (varsayılan: `BUNDLE_PATH`'teki bundle, şu an S4; `--mode check3/five_way/historical` eskiler); `--live` iki sabit B taslağını tekrarlar | CPU |
| `40_check_crop_gpu.py` | Dondurulmuş attention kırpımlarında aynı OWLv2'yi yeniden çalıştırır; `--dry-run` model yüklemez | **CUDA zorunlu**, henüz çalıştırılmadı |
| `41_prepare_check_gpu.py` | Referans etiketi içermeyen sabit B pilotunu ve Colab paketini hazırlar | CPU, akışlı dosya okuma |
| `42_check_complete.py` | Tamamlanmış Check (beş eylem, z ölçeği): görüntü vetosu, replace/remove, "camera" kalıbı, tüm B'de çapraz ayar → `lcr_check_bundle.json`, `A_B_lcr2.json` | CPU |
| `43_check_fix.py` | Pozitif-çapalı Check (Fable düzeltmesi): ham OWLv2 logit, sınıf eşiği temiz pozitiflerden, çekimser sınıflar, karışma kümesi ikamesi, FCR ≤ %5; S2 ile FCR + swap kıyası → `lcr_check3_bundle.json`, `check_fix_B.json`, `A_B_lcr3.json` | CPU |
| `44_check_fix_ablation.py` | 43'ün sonradan çekimserlik ablation'ı (seçim değil) | CPU |
| `45_check_s4.py` | **Check S4 (varsayılan):** rakip-farkında kanıt, maliyete duyarlı sınıf eşikleri, Look top-1 merkezli ikame, temiz kümeler; S2/S3 kıyası → `lcr_check4_bundle.json`, `check_s4_B.json`, `A_B_lcr4.json` | CPU |
| `46_look_referent.py` | Look L1 (`docs/LOOK_L1_PROTOKOL.md`): önbellekteki E3 işaretli başlıklardan güç/yoğunluk/boş-hipotez sinyalleri, LCR `candidates` sıralaması; referansta destekli/desteksiz claim ayrımı (otomatik proxy) × Look top-1 kutu alanı üçte birlikleri → `look_referent_B.json`, `docs/generated/LOOK_REFERENT_B.md`. Yalnız B. | CPU |
| `47_look_forced_hallucination.py` | Look L1: zorlanmış halüsinasyon testi — `decoder_A` ile takaslı/eklemeli başlıkları teacher-force eder; kelimeyi üreten (*predict*, v4) ve okuyan (*read*) adımda tek sıralamayla top-1/top-3'ü B3'e karşı F30kE referansıyla, S4'ün eşiğiyle yokluğu onaylanmış uydurulmuş claim'leri doğru claim'lere karşı AUROC ile ölçer → `look_forced_hallucination_B.json`. Yalnız B. | CPU (birkaç yüz claim), yoksa Colab |
| `49_look_small_object.py` | B_fit'te Look'un insan kutusu kapsaması, küçük nesne isabeti yalnız küçük kutuya gelirse (script 37'nin ilk sayımı büyük kutuyu da sayıyordu); filtreli/filtresiz → `look_small_object_B.json`, `docs/generated/LOOK_SMALL_OBJECT_B.md`; script 38 rapor makrolarını buradan alır | CPU |
| `50_h1_power.py` | H1 örneklem tablosu, etki tam %15'te tutularak (düzeltme sabiti hesaplanır) → `h1_power.json`, `report/generated/power_table.tex`, `power_macros.tex` (eskiden script 08) | CPU, veri gerekmez |
| `review_first_principles_B.py` | İlk ilkeler tanısı (eylem türü, bölge dışı varlık, alternatif kalitesi, görsel denetim özeti) | CPU |
| `review_look_e3.py` | Donmuş E3/v4 için kayıt eşitliği, B tür/başlık/SAM tanıları ve `--live` yeniden çıkarım kontrolü | CPU/MPS |
| `review_look_e3_drift.py` | İlk 16 B taslağında SEP ve sayısal duyarlık sapma tanıları | CPU |
| `look_e3_review_table.py` | Kayıtlı inceleme sonuçlarından otomatik sayısal ek | CPU |
| `review_look_v2.py` | B'de katkı ayrımı, kişi hataları ve örnek galerisi | CPU |
| `verify_look_artifacts.py` | Düzeltme öncesi/sonrası kayıt ve metrik tutarlılığını denetler | CPU |
| `look_report.py` | İnsan kutusu sonuçlarından Look rapor tablolarını ve sayılarını üretir | CPU |
| `colab_full_metrics.py` | Tam COCO metrikleri (METEOR/SPICE için Java gerekir) | Colab |

**Devam ederken önce `AGENTS.md` ve `docs/CALISMA_GUNLUGU.md` okunmalı** (durum, kararlar, sıradaki adım).

Check incelemesi: `docs/CHECK_V2_INCELEME.md`. Script 20'nin yalnızca kalibrasyon çıktısı `*_check2cal.json`,
script 22'nin öğrenilmiş çıktıları `*_check2.json` / `*_check2vis.json` adını kullanır. Mevcut pipeline
çıktıları açık `--overwrite` olmadan değiştirilmez.

## Look sürümleri ve yeniden üretim

Script 04 varsayılan olarak dondurulmuş **Look v4** yazar: `A_B_look4` / `A_validation_look4`.
V4, decoder A ve sink attention gerektirir; seçili başlıklar + kişi için SAM kullanır. Önceden indirilmiş
`facebook/sam-vit-base` gereklidir. Eski kontroller için `--region-rule v1|v2|v3` açıkça seçilebilir;
raw harita v4 adıyla yazılmaz. Mevcut JSON veya harita `--overwrite` olmadan değiştirilmez;
`--limit` çıktıları ayrı ad alır. Eski Colab akışları v1'i açıkça kullanır.

V4 inceleme sonucu: `docs/LOOK_E3_INCELEME.md`. Canlı çıkarım ve script 29 aynı hesaplama kodunu kullanır.
Script 29 için mevcut dosya koruması vardır; bağımsız yeniden üretim `--suffix look4_review` ile yapılır.
Onaylı kayıt düzeltmesi yedek alınarak `--overwrite --sync-check-metadata` ile yapıldı; kutular/ham skorlar
ve öğrenilmiş Check v2 kararları korundu. SAM `box` alanı ile skorun alındığı OWLv2 `score_box` alanı
farklı olabilir; bu eşleme proxy'dir. Eski dosyalar `data/cache/pipeline/archive/pre_e3_review/` içindedir.

Check/Retell'e mevcut kayıtları vermek için `--attn look4_check2` gibi ilgili stem eki kullanılır;
kanıt şeması ve dış-kat eğitim özellikleri henüz tamamlanmadığından bu bir yeni eğitim talimatı değildir.
Canlı script 04 ham Check kaydı üretir; öğrenilmiş doğrulayıcı ayrı uygulanır.

Check v4 hedefli ön incelemesi: `docs/CHECK_V4_ON_INCELEME.md`. B'de üç kutuyu ayrı incelemek ve frizbi
örneklerini görmek için `scripts/review_check_v4_boxes_B.py`; mevcut Retell girişinin karar/alternatif
tutarlılığı için `scripts/review_check_v4_contract.py`. Bunlar tanı scriptleridir; yeni yöntem seçmez,
validation/test okumaz, dondurulmuş v4 kayıtlarını değiştirmez. Canlı Retell v4'e henüz uyarlanmamıştır.

Script 17, yeni `rec_att` ile birlikte kutu karşılaştırmalarını ve k=4 alanlarını günceller.
Haritaları saklanmamış eski k=1/12 sonuçları `legacy_look` içinde korunur; güncel sonuç gibi kullanılmaz.
`look_metadata`, harita düzeltmesini ve kutu kuralını ayrı kaydeder. Düzeltme öncesi kayıtlar ve ölçümler
`data/cache/pipeline/archive/pre_consistency_fix` ve `results/archive/pre_consistency_fix` içindedir.

Raporun sayılarını üretmek için sırasıyla `python scripts/08_analyse.py` ve
`python scripts/look_report.py` çalıştırın. Yerelde tam COCO sonuçları bulunmuyorsa mevcut BLEU/CIDEr
ölçümleri kullanılır; METEOR/SPICE gibi eksik değerler uydurulmaz ve tabloda eksik gösterilir.

## Colab'da çalıştırma
1. Kod zip'ini `/content`'e yükleyin, `/content/lcr` içine açın.
2. `pip -q install transformers==5.23.0 pycocoevalcap`
3. `bash colab/run_all.sh` (A100'de ~1 saat), sonra `bash colab/run_sink.sh`.
4. Oluşan `lcr_results.zip`'i indirip proje klasörüne açın, sonra yerelde `python scripts/08_analyse.py`.

## Raporu derleme
```
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Overleaf'e `report/` klasörünü olduğu gibi yükleyebilirsiniz.
