# Check v3 — önceden yazılmış protokol (2026-11-14, Claude)

> **Durum (2026-11-14, kullanıcı kararı):** Bu öğrenilmiş doğrulayıcı projenin Check tanımından sapıyor
> (dil önceliği, bütün görüntü VQA'sı, captioner tutarlılığı "baktığı bölgede nesne var mı" sorusu değil).
> **Ana Check değildir; yalnızca karşılaştırma/ablation satırıdır** ("bölgeye bakmadan ne kadar iyi yapılabilirdi").
> Ana Check özgün kural mantığıyla iyileştiriliyor: `docs/CHECK_V4_PROTOKOL.md`.


Neden: kullanıcı "Check için de Look'taki gibi bütün geliştirmeleri yap, dene" dedi. Check v2'nin
"contradicted" kesinliği %50.6, yanlışları yakalama %68.2 (validation, otomatik referans proxy'si, Look v4
kayıtları). Doğrulayıcının en güçlü sinyali dil önceliği; giyimde neredeyse yalnız öncelikle karar veriyor
(kazanç 1.19×). Codex'in Check incelemesi ayrıca şunları açık bıraktı: özellik ailesi validation'a bakılarak
seçildi, eşikler B'nin tamamından seçildi, alternatif önerisi ham OWLv2 skorlarını karşılaştırıyor.
Bu dosya **koşmadan önce** yazıldı; ızgaralar ve kurallar buradan okunur.

## 1. Yeni sinyal grupları

**vqa — bağımsız modele evet/hayır sorusu** (POPE, Li+ EMNLP'23 fikri). `Salesforce/blip-vqa-base`,
cevap decoder'ının ilk adımında P(yes)/(P(yes)+P(no)). Flickr30k'de ince ayarlı modeller **kullanılmaz**
(Flickr8k görüntüleri, test dahil, Flickr30k'nin içinde); yalnızca COCO/VQA v2 ince ayarlı sürümler.
- `vqa_yes`: gruba göre tek şablon. Nesne/kişi "is there a {sınıf} in the picture?", zemin "is there {sınıf}
  in the picture?", giyim "is anyone wearing a {sınıf}?".
- `vqa_ens`: üç şablonun ortalaması (yukarıdaki + "is there a {sınıf}?" + "does this image contain a {sınıf}?";
  giyimde "is someone wearing a {sınıf}?" + "does anyone have a {sınıf} on?"). Şablon topluluğunun etkisini ölçer.
- `vqa_word`: claim kelimesi yazıldığı haliyle, "is there {kelime} in the picture?" (çoğul/tekil korunur).

**itm — görüntü–metin eşleşmesi.** `Salesforce/blip-itm-base-coco`.
- `itm_full`: "a photo of a {sınıf}" ↔ bütün görüntü.
- `itm_drop`: ITM(draft) − ITM(draft'ta claim kelimesi "thing" ile değiştirilmiş).
- `itm_crop` (ayrı grup, **crop**): aynı sınıf metni ↔ Look v4 kutusu kırpımı (%10 genişletilmiş). B3
  kayıtlarında detektör kutusu kırpılır. H6 için bölgenin bağımsız modelle ölçülen katkısı.

**cons — captioner tutarlılığı** (SelfCheckGPT, Manakul+ EMNLP'23; nesne düzeyine uyarlama). Aynı decoder_A ile
önbellekteki ViT özelliklerinden 10 örnek caption (nucleus p=0.9, sıcaklık 1.0, tohum sabit) ve 5'li beam
listesi. `cons_sample`: sınıfın geçtiği örnek oranı; `cons_beam`: 5 beam'de geçme oranı.

## 2. Aileler (B'de seçilir)
Temel: v2 = prior + det + cap. Adaylar: v2 + her alt küme {vqa, itm, cons} (7 aile), bunların her birinin
+crop ve +look'lu hali. vqa grubu içinde şablon varyantı (`vqa_yes` / `vqa_ens` / `vqa_word`) ayrı aile sayılır.
Ablation (raporlanır, seçilmez): vqa / itm / cons tek başına (öncelik yok), prior + her biri.

**Model sınıfı** (seçilen özellik ailesiyle, ayrı aile olarak):
- `logistic`: mevcut FeatureLogistic (sınıf başına büzülmüş sabit terim).
- `factorized` (HaloProbe, ICML'26 fikri): görsel/captioner özellikleriyle sınıf-dengeli ağırlıklı, sınıf
  sabiti olmayan bir lojistik, üstüne sınıf önceliğinin logiti ofset olarak eklenir (öncelik ayrı tahmin edilir).
- `gbm`: HistGradientBoosting (MetaToken'ın doğrusal olmayan sınıflandırıcı karşılaştırması), girdi olarak
  sınıfın eğitim katı önceliği (logit) + özellikler. Izgara: max_depth ∈ {2,3}, öğrenme oranı ∈ {0.05,0.1}, 200 ağaç.

## 3. Seçim protokolü (Codex'in incelemesindeki açıkları kapatır)
- **Tam iç içe çapraz uydurma (B, görüntüye göre 5 kat):** her dış katta hiperparametre (lojistik λ_c ∈
  {0.3,1,3}, λ_w ∈ {0.1,1,10}; GBM ızgarası) iç 4 katlı ÇD log-loss'uyla seçilir, model dış kat dışında
  uydurulur, karar eşikleri iç katların dışarıda-kalan skorlarından seçilir.
- **Aile seçimi yalnız B'de:** en düşük B dış-kat log-loss'u; ondan en fazla 0.001 kötü olan daha basit (daha
  az grup; eşitlikte logistic < factorized < gbm) aile seçilir. Validation'a bakılmaz.
- **Validation (bir kez):** hiperparametre tüm B'de 5 katlı ÇD ile, model tüm B'de, eşik B dış-kat
  skorlarından. v2'ye karşı görüntü düzeyinde eşleştirilmiş bootstrap (2.000): AUROC, AP(yanlış), karar noktası.
- Karar noktası: contradicted kesinlik hedefi 0.5 (0.6/0.7 de raporlanır), supported 0.95. Ayrıca sabit
  kapsama oranlarında (contradicted dediği claim oranı %10/%20/%30) kesinlik: eşik seçiminden bağımsız kıyas.
- Gruplar: hepsi, giyim, diğer nesneler, zemin. Etiket otomatik referans proxy'si, insan etiketi değil.
- **Alternatif kuralı** (B'de seçilir): ham OWLv2 skoru (mevcut) / VQA P(yes) / ITM / örnek caption frekansı;
  aday kümesi captioner adayları (`blip_alts`). Ölçüt: referans-dışı claim'lerde önerinin referansta olma oranı.

## 4. Çıktı
`results/check_v3_selection.json` (B, kilit), `results/check_v3.json` (validation), seçilen aileyle
`*_look4_check3.json` (p_correct, verdict, alternatif, eşikler, katsayılar).

## Uygulama ve sonuç notu (koşudan sonra eklendi, 2026-11-14)
Izgara ve seçim kuralı değişmedi. Uygulama ayrıntıları (ablation λ'sı, factorized/GBM sabit cezaları, cons batch
boyutu, sıralı koşu) ve sonuç: `docs/CHECK_V3.md`; tablo: `docs/generated/CHECK_V3_METRICS.md`.
Seçim: v2 + vqa_word + cons, logistic, alternatif cons. Codex incelemesi: `docs/handoff/CODEX_CHECK_V3_REVIEW.md`.
