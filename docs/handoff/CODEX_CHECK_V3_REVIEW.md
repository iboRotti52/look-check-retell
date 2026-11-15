# Codex: Check v3'ü incele (2026-11-14, Claude)

> **Durum (2026-11-14, kullanıcı kararı):** Bu öğrenilmiş doğrulayıcı projenin Check tanımından sapıyor
> (dil önceliği, bütün görüntü VQA'sı, captioner tutarlılığı "baktığı bölgede nesne var mı" sorusu değil).
> **Ana Check değildir; yalnızca karşılaştırma/ablation satırıdır** ("bölgeye bakmadan ne kadar iyi yapılabilirdi").
> Ana Check özgün kural mantığıyla iyileştiriliyor: `docs/CHECK_V4_PROTOKOL.md`.


Kullanıcı: "Look'ta olduğu gibi Check için de literatürdeki geliştirmeleri dene; sonra Codex Check'e baksın."
Claude, Check için önceden yazılmış bir protokolle yeni sinyaller (VQA, ITM, captioner tutarlılığı) ve model
sınıflarını denedi, B'de seçip kilitledi, validation'ı bir kez raporladı. İnceleme kapsamı serbest:
aşağıdakiler yön ve şüphe listesi, sınır değil. Gerekirse başka yöntem dene, düzelt, yeniden koş.
Bağlayıcı olan yalnızca AGENTS.md kuralları.

## Sonuç (validation, otomatik referans proxy'si; tablo `docs/generated/CHECK_V3_METRICS.md`)
- Seçilen: v2 + vqa_word + cons, logistic; alternatif kuralı cons.
- AUROC 0.826 → 0.862 (+0.036, GA95 [+0.022, +0.051]); en düşük %20'de yanlış oranı +6.0 puan.
- Hedef 0.5'te contradicted kesinliği 0.506 → 0.484 (GA sıfırı içeriyor), yanlış yakalama 0.682 → 0.841.
- Alternatif önerisinin referansta olma oranı OWLv2 0.330 → cons 0.480.
- Ayrıntı ve yorum: `docs/CHECK_V3.md`. Protokol: `docs/CHECK_V3_PROTOKOL.md`.

## Dosyalar
- Sinyaller: `scripts/31_check_vlm_signals.py`, `scripts/32_check_consistency.py`, Colab `colab/run_check3.sh`
  → `data/cache/pipeline/A_{B,validation}_look4_{vlm,cons}.json`.
- Seçim/rapor: `scripts/33_check_v3.py` → `results/check_v3_selection.json`, `results/check_v3.json`,
  `*_look4_check3.json`. Tablo: `scripts/34_check_v3_table.py`.
- Testler: `tests/test_check_v3.py`.

## Özellikle bakılmasını istediklerim
1. **cons ne ölçüyor?** Kazancın çoğu captioner'ın 10 örnekte claim'i tekrar etme oranından (katsayı +1.05).
   Etiket "referansta geçiyor mu" olduğundan bu, varlık yerine belirginliği (anlatıcıların söyleme eğilimi)
   öğreniyor olabilir. B'de bunu ayırmanın etiketsiz ya da F30kE kutularıyla (yalnız ölçüm için) bir yolu var mı?
   Örn. F30kE'de kutusu olan ama referansta geçmeyen sınıflar; insan pilotuyla nasıl ölçülmeli?
2. **Döngüsellik ve Retell:** cons, Retell'in düzelteceği captioner'dan geliyor. Retell'in girdisi olarak
   kullanmak sorun mu? cons'suz bir doğrulayıcı da Retell için ayrıca kaydedilmeli mi?
3. **Seçim protokolü:** 48 aile × tam iç içe ÇD, log-loss + 0.001 tolerans. Factorized (HaloProbe uyarlaması)
   ve GBM uygulamaları adil mi (`Factorized`, `GBM` sınıfları; sabit cezalar `docs/CHECK_V3.md` §5)?
4. **VQA uygulaması:** ilk decoder adımında P(yes)/(P(yes)+P(no)), soru yanıt kodlayıcısından geçiyor
   (`VLM.vqa`). transformers 5.23 BLIP VQA'nın resmî `generate` yoluyla aynı olasılığı veriyor mu?
   vqa_word (yüzey kelime) şablonlu sorudan neden daha iyi?
5. **ITM ve kırpım:** itm_crop hiçbir yerde katkı vermedi. Kırpma (%10 pay, Look v4 kutusu) mı kötü, yoksa
   bölge gerçekten bilgi taşımıyor mu (H6)? Detektör kutusu kırpımıyla (itm_crop_det) karşılaştır.
6. **Karar noktası:** validation eşikleri tc 0.784 / ts 0.809; belirsiz bandı çok dar ve contradicted oranı
   %40.7. Retell için 0.5 doğru hedef mi? Sabit kapsama kıyası mı esas alınmalı?
7. **Cons batch boyutu / tohum:** örnekler batch başına tohumlanıyor; batch 16 ile koşuldu. Kararlılık
   (farklı tohumla AUROC değişimi) ölçülmeli mi?
8. Canlı pipeline (`lcr/pipeline.py`) henüz v3 doğrulayıcıyı üretmiyor; test değerlendirmesinden önce taşınmalı.
