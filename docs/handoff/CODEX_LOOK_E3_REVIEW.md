# Codex: Look E3 / v4'ü incele (2026-11-14, Claude)

> **Tamamlandı — Codex, 2026-11-14.** Kapanış: `docs/LOOK_E3_INCELEME.md`; sayısal ek:
> `docs/generated/LOOK_E3_REVIEW_METRICS.md`. Canlı v4 taşındı, dondurmayla birebir eşitlik doğrulandı;
> SAM yardımcı alanları ve skor geometrisi düzeltildi. Başlıkların iki eksende 1-tabanlı doğru adları
> L9H8, L10H8, L10H9, L9H3; düz indeksler değişmedi. Arşiv S3 sapmasının kök nedeni açık bir sınırlama.
> Aşağıdaki liste özgün devir notudur; sonuç ve güncel devam yönü kapanış belgesindedir.


Kullanıcı: "vakit var, her adımı olabildiğince iyi hale getirelim; eksikleri deneyelim; Codex review'lasın."
Claude, Look'ta literatürden hiç denenmemiş yöntemleri önceden yazılmış tek bir protokolle denedi ve Look v4'ü dondurdu.
İnceleme kapsamı serbest. Aşağıdakiler yön ve şüphe listesi, sınır değil. Gerekirse başka yöntem dene,
düzelt, yeniden koş. Bağlayıcı olan yalnızca AGENTS.md kuralları.

## Sonuç (validation, insan kutuları; tablo `docs/generated/LOOK_E3_METRICS.md`)
- Doğru bölge v3 %76.10 → **v4 %80.39** (+4.29, görüntü bootstrap GA95 [+2.46, +6.09]).
  Kişi 67.15 → 76.48, thing 83.77 → 85.09, stuff 79.71 → 78.26 (GA sıfırı içeriyor).
- Kabul edilenler: (A) etiketsiz seçilmiş 4 cross-attention başlığı (L9H7, L10H7, L10H8, L9H2) ve (B) kişi için
  harita tepe noktasıyla SAM, 3 maskeden kütle/alan^0.25 en yüksek olan.
- Reddedilenler: denetimli başlık, katman seçimi, isim öbeği, oran tipi sink, Chefer relevance, Salience DropOut,
  topluluk; thing/stuff için SAM.
- Check v4 kayıtlarıyla yeniden koşuldu: seçilen doğrulayıcı AUROC 0.829 → 0.826. Kutu iyileşti, varlık doğrulaması değişmedi.

## Dosyalar
- Protokol: `docs/LOOK_E3_PROTOKOL.md`. Sonuç ve yorum: `docs/LOOK_V2_PLAN.md` §7.8.
- Çıkarım: `lcr/look_e3.py`, `scripts/25_look_e3_extract.py` → `data/cache/look_e3/`.
- Değerlendirme: `lcr/look_e3_eval.py` (v3'ün B sayısını birebir veriyor).
- Seçim/rapor: `scripts/26_look_e3_select.py`, `27_look_e3_sam.py`, `28_look_e3_boxes.py`, `30_look_e3_table.py`.
- Dondurma: `scripts/29_freeze_look4.py` → `*_look4.json`; Check: `results/check_verifier_look4.json`.
- Testler: `tests/test_look_e3.py`.

## Özellikle bakılmasını istediklerim
1. **Seçim protokolü:** aile başına 5 katlı ÇD ve ≥ 1 puan eşiği adil mi? A aşamasında 9 aile, B'de 3 tür ayrı
   kabul edildi; çoklu karşılaştırma açısından validation GA'ları yeterli mi? Başlık ızgarasının en küçüğü (m=4)
   her katta seçildi. Daha küçük m ya da başka seçim kuralı denenmeli mi (validation'a bakmadan, B'de)?
2. **Etiketsiz başlık skoru** = özgüllük / (1 + bağlı-bileşen entropisi). Kang+ CVPR'25'in makul bir BLIP
   uyarlaması mı? Skor B pointing'iyle 0.81 korelasyonlu. Seçilen başlıkların anlamlı olup olmadığını
   (örnek haritalar, kelime türlerine göre) ayrıca incele.
3. **SAM kişi kuralı:** tepe noktası tek nokta; SAM'in kendi IoU tahmini en küçük parçayı seçiyor, bu yüzden
   düşük α ile büyük maske tercih ediliyor. Çok kişili görüntülerde ne oluyor? SAM kutusu OWLv2 önerisine
   eşlenerek Check skorları alınıyor (`sam_proposal_iou`). Bu eşleme Check için doğru mu?
4. **Chefer uyarlaması** (`E3Decoder.chefer`): görüntü tarafında R^ii = I. Kural sırası ve normalizasyon makaleye
   uygun mu? Pointing en yüksek (92.3) ama kutu kötü; bu bir hata mı, gerçek bir özellik mi?
5. **Salience DropOut:** maskeden sonra haritanın tek yamaya çökmesi (yeni sink) görüldü. Uyarlama yöntemin amacına uygun mu?
6. **Arşiv/yeniden hesap sapması:** yeniden hesaplanan v2 haritası B'de 1.4 puan kötü. Kaynağını bul (fp16 GPU mu,
   konum ortalaması mı). v4 yeniden hesap hattında; arşivle karışık bir karşılaştırma yanıltıcı mı?
7. **Canlı pipeline:** `lcr/pipeline.py: run` hâlâ v3 üretiyor. Test değerlendirmesinden önce v4 (seçili başlıklar +
   kişi için SAM) pipeline'a taşınmalı. Taşıyıp `29_freeze_look4.py` ile aynı kutuları verdiğini doğrula.
8. stuff'ta −1.45 (GA sıfırı içeriyor): başlık haritası stuff için tür bazında ayrı seçilmeli mi?
