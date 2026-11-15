# Check v3 — sonuç ve yorum (2026-11-14, Claude)

> **Durum (2026-11-14, kullanıcı kararı):** Bu öğrenilmiş doğrulayıcı projenin Check tanımından sapıyor
> (dil önceliği, bütün görüntü VQA'sı, captioner tutarlılığı "baktığı bölgede nesne var mı" sorusu değil).
> **Ana Check değildir; yalnızca karşılaştırma/ablation satırıdır** ("bölgeye bakmadan ne kadar iyi yapılabilirdi").
> Ana Check özgün kural mantığıyla iyileştiriliyor: `docs/CHECK_V4_PROTOKOL.md`.


Protokol (koşmadan önce yazıldı): `docs/CHECK_V3_PROTOKOL.md`. Tüm sayılar otomatik tablodan:
`docs/generated/CHECK_V3_METRICS.md` (`scripts/34_check_v3_table.py`). Kaynak: `results/check_v3_selection.json`
(B, kilit), `results/check_v3.json` (validation, bir kez). Etiket **otomatik referans proxy'si**, insan etiketi değil.
Değerlendirme kişi dışı claim'ler (v2 ile aynı).

## 1. Ne yapıldı
- Yeni sinyaller (Colab T4, `colab/run_check3.sh`):
  - `scripts/31_check_vlm_signals.py`: bağımsız BLIP VQA evet/hayır (POPE fikri; 3 şablon varyantı) ve
    BLIP-ITM (tam görüntü, kelime düşürme, Look v4 kutusu kırpımı). Yalnız COCO/VQA v2 ince ayarlı modeller.
  - `scripts/32_check_consistency.py`: decoder_A'dan 10 nucleus örnek + 5 beam; sınıfın geçme oranı
    (SelfCheckGPT'nin nesne düzeyine uyarlaması).
- `scripts/33_check_v3.py`: 48 özellik ailesi × tam iç içe 5 katlı çapraz uydurma (hiperparametre iç 4 katlı
  log-loss, eşikler iç katların dışarıda-kalan skorlarından). Aile ve model sınıfı yalnız B'de seçildi, kilitlendi,
  sonra validation bir kez okundu. v2'ye karşı görüntü düzeyinde eşleştirilmiş bootstrap (2.000).

## 2. Seçim (B)
- Seçilen aile **v2 + vqa_word + cons**, model **logistic** (factorized ve GBM daha kötü log-loss),
  alternatif kuralı **cons** (örnek caption frekansı).
- B dış-kat: AUROC v2 0.825 → 0.872; aynı contradicted kesinlik hedefinde (0.5) yanlış yakalama 0.689 → 0.850.
- Tek başına sinyaller (B): cons ~0.84 AUROC (öncelikten ve v2'nin tamamından yüksek), VQA ~0.71, ITM ~0.68.
  `crop` (Look kutusu kırpımı) ve `look` grubu hiçbir ailede anlamlı katkı vermedi.

## 3. Validation (bir kez)
Tam tablo: `docs/generated/CHECK_V3_METRICS.md`. Özet (kişi dışı 1.503 claim):
- AUROC 0.826 → 0.862 (+0.036, GA95 [+0.022, +0.051]); AP(yanlış) +0.065 [+0.029, +0.103].
- Eşikten bağımsız kıyas, en düşük skorlu %20: yanlış oranı 0.561 → 0.621 (+0.060 [+0.013, +0.104]).
- Hedef 0.5'te: contradicted kesinliği 0.506 → 0.484 (fark GA sıfırı içeriyor), yanlış yakalama 0.682 → 0.841.
  Hedef 0.6'da v3, v2'ye yakın yakalamada (0.656; v2 0.682) daha yüksek kesinlik (0.555; v2 0.506) veriyor.
- Gruplar: zemin (stuff) en çok kazanan (AUROC 0.728 → 0.803, yakalama 0.362 → 0.702); giyim 0.662 → 0.727;
  nesne 0.904 → 0.918.
- Alternatif önerisi (referans-dışı claim'de önerinin referansta olma oranı): OWLv2 0.330 → cons 0.480.

## 4. Yorum ve dikkat edilmesi gerekenler
1. **Kazancın çoğu captioner'ın kendi tutarlılığından** (standart katsayı cons_sample +1.05; vqa_word +0.47;
   detektör s_image +0.20). Bu sinyal görsel kanıt değil, captioner'ın belirsizliği. Etiket "referansta geçiyor
   mu" olduğu için cons, "nesne görüntüde var mı"nın yanında "anlatıcılar bunu söyler mi"yi (belirginlik)
   de öğreniyor olabilir. İnsan pilotu bu ikisini ayıracak tek kapı. Proxy diye yazılmalı.
2. **Look'un katkısı yine yok:** look ve crop grupları seçilmedi. H6 (bölgenin katkısı) için olumsuz bir bulgu:
   bağımsız ITM bile kutu kırpımında tam görüntüden daha iyi değil.
3. **Retell'e etkisi:** v3 hedef 0.5'te kişi dışı claim'lerin %40.7'sine contradicted diyor (v2 %31.5).
   Retell daha çok değiştirme sinyali alacak; kesinlik 0.5 olduğundan yarısı yanlış alarm. Retell için
   karar noktası (0.5/0.6/0.7) ve `p_correct`'in sürekli kullanımı yeniden düşünülmeli.
4. **Döngüsellik:** cons, Retell'in düzelteceği captioner'ın kendisinden geliyor. Retell eğitiminde bu özellik
   modelin kendi örneklemesine dayanıyor; ablation (cons'suz doğrulayıcı) ile ayrıca ölçülmeli.
5. Validation'da hiçbir seçim yapılmadı; seçim kuralı dosyada önceden yazılı. Bootstrap aralıkları tek tek;
   48 aile arasından seçim B'de yapıldığı için validation farkı seçim yanlılığı içermez.

## 5. Uygulama notları (protokole ek, ızgarayı değiştirmez)
- Ablation satırları sabit λ (1, 1) ile; protokol λ belirtmiyordu.
- Factorized: öncelik modelinin özellik cezası 1 sabit; görsel kısım sklearn LogisticRegression,
  C = 1/(2λ_w), sınıf-dengeli ağırlık. GBM: sınıf önceliği λ_c = 1 sabit.
- Cons: Colab'da `--bs 64` T4'te bellek taşırdı, `--bs 16` ile koşuldu (tohum batch başına, sonuç batch
  boyutuna bağlı). Cons için ViT özellikleri Colab'da yeniden hesaplandı (draft'lar önceki önbellekten).
- joblib paralelliği betik adı nedeniyle çalışmadı; aileler sıralı koşuldu (sonuç etkilenmez).
- Kayıtlar: `data/cache/pipeline/A_{B,validation}_look4_check3.json` (B'de iç içe dış-kat skorlar). Validation
  eşikleri tc 0.784 / ts 0.809 (hedef 0.5): belirsiz bandı dar. Seçilen λ_c=1, λ_w=10.
