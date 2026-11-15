# Tarihsel durum — birleşik Look/Check öncesi

Güncel yöntem için `docs/LCR.md` okunur. Bu dosya eski durum/adımların kayıpsız arşividir.

## Şu anki durum (2026-11-14)

**Codex hedefli Check v4 ön incelemesi:** `docs/CHECK_V4_ON_INCELEME.md`. Kullanıcının üç kutuyu ayrı
inceleme/frizbi sorusu yalnız B'de araştırıldı; sayısal ekler `docs/generated/CHECK_V4_BOXES_B.md` ve
`CHECK_V4_CONTRACT_B.md`. Bazı güçlü küçük frizbi adayları Look'un minimum kenar filtresinde eleniyor;
ilk dört hata örneğinde frizbiye oturan OWLv2 kutusu görsel doğrulandı. Ayrıca mevcut Retell girişi v4
kararlarını ham skorlarla yeniden yazıyor ve eski skor alanlarını okuyor. Bunlar tespit edildi, canlı
kod henüz düzeltilmedi. Seçili v4 ve kayıtlar korunuyor; tam v4 incelemesi henüz kapanmadı.

**Ana Check = Check v4 (özgün kural mantığı), Codex incelemesi bekliyor** (`docs/CHECK_V4.md`, görev
`docs/handoff/CODEX_CHECK_V4_REVIEW.md`). Kullanıcı kararı: Check "captioner'ın baktığı bölgede (Look) OWLv2'ye claim
var mı / yerine geçen başka bir şey var mı" sorusu olarak kalır; mantığı değiştiren öğrenilmiş doğrulayıcılar
(v2/v3) yalnız karşılaştırma. Önceden yazılmış protokolle (`docs/CHECK_V4_PROTOKOL.md`, §5 eşik değişikliği seçimden
önce) 144 konfigürasyon B'de ÇD ile denendi; seçilen **Q1 C2 A1 R2 G0 E0** (sınıf başına etiketsiz z ölçeği + Look'un
r-ağırlıklı ilk 3 önerisi). Validation (bir kez, proxy): canlı özgün kurala göre contradicted kesinliği 0.440 → 0.523
(GA [+0.007, +0.157]), yanlış yakalama 0.219 → 0.287 (GA [+0.015, +0.120]), J farkı GA sıfırı içeriyor. Zemin
claim'lerine artık contradicted denmiyor (B'de 0.5 kesinliğe ulaşan eşik yok). Öğrenilmiş doğrulayıcılar çok daha
güçlü (H6 tipi olumsuz bulgu). Look denetlendi, özgün mantıktan sapma yok; değişmedi. Testler: **74 geçti**.

Aşağıdaki Check v3 paragrafı artık yalnız karşılaştırma satırıdır:
**Check v3 tamamlandı (karşılaştırma)** (`docs/CHECK_V3.md`, görev `docs/handoff/CODEX_CHECK_V3_REVIEW.md`).
Önceden yazılmış protokolle (`docs/CHECK_V3_PROTOKOL.md`) bağımsız BLIP VQA/ITM ve captioner tutarlılığı (10 örnek
caption) eklendi; 48 aile tam iç içe ÇD ile B'de seçildi: **v2 + vqa_word + cons, logistic**, alternatif kuralı cons.
Validation (bir kez, otomatik proxy): AUROC 0.826 → 0.862 (GA [+0.022, +0.051]); hedef 0.5'te yanlış yakalama
0.682 → 0.841, kesinlik 0.506 → 0.484 (fark GA sıfırı içeriyor). Kazancın çoğu captioner'ın kendi tutarlılığından
(belirginlik/varlık ayrımı insan pilotuyla ölçülmeli). Look kutusu kırpımı ve look grubu katkı vermedi.
Sinyaller Colab T4'te çıkarıldı (`colab/run_check3.sh`); ağır işler artık yerelde koşulmuyor. Testler: **70 geçti**.

**Look E3/v4 Codex incelemesi tamamlandı.** Sonuç: `docs/LOOK_E3_INCELEME.md`;
otomatik sayısal ek: `docs/generated/LOOK_E3_REVIEW_METRICS.md`. Dondurulmuş dört başlık + kişi SAM
tercihi korundu. Canlı `pipeline.run` ve script 04 artık v4 üretir; script 29 aynı ortak kodu kullanır.
B/validation'daki 9.128 claim'in haritaları, kutuları, bölge indeksleri ve ham Check skorları birebir
korundu. SAM sonrası 3.031 eski yardımcı excess alanı düzeltildi; skor kutusunun proxy kaynağı açıklandı.
Dört sabit B draft'ı sıfırdan decoder/SAM ile, iki B görüntüsü gerçek taslak üretimiyle birebir eşleşti.
Bağımsız validation benchmark'ı doğru bölgeyi tekrar **%80.39** verdi. CPU testleri: **64 geçti**.
Test split'e bakılmadı; yeni validation seçimi veya Check/Retell eğitimi yapılmadı.

Başlık adları iki eksende de 1'den sayılınca L9H8, L10H8, L10H9, L9H3; düz indeksler değişmedi.
Etiketsiz olan başlık sıralaması; m/aile seçimi B insan kutularını kullanıyor. Aşamalı B ÇD tam iç içe
değerlendirme değildir. SAM seçimi maske içini değil çevreleyen kutuyu puanlar; Check skoru en yakın
OWLv2 önerisinden gelir. Chefer ve Salience DropOut uyarlamalarının kapsamı incelemede daraltıldı.
Tarihsel GPU arşivi ile yeniden hesaplanan S3 arasındaki sapmanın kök nedeni **açık**;
SEP/yuvarlama/tam CPU float16 tanıları arşivi bütünüyle açıklamadı. V4'ün kendi hattı birebir üretilebilir.

**Check incelemesi de tamamlandı:** `docs/CHECK_V2_INCELEME.md`. Mevcut Look v4 Check v2 olasılıkları,
kararları, alternatifleri ve eşikleri korundu; yalnızca açıklayıcı metadata/yardımcı alanlar güncellendi.
Sıradaki iş Check/Retell arasındaki açık kapıları kapatmaktır: dış görüntü katından bağımsız özellikler,
doğrulayıcıların kaydı, `p_correct` kanıt şeması ve sentetik bozmanın yeniden skorlanması.
Canlı Look v4'ün ham Check kayıtları öğrenilmiş Check v2 yerine geçmez. Retell henüz yeniden eğitilmedi.

Aşağıdaki gece notları ve eski deney sayıları tarihseldir; güncel Look kararı §7.8 ve E3 incelemesindedir.


## Sıradaki adımlar (sırayla)

0. **Codex: Check v4 incelemesi** (`docs/handoff/CODEX_CHECK_V4_REVIEW.md`). Ardından v4'ü canlı pipeline'a taşı
   (sınıf başına μ/σ dosyası + yumuşak bölge), alternatif önerisinin referansta olma oranını ölç, Retell'in girdi
   şemasını v4 kararları + alternatif üzerine kur. v2/v3 öğrenilmiş doğrulayıcılar yalnız ablation satırı.
   Ön incelemede bulunan küçük kutu filtresi kaybını B'de ölç; üç kutunun ayrı kontrolünü ortalamayla ayrıca
   kıyasla. Retell'in v4 kararını yeniden yazmasını ve alternatif–skor tutarsızlığını taşıma sırasında düzelt.
1. **Look v4 kararı korunacak.** İnceleme kapandı: `docs/LOOK_E3_INCELEME.md`. Canlı/dondurulmuş
   kutu eşitliği doğrulandı. Eski S3 arşiv sapması açık bir yeniden üretim sınırlamasıdır; validation'a
   göre yeni başlık, m veya stuff kuralı seçilmez. E1/E2/E3 geçmişi `LOOK_V2_PLAN.md` §7'de.
2. **Check/Retell eğitim verisinin dış kat bağımsızlığı.** `docs/CHECK_V2_INCELEME.md` ve `CHECK_V2.md`
   §5: model/lambda/eşiğin tamamını dış görüntü katından bağımsız üret ve doğrulayıcıları kaydet.
   Look seçicisinin bütün B'yi görmesi de bu tasarıma dahil edilmeli. Ana özellik ailesi mevcut
   keşif tercihi olarak kalır; validation üzerinden yeni seçim yapılmaz.
3. **Kanıt şeması ve bozma.** `p_correct` değerinin Retell'e girişini ve sentetik yanlış etiket için
   Check v2'nin yeniden skorlanmasını tanımla. Mevcut hard verdict doğrudan kelime değiştirme emri
   değildir. Canlı script 04'ün ham Check çıktıları öğrenilmiş Check v2 yerine kullanılmaz.
4. **Bağımsız kapılar.** İnsan pilotu (100 draft), proxy kesinlik/duyarlılığı ve alternatif önerilerin
   başarısı hâlâ gerekli. Retell öncesi oracle kanıt deneyi ve constrained-beam baseline; sonra gate
   weight decay, kanıt bozma ve s_image karıştırıcısı ablation'ları. İşlem kapıları:
   `docs/LITERATUR_TARAMASI.md`; mevcut Look-v4 Check kayıtları `*_look4_check2{,vis}.json`.
5. **Sonra Retell, ardından hipotez testleri/final rapor.** Epoch seçimi B'de ayrılmış kalibrasyon
   bölümünde yapılmalı. Test split geliştirmede okunmaz; final değerlendirmede bir kez kullanılır.

Paralel yürüyebilecek ekip işi: `annotation/pilot/annotate.html` ile kör insan etiketlemesi.
Eski Look önerileri ve kapanış ölçütleri tarihsel devir dosyalarında ve günlük kayıtlarında korunur.

---

