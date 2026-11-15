# Look E3/v4 incelemesi — Codex, 2026-11-14

## Karar ve kapsam

Dondurulmuş v4 tercihi korunuyor: sıralaması etiketsiz hesaplanan dört başlık ve kişi için SAM kutusu.
Canlı `pipeline.run` ve script 04 artık v4 üretiyor. Yeni bir validation seçimi, test split erişimi,
Check model seçimi veya Retell eğitimi yapılmadı. Sayısal ek:
[`generated/LOOK_E3_REVIEW_METRICS.md`](generated/LOOK_E3_REVIEW_METRICS.md).

Kutulara dayalı bağımsız validation yeniden ölçümü doğru bölgeyi yeniden %80.39 verdi;
önceden dondurulmuş karşılaştırma %76.10 → %80.39 olarak kalıyor. Bu ölçü insan çizimi sınıf kutularıyla
geometrik eşleşmedir; yanlış caption iddiaları için insan doğruluk etiketi değildir.

## Düzeltilen uygulama sorunları

- **Canlı v4 eksikti.** `lcr/look_v4.py` canlı çıkarım ve script 29 için ortak hesaplama sağlar.
  Bitmiş draft yeniden tokenize edilir; BOS değiştirilir; ortalama SEP'i dışarıda bırakır.
  İmzalı başlık haritaları, E3 önbelleği gibi, ortalamadan önce float16'ya çevrilir.
  Tür düzleştirmesi ve kişi SAM istemi dondurulmuş kuralla hesaplanır. SAM görüntü başına bir kez kodlar.
- **SAM sonrası yardımcı excess yanlıştı.** `rec_att.excess` SAM kutusunu, `excess_by_k["4"]` önceki
  OWLv2 kutusunu anlatıyordu. B'de 2.262, validation'da 769 alan düzeltildi. `relabel` da SAM geometrisini
  koruyarak eşlenen önerinin yeni sınıf skorunu kullanıyor. Check v2 bozma kısıtı devam ediyor.
- **Geometri ile skor kaynağı ayrıldı.** `box` SAM geometrisi; `region` OWLv2 skorunun öneri indeksi.
  `score_box`, `score_source=owl_nearest_proposal_proxy` ve `sam_proposal_iou` bu ayrımı açıklar.
  Sınıf skoruna bakmadan en çok örtüşen geçerli öneriyi eşlemek, mevcut önbellekle uygulanabilir bir
  yaklaşımdır; SAM kırpımının yeniden doğrulanması değildir. B'deki GT'li kişilerde 215/2.244 eşlemenin
  IoU'su 0.5'in altında. Bu yüzden yeni skor-kutusu eşiği koymadım veya Check skorunu olduğundan güçlü sunmadım.
- **Başlık isimleri tutarsızdı.** Düz indeksler `[103,115,116,98]` değişmedi. İki eksen de 1'den
  başlarsa adlar **L9H8, L10H8, L10H9, L9H3** olur. Önceki metin katmanı 1'den, başlığı 0'dan sayıyordu.
- **B2 maske skoru diye fazla geniş tanımlanmıştı.** Kod, maskenin içindeki kütleyi kullanmıyor;
  maskenin çevreleyen **dikdörtgeni** içindeki attention kütlesini dikdörtgen alanının kuvvetine bölüyor.
  Dokümantasyon ve metadata bu tanımı kullanıyor. Yöntemi değiştirip yeniden seçmedim.
- **Sıra topluluğunda eşitlikler bozuluyordu.** Çift `argsort`, eşit sıfır arka plana farklı uzamsal
  sıralar veriyordu. `rank_maps` artık ortalama sıra kullanır. Bu, reddedilen A6'nın rank dallarının
  tarihsel sonuçlarına uygulama sınırlaması getirir. Seçilen A6 dalı olasılık ortalamasıydı; v4 bu işlevi
  kullanmaz. Tarihsel sayılar yeniden yazılmadı; yeni genişletilmiş seçim yapılmadı.
- **Çıktı koruması.** Script 29 mevcut sonuçları `--overwrite` olmadan değiştirmez; `--suffix look4_review`
  bağımsız yeniden üretim sağlar. Script 04 raw/başka decoder ile dondurulmuş v4'ü kullanmayı reddeder.
  V4 yalnızca mevcut seçili haritayı sunar; k=4 alanı uyumluluk anahtarıdır, son dört katman ortalaması değildir.

Orijinaller `data/cache/pipeline/archive/pre_e3_review/` altında saklandı. Ana v4 kayıtları ve
`*_look4_check2{,vis}.json` yardımcı alanları güncellendi. Check dosyalarında tüm eski `rec_att`/`rec_det`
alanlarının (olasılık, karar, alternatif, eşik dahil) değişmediği otomatik olarak doğrulandı.

## Seçim protokolünün değerlendirmesi

Görüntü katları ve önceden sabitlenmiş bir puan kabul eşiği makul bir geliştirme kuralı;
p-değeri ya da aşırı uyum garantisi değildir. Aile içi ÇD farklı ızgara boyutlarını kısmen dengeler.
Ancak aile kazançlarının en büyüğünü seçmek bir üst seçim katıdır. Etiketsiz sıralama tüm B'nin
özelliklerinden hesaplandığı için transdüktiftir. B aşamasının haritası da A'da bütün B'nin kutularıyla
seçilmiştir. Dolayısıyla B ÇD değerleri bütün A→B seçicisinin tamamen bağımsız dış-kat performansı
olarak sunulmamalıdır. Bu ayrım, bağımsız validation raporunun yerini değiştirmez.

Validation öncesi dondurulmuş tek son yöntem için ana toplam karşılaştırmanın görüntü bootstrap
aralığı kullanılabilir. Dokuz aileye ve üç türe ait ayrı %95 aralıklar eşzamanlı %95 güvence sağlamaz;
reddedilen aileler, alt gruplar ve sonradan yapılan B tanıları keşifsel kabul edilmelidir. E1/E2/E3
boyunca tekrar kullanılan validation, artık tamamen dokunulmamış bir seçim kaynağı sayılamaz.
Final test gelecekte, bir kez ve önceden sabitlenmiş raporlama planıyla kullanılmalı.

m=1/2/3 yalnızca B'de, mevcut sıralama üzerinde tanısal olarak ölçüldü; hiçbiri dört başlığı geçmedi
(otomatik ekte tablo). Validation'a bakarak daha küçük m veya stuff'a özel yeni sıralama seçilmedi.
Stuff kaybının aralığının sıfırı içermesi, eşdeğerlik ya da zararsızlık kanıtı değildir.
Retell'e taşınırken B doğrulayıcılarının dış katından bağımsız Look özellikleri de düşünülmelidir;
Look seçimi bütün B'yi gördüğü için Check çapraz uydurması tek başına bütün zinciri dışarıda bırakmaz.

## Başlıkların ve SAM'in anlamı

[Kang ve ark.](https://arxiv.org/html/2503.06287v1) görsele ayrılan attention toplamı ve bağlı-bileşen
entropisiyle başlıkları sıralar. BLIP cross-attention'da bütün anahtarlar görsel olduğundan toplam
ölçütünün ayırt ediciliği düşüktür. Aynı görüntüde farklı sınıfların harita ayrışmasını kullanmak
savunulabilir bir uyarlama; `specificity/(1+entropy)` makalenin birebir skoru değildir. Entropinin
8-komşuluk/ortalama eşik tanımı kaynakla uyumlu. Tek geniş bileşen de düşük entropi alabilir;
skorun her mimaride yerelleştirmeyi garanti ettiği söylenemez. Başlık sıralaması etiketsizdir;
**başlık sayısı ve aile seçimi B'nin insan kutularıyla ayarlanmıştır**.

`results/look_e3_review_B.png` deterministik B örneklerini ve dört ayrı haritayı gösterir. L10H8 ve
L9H3 kişi, L9H8 ve L10H9 thing/stuff üzerinde farklı kuvvetler gösteriyor; sayısal ayrım ekte.
Bu gözlem başlıkların nedensel veya evrensel olarak özelleştiğini kanıtlamaz. Galeride başa daralan
SAM kutuları ve kısmen örtülü kişilerde parça seçimi de görülüyor. Birden çok GT kişi kutusu bulunan
B iddialarında doğruluk %78.80, tek kutulularda %81.14. Bu ölçü **herhangi bir aynı sınıf kutusuna**
IoU ile ölçülür; “iki kişi”nin tamamını bulma veya belirli kişinin referansını çözme testi değildir.

## Chefer ve Salience DropOut

[Chefer makalesi](https://arxiv.org/html/2103.15679v1) ve
[yazarların kodu](https://raw.githubusercontent.com/hila-chefer/Transformer-MM-Explainability/main/DETR/modules/ExplanationGenerator.py)
ile karşılaştırınca self-attention güncellemeleri → artık çıkar/normalize et/kimlik ekle →
transpozlu cross-attention güncellemesi sırası uyumlu. Sıfır satırı boş tutmak sayısal korumadır.
`R_ii=I` yalnızca önbellekteki **bağlamsallaştırılmış ViT özellikleri** düzeyindeki açıklama için
savunulabilir; tam görüntü encoder'ının piksel açıklaması değildir. Hedef de özgün sınıf logiti yerine
kelimenin log olasılığıdır. Yüksek pointing ve daha kötü kutu sonucu matematiksel çelişki değil:
tepe konumu ve kutu içi toplam kütle farklı ölçülerdir. Bunu Chefer'in genel başarısızlığı diye yazmamalıyız.

[PnP-OVSS'nin güncel yazar sürümü](https://arxiv.org/html/2311.17095v4) ITM-GradCAM haritasıyla,
tekrarlı görüntü yamaları silme ve yeniden encoder geçişi kullanıyor. E3 ise tek tur, claim başına
%5/%10 **bağlamsallaştırılmış özellik maskelemesi**, yeniden sink haritası ve eşit ağırlıklı birleştirme
uyguluyor. Niyet benzer; aynı yöntem değildir. CLS ve diğer ViT token'ları maskelenen bölgeden bilgi
taşımayı sürdürebilir. Yeni sink gözlemi bu sınırlı uyarlamanın negatif sonucu olarak kalmalı;
literatürdeki yöntemin bütünüyle denenip elendiği iddiası düzeltilmeli. Yeni encoder hattı/deney ızgarası açılmadı.

## Arşiv sapması ve yeniden üretim

`review_look_e3_drift.py`, B'deki sabit küçük bir örnekte SEP dahil/harici ortalamayı, attention/ortalama
ve çıkarma yuvarlamasını ve tam CPU float16 decoder'ını karşılaştırır. SEP dahil etmek bazı farkları
azaltıyor; yarım duyarlık arşivi yeniden üretmiyor. Eski GPU hattının kesin kök nedeni **çözülmedi**.
“Muhtemelen fp16” artık sonuç gibi kullanılmıyor. Tarihsel arşiv karşılaştırması ile aynı yeniden hesap
hattında yalnızca başlık havuzunu değiştiren karşılaştırma ekte ayrı verildi. Arşiv-v4 farkını yalnızca
başlık seçiminin etkisi diye adlandırmıyoruz.

Buna karşılık yeni canlı v4 → E3 dondurma eşitliği doğrulandı: iki split'te 9.128 iddianın harita, kutu,
bölge ve ham skorları birebir; dört sabit B draft'ında sıfırdan decoder+SAM ile birebir; iki gerçek B
üretiminde taslak ve kutular birebir. Bunlar tüm GPU/back-end koşulları için bit düzeyi eşitlik garantisi değildir.

## Doğrulama ve devam

- CPU testleri: **64 geçti**.
- Bağımsız insan-kutusu benchmark'ı tekrar çalıştı; v4 toplam bölge doğruluğu değişmedi.
- Test split okunmadı; dondurulmuş seçim JSON'ları ve önceki validation deney sonuçları değiştirilmedi.
- Yeniden üretim: script 29 `--suffix look4_review`; `review_look_e3.py --live`; `review_look_e3_drift.py`;
  ana düzeltmeden sonra `review_look_e3.py --verify-only`; `look_e3_review_table.py`.
- Ana kayıtları güncelleme: yedek sonrası script 29 `--overwrite --sync-check-metadata`.
- Sonraki iş Check/Retell devir noktalarıdır: Check modelinin/eşiklerinin kaydı, dış görüntü katından
  bağımsız eğitim özellikleri, `p_correct` kanıt şeması ve sentetik bozmanın yeniden skorlanması;
  bkz. `CHECK_V2_INCELEME.md`. Canlı Look v4'ün ürettiği ham Check kayıtları kalibre Check v2 değildir.
  Bu inceleme yeni Check modeli veya Retell eğitimi başlatmaz.
