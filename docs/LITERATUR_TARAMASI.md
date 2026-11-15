# İşlem bazlı literatür taraması (2026-11-13)

Amaç: pipeline'daki **her işlemi** (sadece Look/Check/Retell aşamalarını değil) literatürle karşılaştırmak,
hangi varsayımımızın literatürde zaten çürütülmüş ya da düzeltilmiş olduğunu **koşmadan önce** görmek.
Böylece "koştur → kötü sonuç → geri dön" döngüsü kısalır. Rapordaki Related Work bölümü de buradan çıkar.

Her işlem için: **Şu an** (kodda ne yapıyoruz) · **Literatür** · **Çıkarım** (ne değişmeli / neyi ablation'la
ölçmeliyiz) · **Kapı** (bir sonraki işleme geçmeden önce bakılacak ölçüm). Kaynakların hepsi web'de varlığı
doğrulanmış makalelerdir; listesi en sonda. `refs.bib`'e eklerken künyeyi (sayfa, DOI) tekrar kontrol et.

Look (işlem 5) ayrıntısı `docs/LOOK_V2_PLAN.md`'de; burada yalnızca özeti var.

---

## 0. Neden "kapı" var: önceki oturumda neden fark edilmedi?

Round-1 sisteminde Look'un seçtiği bölgeler insan kutularına göre **şans seviyesinde** çıktı
(pointing %23.9, şans %37.1; bölge IoU≥0.5 %12.9, detektör %82.7, oracle %95.3). Bunun önceki oturumlarda
(Fable dahil) fark edilmemesinin nedenleri:

1. **Ara aşama için ground-truth kapısı yoktu.** Look hiçbir zaman insan kutusuyla ölçülmedi. Sadece uçtan uca
   sonuçlara (CIDEr, gate değerleri) bakıldı. Uçtan uca bir metrik bozuk bir ara aşamayı saklar: Retell
   kanıtın gürültü olduğunu öğrenip gate'leri kapatıyor ve CIDEr neredeyse değişmiyor.
2. **"Çalışıyor" = "hatasız koştu ve kendi içinde tutarlı".** Kontrol edilen şey script'lerin çökmemesi ve
   sayıların birbirini tutmasıydı, ölçtüğümüz şeyin doğru olup olmadığı değil.
3. **Proxy'ler doğru kabul edildi.** Sink düzeltmesi "pointing %68.5" verdi ama bu ölçüm **detektör kutusuna**
   göreydi. Detektör kutusu Look'un ne olması gerektiğinin değil, OWLv2'nin ne gördüğünün proxy'si.
4. **Önerideki varsayım literatürle test edilmedi.** "Ham cross-attention nesneyi gösterir" önerinin temel
   varsayımıydı. Oysa BLIP'in atası ALBEF bile lokalizasyon için ham attention değil **attention üzerinde
   Grad-CAM** kullanıyor. Attention sink / register literatürü (Darcet ICLR'24, Kang ICLR'25) ve
   HaloProbe (ham attention'ın token konumu ve tekrarla karıştığını gösteriyor) de bu riski açıkça söylüyor.
   OWLv2 model kartı da skorların sorgular arasında karşılaştırılamayacağını yazıyor; bizim "contradicted"
   kuralı tam bunu yapıyor (işlem 6–7).
5. **Bir dil modeli ajanı varsayılan olarak "görevi bitirmeye" optimize olur.** Açıkça istenmedikçe kendi
   ürettiği ara sonucu bağımsız bir kaynağa karşı doğrulamaz. Bu yüzden kural koyuyoruz:
   **her işlemin bir kapısı var ve kapı geçilmeden sonraki işleme geçilmez** (AGENTS.md).

---

## 1. Veri bölme ve captioner fine-tune

**Şu an.** Flickr8k train 6000 görüntü, A/B yarılarına bölündü (SPLIT_SEED=415). BLIP-base decoder A'da
fine-tune edildi, vision encoder donuk. Retell, captioner A'nın **B üzerindeki** draft'larıyla eğitiliyor
(`06_train_retell.py`: `{decoder}_{other}`). Eşikler de B'de seçiliyor.

**Literatür.** Captioner'ın gördüğü veride ürettiği draft'lar, görmediği veridekilerden daha az hata içerir.
Düzeltici (revisor) eğitenler bu yüzden düzelticiyi ayrı veri ya da sentetik hata üzerinde eğitiyor (LURE).
Hata dağılımının eğitimde ve testte aynı olması gerekir.

**Çıkarım.** A/B ayrımı doğru ve korunmalı: B'deki draft'lar, validation'daki draft'larla aynı türden hata
içerir. Ancak hem eşik seçimi hem Retell eğitimi B'yi kullanıyor. Bu küçük bir "aynı veriyi iki kez kullanma"
riski: eşik B'ye uyarlanıyor, Retell de o eşikle B'de eğitiliyor. İki seçenek var: (a) eşik kalibrasyonu
için B'nin içinde sabit bir alt küme ayırmak (ör. B'nin %20'si), ya da (b) bunu raporda açıkça yazmak.
(a) ucuz ve temiz; öneri bu.

**Kapı.** Draft hata oranı (referans proxy'si) B'de ve validation'da benzer olmalı. A'daki oran belirgin
şekilde düşükse ayrımın neden gerekli olduğu da gösterilmiş olur.

## 2. Draft üretimi (decoding)

**Şu an.** Beam search, `num_beams=3, max_new_tokens=30, min_new_tokens=5`, sampling yok.

**Literatür.** CHAIR makalesi (Rohrbach EMNLP'18): standart cümle metriklerinde en iyi olan modeller her
zaman daha az hallucinate etmiyor, hallucinasyon büyük ölçüde dil önceliklerinden geliyor. OPERA (CVPR'24)
ve VCD (CVPR'24) hallucinasyonu decoding sırasında azaltan, eğitim gerektirmeyen yöntemler.

**Çıkarım.** Decoding ayarları draft'taki hata oranını değiştirir, dolayısıyla Retell'in düzeltecek ne kadar
hatası olduğunu da. Draft ayarını sabitleyip raporlamak yeterli. Ancak "eğitimsiz decoding düzeltmesi"
(VCD / OPERA tarzı) Retell'e karşı **doğal bir baseline**: "kanıta gerek var mı, sadece decoding ile aynı
kazanç alınır mı?" sorusunu cevaplar. OPERA'nın resmi kodu eski bir transformers sürümüne bağlı; VCD'nin
fikri (görüntü bozulmuşken ve sağlamken logit farkı) BLIP'e kendimiz uygulayabileceğimiz kadar basit.
Bunu Retell çalıştıktan sonraki bir baseline olarak not ediyoruz, şimdi değil.

**Kapı.** Yok (sabit ayar). Draft'ın hata oranı raporlanır.

## 3. Claim çıkarma ve sözlük

**Şu an.** NLTK POS tagger ile isimler alınıyor, 169 sınıflık sözlüğe (1 person, 130 thing, 38 stuff)
eşleniyor. Bir sonraki sözlük isminin niteleyicisiyse atlanıyor ("water bottle" → bottle). Her sınıf için
ilk geçiş tutuluyor.

**Literatür.** CHAIR sabit bir nesne listesi ve eşanlamlı listeleri kullanıyor. ALOHa (NAACL'24) nesneleri
LLM ile ayrıştırıp referanslarla Hungarian eşlemesi yapıyor ve sabit listenin kaçırdıklarını gösteriyor.
FaithScore (Findings EMNLP'24) atomik olgulara ayırıyor. SPICE (ECCV'16) cümleyi sahne grafiğine çeviriyor.
AMBER nesne varlığı, nitelik ve ilişki hallucinasyonunu ayrı ölçüyor.

**Çıkarım.** Kapsamımız (yalnızca nesne varlığı) bilinçli bir sınırlama ve CHAIR / POPE ile aynı düzlemde.
Raporda "attribute ve relation hallucination kapsam dışı" diye açıkça yazılmalı. Asıl risk ölçüm hatası:
claim çıkarıcı bir nesneyi yanlış sınıfa eşlerse Look ve Check yanlış şeyi doğrular. ALOHa'nın gösterdiği
gibi sabit liste eşanlamlıları kaçırır ("puppy" → dog?). Bu bir kapıyla ölçülmeli.

**Kapı (yeni, ucuz).** 100 rastgele validation draft'ında claim çıkarıcının kesinliği ve kapsamı elle
kontrol edilir. Bunu kör etiketleme pilotuna eklemek mümkün. Hedef: kesinlik ≥ %95. Kaçırılan isimler
listelenip sözlüğe eklenir.

## 4. Doğruluk proxy'si (referansta geçiyor mu?)

**Şu an.** Bir claim, beş referanstan birinde aynı sınıf geçiyorsa `in_refs=True` sayılıyor. Hem eşik
kalibrasyonunda hem B1'de bu proxy kullanılıyor.

**Literatür.** CHAIR COCO'da segmentasyon etiketleri ile referans nesnelerini birleştiriyor. Yalnızca
referansa dayanmak kaçırılan (eksik) nesneleri "hallucination" sayar. ALOHa ve FaithScore da referans veya
görüntü üzerinden doğrulama yapıyor ve iki kaynağın farklı hatalar verdiğini gösteriyor.

**Çıkarım.** Referans proxy'si **tek yönlü yanlış**: referansta geçmeyen ama görüntüde olan nesneler
"hatalı" sayılıyor (yanlış pozitif hallucination). Elimizde artık ikinci bir kaynak var: Flickr30k Entities
kutuları. F30kE ifadeleri de referans cümlelerinden geldiği için tamamen bağımsız değil, ama **kutunun
varlığı** görüntüye dair bir insan kararı. Öneri: "in_refs" yanında "in_f30k_boxes" alanı. İkisinin
anlaşmadığı claim'lerin oranı raporlanır ve insan etiketi pilotunda bu claim'lere ağırlık verilir. Her
yerde "automatic proxy" diye yazılmaya devam edilir.

**Kapı.** Pilot insan etiketleriyle (100 draft) proxy'nin kesinlik/duyarlılığı raporlanır. Bu olmadan
proxy'ye göre kalibre edilen eşik raporda "proxy-calibrated" diye geçer.

## 5. Lokalizasyon (Look)

Ayrıntı: `docs/LOOK_V2_PLAN.md`. Özet:

**Şu an (2026-11-14).** Look v2: son 4 katmanın ortalama cross-attention'ı, caption ortalamasını çıkaran
sink düzeltmesi, ardından kütle / alan^0.5 × sorgudan bağımsız objectness^0.5 ile kutu seçimi.
İnsan kutusuyla validation bölge doğruluğu %75.3; eski ham/excess-mass yönteminin şans düzeyindeki
sonucu tarihsel taban çizgisidir. B'deki yeni katkı ayrımı ve kişi hata analizi: `docs/LOOK_V2_INCELEME.md`.

**Güncelleme (2026-11-14, E3).** Look v4: B'de etiketsiz seçilen 4 başlık (Kang+ CVPR'25 uyarlaması) + kişi için
haritanın tepe noktasıyla SAM (Kirillov+ ICCV'23) kutusu. Validation doğru bölge %80.4 (v3 %76.1; +4.3, GA95
[+2.5, +6.1]); kişi %76.5 (v3 %67.1). Chefer relevance, Salience DropOut, oran tipi sink, isim öbeği, katman
seçimi ve topluluk için sınırlı uyarlamalar denendi, kabul eşiğini geçmedi. Bu, özgün yöntemlerin
tam tekrarlandığı anlamına gelmez. Codex incelemesi ve canlı v4 doğrulaması: `docs/LOOK_E3_INCELEME.md`. Ayrıntı: `docs/LOOK_V2_PLAN.md` §7.8, `docs/LOOK_E3_PROTOKOL.md`.

**Literatür.** ALBEF (NeurIPS'21) ve PnP-OVSS (CVPR'24): attention × gradyan (Grad-CAM). Chefer (ICCV'21):
gradyanla ağırlıklandırılmış attention relevance. Abnar & Zuidema (ACL'20): katmanlar arası attention
rollout. Darcet (ICLR'24) ve Kang (ICLR'25): sink/register token'ları haritayı kirletiyor. Kang (CVPR'25):
yalnızca birkaç "lokalizasyon head'i" nesneyi gösteriyor. Zhang (ICLR'25): MLLM'ler nereye bakacağını
biliyor, attention ile kırpma işe yarıyor. Jacovi (EMNLP'21): kontrastif açıklama ("neden *X* değil de
*Y*?"). Zhou (CVPR'20, POS-SCAN) ve Liu (AAAI'17): captioning attention'ı zayıf grounded, denetimle
iyileşiyor. Ölçüm: pointing game (Zhang ECCV'16), enerji tabanlı pointing (Score-CAM), Flickr30k Entities
(Plummer ICCV'15). HaloProbe (ICML'26): ham attention token konumu ve tekrarla karışık.

**Çıkarım.** Plan: harita kaynakları (S1–S5) × head seçimi × düzleştirme × bölge seçimi. B'de aşamalı
ileri seçim, validation'da raporlama. Kesin ilke: Look **iddia edilen sınıfın detektör skorunu**
kullanmaz, yoksa doğrulama döngüsel olur. Objectness (sınıftan bağımsız) kullanılabilir.

**Kapı.** Validation'da pointing ve bölge-IoU şansın açıkça üstünde, B3 detektörüne yakın olmalı.
Hedef sayı LOOK_V2_PLAN §5'te.

## 5a. Attention'ı okuma (Look'un içindeki gizli kararlar)

Look tek bir işlem gibi görünüyor ama içinde ayrı ayrı yanlış yapılabilecek birkaç karar var. İlk taramada
bunları ayrı yazmamıştık:

| Karar | Şu an | Alternatif / risk | Kapı |
|---|---|---|---|
| Hangi geçiş | Bitmiş draft üzerinde teacher-forced geçiş | Ham p−1 attention'ı nedensel olarak üretim önekiyle uyumlu; bütün caption ortalaması çıkarılmış sink haritası ise sonraki konumları da kullanan post-hoc ölçümdür | Prefix baseline ayrı ablation olabilir |
| Hangi zaman adımı | Kelimenin **üretildiği** adım (p−1 konumu p'yi tahmin ediyor) | Kelime okunduktan sonraki adım (p). Liu+ AAAI'17 ve Zhou+ CVPR'20 üretim adımını kullanıyor | ablation (S-kaynakları ile birlikte) |
| Alt-kelimeler | Kelimenin **ilk** parçası ("skate" ##board) | Parçaların ortalaması ya da son parça. İlk parça üretilirken kelime henüz belirsiz olabilir | ablation, çok parçalı kelimelerde ayrıca raporla |
| Öbek | Sadece isim | İsim öbeği (sıfat + isim) ortalaması. F30kE kutuları öbek düzeyinde | ablation (bizim ekleme) |
| Katman / baş | Son 4 katman × tüm başlıklar ortalaması | §3.2 (baş seçimi, Kang+ CVPR'25) | LOOK_V2_PLAN §5 |
| CLS ve sink | CLS atılıyor. Sink: caption ortalaması çıkarılıp kırpılıyor | Darcet+, Kang+ | yapıldı |
| Koordinat | 24×24 ızgara, satır = y. BLIP görüntüyü kırpmadan 384×384'e yeniden boyutluyor, normalize koordinat doğrusal | Transpoze ya da eksen hatası haritayı şans seviyesine indirir | **yapıldı**: transpoze → %51.9, kendi → %88.3 |
| Kelimeye özgüllük | — | Harita sadece göze çarpan nesneyi gösteriyor olabilir | **yapıldı**: takas testi, kendi %87.5, başka claim %45.4 |

Son iki kapının script'i `scripts/14_look_sanity.py`. Her yeni harita varyantında tekrar koşulmalı.

## 5b. Alternatif aday üretimi

**Şu an.** `blip_alternatives`: claim konumunda captioner'ın kendi logit'lerinde ilk 30 token içindeki,
sözlükte karşılığı olan en iyi 3 farklı sınıf. Check'teki "contradicted" kararı ve B1 bunları kullanıyor.

**Literatür.** LURE: hallucination belirsizlikle (token olasılığı) ilişkili. Petryk (WACV'24): token düzeyinde
güven caption doğruluğunu öngörüyor. Captioner'ın ikinci adayı bu belirsizliğin doğal bir ölçüsü.

**Çıkarım.** Alternatif, kanıtın en önemli parçalarından biri (Retell "dog yerine ne?" sorusunun cevabını buradan
alıyor), ama hiç ölçülmedi. Kapı: referansta geçmeyen claim'lerde, ilk 3 alternatiften biri referansta geçen
doğru sınıf mı? Bu oran düşükse "contradicted → alternatifi öner" yolu baştan sınırlı demektir. Alternatifler ayrıca
detektörün tüm-görüntü sınıflarından da alınabilir (B2'nin kanıtı gibi). Bu bir ablation olur. Ucuz, CPU'da,
mevcut pipeline dosyalarıyla yapılabilir.

**Ölçüldü (2026-11-13, `scripts/15_alt_quality.py`, validation, referans proxy'si):** referansta geçmeyen 352
thing/stuff claim'inde 1. alternatif %36.6 oranında referansta geçiyor, ilk 3'ten biri %58.2 oranında.
1. alternatifin %6.5'i zaten draft'ın başka bir yerinde geçiyor. Yani "alternatifi öner" yolu yanlış claim'lerin
en fazla üçte biri ile yarısı arasında doğru kelimeyi sunabilir. Geri kalanında en iyi hamle kelimeyi
düşürmek ya da genelleştirmek olur. Detektör tabanlı alternatif kaynağı bu yüzden ablation'a değer.

## 6. Öneri kutuları ve detektör skorlaması

**Şu an.** OWLv2: görüntü başına top-100 öneri (objectness'a göre NMS). Gömmeler, logit shift ve scale
önbellekte. Sorgu şablonu tek: `"a photo of a {}"`. Skor = `sigmoid((e·q + shift)·scale)`.

**Literatür.** OWL-ViT (ECCV'22) ve OWLv2 (NeurIPS'23) açık sözlüklü detektörler. OWLv2'nin objectness
head'i, sorgudan bağımsız olarak tahminleri sıralamak ya da filtrelemek için kullanılabilir. **Doğrulanmış
uyarı:** OWLv2 model kartına göre skorlar **farklı metin sorguları arasında karşılaştırılamaz**. Petryk
(WACV'24) token düzeyinde güvenin caption doğruluğunu öngördüğünü gösteriyor. Bu, detektöre ek olarak
captioner'ın kendi olasılığının da bir sinyal olduğu anlamına geliyor.

**Çıkarım (önemli).**
- Öneri havuzu sorun değil: oracle IoU≥0.5 %95.3.
- **Sınıflar arası skor karşılaştırması düzeltilmeli.** `contradicted` kararı `s_alt > s_claim` türünde
  bir karşılaştırma içeriyor, oysa bu iki skor farklı sorgulardan geliyor. Bunun için seçenekler:
  (i) Sınıf başına normalizasyon: her sınıfın skorunu B'deki dağılımına göre yüzdeliğe ya da z-skora
  çevirmek (sınıf başına referansta-geçen / geçmeyen dağılımlarından).
  (ii) Sınıf başına kalibrasyon: Guo (ICML'17) temperature/Platt scaling'i sınıf başına bir logistic
  regresyonla yapmak. Az örnekli sınıflar için ortak bir parametreye büzülme gerekir.
  (iii) Karşılaştırmayı tamamen bırakıp yalnızca "bu sınıf bu bölgede var mı" sorusunu kalibre etmek.
  Ablation: ham / (i) / (ii).
- **Şablon ensemble'ı** (birkaç şablonun ortalaması) CLIP tarzı modellerde yaygın bir uygulama. Bunun için
  burada ayrıca bir makale doğrulamadık; denenirse "engineering choice" olarak yazılır.
- Captioner'ın kendi token olasılığı (Petryk) Check'e ikinci bir özellik olarak eklenebilir. Ablation'la
  ölçülür.

**Kapı.** Detektör skoru tek başına, sınıf başına AUROC ile (in_refs ve F30kE kutularına karşı) ölçülür.
Normalizasyon sonrası sınıflar arası karar kalitesi (aşağıda J) artmalı.

## 7. Karar kuralı (supported / contradicted / uncertain) ve çekimser kalma

**Şu an.** İki eşik var: `t_support`, `t_contra`. Ayrıca bölge bulunamazsa ya da excess-mass düşükse
"uncertain". Geçici değerler: 0.2 / 0.05.

**Literatür.** Selective classification (Geifman & El-Yaniv NeurIPS'17): risk–kapsam eğrisi, hedef risk
için eşik seçimi. Woodpecker ve LURE nesne düzeyinde doğrulama yapıyor. POPE varlık sorularını ikili
soruyor ve dengeli ölçüyor.

**Çıkarım.** "Uncertain" bir çekimser kalma seçeneği. Doğru değerlendirme risk–kapsam eğrisi: kapsam =
karar verilen claim oranı, risk = verilen kararlardaki hata. Tek bir J değeri yerine (ya da yanında)
bu eğri raporlanırsa "her şeye uncertain demek" ya da "her şeyi contradicted yapmak" gibi dejenere
çözümler görünür hale gelir. Önceki kalibrasyonda tam olarak bu oldu: balanced accuracy her şeyi
çelişkili saydı. Youden J'ye geçtik, eğri de eklenmeli.

**Kapı.** Validation'da: (a) risk–kapsam eğrisi, B3 ve rastgele bölgeye göre; (b) "supported" kesinliği
ve "contradicted" kesinliği ayrı ayrı, taban oranın üstünde.

## 8. Eşik kalibrasyonu

**Şu an.** `11_proxy_thresholds.py`: B'de ızgara araması, Youden informedness maksimize ediliyor.
Sonuç `LCR_THRESHOLDS` ile veriliyor.

**Literatür.** Guo (ICML'17): post-hoc kalibrasyon (temperature/Platt) ve ECE. Geifman: hedef riske göre
eşik.

**Çıkarım.** Izgara + J makul ve şeffaf. İki ekleme öneriliyor: (1) eşikler B'nin ayrılmış alt kümesinde
seçilir (işlem 1). (2) Seçilen eşiğin belirsizliği bootstrap ile raporlanır (B'yi yeniden örnekleyip eşiği
yeniden seçmek), çünkü düz bir J yüzeyinde eşik çok oynar. İşlem 6'daki normalizasyon yapılırsa eşikler
artık sınıflar arası ortak bir ölçekte olur.

**Kapı.** Bootstrap'ta eşik aralığı dar olmalı. Validation J'si B J'sine yakın olmalı (aşırı uyum yok).

## 8a. Check'in içindeki gizli kararlar (kod okuması, 2026-11-14)

`lcr/check.py` (`make_record`, `verdict`) ve `lcr/pipeline.py` (`finalise`) satır satır okunarak çıkarıldı:

| Karar | Şu an | Sorun / alternatif | Ne zaman |
|---|---|---|---|
| Sorgu metni | Tek şablon `"a photo of a {}"` | Şablon ensemble'ı; sınıf adının çok anlamlı olduğu durumlar ("glass", "bat") | Check, ablation |
| Skor ölçeği | Sigmoid olasılık, **tüm sınıflara tek eşik** | Sınıflar arası karşılaştırılamaz (işlem 6) → sınıf başına normalizasyon | Check, öncelikli |
| Alternatifin seçimi | Bölgede detektörün ilk 6 sınıfı ∪ captioner'ın 3 alternatifi, içlerinden **detektör skoru en yüksek** olan | Bu seçim de sorgular arası karşılaştırma. Ayrıca en yüksek skorlu sınıf çoğu zaman bölgedeki **başka** bir nesne (ör. köpeğin bölgesinde "grass") ya da claim'in üst/alt sınıfı | Check: alternatifi sınıf türüyle (thing/stuff) sınırlamak ve 5b'deki ölçümle karşılaştırmak |
| Bölge skoru | Seçilen tek kutunun skoru | Yumuşak bölge (R4): ilk m kutunun ağırlıklı skoru | Check |
| `s_image` | Claim sınıfının tüm görüntüdeki en yüksek detektör skoru, **"ours" kaydına da giriyor** | Look'tan bağımsız bir detektör sinyali. Retell "ours" bunu kullanarak Look'u yok sayabilir. H6'da (attention vs detektör) **karıştırıcı**. Ablation: `s_image`'sız "ours" | Retell |
| Excess-mass kapısı | `t_excess = 0` (fiilen kapalı) | Haritadan kutuya varyantı değişince yeniden tanımlanmalı | Look dondurulunca |
| Bölge yok | → uncertain | Doğru (bulamamak yokluk kanıtı değil). Oranı raporlanmalı | — |
| Tekilleştirme | Sınıf başına ilk geçiş | "two dogs ... a dog" tek claim. Doğru, ama sayı hataları kapsam dışı (yazılmalı) | rapor |

### 8b. Check v3 (2026-11-14): bağımsız VQA/ITM ve captioner tutarlılığı
POPE tarzı evet/hayır sorusu (BLIP-VQA), BLIP-ITM ve SelfCheckGPT'nin nesne düzeyine uyarlaması (decoder_A'dan
10 örnek caption) önceden yazılmış protokolle denendi (`docs/CHECK_V3_PROTOKOL.md`). Seçilen: v2 + vqa_word + cons.
Validation AUROC 0.826 → 0.862; ayrıntı `docs/CHECK_V3.md`. **Kapı uyarısı:** cons captioner'ın kendi sinyali;
referans proxy'siyle ölçülen kazancın ne kadarının varlık, ne kadarının belirginlik olduğu yalnız insan pilotuyla
ayrılabilir. Look kutusu kırpımı (ITM) katkı vermedi. **Kullanıcı kararıyla v3 ana Check değil, karşılaştırma satırı.**

### 8c. Check v4 (2026-11-14): 8a'daki gizli kararlar, özgün kural içinde
8a tablosundaki seçenekler (sorgu ensemble'ı, sınıf başına ölçek, alternatif türü, yumuşak bölge R4, `s_image`
koruması, excess kapısı) önceden yazılmış protokolle tek ızgarada denendi (`docs/CHECK_V4_PROTOKOL.md`). Kazanan:
sınıf başına etiketsiz z ölçeği + yumuşak bölge (ikisi tek başına yetersiz, birlikte J 0.341 → 0.473, B ÇD).
Alternatifi türle sınırlamak kalibre skorla kazanmadı; prompt ensemble etkisiz; excess kapısı zararlı.
Validation: `docs/CHECK_V4.md` (artık tarihsel kıyas). Öğrenilmiş doğrulayıcılarla fark büyük;
bu belirli uygulama/proxy altında bir olumsuz sonuçtur, bütün bölge kanıtı yöntemlerinin üst sınırı değildir.

### 8d. Birleşik attention–nesne yolu (2026-11-14, Codex)

Güncel tanım `docs/LCR.md`, önceden yazılan B protokolü `docs/LCR_BIRLESTIRME_PROTOKOL.md`.
Literatür ana işlemlerin uygulanmasını destekler; Check'in yerine VQA/ITM, dil önceliği veya captioner
tutarlılığı konmaz. [OWLv2](https://arxiv.org/abs/2306.09683) ve
[model kartı](https://huggingface.co/google/owlv2-base-patch16-ensemble) nesne önerileri ve sorgu skorlarını
tanımlar. Kayıtlı öneri özelliğinin sorgulanması, seçili kırpımın yeniden kodlanmasıyla aynı işlem değildir.
Minimum iki-patch kenar, ilk üç kutu, max/ağırlıklı birleştirme ve z ölçeği bu projedeki mühendislik
tercihleridir; kaynaklardan alınmış doğruluk garantileri değildir.

**B kapısı:** Küçük kutu filtresi, attention haritası sabitken B_fit insan kutularına karşı kaldırıldı;
alan kuvveti 0.5 kaldı. Check, yalnız Look adayları arasında max kullanır. Sayılar otomatik ekte:
`docs/generated/LCR_DEVELOPMENT.md`. Her kutu ayrı kaydedilir; başka kutudaki düşük claim skoru,
bir kutudaki güçlü desteği tek başına geçersiz kılmaz. Normalizasyon ve eşikler fit görüntülerinden
bağımsız tutulur; Look seçimi B_fit'i gördüğü için bütün zincirin tam dış-kat genellemesi diye sunulmaz.

**Açık kapı:** Alternatifin bölgede görünmesi, semantik ikame olmasını kanıtlamaz. Proxy'deki alternatif
başarısı sınırlıdır. Sıradaki GPU işi, aynı donmuş OWLv2'nin sabit attention kırpımlarındaki nesne
önerilerini yeniden çıkarmasıdır (`docs/handoff/CLAUDE_LOOK_CHECK_GPU.md`). Henüz yapılmadı veya
kazandığı gösterilmedi. Retell kullanıcı kararıyla bu sınırın dışında bekliyor.

## 9. Sentetik bozma ve zor negatifler

**Şu an.** Eğitimde claim'lerin ρ oranı, ikinci derece eş-geçişe göre en yakın aynı türden yanlış sınıfla
değiştiriliyor (frisbee → ball). Bölge aynı kalıyor, skorlar yeni etiketle yeniden hesaplanıyor. Person
claim'leri bozulmuyor.

**Literatür.** LURE: hallucination'ın eş-geçiş, belirsizlik ve konumla ilişkili olduğunu gösteriyor ve
düzelticiyi bu istatistiklerle üretilmiş sentetik hallucination'larla eğitiyor. Bizim eş-geçiş tabanlı
bozmamız doğrudan bu fikrin bir benzeri. NegCLIP (ICLR'23): kompozisyonel zor negatifler olmadan modeller
"kelime torbası" gibi davranıyor. Longpre (EMNLP'21): bağlamdaki varlık değiştirildiğinde modeller bağlamı
yok sayıp ezberine dönüyor (knowledge conflict). Bu, Retell'in yanlış kanıtı görmezden gelmesinin metindeki
karşılığı.

**Çıkarım.** Bozma tasarımı literatürle uyumlu, ama iki boşluk var:
- **Bozma Check'in yakalayabileceği türde olmalı.** Bozulan etiketin skoru düşük çıkmazsa (Look kötü
  olduğu için ya da skorlar sınıflar arası karşılaştırılamadığı için) Retell "contradicted" etiketiyle
  gerçek bir çelişki görmez. Yani bozma ancak Look ve Check düzeldikten sonra anlamlı. Sıranın gerekçesi
  bu.
- **Bozma yalnızca etiket değiştiriyor.** LURE belirsizlik ve konum (cümle sonu) etkisini de kullanıyor.
  Bozmayı cümlenin ilerisindeki claim'lerde biraz daha sık yapmak, gerçek hata dağılımına daha yakın bir
  eğitim verebilir. Ablation olarak not edildi; önce gerçek hata dağılımı (konuma göre) B'de ölçülür.

**Kapı.** Bozulmuş claim'lerde Check'in "contradicted" oranı gerçek hatalardakine yakın olmalı. Çok daha
kolaysa Retell sentetik hatayı öğrenir, gerçeği öğrenmez.

## 10. Kanıtın captioner'a verilmesi

**Şu an.** Her claim kaydı bir token'a çevriliyor (sınıf, alternatif, skorlar, kutu, karar). Decoder'ın
6–11. katmanlarına Flamingo tarzı, sıfırla başlatılmış tanh-gate'li cross-attention ekleniyor. Bir de
null token var.

**Literatür.** Flamingo: gated cross-attention, tanh(α) ile α=0'dan başlayarak donuk dil modelini bozmadan
yeni modalite ekleme. ControlNet (ICCV'23): sıfır konvolüsyon, aynı amaç. LLaMA-Adapter (ICLR'24): sıfırla
başlatılmış gating. OSCAR (ECCV'20): nesne etiketlerini metin token'ı olarak vermek. SmallCap (CVPR'23):
donuk decoder + yeni cross-attention ile getirilmiş metni koşul yapmak.

**Çıkarım.** Mimari literatürde iyi temellenmiş. Ama sıfırla başlatmanın bilinen bir yan etkisi var:
gradyan sinyali zayıfsa gate kapalı kalır. Bizde de bu oldu (sensitivity: masked %10.9, false %0.0).
Kanıtın bilgi taşımaması (Look/Check) ile birleşince gate'in açılması için sebep kalmıyor. Ek bir ablation:
OSCAR tarzı **metin olarak** kanıt ("[supported] dog [contradicted] ball → frisbee") vermek. Mimari
eklemeden aynı soruyu yanıtlar ve B0+/B2 ile daha doğrudan karşılaştırılabilir.

**Kapı.** Önce (Look ve Check düzeldikten sonra) oracle kanıtla eğitim: kanıt doğrudan referanslardan
türetilir. Oracle kanıtla bile sensitivity düşükse sorun mimari/eğitimde, değilse kanıt kalitesinde.
Bu, hatayı ayırt etmenin en ucuz yolu.

## 11. Kanıtın kullanılmasını sağlayan eğitim

**Şu an.** Eğitim hedefi beş referans. `--img-drop` ile örneklerin bir kısmında görüntü yamaları
düşürülüyor. Kanıt token'ları `p_drop` ile düşürülüyor.

**Literatür.** Wang (CVPR'20, "What makes training multi-modal classification networks hard?"): çok
modlu ağlar güçlü modaliteye aşırı uyuyor (Gradient-Blending). ModDrop (Neverova TPAMI 2016): modaliteleri
rastgele düşürerek her birini kullanmaya zorlamak. Longpre (EMNLP'21): bağlam ile ezber çeliştiğinde
çelişkili örneklerle eğitim bağlama bağlılığı artırıyor.

**Çıkarım.** Görüntü yaması düşürme ModDrop'un gerekçesine dayanıyor: görüntü güçlü modalite, kanıt zayıf
modalite. Longpre'nin karşılığı ise "kanıt ile görüntü/ezber çeliştiğinde hedefin kanıtı izlemesi". Bizde
hedef beş referans olduğu için, bozulmuş bir claim'de referanslar zaten doğru sınıfı içerir ve model "kanıtı
izlemek" yerine "görüntüyü izlemek" ile aynı kaybı alır. Yani **kanıtı kullanmak için doğrudan bir teşvik
yok.** İki seçenek:
- (a) Görüntü yamaları düşürüldüğünde kanıt tek bilgi kaynağı kalır (şu anki `--img-drop`).
- (b) Kontrastif bir yardımcı kayıp: aynı örnek gerçek ve bozulmuş kanıtla verildiğinde referansın olasılığı
  gerçek kanıtla daha yüksek olmalı (Jacovi'deki kontrastif fikre ve NegCLIP'teki zor negatife paralel).
  Bu bizim eklememiz olur. Ablation'la ölçülür.

**Kapı.** Validation'da "false evidence" müdahalesinde caption'ların değişme oranı (sensitivity) sıfırdan
anlamlı şekilde büyük olmalı ve doğru yöne gitmeli: yanlış kanıtla hallucination artmalı, doğru kanıtla
azalmalı.

## 12. Yeniden üretim / düzeltme

**Şu an.** Retell tam caption'ı kanıtla koşullu olarak baştan üretiyor (beam=3).

**Literatür.** Üç aile var:
- Yeniden yazma: Woodpecker (LLM ile düzeltme), LURE (revisor), Volcano (NAACL'24, kendi geri bildirimiyle
  düzeltme).
- Kısıtlı decoding: Anderson (EMNLP'17) constrained beam search, etiketleri caption'a zorla sokuyor ve
  yeniden eğitim istemiyor.
- Kontrastif / ceza tabanlı decoding: VCD, OPERA.

**Çıkarım.** Kısıtlı beam search, kanıtın "ne yapılacağını" kurala bağlayan, eğitimsiz bir baseline verir:
"contradicted" sınıfı yasakla, alternatifi zorla. Bu **B4 (kural tabanlı)** olarak eklenebilir ve
"öğrenilmiş kanıt kullanımının (Retell) basit bir kuraldan iyi olup olmadığı" sorusunu yanıtlar. Önerideki
"learning to use feedback" iddiası ancak böyle bir kural baseline'ını geçerse anlamlı. transformers 5.23'te
`bad_words_ids` var. `force_words_ids` constrained beam search moduna yönlendiriyor, ama bu sürümde gerçekten
çalıştığı ayrıca test edilmeli; çalışmazsa yasaklama tek başına da bir baseline olur. Kapsamı Retell aşamasında.

**Kapı.** Retell, B0+ / B2 / B3'ün yanında bu kural baseline'ını da geçmeli (ya da geçemediği raporlanır).

## 12a. Retell'in içindeki gizli kararlar (kod okuması, 2026-11-14)

`lcr/retell.py`, `lcr/evidence.py`, `scripts/06_train_retell.py`:

| Karar | Şu an | Sorun / alternatif | Ne zaman |
|---|---|---|---|
| Gate'lerde weight decay | AdamW `weight_decay=0.01` **tüm** parametrelere, gate α'ları dahil | Weight decay sinyal zayıfken gate'i sıfıra çeker. Yaygın uygulama: gate, bias ve LayerNorm parametreleri decay'siz. **Gözlenen kapalı gate'lerle (≈0.001) tutarlı bir ek neden** | Retell, ilk düzeltme (ucuz) |
| Gate öğrenme hızı | Diğer parametrelerle aynı (1e-4, OneCycle, 3 epoch) | Adam'da bir parametre adım başına en fazla ~lr kadar hareket eder. Gate'e ayrı (daha yüksek) lr ya da küçük sıfırdan farklı başlatma denenebilir. Sıfırla başlatmanın amacı (başlangıçta captioner'ı bozmamak) korunur | Retell, ablation |
| Kanıt token'ı | claim ve alternatif kelime gömmesi (BLIP'in donuk embedding'i) + OWLv2 bölge gömmesi (512) + 9 skaler + karar tipi | Token, claim'in caption'daki **konumunu** taşımıyor. Eşleşme kelime gömmesi üzerinden olmalı. Muhtemelen yeterli, ama metin olarak kanıt ablation'ı (işlem 10) bunu da test eder | Retell |
| Bölge gömmesi | Seçilen kutunun OWLv2 sınıf gömmesi | Retell bölgenin içeriğini kendisi "okuyabilir", yani kararı ve skoru atlayıp kendi küçük detektörünü öğrenebilir. Ablation: gömmesiz | Retell |
| Taslak girdi mi | Hayır. Retell görüntü + kanıtla **baştan üretiyor**, draft'ı görmüyor | Bu bir "edit" değil, "yeniden üretim". Raporda açık yazılmalı. Edit tarzı alternatif: B1 ve constrained beam (işlem 12) | rapor |
| Hedef | Her adımda 5 referanstan biri | İşlem 11: kanıtı kullanmak için doğrudan teşvik yok | Retell |
| Eğitim verisi | B draft'ları + ρ bozma, eşikler de B'de | İşlem 1 ve 8: kalibrasyon alt kümesi ayrılmalı | Check/Retell |
| Seçim ölçütü | En iyi epoch validation CIDEr'e göre | Validation hem seçim hem raporlama için kullanılıyor. Epoch seçimi B'nin ayrılmış bir parçasında yapılmalı ya da sabit epoch kullanılmalı | Retell |

Son satır kuralımızla ilgili küçük bir ihlal: validation'a göre seçip validation'da raporlamak. Düzeltilecekler listesine eklendi.

## 13. Otomatik caption metrikleri

**Şu an.** BLEU-4, METEOR, ROUGE-L, CIDEr (pycocoevalcap), SPICE Colab'da (Java).

**Literatür.** SPICE (ECCV'16): sahne grafiği önermeleri üzerinden F-skoru, insan yargısıyla daha yüksek
korelasyon. CLIPScore (EMNLP'21): referanssız, CLIP tabanlı. RefCLIPScore referanslı varyant. Haber gibi
bağlam gerektiren caption'larda zayıf.

**Çıkarım.** Faithfulness projesinde CIDEr tek başına yanıltıcı (CHAIR makalesinin bulgusu). SPICE'ın
nesne alt skoru ve CLIPScore, referans proxy'sinden bağımsız ikinci bir sinyal olarak eklenmeli. CLIPScore
ucuz (CLIP ViT-B/32 Colab'da dakikalar). Ancak CLIP de kelime torbası gibi davranabiliyor (NegCLIP
bulgusu), bu yüzden tek bir nesne değişikliğine duyarlılığı sınırlı olabilir. Bu bir sınırlama olarak
yazılır.

## 14. Hallucination metrikleri

**Şu an.** CHAIR tarzı oranlar referans proxy'sine göre (CHAIR-i / CHAIR-s benzeri) ve kör insan
etiketlemesi (HICR) için pilot.

**Literatür.** CHAIR (EMNLP'18), POPE (EMNLP'23, dengeli evet/hayır sorgulama), ALOHa (NAACL'24, HAT
veri seti), FaithScore, AMBER (LLM gerektirmeyen, varlık / nitelik / ilişki).

**Çıkarım.** CHAIR'in Flickr8k versiyonunu kendi sözlüğümüzle hesaplamak meşru, ama adı "CHAIR-style,
reference-based proxy" olmalı. F30kE kutularıyla ikinci bir versiyon ("box-based proxy") eklenebilir
(işlem 4). POPE ve AMBER captioning değil soru-cevap benchmark'ları. İlgili çalışma olarak anılır,
uygulanmaz.

## 15. İnsan değerlendirmesi

**Şu an.** 100 draft'lık kör etiketleme sayfası (annotate.html). İki etiketçi, Cohen κ.

**Literatür.** Card (EMNLP'20): tipik insan değerlendirme tasarımları küçük farkları yakalamak için
düşük güçlü.

**Çıkarım.** 100 draft, sistemler arası küçük farkları ayırt etmek için büyük olasılıkla yetersiz. Bu yüzden
insan etiketini **sistemleri karşılaştırmak için değil, proxy'leri doğrulamak için** (işlem 3–4: claim
çıkarma, in_refs, Check kararları) kullanmak daha akıllıca. Sistem karşılaştırması otomatik metriklerle
ve tam validation'da yapılır. Etiketlemeye başlamadan önce Card'ın notebook'larıyla bir güç analizi
yapılabilir.

## 16. İstatistik ve güç

**Şu an.** Hipotez testleri ileride. Planlanan: eşleştirilmiş testler.

**Literatür.** Dror (ACL'18): metrik türüne göre test seçimi protokolü, NLP'de testlerin sık atlandığı ya
da yanlış uygulandığı. Koehn (EMNLP'04): eşleştirilmiş bootstrap yeniden örnekleme, MT'de standart.
Card (EMNLP'20): güç analizi.

**Çıkarım.** Corpus düzeyindeki metrikler (CIDEr, SPICE) için eşleştirilmiş bootstrap (Koehn). Claim
düzeyindeki oranlar (hallucination, sensitivity) için eşleştirilmiş permütasyon ya da McNemar (Dror
protokolü). Birden çok karşılaştırma varsa düzeltme (Holm). Validation 1000 görüntü, ~1800 claim:
hipotez testlerinden önce gözlenen etki büyüklüğüne göre güç hesaplanır. Bu, "fark yok" sonucunun
yorumlanması için gerekli.

## 17. Kanıt müdahaleleri (H4)

**Şu an.** Kanıt maskeleme, yanlış kanıt, sensitivity = değişen caption yüzdesi.

**Literatür.** Longpre (EMNLP'21): karşı-olgusal bağlam ile bağlama bağlılığı ölçmek. Jacovi (EMNLP'21):
kontrastif açıklama. Darcet ve Kang (ICLR'25) çalışmalarındaki müdahale (attention'ı yeniden dağıtma)
mantığı da aynı.

**Çıkarım.** Müdahale tasarımı literatürle uyumlu. Eklenmesi gerekenler: (a) "değişme oranı"nın yanında
**değişimin yönü** (yanlış kanıtla eklenen sınıf, kanıttaki yanlış sınıf mı?). (b) Oracle kanıt
müdahalesi (işlem 10'daki kapı). (c) Görüntü yokken (yamalar maskeli) kanıta bağlılık.

---

## Özet: değişiklik listesi (öncelik sırasıyla, sıra kuralına uygun)

| # | İşlem | Değişiklik | Ne zaman |
|---|---|---|---|
| 1 | Look | LOOK_V2_PLAN (Grad-CAM, head seçimi, ensemble, …), insan kutusu kapısı | **şimdi** |
| 2 | Proxy | `in_f30k_boxes` ikinci proxy; proxy anlaşmazlığı raporu | Look ile birlikte (aynı veri) |
| 3 | Claim | 100 draft'ta claim çıkarma kesinliği (kapı) | Look ile paralel, ucuz |
| 4 | Detektör | Sınıflar arası skor normalizasyonu / sınıf başına kalibrasyon (ablation) | Check |
| 5 | Karar | Risk–kapsam eğrisi; supported/contradicted kesinliği ayrı | Check |
| 6 | Eşik | B içinde ayrılmış kalibrasyon alt kümesi + bootstrap aralık | Check |
| 7 | Bozma | Sentetik ve gerçek hatada Check davranışı karşılaştırması (kapı) | Retell öncesi |
| 8 | Kanıt girişi | Oracle kanıt deneyi; metin olarak kanıt ablation'ı | Retell |
| 9 | Eğitim | Kontrastif yardımcı kayıp (bizim eklememiz), img-drop ablation | Retell |
| 10 | Yeniden üretim | Kural tabanlı constrained beam baseline (B4) | Retell |
| 11 | Metrik | SPICE nesne alt skoru + CLIPScore | Final değerlendirme |
| 12 | İstatistik | Paired bootstrap + permütasyon/McNemar + güç analizi | Hipotez testleri |
| 13 | Decoding | VCD tarzı eğitimsiz baseline | İsteğe bağlı, sonda |
| 14 | Check (8a) | Alternatif seçimini türle sınırla; `s_image`'sız "ours" ablation'ı (H6 karıştırıcısı) | Check / Retell |
| 15 | Retell (12a) | Gate'leri weight decay'den çıkar; gate lr ablation'ı; gömmesiz bölge ablation'ı | Retell |
| 16 | Retell (12a) | Epoch seçimini validation yerine B'nin ayrılmış parçasında yap (kural düzeltmesi) | Retell |

---

## Doğrulanmış kaynaklar

(Varlığı bu oturumda veya bir önceki oturumda web'de kontrol edildi. `refs.bib`'e eklerken künyeyi tekrar kontrol et.)

- Kirillov et al., *Segment Anything*, ICCV 2023, pp. 3992–4003, doi 10.1109/ICCV51070.2023.00371 (Crossref, 2026-11-14).

**Lokalizasyon / açıklanabilirlik**
- Li et al., *Align before Fuse (ALBEF)*, NeurIPS 2021: Grad-CAM ile lokalizasyon.
- Luo et al., *Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models (PnP-OVSS)*, CVPR 2024.
- Kang et al., *See What You Are Told: Visual Attention Sink in Large Multimodal Models*, ICLR 2025.
- Kang et al., *Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding*, CVPR 2025.
- Zhang et al., *MLLMs Know Where to Look: Training-free Perception of Small Visual Details*, ICLR 2025.
- Chefer, Gur, Wolf, *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers*, ICCV 2021.
- Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020.
- Darcet et al., *Vision Transformers Need Registers*, ICLR 2024.
- Liu et al., *Attention Correctness in Neural Image Captioning*, AAAI 2017.
- Zhou et al., *More Grounded Image Captioning by Distilling Image-Text Matching Model (POS-SCAN)*, CVPR 2020.
- Zhang et al., *Top-down Neural Attention by Excitation Backprop* (pointing game), ECCV 2016.
- Wang et al., *Score-CAM*, CVPR Workshops 2020 (enerji tabanlı pointing).
- Zhou et al., *Learning Deep Features for Discriminative Localization (CAM)*, CVPR 2016.
- Jacovi et al., *Contrastive Explanations for Model Interpretability*, EMNLP 2021.
- Plummer et al., *Flickr30k Entities*, ICCV 2015.
- *HaloProbe*, ICML 2026 (ham attention'ın konum ve tekrarla karışması). Tam künye aşağıda "Öğrenilmiş doğrulama" altında.

**Doğrulama / claim / hallucination**
- Rohrbach et al., *Object Hallucination in Image Captioning (CHAIR)*, EMNLP 2018.
- Li et al., *Evaluating Object Hallucination in Large Vision-Language Models (POPE)*, EMNLP 2023, doi 10.18653/v1/2023.emnlp-main.20.
- Manakul, Liusie, Gales, *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs*, EMNLP 2023, doi 10.18653/v1/2023.emnlp-main.557.
- Yin et al., *Woodpecker: Hallucination Correction for Multimodal Large Language Models*, arXiv 2310.16045.
- Zhou et al., *Analyzing and Mitigating Object Hallucination in Large Vision-Language Models (LURE)*, ICLR 2024.
- Petryk et al., *ALOHa: A New Measure for Hallucination in Captioning Models*, NAACL 2024 (short).
- Jing et al., *FaithScore*, Findings of EMNLP 2024.
- Wang et al., *AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation*, arXiv 2311.07397.
- Petryk et al., *Simple Token-Level Confidence Improves Caption Correctness*, WACV 2024.
- Minderer et al., *Simple Open-Vocabulary Object Detection with Vision Transformers (OWL-ViT)*, ECCV 2022.
- Minderer et al., *Scaling Open-Vocabulary Object Detection (OWLv2)*, NeurIPS 2023.

**Karar, kalibrasyon**
- Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*, NeurIPS 2017.
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 (PMLR 70:1321–1330).

**Negatifler, bağlam çelişkisi**
- Yuksekgonul et al., *When and Why Vision-Language Models Behave like Bags-of-Words, and What to Do About It? (NegCLIP)*, ICLR 2023.
- Longpre et al., *Entity-Based Knowledge Conflicts in Question Answering*, EMNLP 2021.

**Kanıt girişi ve eğitim**
- Alayrac et al., *Flamingo*, NeurIPS 2022.
- Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)*, ICCV 2023.
- Zhang et al., *LLaMA-Adapter*, ICLR 2024.
- Li et al., *OSCAR*, ECCV 2020.
- Ramos et al., *SmallCap*, CVPR 2023.
- Wang, Tran, Feiszli, *What Makes Training Multi-modal Classification Networks Hard?*, CVPR 2020.
- Neverova et al., *ModDrop: Adaptive Multi-modal Gesture Recognition*, IEEE TPAMI 2016.

**Yeniden üretim / decoding**
- Anderson et al., *Guided Open Vocabulary Image Captioning with Constrained Beam Search*, EMNLP 2017, pp. 936–945.
- Leng et al., *Mitigating Object Hallucinations in LVLMs through Visual Contrastive Decoding (VCD)*, CVPR 2024.
- Huang et al., *OPERA*, CVPR 2024, pp. 13418–13427.
- Li et al., *Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision*, NAACL 2024.

**Öğrenilmiş doğrulama (Check v2 için eklendi, 2026-11-14)**
- Fieback, Spiegelberg, Gottschalk, *MetaToken: Detecting Hallucination in Image Descriptions by Meta Classification*, arXiv 2405.19186 (SciTePress 2025, DOI 10.5220/0013165700003912).
- Zohrabi et al., *HaloProbe: Bayesian Detection and Mitigation of Object Hallucinations in Vision-Language Models*, arXiv 2604.06165, ICML 2026 (poster sayfası icml.cc/virtual/2026/poster/62157).
- Gu et al., *Open-vocabulary Object Detection via Vision and Language Knowledge Distillation (ViLD)*, arXiv 2104.13921 (ICLR 2022), prompt ensemble.

**Metrikler ve istatistik**
- Anderson et al., *SPICE*, ECCV 2016 (LNCS 9909, pp. 382–398).
- Hessel et al., *CLIPScore*, EMNLP 2021, pp. 7514–7528.
- Dror et al., *The Hitchhiker's Guide to Testing Statistical Significance in NLP*, ACL 2018, pp. 1383–1392.
- Koehn, *Statistical Significance Tests for Machine Translation Evaluation*, EMNLP 2004, pp. 388–395.
- Card et al., *With Little Power Comes Great Responsibility*, EMNLP 2020, pp. 9263–9274.
