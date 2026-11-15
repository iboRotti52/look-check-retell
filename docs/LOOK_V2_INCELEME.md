# Look v2 — bağımsız inceleme ve öneriler

**Look kapanışı:** E1/E2 tamamlandı ve v3 donduruldu; güncel sonuç `docs/LOOK_V2_PLAN.md` §7.8 ve `results/look_closed.json`. Aşağıdaki inceleme tarihsel kayıttır; yeni denetim görevi değildir.

**Düzeltme durumu (2026-11-14):** aşağıdaki incelemede bulunan kayıt/çıktı sürümü, kutu birleşimi,
AUROC ve rapor sorunları giderildi. `results/look_consistency_fix.json` tüm B/validation kayıtlarını
doğruluyor; mevcut kutular ve skorlar aynı. Önceki sayılar aşağıda incelemenin tarihsel kaydıdır;
güncel enerji/alan sonuçları `metric_version="exact_box_union_v2"` etiketli sonuç dosyalarındadır.
Yeni yöntem seçimi, gradyan çıkarımı ve kişi için kutu kuralı denemesi ayrı deney işleridir.

Tarih: 2026-11-14. Kapsam: kod, kayıtlı sonuçlar, birincil kaynaklar ve B yarısında yeni tanısal ölçümler.
Yeni bir Look konfigürasyonu seçilmedi; üretim kodu, Check ve Retell değiştirilmedi. T1–T5 uygulama
paketinin tamamlandığı iddia edilmiyor. Yeni ölçümler yalnızca B'de; validation için mevcut JSON sonuçları kullanıldı.

## Değerlendirme

Look v2'yi korumak için iyi kanıt var. Sink düzeltmesi kelimeye özgü bir harita sağlıyor; sorgudan
bağımsız objectness ile kutu seçimi bunu tamamlıyor. İddia edilen sınıfın detektör skoru Look'a girmiyor.
OWLv2'nin ayrı objectness başlığını kullanmak, sözlük sınıflarındaki en yüksek skoru objectness diye
adlandırmaktan daha temiz: [OWLv2 makalesi](https://arxiv.org/abs/2306.09683).

Mevcut validation sonucu `results/look_benchmark_A_validation_look2.json`: pointing %88.3, kutu
doğruluğu %75.3; person %66.4. Bunlar insan kutularına göre **sınıf düzeyinde lokalizasyon** sonuçları.
Yanlış nesne iddialarının ne kadarını düzelttiğimizi veya modelin kararının nedensel açıklamasını ölçmüyor.
F30kE kutusu olmayan iddialar bu lokalizasyon değerlendirmesinin dışında kalıyor.

Öncelik yeni bir harita ailesi eklemekten önce mevcut kişi kutularını ve kayıt tutarlılığını çözmek olmalı.

## Yeni ölçümler: attention gerçekten katkı sağlıyor mu?

Tekrar üretim: `LCR_DEVICE=cpu .venv/bin/python scripts/review_look_v2.py`.
Kaynak: `results/look_v2_review_B.json`. B'de insan kutusuyla eşleşen 5.383 iddia; aynı aday kutuları ve
aynı boyut filtresi kullanılıyor. Kontrollerin hiperparametreleri yeniden ayarlanmadı.

| Sabit karşılaştırma | Doğru kutu (%) |
|---|---:|
| Look v2: attention + alan cezası + objectness | 74.64 |
| Yalnızca objectness: en yüksek nesnellikli uygun kutu | 35.95 |
| Tekdüze harita + v2 alan/objectness kuralı | 39.61 |
| Attention + aynı alan cezası, objectness yok | 50.77 |
| Sınıfa özel detektör seçimi, B3 referansı | 82.91 |
| Aday havuzunda doğru kutu var mı, oracle | 94.89 |

Ek kontrol: başka sınıftan iddiası olan **aynı 4.991 örnekte**, kendi haritasıyla kutu doğruluğu %74.65;
aynı görüntüde başka sınıfın haritasını kullanınca %22.12. Burada yalnızca harita değişiyor; adaylar ve
objectness sabit. Bu, kutu seçimi düzeyinde de kelimeye özgü haritanın önemli olduğunu gösteriyor.
Haritası değiştirilen diğer iddia, dosya sırasındaki ilk farklı sınıf; bütün olası takasların ortalaması değil.

Sonuç: v2'de kazanım yalnızca detektörün nesnellik önceliğinden gelmiyor. Attention ve objectness
birlikte daha başarılı. Bunlar geliştirme verisindeki betimleyici sonuçlar; yeni validation sonucu veya
istatistiksel anlamlılık testi olarak sunulmamalı. Objectness'sız bu kontrolün alan üssü v2 ile aynı;
önceki grid'deki ayrıca optimize edilmiş objectness'sız satırla karıştırılmamalı.

## Kişi kutularındaki hata nereden geliyor?

B'de 2.244 person iddiası incelendi. İnsan kutusu sayısına göre:

| Aynı sınıfa eşlenen GT kutusu sayısı | n | Look v2 (%) | B3 (%) |
|---|---:|---:|---:|
| 1 | 1.145 | 72.14 | 86.55 |
| 2 | 489 | 72.80 | 88.34 |
| 3+ | 610 | 57.38 | 90.16 |

Bunlar görüntüdeki gerçek kişi sayıları değil, eşlenen GT kutusu sayılarıdır. Topluluk ve tekil kişi
anotasyonları örtüşebilir. Ölçüt herhangi bir GT kutusuyla yeterli IoU'yu kabul ediyor; dolayısıyla
"başka bir kişiyi seçtiği için başarısız" açıklaması tek başına geçerli değil.

712 başarısız kişi seçiminin 569'unda attention tepe noktası yine bir doğru kişi kutusunun içinde.
Geometriye göre otomatik ayrım:

- 206: küçük parça. Tahmin/GT alanı < 0.5 ve tahminin en az %80'i GT içinde.
- 201: büyük kapsayıcı. Tahmin/GT alanı > 2 ve GT'nin en az %80'i tahmin içinde.
- 266: diğer örtüşmeler; kayık kutular, karışık nesneler, eşik yakınındaki kutular vb.
- 39: hiçbir eşleşen GT kutusuyla örtüşme yok.

Bu sınıflandırma insan hata etiketi değil; açık eşiklerle tanımlı geometri özeti. Galeri:
`results/look_v2_person_review_B.png`. Her hata grubundan sabit tohumla beşer örnek seçildi; görülme
sıklığını temsil eden rastgele örneklem değil. Yeşil GT, kırmızı Look, camgöbeği B3. Yirmi örnek görsel
olarak incelendi: yüz/baş kutuları ile birkaç kişiyi ve çevreyi birlikte alan geniş kutular belirgin.

**Çıkarım:** kutu ölçeği güçlü bir darboğaz. Sadece küçük kutuyu cezalandırmak büyük-kutu hatalarını,
sadece büyük kutuyu cezalandırmak parça hatalarını artırabilir. Tür bazında sınırlı bir (alpha, gamma)
denemesi B'de yapılabilir; çok sayıda elle kişi-geometrisi kuralı eklemeden önce bu denenmeli.
Kutu birleşimi tek başına öncelikli değil: zaten fazla geniş seçimler de var.

## Literatürün plana getirdiği düzeltmeler

### Başlık seçimi BLIP'e doğrudan taşınamaz

[Kang ve ark., CVPR 2025](https://arxiv.org/abs/2503.06287) görüntü ve metin token'larını birlikte
işleyen self-attention üzerinde önce görüntüye giden toplam attention'ı, sonra uzamsal entropiyi kullanıyor.
BLIP'in cross-attention anahtarları zaten yalnızca görüntü token'ları: CLS dahil toplamı bire eşit.
CLS hariç toplam ise yalnızca CLS'ye gitmeyen payı ölçer. Bu nedenle aynı seçim ölçütü aynı anlamı taşımıyor.

Ayrıca makaledeki uzamsal entropi, yamaların Shannon entropisi değil: ortalamada eşiklenen haritadaki
sekiz-komşuluk bağlı bileşenlerinin alan dağılımından hesaplanıyor. Mevcut `att_entropy` bunun eşdeğeri
değil. BLIP uyarlaması açıkça tanımlanıp başlıkların kelimeler arasında ayrıştırıcı olup olmadığıyla
kontrol edilmeli. İnsan kutusuna göre başlık seçimi ayrı bir denetimli karşılaştırma olarak kalmalı.

### Bizim sink düzeltmemiz bir uyarlama

[Kang ve ark., ICLR 2025](https://arxiv.org/abs/2503.03321) sink token'larını ve attention yeniden
dağıtımını inceliyor. Caption ortalamasını çıkarıp kırpmak bu çalışmanın birebir VAR algoritması değil;
bu projede ölçülmüş bir uyarlama. Kaynak motivasyonu ile kendi formülümüz ayrılmalı.

[Zhang ve ark., ICLR 2025](https://arxiv.org/abs/2502.17422) göreli attention'ın paydasını aynı görüntüye
verilen genel bir açıklama talimatından alıyor. Bizdeki caption ortalaması aynı referans sinyali değil.
Oranlama denenebilir ama "makaleyi aynen uyguladık" denmemeli.

### İyi lokalizasyon nedensel açıklama kanıtı değil

[Jain ve Wallace, NAACL 2019](https://aclanthology.org/N19-1357/) NLP modellerinde attention ile
çıktıya katkının ayrışabildiğini gösteriyor; bu BLIP'e ilişkin doğrudan bir deney sonucu değil.
Buradaki çıkarımımız: insan kutusuyla uyum ve harita takası, yerleşim/özgüllük kapılarıdır; kararın
gerçekten o bölgeye dayandığını göstermek için çıktı olasılığına müdahale de gerekir.

Küçük bir B alt kümesinde, sabit caption öneki altında hedef kelimenin log-olasılığını ölçmek yararlı:
Look bölgesini bozma, aynı büyüklükte rastgele bölgeyi bozma, detektör bölgesini bozma. Piksel düzeyinde
yer iddiası için görüntüyü yeniden encode etmek gerekir; önbellekteki ViT yamasını silmek aynı deney
değil, çünkü encoder özellikleri başka bölgelerden de bilgi taşır.

Ham üretim attention'ının p-1 konumunda okunması doğru. Ancak düzeltmede kullanılan **bütün caption
ortalaması** sonraki konumları da içeriyor. Bu, bitmiş draft'a uygulanan geriye dönük bir haritadır;
üretim anındaki çevrimiçi haritayla birebir aynı olduğu söylenmemeli. Prefix ortalaması ayrı ablation olabilir.

### Grad-CAM ikinci aşama olmalı

[Chefer ve ark., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chefer_Generic_Attention-Model_Explainability_for_Interpreting_Bi-Modal_and_Encoder-Decoder_Transformers_ICCV_2021_paper.html)
encoder–decoder için gradient/relevance yöntemlerini temellendiriyor. Yeni harita kaynağı olarak makul;
fakat mevcut p-1 haritasının pointing'i zaten güçlü olduğundan önce kutu seçimi denemesi daha doğrudan.

Teknik ön koşul: `Captioner.cross_attention` no-grad çalışıyor, hook attention'ı detach ediyor ve
parametreler donuk. Yeni çıkarımda bunlar ele alınmalı; örneğin görüntü özellikleri `requires_grad=True`
yapılmalı ve detach edilmemiş attention üzerinden gradyan gerçekten doğrulanmalı. Başarılı backward
tek başına lokalizasyon katkısı kanıtlamaz.

## Kod ve kayıt sorunları (inceleme bulguları; sonradan düzeltildi)

1. **V2 kayıtlarında eski yardımcı alanlar.** `scripts/17_freeze_look.py` sadece `rec_att` değiştiriyor.
   B'deki kutulu iddialarda `iou_att_det` 2.469 kayıtta yeniden hesaplanan IoU ile uyuşmuyor;
   `regions_by_k["4"]` 2.571 kayıtta yeni `rec_att.region` ile farklı. `excess_by_k` de kural değişiminden
   önceki kutuya ait. Harita entropisi ise harita değişmediğinden aynı kalabilir.
   Ana benchmark `rec_att.box` okuduğu için bu bulgu mevcut kutu doğruluğunu geçersiz kılmıyor.
   Eski alanlar yeniden hesaplanmalı veya açıkça eski sürüm altında saklanmalı; diğer k haritaları
   saklanmadığı için hepsi tek k=4 haritadan yeniden üretilemez.

2. **Yeniden çalıştırmada sürüm karışması.** `scripts/04_draft_look_check.py` hâlâ `_sink` / ham dosya
   adlarına yazıyor, fakat `pipeline.run` artık v2 kutu kuralını kullanıyor. Eski v1 dosyalarının üstüne
   aynı adla farklı yöntem yazılabilir. Sonraki üretimden önce harita ve kutu sürümleri ayrı metadata
   ve dosya adlarıyla kaydedilmeli.

3. **Enerji/şans maskesinde birleşim yaklaşık.** `grounding.coverage_mask`, bir yamadaki kutu kapsama
   oranlarının maksimumunu alıyor; gerçek geometrik birleşimi değil. Aynı yamayı ayrı yarımlarıyla
   örten iki kutuda gerçek kapsama 1 iken fonksiyon 0.5 veriyor. Pointing ve kutu-IoU bundan etkilenmiyor;
   enerji ve alan tabanlı şans ölçümü düzeltilene kadar yaklaşık diye yorumlanmalı.

4. **AUROC'ta eşit skorlar.** `scripts/16_look_regions.py: auroc` ordinal sıralama kullanıyor; eşit
   skorlara ortalama sıra vermiyor. Bu kontrol edilmeli. Bölge seçimi bölge doğruluğuna göre yapıldığı
   için seçilen konfigürasyon bundan etkilenmiyor; AUROC farkının büyüklüğü ayrıca ölçülmeli.

5. **Rapor geride.** `report/sections/3_design.tex` hâlâ excess-mass kutu kuralını ana yöntem olarak
   tanımlıyor; deney anlatısı insan kutuları öncesindeki detektör proxy ölçümlerine dayanıyor. V2 ve
   bağımsız insan-kutusu benchmark'ı rapora script üretimli tablolarla geçirilmeden son sürüm hazır sayılmaz.

## Önerilen sıra

1. Kayıt sürümlerini ve yardımcı alanları düzelt; mevcut v2 kutuları değişmeden yeniden ölçüm tutarlılığını doğrula.
2. B'de küçük, önceden belirlenmiş tür-bazlı kutu kuralı denemesi ve Gauss yumuşatma ablation'ı yap.
   Kişi iyileşmesi yanında thing/stuff kaybını da raporla. Başarısızlık geçerli bir sonuç.
3. Gerekirse küçük B alt kümesinde gradyan/kontrastif harita ve müdahale deneyi yap; katkı varsa ölçekle.
4. Son konfigürasyonu B'de seç; validation'da önceden belirlenen karşılaştırmayı raporla.
   Belirsizlik için görüntü düzeyinde eşleştirilmiş bootstrap kullan; aynı görüntünün iddialarını bağımsız sayma.
5. Ardından Check'e geç. Look AUROC'unun B3'e yaklaşması, Check'in yanlış karar oranını çözmüş sayılmaz.

T2 için veri sınırı: kayıtlı, kırpılmış sink haritalarıyla Gauss ve üs dönüşümü denenebilir; token ortalaması
yerine medyan veya farklı oran paydası bu dosyalardan geri kazanılamaz. Bunlar için token haritaları
veya yeni decoder geçişi gerekir. B/validation oranlarının yakın olması da tek başına aşırı uyumun
olmadığını kanıtlamaz.

Doğrulama: mevcut 33 test geçti; tanısal script B üzerinde tamamlandı ve dondurulmuş v2 kutu seçimini
kutulu iddialarda birebir yeniden üretti. Yeni bir Check/Retell eğitimi veya Look v3 dondurması yapılmadı.
