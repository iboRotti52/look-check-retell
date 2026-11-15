# Check v2 bağımsız incelemesi — 2026-11-14

**Karar:** öğrenilmiş doğrulayıcı yönü korunmalı. Mevcut skorlar bağımsız hesaplamayla yeniden üretildi;
detektör ve captioner sinyallerinin ek katkısı görüntü düzeyindeki eşleştirilmiş aralıklarda da görülüyor.
Ancak Check henüz gerçek nesne doğruluğu için kalibre edilmiş bir doğrulayıcı değil: hedefi referansta
sınıfın anılması. Retell'e önerilen aktarım sürekli olasılık + ayrı alternatif önerisi; otomatik değiştirme değil.
Mevcut özellik ailesi keşif sonucu olarak dondurulmalı; validation'a bakılarak yeni varyant seçilmemeli.

## Kapsam ve yeniden üretim

`lcr/calibrate.py`, Check scriptleri 20–23, `pipeline.finalise`, `evidence.features`, Retell kanıt şeması,
mevcut B/validation kayıtları ve ilgili literatür incelendi. Test split'i ve kör pilotun `key.json` dosyası
açılmadı. Yeni Look yöntemi, Retell eğitimi veya Check model ailesi seçilmedi.

Sayıların kaynağı `scripts/review_check_v2.py` → `results/check_v2_review.json`.
**Otomatik üretilen sayısal tablolar:** [inceleme ölçümleri](generated/CHECK_V2_REVIEW_METRICS.md).
Yazma düzeltmesinin tam script doğrulaması: `results/check_v2_writer_review.json`.

Çalıştırma:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LCR_DEVICE=cpu .venv/bin/python scripts/review_check_v2.py
.venv/bin/python -m pytest -q
```

## 1. Desteklenen sonuçlar

- `prior → prior+det` ve `prior+det → prior+det+cap` AUROC farklarının eşleştirilmiş aralıkları sıfırın
  üzerinde. Artış yalnızca sınıf önceliğinden kaynaklanmıyor; ek görsel/dil-modeli sinyali katkı sağlıyor.
- `prior+det → prior+det+look` farkı sıfır çevresinde. Bu özellik seti ve doğrusal doğrulayıcı altında
  excess/attention entropisi/IoU grubunun ek katkısı gösterilemedi. Attention'ın genel olarak yararsız olduğu
  sonucu çıkarılmaz; det grubunun bölgesi zaten Look'tan geliyor.
- Attention bölgesi ile detektör bölgesinin karşılaştırma aralığı sıfırı içeriyor. **H6 doğrulandı** veya
  **iki yöntem eşdeğer** denmemeli. Eşdeğerlik için önceden belirlenmiş marj ve uygun test gerekir.
- Round-1 ve mevcut v2 sistemleri arasındaki contradicted kesinliği, yanlışları yakalama ve supported
  kesinliği farklarının aralıkları pozitif. Ancak eşikleri/kapsamları farklıdır; bu bir **sistem karşılaştırması**,
  eşit kesinlikte/eşit kapsamda saf model karşılaştırması değildir.
- Bootstrap birimi görüntü; aynı görüntünün claim'leri birlikte taşındı. Aralıklar sabit modeller/kararlar
  koşulundadır. Eğitim, yöntem seçimi veya etiket belirsizliğini kapsamaz; çoklu karşılaştırma düzeltmesi yoktur.

## 2. C0 ve seçim: çalışan kısım, kalan sınırlar

**Görüntüye göre katlama doğru.** Aynı görüntünün tüm claim'leri aynı kata gidiyor. A/B ayrılığı ve B'nin
beklenen görüntülerle birebir eşleşmesi ayrıca doğrulandı. A referanslarından sabitlenen PMI, B/validation
etiketlerini kullanmıyor. Captioner'ın A'da eğitilmiş olması PMI'ı sızıntı yapmaz; ölçülen popülasyon aynı
captioner'ın ürettiği claim'lerdir.

**Global lambda seçimi tam iç içe çapraz doğrulama değil.** B-OOF kaybıyla lambda seçilip aynı OOF
skorların eşik/performans için kullanılması, B etiketlerinin seçim üzerinden dolaylı etkisini taşır.
Validation'ın B'de uydurulmuş modele girmesi söz konusu değildir. Referans:
[scikit-learn iç içe çapraz doğrulama açıklaması](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

B'de ek tanı yapıldı: her dış kat için lambda yalnızca diğer katların iç çapraz doğrulamasından seçildi;
karar eşikleri de dış katı görmeyen iç skorlardan belirlendi. Her dış katta aynı lambda çifti seçildi ve
olasılık sıralaması değişmedi. Eşikleri de dışarıda seçmek küçük karar farkları yarattı; ayrıntı JSON'da.
Bu bulgu mevcut lambda aramasının büyük bir iyimserlik kaynağı olduğuna işaret etmiyor; yine de tüm
özellik-ailesi seçiminin bağımsızlığı anlamına gelmez. Tanı üretim kayıtlarının üzerine yazılmadı.

**Eşiklerin de OOF olması gerekir.** Mevcut B olasılıkları kendi görüntü etiketini doğrudan görmüyor;
ama tüm B'den seçilen ortak eşik kendi etiketini dolaylı olarak içeriyor. Retell eğitimi için sıkı çözüm,
görüntü katına özgü iç-seçilmiş eşik + model veya ayrı B_fit/B_cal düzenidir. Bu incelemede ana kayıtların
eğitim protokolü sessizce değiştirilmedi.

**Özellik ailesi seçimi B-only olarak doğrulanamıyor.** Script 22, yazılacak aileyi sabit `WRITE`
sözlüğünden alıyor. B'de `all` hem AUROC hem log-loss açısından seçilen aileden daha iyi; dokümandaki
sadelik gerekçesi ise validation AUROC'larının yakınlığına dayanıyor. Önceden yazılmış sadelik/tolerans
kuralı bulunmadığından bunun bağımsız bir B-only seçim olduğu söylenemez. Geçmişte validation'ın görülmesi
geri alınamaz: mevcut aile keşif tercihi olarak dondurulmalı, final test geliştirme için açılmamalı.

**B3 için gizli attention bağımlılığı:** script 22 `rec_det` modellerinin lambdasını seçerken `names_all`
üzerinde `look` grubunu da kullanıyor, sonra bu grubu B3 özelliklerinden çıkarıyor. Son katsayılarda
attention yok; yöntem-seçimi yolunda var. Sıkı H6/B3 üretim koşusunda lambda yalnızca izin verilen
özelliklerle seçilmeli veya önceden ortak sabitlenmeli. Mevcut lambda çifti değiştirilmedi.

## 3. Matematik ve captioner sinyalleri

- ClassLogistic'in ortak/sınıf eğimleri ve FeatureLogistic'in ağırlık/sabit terim gradyanları sonlu farkla
  doğrulandı. Ceza terimleri ve gradyanlarının aynı örnek sayısına bölünmesi tutarlı. Lambda burada toplam
  kayıp üzerinden tanımlıdır; başka paketlerin `C`/ortalama-kayıp katsayısıyla doğrudan eşitlenmemeli.
- Script 20'nin her durumda aynı lambda seçmesi hata değil: tüm adayların OOF kayıpları yeniden
  hesaplandı; seçilen değer gerçekten en düşük kaybı veriyor. Kayıplar inceleme JSON'unda saklı.
- Yeniden tokenleştirmede **sınıf yanında kelime, ilk parça konumu ve göreli konum** da kontrol edildi.
  B ve validation'da uyumsuzluk yok. Çok parçalı kelimeler de kontrolde var. Bu kontrol decoder
  logit'lerini yeniden çalıştırmaz; önbellekteki logit sayılarının model geçişi düzeyindeki doğrulaması değildir.
- `p-1` logit satırı doğru: teacher-forced decoder o satırda p konumundaki kelimeyi tahmin ediyor.
  Sinyal bütün sözcüğün olasılığı değil, **ilk wordpiece'in** olasılığıdır; raporda böyle yazılmalı.
- Ortalama görüntü gömmesi gerçek bir görüntüsüz koşul değildir. Konuma göre ortalanmış ViT gömmeleri
  dağılım dışında olabilir; CLS de görüntü bilgisi taşıdığı için CLS-only gerçek bir null değildir.
  B ortalamasının OOF görüntüsünü içermesi etiketsiz/transdüktif bir tercihtir. Sıkı OOF koşusunda null
  yalnızca dış-eğitim görüntülerinden, validation için B'den hesaplanmalı ve metadata'da kaydedilmelidir.
- `vis_gain` ile `lp` arasında varsayılan güçlü doğrusallık veride görülmüyor; korelasyonlar JSON'da.
  Buna karşılık lp/entropy/margin ilişkili. Seçilen modelde vis_gain katsayısı negatiftir. Katsayı
  koşullu, standartlaştırılmış bir ilişkidir; "daha çok görsel kazanç = daha doğru" nedensel yorumu yapılmaz.
  Ayrı bir B ablation'ı olmadan cap grubunun kazanımı doğrudan görsel bağımlılığa bağlanmamalı.

## 4. Etiketler, sınıf önceliği ve alternatifler

`p_correct` ismi sözleşmede korunuyor; gerçek hedefi **P(sınıf referanslarda anılıyor | draft claim'i ve
özellikler)**. Gerçek nesne varlığı olasılığı değildir. Kalibrasyon/ECE de bu otomatik hedef içindir.
Referanslarda anılmayan nesne görüntüde bulunabilir; buna ek olarak sözlük ve referans eşlemesi hata yapabilir.

F30k Entities kontrolü bağımsız ikinci insan-doğruluk kaynağı değildir. Aynı caption'ların isim öbeklerine
eklenen kutular kullanılıyor; görünmez/kutusuz entity ve sınıfa eşleme farkları var. Rapor adı
**"aynı referanslardan türetilen kutulu-entity duyarlılık kontrolü"** olmalı. Negatif sonuç nesnenin
görüntüde yokluğunu kanıtlamaz. Bu oturumda yeni F30kE etiketleri eğitime katılmadı.

`prior` yalnızca sınıf sabiti değildir: PMI ve draft claim sayısını da içerir. "Saf sınıf önceliği tek başına"
diye sunulmamalı; **draft'a koşullu, görüntüye doğrudan bakmayan öncelik modeli** denmeli. Draft'ın kendisi
görüntüden üretilmiştir. Görsel katkı ilgili grubu ekleyen eşleştirilmiş ablation ile sunulmalı.

**Alternatif seçimi hâlâ ham sorgular-arası skora dayanıyor.** Script 22 captioner adaylarını raw OWLv2
skoruyla sıralıyor. Ana doğru/yanlış kararı bu karşılaştırmadan kurtarılmış; alternatif önerisi kurtarılmamış.
OWLv2'nin kendi model kartındaki uyarı geçerlidir:
[OWLv2, Ek A.1](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6d58fc68c0f3c36ae6e0e64478a69c0-Paper-Conference.pdf).

Gerçek yazılmış alternatifler ayrıca ölçüldü. Referans-dışı claim'lerde önerinin referansta bulunma oranı,
literatür günlüğündeki captioner ilk-aday oranıyla karıştırılmamalı; bunlar farklı seçim kurallarıdır.
Hem tüm referans-dışı claim'lerde hem Check'in işaretlediklerinde öneri başarı proxy'si sınırlı.
Üstelik referansta geçmek doğru yerde/doğru anlamda bir ikameyi kanıtlamaz. Tam sayımlar otomatik tabloda.
Kalibre claim modeli alternatiflere doğrudan uygulanırsa da **draft edilmiş claim → önerilen alternatif**
dağılım değişimi vardır; alternatif doğruluğu ayrıca ölçülmelidir.

## 5. Düzeltilen somut kod sorunları

1. **Script 20 yazma kaybı:** her kayıt türü için satırlar yeniden yüklendiğinden `rec_det` yazımı önceki
   `rec_att` güncellemelerini atıyordu. Artık iki tür aynı satır yapısı üzerinde biriktiriliyor.
2. **Çıktı çakışması:** script 20 kalibrasyon deneyi ve script 22 öğrenilmiş doğrulayıcı aynı dosya adına
   yazıyordu. Script 20 artık `*_check2cal.json`; öğrenilmiş kayıtlar `*_check2.json`/`*_check2vis.json`.
   İki scriptte de mevcut pipeline çıktısı için açık `--overwrite` gerekir.
3. **Eksik özellik doldurma:** script 22 B/validation matrislerini ayrı ayrı kendi ortalamalarıyla
   dolduruyordu; B'de katların dışındaki özellikler de doldurmaya katılıyordu. Doldurma modelin `fit`
   aşamasına taşındı; tahmin her zaman eğitim ortalamasını kullanıyor. Tamamen eksik eğitim sütunu için
   sabit sıfır kullanılıyor. Mevcut veride eksik sinyal olmadığından ana olasılıklar değişmedi.
4. **AP eşit skor hatası:** sıralama eşit skorları örnek örnek kırıyor, AP satır sırasına bağlı olabiliyordu.
   Artık aynı skor tek eşik olarak gruplanıyor (`calibrate.ap_wrong`). Öncelik modelinin düzeltilmiş AP'si
   otomatik tabloda; eski JSON sonuçları tarihsel olarak korunuyor.
5. Sigmoid sayısal olarak kararlı `scipy.special.expit` ile hesaplanıyor.
6. Script 21 bundan sonra eşleşmeyi yalnızca sınıf üzerinden kabul etmiyor; saklanmış kelime/konum
   değişirse açık hata veriyor.

Regresyonlar: gradyanlar, doldurmanın tahmin grubundan bağımsızlığı ve AP eşitlikleri.
Tam script 20, geçici çıktı dizininde çalıştırıldı: her iki kayıt türü B/validation'daki tüm claim'lerde v2;
mevcut öğrenilmiş kayıtların dosya özetleri değişmedi. Script 22'nin mevcut çıktıya yazma koruması da denendi.
Test sonucu: bu oturumdaki tam koşuda **54 test geçti**; bunun üçü bu incelemede eklendi.

## 6. Retell'e geçiş için öneri ve bu oturumda değiştirilmemiş işler

- `p_correct` ve yoksa ayrı bir kullanılabilirlik göstergesi kanıt şemasına eklenmeli. Eski kayıt/B2 için
  sessizce sıfır atamak anlamı değiştirir; açık şema ve eşleşik ablation gerekir. Bu bir Retell değişikliğidir,
  bu incelemede eğitim girişi değiştirilmedi.
- `contradicted` kararı şu anda kesin bir mantıksal çelişki değil, referans-dışı olma risk bayrağıdır.
  Seçilen precision hedefi otomatik ikameye yetmez. supported/uncertain/flagged ayrımıyla sürekli risk ve
  alternatifin güvenilirliği ayrı aktarılmalı. İnsan pilotu gerçek kesinlik kalibrasyonu için hâlâ gerekli.
- Eşik, Retell'deki düzeltmenin faydası ve doğru claim'i bozmanın maliyetiyle seçilmeli. Aynı kesinlik
  hedefi B'de sağlansa bile validation'da garanti değildir. Kesinlik-kapsam eğrisi ve insan etiketleriyle
  maliyete duyarlı çalışma noktası gerekir; yalnızca daha yüksek hedef seçmek yeterli değildir.
- Sentetik bozma için mevcut `NotImplementedError` koruması doğru; kaldırılmadı. Üretim doğrulayıcıları
  (katsayılar, standartlaştırma/doldurma, sınıf dizini, PMI/null kökeni, fold ve eşikler) önce saklanmalı.
  Bozulmuş caption tokenleştirilip tüm ilgili sinyaller/adaylar yeniden çıkarılmalı; B görüntüsü kendisini
  görmeyen doğrulayıcıyla skorlanmalı. Token uzunluğu/compound değişimleri hesaba katılmalı. Eski bölgenin
  korunması seçilirse açık bir müdahale tanımıdır; yeni kelimenin Look bölgesiymiş gibi sunulmamalı.
  Doğal ve sentetik hatalarda skor/karar dağılımı B'de karşılaştırılmadan Retell'e aktarılmamalı.
- Sentetik ve gerçek kayıtlarda aynı yöntem kullanılması gereklidir, ama tek başına yeterli değildir:
  captioner sentetik yeni sözcüğe çok düşük olasılık verebilir ve hata gereğinden kolaylaşabilir.
  İnsan kutuları veya referanslardan türetilen yanlışlık etiketi, gerçek yokluk garantisi değildir.
- H6 için `s_image`'sız `prior+region` doğrulayıcı ve Retell kanıtındaki `s_image` alanının kapatılması
  **birlikte** gerekir. Yalnızca doğrulayıcıdan çıkarmak kanıt token'ındaki görüntü skoru yolunu kapatmaz.
  `check2vis`, öncelik + detektör içerir; “önceliksiz, tamamen görsel doğrulayıcı” diye adlandırılmamalı.
- Sorgu şablonu sonuçları mevcut tek şablonu değiştirmek için anlamlı bir kazanım göstermiyor.
  JSON'daki en küçük farkla seçilen şablon ile üretimde tutulan şablon ayrımı metadata'da belirtilmeli;
  sayısal en iyi sonucu pratik üstünlük sayıp yeni deney döngüsü açılmamalı.
- Raporun ana Check tasarımı hâlâ round-1 kuralını anlatıyor. Progress report güncellemesinde bu sonuçlar
  yeni bir Check v2 bölümü olarak, eski Retell koşularından ayrılarak eklenmeli. Bu incelemede PDF yeniden
  yayımlanmadı ve eski Retell başarısı yeni Check'e mal edilmedi.

## 7. Literatürden çıkarım

[Petryk ve ark., TLC](https://arxiv.org/abs/2305.07021), token güveninin doğruluk kestiriminde anlamlı bir
sinyal olduğunu destekliyor. [MetaToken](https://arxiv.org/html/2405.19186v2), küçük bir sınıflandırıcıda
token olasılığı, entropi, konum ve benzeri sinyalleri birleştirme fikrine doğrudan emsal. Bizdeki
lojistik doğrulayıcı bu çizgide makul; kendi başına yeni bir hallucination tespit yöntemi olarak sunulmamalı.

[HaloProbe](https://arxiv.org/html/2604.06165v3) konum/tekrar gibi dış istatistiklerin iç attention
sinyallerini karıştırabildiğini gösteriyor; dengeli öğrenme ve ayrı prior üzerinden posterior kuruyor.
Bizim tek lojistik modelimiz bu faktörleştirmeyi uygulamıyor. Dolayısıyla “HaloProbe uygulandı” değil,
“bu sinyallerden esinlenen bir doğrulayıcı” denmeli. Öncelik ve görsel katkıyı ayıran ablation'lar bunun
için gerekli; attention katkısının yokluğunu bu çalışma evrensel olarak kanıtlamaz.

[LURE](https://arxiv.org/abs/2310.00754), eş-geçiş/belirsizlik/konumdan esinlenen bozmayı destekliyor;
bizim sentetik bozma düzenimizin doğal hataya denk olduğunu kanıtlamaz. Projenin savunulabilir katkısı,
Look bölgesi + harici detektör + captioner sinyallerinin ölçülmüş katkıları ve bunların kanıt koşullu
yeniden üretime aktarılmasıdır. Son halkadaki fayda yeni Retell deneyleriyle ayrıca gösterilmelidir.
