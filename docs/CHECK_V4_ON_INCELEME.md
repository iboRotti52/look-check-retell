# Check v4 ön inceleme — üç kutu, bulunamama ve Retell (Codex, 2026-11-14)

## Kapsam

Kullanıcının sorusu: Look'un ilk üç kutusunu ortalamak yerine ayrı ayrı kontrol etsek ne olur;
frizbi bulunan bir görüntüde detektör neden frizbiyi bulamıyor gibi görünüyor?
Bu dosya bu soruların hedefli incelemesidir. Devir dosyasındaki tam Check v4 incelemesinin,
canlı taşımanın veya insan pilotunun tamamlandığı anlamına gelmez.
Seçili v4, eşikler ve dondurulmuş kayıtlar değiştirilmedi. Yeni tanılar yalnızca B üzerinde;
validation/test görüntüleri ve referansları açılmadı, validation değerlendirmesi tekrarlanmadı.

## Mevcut kod gerçekte ne yapıyor?

1. OWLv2 bütün görüntüden sınıftan bağımsız aday kutular ve özellikler çıkarır; önbelleğe yazar.
2. Look, attention kütlesi, alan ve objectness ile adayları sıralar. En küçük kutu kenarı filtresi de vardır.
3. Check v4, ilk üç adayın **her sınıf için skorunu** r-ağırlıklarıyla ortalar. Claim skoru ve en güçlü
   alternatif bu ortalama üzerinden bulunur. Üç bağımsız kararın birleştirilmesi uygulanmaz.
4. OWLv2'ye Look kırpımı yeniden gösterilmez. Kutunun içini ayrı bir görüntü gibi yeniden doğrulamak,
   mevcut hesaplamadan farklı bir deney olur.

Dolayısıyla düşük Check skoru tek başına üç olasılığı ayırmaz: iyi aday Look havuzuna girmemiştir;
aday girmiştir ama ortalamada seyrelmiştir; adayın kendi detektör skoru düşüktür.
Görüntüde en yüksek frizbi skorunu alan kutunun gerçekten frizbiye oturup oturmadığı da görsel denetlenmelidir.
Tüm görüntüde yüksek skor bulmak, ana yöntemi detektörün seçtiği kutuya geçirmek için gerekçe sayılmadı.

## Hata işaretlemek ile doğru ikameyi bulmak

Gerçekte claim nesnesi yoksa bunu yakalamak projenin amacıdır. Ancak kodda `contradicted` kararı için
**hem düşük claim skoru hem yüksek alternatif skoru** gerekir (`lcr/check.py:verdict`). Düşük skor,
alternatif yoksa `uncertain` olur. Bu güvenlik tercihi bazı gerçek hataları da kaçırır.
Başka bir nesnenin bölgede bulunması claim'i mantıken dışlamaz: köpek ile top, kişi ile gömlek birlikte
bulunabilir. A1 bu ilişkiyi doğrulamaz. Üç kutuyu ayrı sorgulamak da bu semantik sorunu tek başına çözmez.

Üç kutuyu ayrı kontrol ederken önce her kutunun claim/alternatif skoru ve kararı saklanmalı.
Bir kutu desteklerken başka kutu çelişiyorsa bu uyuşmazlık görünür olmalı; tek bir düşük skorla
caption hatası ilan edilmemeli. Nihai birleştirme ve eşik politikası B'de ayrıca belirlenip sınanmalıdır.
Özellikle ortalama için kalibre edilmiş eşiği tek kutuya taşımak, yeni yöntemin kalibre edildiği anlamına gelmez.

`scripts/review_check_v4_boxes_B.py` mevcut B OOF kararlarını kayıtlı skorlardan yeniden üretir;
ardından aynı Look adaylarını tek tek inceler. Kutulara uygulanan eşikler mevcut ortalamanın OOF eşikleridir.
Çıktı `results/check_v4_boxes_B.json`, frizbi örnekleri `results/check_v4_frisbee_B.png`.
Bu bir tanı koşusudur; yeni yöntem seçimi veya başarı iddiası değildir.

**Tanı sonucu:** [`generated/CHECK_V4_BOXES_B.md`](generated/CHECK_V4_BOXES_B.md).
Referansta desteklenen ama v4'ün desteklemediği frizbi örneklerinde güçlü aday, Look'un ilk üçünde yok.
Bu adayların hepsi `size_ok` filtresinden eleniyor: her kenar en az `MIN_BOX_PATCHES/BLIP_GRID = 2/24`
olmalı. Filtre, attention sıralaması yapılmadan uygulanıyor. İlk dört başarısız örnek görüntü üzerinde
incelendi; OWLv2'nin en yüksek frizbi skorunu alan yeşil kutuları gerçekten frizbiye oturuyor.
Diğer örnekler için yalnız sayısal filtre tanısı yapıldı.

Bu örneklerde sorun ortalama yüzünden güçlü adayın seyrelmesi değil, güçlü adayın havuzdan çıkarılması.
Dolayısıyla aynı üç kutuya ayrı ayrı karar vermek bu örnekleri kurtarmıyor. Bununla birlikte B'nin genelinde
ortalamada desteklenmeyen ama tek kutuda eşiği aşan claim'ler de bulunuyor; bunların bir kısmı referansta
desteklenmiyor. `any-box` desteğinin daha iyi olduğuna bu tanıdan karar verilemez.
Küçük kutu filtresinin attention'ın kaba çözünürlüğüyle ilişkisi makul bir tasarım gerekçesi olabilir;
fakat bu örneklerde yarattığı kayıp açıkça ölçülmeli. Yeni bir Look seçimi henüz yapılmadı.

## Retell'e geçişte doğrulanan uygulama açıkları

- `scripts/35_check_v4.py:write_records`, yeni alternatifi `alt` alanına ve yeni skorları `*_cal`
  alanlarına yazar; eski `s_claim/s_alt/s_image` korunur. `lcr/retell.py:record_features` eski alanları okur.
  Böylece alternatif adı ile eski `s_alt` farklı nesneleri anlatabilir.
- `lcr/evidence.py:records_for/features`, kaydı `pipeline.finalise` üzerinden geçirir. `finalise` yalnız
  Check v2'yi olduğu gibi geçirir; v4'ün kararını eski ham skor/eşik kuralıyla yeniden hesaplar.
  V4 dosyasını doğrudan eğitim girişine vermek, v4 kararlarının kullanıldığını garanti etmez.
- Sentetik bozma yolu da v4 üç-kutu skorlarını yeniden hesaplamaz. Gerçek ve bozulmuş kanıtın aynı
  şemayla üretilmesi canlı taşımanın parçası olmalı.

Bu bulgular `scripts/review_check_v4_contract.py` ile yalnız B'de ölçüldü. Otomatik sayısal ek:
[`generated/CHECK_V4_CONTRACT_B.md`](generated/CHECK_V4_CONTRACT_B.md);
makine çıktısı `results/check_v4_contract_B.json`.
Alternatifin referansta geçmesi görüntü düzeyinde leksik bir proxy'dir; aynı kutuda bulunduğunu veya
yanlış kelimenin yerine uygun olduğunu kanıtlamaz.

## Yorum sınırı ve sıradaki iş

Sınıf başına z ölçekleme skor dağılımlarını standartlaştırır; kendi başına bir varlık olasılığı kalibrasyonu
veya bütün sınıflar için eşit güvenilirlik garantisi değildir. Validation'da zemin için hiç `contradicted`
verilmemesi hata yakalama başarısı değildir; yanlış alarm karşılığında yakalamadan vazgeçilen bir noktadır.
Öğrenilmiş doğrulayıcılarla fark, bu uygulama ve proxy altında daha düşük başarıyı gösterir;
tek başına bütün bölge kanıtı yöntemlerinin ulaşabileceği üst sınırı kanıtlamaz.

Sonraki adım küçük nesne aday filtresini ve attention'dan kutuya geçişi B üzerinde ayrıca kıyaslamak;
ardından yeni bir B protokolüyle kutu başına kontrolü değerlendirmek. Yeni yöntemi mevcut v4'ün
tek seferlik validation sonucu gibi sunma. Retell'den önce ortak canlı Check hesaplaması ve
karar–skor–alternatif uyumu kapatılmalı; sonrasında insan pilotu ve doğru ikame değerlendirmesi gerekir.
