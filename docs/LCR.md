# Ana yol: Look → Check

> **Güncelleme 2026-11-15 (3):** Varsayılan Check artık S4 (`docs/CHECK_S4.md`): rakip-farkında OWLv2 kanıtı, maliyete
> duyarlı sınıf eşikleri, Look top-1'de merkezli ikame. Mantık aynı: yer Look'tan, karar donmuş OWLv2'den.
>
> **Güncelleme 2026-11-15 (2):** Fable'ın pozitif-çapalı Check düzeltmesi denendi; ön kayıtlı ölçüt tutmadı, varsayılan
> değişmedi. Giysi için açık iyileşme, yaygın sınıfa kayan yanlışlarda kayıp. Ayrıntı `docs/CHECK_DUZELTME.md`.
>
> **Güncelleme 2026-11-15:** Check tamamlandı: beş eylem, görüntü vetosu, çapraz ayar. Ayrıntı `docs/CHECK_TAMAMLAMA.md`;
> varsayılan bundle artık `results/lcr_check_bundle.json`. Aşağıdaki B_cal anlatımı ilk birleşik sürüme aittir.

**Güncel kapsam: önce Look, sonra Check. Retell kullanıcı kararıyla bekliyor.**
Bu dosya güncel yöntemin tek giriş noktasıdır. Eski Look v1–v4 ve öğrenilmiş Check v2/v3 dosyaları
tarihsel deney/kıyas olarak korunur. Yeni ana kayıt adı `A_B_lcr`, ortak uygulama `lcr/region_evidence.py`.

## Amaç ve sınırlar

Captioner bir nesne kelimesi üretir. Look onun kendi cross-attention'ından ilgili yerleri bulur.
Check aynı yerlerde donmuş OWLv2 ile kelimeyi ve olası alternatifleri sorgular. Doğrulama kaydı,
neyin bulunduğunu, bulunamadığını ve kararsız kalındığını açıkça ayırır.
Literatür bu işlemlerin içini kurmak içindir; ana Check'te VQA, captioner tutarlılığı veya dil önceliği yoktur.

Detektörün bir nesneyi bulamaması, görüntüde kesinlikle olmadığı anlamına gelmez. Başka bir nesnenin
görünmesi de claim'i otomatik dışlamaz. `contradicted` bu nedenle kalibre edilmiş bir hata işaretidir;
mantıksal kesinlik veya doğru kelimeyi otomatik değiştirme emri değildir.

## Look

- Captioner'ın seçili dört başlığı, kelime zaman adımı ve sink düzeltmesi korunur.
- Harita nesne sınıfının detektör skoruna bakmadan, attention kütlesi/alan^alpha × objectness^gamma
  ile OWLv2 adaylarını sıralar.
- Kişi dışı nesnelerde küçük kutuları peşinen eleyen iki-patch kenar tabanı kaldırılır.
  Yalnız sonlu, pozitif alanlı, pozitif attention kanıtı olan kutular alınır; en fazla üç aday vardır.
- Alan kuvveti mevcut 0.5 değerinde kalır. Kişi için eski SAM tercihi/proxy kaynak açıklaması korunur.
- Seçim yalnız B_fit'te insan kutularıyla yapılmıştır. İnsan kutuları Check etiketi veya model eğitim hedefi değildir.

Otomatik katkı tablosu: [`generated/LCR_DEVELOPMENT.md`](generated/LCR_DEVELOPMENT.md).
**Düzeltme (2026-11-19):** o tablodaki küçük nesne kapsaması (%39,5 → %83,3) bir sınıfın büyük kutusuna gelen isabeti de "küçük nesne bulundu" sayıyordu. Doğru sayım `scripts/49_look_small_object.py` → `docs/generated/LOOK_SMALL_OBJECT_B.md` (isabet küçük kutuya gelmeli). Astra'nın B'deki yeniden sayımı %4,9 → %55,6 (n = 162; burada yeniden üretilmedi). Filtreyi kaldırmanın faydası sürüyor; Look seçimi top-3'e dayandığı için değişmiyor. Rapor makroları script 49 koşulana kadar `--` gösterir.
Bu, tüm zincirin bağımsız test başarısı değil, B üzerinde bir Look geliştirme sonucudur.

## Check

Her aday kutunun claim ve alternatif skorları ayrı hesaplanır ve kaydedilir. Üç kutunun sınıf skorları
ortalamak yerine sınıf başına maksimumla birleştirilir. Böylece destek için bir kutudaki güçlü kanıt yeterlidir;
contradicted için claim'in bütün adaylarda düşük kalması gerekir. Bu maksimum yalnız Look'un üç adayı
arasındadır; bütün görüntüde detektörün seçtiği nesneye geçiş yapılmaz.

Skor ölçeği, sınıf başına görüntü-maksimum logit dağılımından etiketsiz z ölçeğidir. Bu bir varlık olasılığı
garantisi değildir. B_fit ÇD'de ölçek ve eşikler yalnız eğitim görüntülerinden hesaplanır. Son kalibrasyon
ayrılmış B_cal'dan gelir. Destek ve alternatif için ortak varlık eşiği kullanılır:

- Claim skoru ≥ destek eşiği: `supported`.
- Claim skoru < çelişki eşiği ve alternatif skoru ≥ destek eşiği: `contradicted`.
- Diğer durumlar: `uncertain`; sebebi ayrıca kaydedilir.

Kayıtta bütün adaylar, skorlar, alternatifin sınıfı, karar, kararın hangi kutuyu tanık seçtiği, kullanılan
eşikler ve yöntem kimliği vardır. Alternatifin referansta geçmesi ayrı bir görüntü düzeyi proxy'si olarak
ölçülür; doğru ikame diye sunulmaz. B kıyasında ikame grubu kısıtı daha düşük J verdiğinden ana karar
hâlâ tüm alternatif sınıflarını kullanır; **birlikte bulunabilen nesnelerin alternatif sayılması açık bir
güvenilirlik sınırlamasıdır.** Check'in bunu çözmüş olduğu veya Retell'in güvenle düzeltme yapabileceği iddia edilmez.

## Tek hesaplama ve tekrar üretim

- `scripts/37_unify_lcr.py`: B'de önceden tanımlanmış Look/Check kıyasları, tek kalibrasyon bundle'ı ve kayıtlar.
- `results/lcr_bundle.json`: sorgu, sınıf sırası, attention başlıkları/kuralları, decoder dosyası kimliği,
  ölçek, eşikler ve B_fit/B_cal görüntüleri. Kaynak değişirse sessizce kullanılmaz.
- `pipeline.run(..., region_rule="lcr")`: yeni attention'dan aynı Check uygulamasına gider. Referans okumaz.
- `scripts/04_draft_look_check.py`: ana yol varsayılan `lcr`; eski yollar açık sürüm adı gerektirir.
- `scripts/39_verify_lcr.py`: B kayıtlarında kutu–skor–alternatif tutarlılığı ve kararların korunması.
  `--live`, iki sabit B taslağının attention'ını yeniden hesaplayarak eşitliği kontrol eder; yeni yöntem seçmez.
  Sonuçlar `results/lcr_contract.json`; önceki canlı kontrolün kaydı `results/lcr_live_replay.json`.
- `scripts/38_lcr_table.py`: kayıtlı sonuçları rapor sayılarına çevirir; veri seti okumaz.

Seçim protokolü: [`LCR_BIRLESTIRME_PROTOKOL.md`](LCR_BIRLESTIRME_PROTOKOL.md).
Kalıcılık kimliğine attention/decoder kaynak bilgisi ilk B koşusundan sonra eklendi; seçim, skor ve kararlar
değiştirilmedi. Eski validation dosyaları ve eski Check v4 kilidi korunmuştur.

## Ağır hesap sınırı ve Claude'a devir

Mevcut kayıtlarla yapılabilen Look düzeltmesi ve ortak Check yolu hazır. Kalan Check sorusu: **OWL'in
önceden çıkardığı bir önerinin özelliğini sorgulamak yerine, attention ile seçilmiş alanı gerçekten
kırpıp aynı donmuş OWLv2'ye yeniden göstermek ne değiştirir?** Bu ikinci bir doğrulayıcı değildir;
aynı Check'in görsel kanıtını çıkarma biçiminin kıyasıdır. Henüz yapılmadı veya kazandığı gösterilmedi.

Sabit B pilotu, CUDA zorunlu çıkarıcı ve Colab defteri hazırdır. Ayrıntı:
[`handoff/CLAUDE_LOOK_CHECK_GPU.md`](handoff/CLAUDE_LOOK_CHECK_GPU.md).
Yeni model hesaplaması bu sınırda durduruldu. Retell eğitimi kapalıdır; ana LCR girdisiyle script 06
eğitim başlatmayı açık hatayla reddeder. Eski Retell deneyleri tarihsel olarak durur.

Kaynaklar: [`LITERATUR_TARAMASI.md`](LITERATUR_TARAMASI.md) §5–8c,
[OWLv2 model kartı](https://huggingface.co/google/owlv2-base-patch16-ensemble),
[OWLv2 makalesi](https://arxiv.org/abs/2306.09683).
