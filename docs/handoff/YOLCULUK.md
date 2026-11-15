# Yolculuk: Look–Check–Retell'de bugüne kadar ne yaptık (2026-11-15)

Bu doküman projeyi devralan ekip arkadaşları ve onların ajanları (Claude, Codex, …) içindir. Neyi neden yaptığımızı,
nerede yanıldığımızı ve şu an nerede durduğumuzu kısaca anlatır. Ayrıntı her zaman bağlantılı dosyalardadır.
Sayılar ilgili `results/*.json` ve `docs/generated/*.md` dosyalarından alınmıştır; elle sayı uydurulmaz.

## 0. Önce bunu oku (ajanlar dahil)
1. `AGENTS.md`: değişmez kurallar. Test asla okunmaz; seçim B'de, rapor validation'da; proxy'ler proxy diye yazılır;
   sonuç uydurma yok.
2. `docs/CALISMA_GUNLUGU.md`: "Şu anki durum", "Sıradaki adımlar" ve tarihli günlük.
3. Bu dosya, sonra `docs/CHECK_S4.md` (güncel Check) ve `docs/LCR.md` (tek yöntem).
4. Konuşmaların kendisi: `docs/handoff/sohbetler/` (Claude oturumları, kronolojik) ve `docs/handoff/transcript_main.md`
   (ilk büyük oturum).

## 1. Fikir
CS415 (Sabancı) projesi. Bir görüntü altyazıcısı (BLIP) bazen görüntüde olmayan nesneleri yazar (hallucination).
- **Look:** altyazıcı bir nesne kelimesini üretirken kendi cross-attention'ı görüntünün neresine bakıyor?
- **Check:** bakılan yere **donmuş bir nesne detektörü (OWLv2)** bakar. Orada iddia edilen nesne var mı, yoksa ne var?
  Kararı detektör verir, altyazıcı değil.
- **Retell:** bu bilgiyle altyazı yeniden üretilir ya da düzeltilir.
- **Veri:** Flickr8k. Train iki yarıya bölündü: A'da altyazıcı fine-tune edildi, **B'de her seçim/ayar yapıldı**;
  validation rapor için, test en sonda bir kez.
- **İnsan kutuları:** Flickr30k Entities kutuları yalnızca Look'u ölçmek için kullanılır.

## 2. Modeller ve araçlar
- **Altyazıcı:** BLIP-base; encoder donuk, decoder A yarısında fine-tune edildi (`data/cache/ckpt/decoder_A.pt`).
  Validation CIDEr 0.669 → 0.979.
- **Detektör:** `google/owlv2-base-patch16-ensemble`, donuk. Öneri kutuları ve gömmeleri önbellekte
  (`data/cache/owl/`).
- **Kişi bölgeleri:** SAM (Look v4'te person için).
- **Hesap:** ağır işler Colab'da (T4/L4 yeterli; A100 gerekmedi), hafif işler Mac M4'te CPU.
- **Ajanlar:**
  - Claude (Opus): ana geliştirme ve Check tasarımı.
  - Codex: bağımsız incelemeler, birleştirme, Retell araştırması.
  - Fable (Claude Fable 5.2): iki bağımsız yöntem incelemesi (`docs/FABLE_INCELEME.md`, `docs/FABLE_INCELEME_2.md`).

## 3. Kronoloji

### 13 Kasım: ilk tur başarısız oldu, kök neden Look çıktı
- İlk uçtan uca Colab turu (fine-tune → taslak → Look/Check → Retell) çalıştı ama işe yaramadı:
  - Retell kanıtı yok saydı.
  - Ham attention bölgesi, doğru/yanlış claim ayırmada şans düzeyindeydi (AUROC 0.51).
  - Check'in bilgi değeri neredeyse sıfırdı.
- **Ekip kararı:** sırayla düzelt. Önce Look ölçülsün ve otursun, sonra Check, sonra Retell. Her ara aşama bağımsız
  bir "kapı" ile ölçülür (`docs/LITERATUR_TARAMASI.md`).

### 13–14 Kasım: Look düzeltildi (insan kutularıyla ölçüldü)
- **Neden bozuktu:** attention sink. Ham BLIP attention'ı kelimeden bağımsız arka plan yamalarına yığılıyor.
  Caption ortalamasını çıkarma düzeltmesi yapıldı (`docs/LOOK_V2_PLAN.md`).
- **Look v2 → E1 → E2 → E3/v4.** Her adım önce protokolle yazıldı, B'de seçildi, validation'da bir kez ölçüldü:
  - E1: kutu kuralı ayarı, kişi/stuff için.
  - E2: yeni harita kaynakları; hiçbiri kabul eşiğini geçmedi.
  - E3: etiketsiz başlık seçimi (4 başlık) ve kişi için SAM.
- **Look v4:** insan kutusuyla validation bölge doğruluğu %80.4 (v3 %76.1).
  - Codex bağımsız inceledi; canlı ve dondurulmuş yolların birebir aynı olduğunu doğruladı
    (`docs/LOOK_E3_INCELEME.md`).
  - B'de ilk üç kutudan biri insan kutusunu %94.4 oranında içeriyor; Look tavana yakın.

### 14 Kasım: Check'te yanlış yola girip döndük
- **Check v2/v3:** öğrenilmiş doğrulayıcılar (kalibrasyon, sınıf önseli, VLM sinyalleri, tutarlılık) proxy'de iyi
  göründü.
- **Neden geri döndük:** kullanıcı itiraz etti: "check yapan herhangi bir şeyle değiştirme." Bu doğrulayıcılar özgün
  mantıktan uzaklaşıyordu; sınıf önseli tek başına AUROC 0.76 veriyordu, yani "görüntüye bakmadan" iyi görünüyordu.
  v2/v3 artık yalnız kıyas satırı.
- **Check v4:** özgün mantığa sadık. Detektör, Look'un seçtiği kutularda karar verir.
- **Codex, birleştirilmiş LCR:** tek ortak kod yolu (`lcr/region_evidence.py`, `RegionChecker`), canlı/önbellek/
  bozma için aynı hesap (`docs/LCR.md`).

### 15 Kasım: Check tamamlandı, iki kez incelendi, S4 bulundu
- **İlk ilkeler** (`docs/ILK_ILKELER_TASARIM.md`).
  - Proxy etiket ("referansta geçiyor mu") çok gürültülü: görsel denetimde proxy-yanlışların çoğu doğruydu.
  - Ayrıca "kamera" gibi kalıplar nesne iddiası değil.
- **Check tamamlama** (`docs/CHECK_TAMAMLAMA.md`):
  - Beş eylem: keep / keep_elsewhere / replace / remove / skip.
  - Görüntü düzeyi veto.
  - "camera" kalıbı.
- **Fable incelemesi 1** (`docs/FABLE_INCELEME.md`):
  - Sınıf başına z ölçeği yaygın sınıflarda yanlış. Doğru giysi claim'lerinin %32'si çelişkili sayılıyordu.
  - "Bölgede başka şey var" koşulu boş.
  - İkame grupları birbirini dışlamıyor.
  - Etiketsiz iki ölçüm aracı önerdi: FCR ve swap testi.
- **Pozitif-çapalı düzeltme, S3** (`docs/CHECK_DUZELTME.md`):
  - Ön kayıtlı ölçütü **tutmadı**, varsayılan değişmedi.
  - Sonradan tanı: "az pozitif" çekimserliği, nadir sınıflara düşen hataları görünmez yapıyordu.
- **Fable incelemesi 2** (`docs/FABLE_INCELEME_2.md`):
  - Swap testinin yarısı yanıltıcı; etiketlerin çoğu birbirini dışlamıyor ve görüntüde gerçekten var.
  - Tek eşik bütçeyi kolay sınıflara harcıyor.
  - Claim tek başına skorlanıyor.
  - Giysi türü detektörle sınırlı.
  - Kırpıp yeniden bakmak (CPU pilotu) kazanç vermedi.
- **Check S4, yeni varsayılan** (`docs/CHECK_S4.md`):
  - Rakip-farkında kanıt: aynı kutuda detektör dışlayan bir adı tercih ediyorsa claim zayıflar.
  - Maliyete duyarlı sınıf eşikleri: tek λ.
  - Look'un ilk kutusunda merkezli ikame.
  - Temiz karışma kümeleri ve birleşik kelime atlama ("hot dog" gibi).
  - Ön kayıtlı üç ölçüt tuttu.

| B, dış-kat (proxy / sentetik) | thing yanlış alarm | giysi yanlış alarm | kesin yanlış etiket yakalama | ikame doğruluğu |
|---|---|---|---|---|
| S2 (önceki varsayılan) | 0.117 | 0.320 | 0.552 | 0.689 |
| **S4 (şimdiki)** | **0.030** | **0.041** | **0.808** | **0.912** |

**Uyarı:** S4'ün bileşenleri aynı B katlarında keşfedildi. Bağımsız doğrulama, yöntem dondurulunca validation'da
bir kez yapılacak.

## 4. Öğrendiklerimiz
- **"Script hatasız koştu" doğrulama değil.** İlk turun hatası uçtan uca metrikle saklandı. Her aşama bağımsız bir
  kaynağa karşı ölçülmeli.
- **Ön kayıt işe yarıyor.** S3 başarısız oldu ve bunu protokol yüzünden dürüstçe raporladık. Sonradan tanıyı seçim için
  kullanmadık.
- **Proxy'ye fazla güvenme.** "Referansta geçmiyor" ≠ "yok". Swap testi de kusurlu. Gerçek hata oranını yalnız insan
  etiketi söyler.
- **Look'un rolü:** nesnenin yokluğunu Look bulamaz; yokluk görüntü düzeyinde bir bilgidir. Look'un işi "baktığı
  yerde aslında ne var" sorusu: ikame doğruluğu Look kutusuyla 0.91, Look olmadan yaklaşık 0.71.
- **Ajanlarla çalışma:**
  - Bağımsız inceleme (Codex, Fable) birkaç gerçek hatayı yakaladı.
  - Görev dosyaları açık uçlu olmalı, kısıtlayıcı değil.
  - Ağır işler Colab'a; Mac'te aynı anda tek hafif CPU işi.

## 5. Sıradaki adımlar
1. **İnsan etiketi (en önemlisi):** ~300–400 B claim.
   - Kapsam: S4 contradicted'ların hepsi; S2 contradicted / S4 keep (özellikle giysi); replace ve keep_elsewhere;
     taban oranı için rastgele keep.
   - Araç: `annotation/pilot/annotate.html`.
   - Sonuç gerçek hata oranını verir; bu da FCR hedefini (λ) belirler.
2. **L1:** Look'u yanlış kelimede ölç. Swap testinde bölge doğru kelimeden geliyor; gerçek hatada attention nereye
   bakıyor? CPU'da yapılabilir.
3. **Validation:** yöntem dondurulunca bir kez. Test en sonda.
4. **İsteğe bağlı (GPU):** OWLv2-large ile giysi ayrımı denemesi (günlükte not).
5. **Retell:** kullanıcı kararıyla bekliyor. Taslak fikir:
   - `remove` → kelimeyi yasakla, öneki koruyup yeniden üret;
   - `replace` → ikameyi öner;
   - yeni altyazıya yeniden Look+Check uygula, yalnız iyileşirse kabul et.

   Codex'in literatür notu: `docs/handoff/RETELL_ARASTIRMA_NOTLARI.md`.
6. **Rapor (`report/`):** Codex'in son bölümleri ve yeni Check sonuçları rapora işlenmeli. Rapordaki her sayı
   scriptten gelir.

## 6. Veri (büyük dosyalar repoda değil)
- `data/cache` (~9 GB: OWLv2 önbelleği, BLIP özellikleri, `decoder_A.pt`, Look haritaları, pipeline kayıtları)
  GitHub release `data-v1` olarak yüklü. `bash scripts/fetch_data.sh` indirir, SHA256 ile doğrular ve birleştirir.
- Flickr8k görüntüleri ve NLTK verisi: `python scripts/00_download.py`.
- HF modelleri ilk kullanımda iner.
- Flickr8k'den türetilen görseller (inceleme galerileri, etiketleme aracı) lisans nedeniyle public repoda yok;
  scriptleriyle yeniden üretilir (`scripts/07_make_annotation.py`).
