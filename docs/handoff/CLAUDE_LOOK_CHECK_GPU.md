# Claude: ağır hesap sınırında Look → Check devri (2026-11-14)

**Kullanıcı:** Mantık korunacak. Önce Look, sonra Check; Retell'e geçme. M4'te ağır model hesabı yapma.
GPU gereken işi kullanıcı Claude'a yaptıracak. Codex yeni GPU işi veya Retell eğitimi başlatmadı.
Güncel yöntemin tek tanımı: `docs/LCR.md`. Bağlayıcı kurallar: `AGENTS.md`.

## Tamamlanan yerel iş

- Look'un attention başlıkları/haritası korundu. Küçük kutu kenar filtresi B insan kutularında ölçülerek
  kaldırıldı; alpha=0.5 kaldı. Üç sonlu, pozitif, attention ile sıralanan aday kullanılıyor.
- Aynı üç aday ayrı skorlanıyor; claim desteği için sınıf başına maksimum kullanılıyor.
  Kutu, skor, alternatif ve yöntem kimliği tek ortak uygulamada: `lcr/region_evidence.py`.
- B_fit / B_cal ayrıldı; eşik ve ölçek son kez yalnız B_cal'da fit edildi. B'deki seçimden sonra validation
  yeniden açılmadı, test okunmadı. B sonuçları bağımsız genel başarı gibi sunulmamalı.
- Kayıtlı skorların doğrudan özellik/sorgu hesabına uyumu ve canlı sabit-taslak eşitliği denetlendi.
  `results/lcr_contract.json`, `results/lcr_development.json`, otomatik tablo `docs/generated/LCR_DEVELOPMENT.md`.
- Eski Check v4 skorlarını/kararlarını Retell sınırında bozan bağlantılar belirlendi. Ortak kanıt arayüzü
  için hazırlık var; Retell eğitimi özellikle engellendi. Bu aşamada Retell çalıştırma.

## Açık Check sorunu

Mevcut Check, OWLv2'nin önceden ürettiği kutuların sınıf özelliklerini okuyor; bakılan alanı yeni bir
görüntü gibi yeniden kodlamıyor. Alternatifin aynı bölgede yüksek skorlu olması, claim'in yerine uygun
olduğunu garanti etmiyor. Alternatifin referansta geçmesi de yalnız görüntü düzeyinde proxy.
Bu nedenle Check/ikame güvenilirliği kapısı tamamen kapanmış değil. "Bölge kanıtının üst sınırı budur"
sonucu çıkarma. Yeni VQA/ITM modeli, dil önceliği veya captioner tutarlılığı ekleme.

## Sıradaki tek GPU işi: aynı OWLv2 ile sabit attention kırpımları

Hazır girişler:

- `output/lcr_check_gpu_pilot.zip`: B_fit ve B_cal'dan etiketsiz hash sırasıyla sabit pilot, görüntüler,
  dondurulmuş crop geometrileri, aynı sınıf sorguları, tam aynı OWLv2 uygulaması.
- `colab/LOOK_CHECK_GPU_PILOT.ipynb`: yükle → CUDA kontrolü → özellik çıkar → sonucu indir.
- `scripts/40_check_crop_gpu.py`: CUDA yoksa çalışmayı reddeder. CPU/MPS'e düşmez.
  `--dry-run` yalnız girdi tutarlılığına bakar, model yüklemez.
- `results/check_gpu_pilot_manifest.json`: otomatik iş büyüklüğü; paketteki `check_crop_job/manifest.json`
  tam görüntü/crop listesini ve kaynak bundle kimliğini içerir.

Crop geometrileri Look'un mevcut adaylarından gelir; her kenara kutu boyunun %10'u kadar sabit bağlam eklenir.
Aynı görüntüde aynı crop tekrarları birleştirilir. Bu padding bir mühendislik tercihi; henüz seçilmiş kazanan
yöntem değildir. Pilot referans caption veya doğruluk etiketi içermez; çıktı modele göre yeniden seçilmez.
Kişi dışı claim'ler kapsanır; kişi/gender/age yönünde yeni bir doğrulama yapılmaz.

Colab GPU'da paketi açıp:

```bash
python scripts/40_check_crop_gpu.py --dry-run
python scripts/40_check_crop_gpu.py --bs 4
```

T4 yeterli bir başlangıçtır; büyük batch gereksiz. Script çıktıları parça parça yazar ve aynı manifest ile
kaldığı yerden devam eder. İş sonunda `results/check_crop_pilot.zip` döner. Bu dosyada crop içi OWLv2
önerileri, sınıf özellikleri, shift/scale, crop geometrisi, orijinal görüntü koordinatlarına eşlenmiş kutular
ve iş kimliği bulunur. Retell eğitimi veya Check eşik seçimi bu GPU işinin parçası değildir.

## GPU sonucu geldikten sonra (önce Look/Check, hâlâ Retell yok)

1. İşin tamamlandığını, manifest/bundle/sorgu kimliğini ve crop→görüntü koordinatlarını doğrula.
2. Aynı sabit B claim'lerinde eski aday özelliği ile yeni crop içi nesne kanıtını karşılaştır.
   Ham crop skorunu eski z-eşiğine doğrudan koyma: dağılım değişti. Yeni ölçek/eşik kıyasının protokolünü
   B'de önce yaz; B_fit/B_cal sınırını koru.
3. Gerçek destek/yanlış alarm ve alternatifin işe yararlığını ayrı incele. İnsan etiketi yoksa proxy de.
   Bir crop içinde birlikte duran nesneleri körlemesine ikame sayma; zorunlu kelime değişimi yaptırma.
4. Pilot kazanç göstermezse sırf crop daha "doğrudan" diye ana yöntem yapma. Eski yolu kıyas olarak tut.
   Tam B çıkarımına ancak gerekçe ve kaynak bütçesi açıkken geç.
5. Check'in kapanışını ve insan pilotundaki açıkları yaz. Retell'e ancak kullanıcının sonraki yönlendirmesiyle geç.

Chrome'da GPU işi başlatılmadı. Codex'in Colab açma denemesi GPU oturumu oluşturmadı; kullanılacak yeni
notebook bu yerel `.ipynb` dosyasıdır. Çalışan eski oturumları veya eski validation/test dosyalarını ezme.
