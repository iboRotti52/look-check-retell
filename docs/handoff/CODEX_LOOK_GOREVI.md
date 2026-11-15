# Codex görevi: Look'u ilerletmek (2026-11-14)

Önce `AGENTS.md`, `docs/CALISMA_GUNLUGU.md` ("Şu anki durum"), `docs/LOOK_V2_PLAN.md` (§3, §5, §7) ve
`docs/LITERATUR_TARAMASI.md` (§0, §5, §5a) okunmalı. Bu dosya sadece "ne yapılacak" listesidir.

## Neredeyiz

Look = captioner'ın bir claim kelimesini üretirken baktığı yerden bir OWLv2 kutusu seçmek. Dondurulan Look v2:

- **Harita:** BLIP decoder cross-attention, son 4 katman × 12 başlık ortalaması, claim kelimesinin ilk parçasının
  üretildiği adım (konum p−1). Sink düzeltmesi: caption'ın tüm adımlarının ortalaması çıkarılıp sıfırda kırpılıyor
  (`lcr/look.py: attention_map`).
- **Kutu:** `look.select_region`: kütle / alan^0.5 × objectness^0.5 (`config.REGION_ALPHA/GAMMA`).

Validation sonuçları, Flickr30k Entities insan kutularıyla (`scripts/13_look_benchmark.py --pipe A_validation_look2`):

| | n | pointing | şans | enerji | doğru kutu | detektör (B3) | oracle |
|---|---|---|---|---|---|---|---|
| hepsi | 1795 | 88.3 | 37.1 | 73.2 | **75.3** | 82.7 | 95.3 |
| person | 761 | 89.2 | 37.7 | 68.6 | **66.4** | 85.9 | 95.5 |
| thing | 758 | 88.3 | 28.3 | 77.1 | 83.8 | 83.8 | 95.4 |
| stuff | 276 | 85.9 | 59.9 | 75.1 | 76.4 | 71.0 | 94.2 |

**Kalan açık:** person (66 vs 86) ve genel olarak detektöre kalan ~7 puan. Haritanın tepesi person'da %89 doğru,
yani sorun büyük olasılıkla kutu seçiminde. Ölçüm, seçilen kutunun **herhangi bir** GT kutusuyla en yüksek IoU'sunu
alıyor (`13_look_benchmark.py: best_iou`). Yani tek bir kişiyi doğru seçmek yeterli. Kişi hatası büyük olasılıkla
kutunun yarım kişi, kalabalık ya da kişi + nesne gibi yanlış ölçekte olmasından geliyor; T1 bunu ölçecek.

## Değişmez kurallar (bu görev için)

- **Seçim B'de** (`A_B_*` dosyaları, `data/cache/owl/train.npz`), **rapor validation'da bir kez.** Test split'i asla.
- **Look, iddia edilen sınıfın detektör skorunu kullanamaz** (Check'i döngüsel yapar). Sınıftan bağımsız sinyaller
  serbest: objectness, kutu geometrisi, claim'in türü (person/thing/stuff; claim'den biliniyor).
- F30kE kutuları yalnızca ölçüm ve az sayıda hiperparametre seçimi için kullanılır (B'de). Bunlarla bir model
  eğitilmez. 144 başlıktan denetimli seçim yapılacaksa bu "üst sınır" diye ayrıca raporlanır, ana yöntem
  denetimsiz olmalı (Kang+ CVPR'25 tarzı).
- Her yeni harita varyantı için **kapılar**: `scripts/14_look_sanity.py`. Yönelim kontrolü: transpoze harita
  düşmeli. Özgüllük kontrolü: başka claim'in haritası belirgin şekilde düşük olmalı.
- Her sayı bir script'ten ve `results/*.json`'dan gelir. Her bileşen ablation satırı olur.
- Check / Retell koduna dokunma. `colab/run_fix.sh`'i çalıştırma.

## Yerelde elinde olanlar

- BLIP görüntü özellikleri: `data/cache/blip/{train,validation}.f16.npy`.
- OWLv2 önbelleği: `data/cache/owl/{train,validation}.npz`.
- Fine-tune edilmiş decoder: `data/cache/ckpt/decoder_A.pt` (Colab'dan indirildi). Yani **yeni attention ve gradyan
  çıkarımı yerelde de yapılabilir**: görüntü encoder'ı gerekmiyor, sadece decoder ileri/geri geçişi. Mac'te
  `LCR_DEVICE=cpu` ile yavaş ama mümkün; çok uzun sürerse Colab (A100).
- Kayıtlı haritalar (sadece son 4 katman ortalaması, sink): `data/cache/pipeline/A_{B,validation}_sink.maps.npz`.
- Dondurulmuş kayıtlar: `data/cache/pipeline/A_{B,validation}_look2.json`.

## Görevler (sırayla; her birinin sonunda LOOK_V2_PLAN §7'ye satır ekle)

### T1. Person hata analizi ve türe özel kutu kuralı (CPU, mevcut dosyalarla)
1. B'de person claim'lerini görüntüdeki F30kE person kutusu sayısına göre ayır (1 / 2 / 3+) ve bölge doğruluğunu
   her grup için ölç.
2. Hatalı seçimlerde seçilen kutu ile en yakın GT kutusunun alan oranına ve kapsamasına bak: kutu çok mu küçük
   (yüz, gövde), çok mu büyük (grup), yoksa yanlış yerde mi? 20 örneği görselleştir (13'teki `--examples`
   figür koduna benzer).
3. `scripts/16_look_regions.py`'ye türe özel (α, γ) seçeneği ekle: person / thing / stuff için ayrı ayrı B'de seç.
  Gerekçe: kişi kutuları dikey ve büyük, stuff kutuları sınırsız.
4. Gerekirse kişi için "tepe noktasını içeren en nesne-benzeri kutu" ya da "birden fazla kutu birleşimi" gibi
  varyantlar. Hepsi B'de, ablation olarak.

### T2. Harita son işleme (CPU, kayıtlı haritalarla)
- Gauss yumuşatma (σ ∈ {0.5, 1, 1.5} yama), üs dönüşümü (att^p), sink düzeltmesinde ortalama yerine medyan ya da
  oranlama (A(p) / ortalama, Zhang+ ICLR'25). LOOK_V2_PLAN §3.3. Her biri için 13 + 14 + 16 koşulur.

### T3. Yeni harita kaynakları (decoder geçişi gerekir: yerel CPU ya da Colab)
Yeni script önerisi `scripts/18_look_extract.py`. B ve validation claim'leri için:
- Tüm katman × başlık cross-attention'ı (12×12×577) ve gradyanları: ∂log p(w)/∂A (S2 Grad-CAM) ve
  ∂[log p(w) − log p(w_alt)]/∂A (S4 kontrastif, w_alt = `blip_alts[0]`).
- Hem p−1 (üretim adımı) hem p (okuma adımı). Hem ilk parça hem tüm parçaların ortalaması (5a ablation'ları).
- Depolama: tüm A ve gradyanları saklamak çok yer tutar (~9 bin claim × 83k × fp16 × birkaç tensör ≈ GB'larca).
  Bunun yerine geçiş sırasında **her varyantın 24×24 haritasını** hesapla ve sakla. Ayrıca başlık başına 24×24
  haritaları sadece B'nin bir alt kümesi için sakla (başlık seçimi için).
- Teknik not: `Captioner._capture_cross_attention` forward hook ile `out[1]` (attention olasılıkları) alıyor ve
  `.detach()` yapıyor. Gradyan için detach'siz bir sürüm ve `retain_grad()` gerekir. transformers 5.23'te BLIP
  cross-attention olasılıklarının dönmesi eager attention implementasyonuna bağlı olabilir; ilk iş, gradyanın
  sıfırdan farklı geldiğini küçük bir testle doğrula.

### T4. Başlık seçimi (T3'ün çıktısıyla)
- Denetimsiz: Kang+ CVPR'25 ölçütleri (görüntüye düşen attention toplamı + uzamsal entropi), B'de.
- Üst sınır: B'de insan kutularıyla başlık başına pointing, en iyi k başlık (ayrıca raporla).

### T5. Topluluk ve seçim (LOOK_V2_PLAN §5)
- Aşamalı ileri seçim B'de: kaynak → katman/başlık → son işleme → topluluk → kutu kuralı. Her aşamada tek değişiklik.
- Kazanan konfigürasyonu validation'da bir kez raporla, ablation tablosunu §7'ye yaz.
- Kodda dondur (`lcr/look.py`, `lcr/config.py`, `lcr/pipeline.py`). Kayıtları yeniden yaz: script 17'yi genelleştir
  ya da yeni haritalarla 04'ü çalıştır. Yeni dosya eki `look3`.

## Bitti sayılma ölçütü
- Validation'da doğru kutu oranı Look v2'den (75.3) anlamlı şekilde yüksek, özellikle person. Ya da denenen her
  şeyin işe yaramadığı ablation tablosuyla gösterilmiş olmalı (bu da geçerli bir sonuç; LOOK_V2_PLAN §6).
- Kapılar (14) geçiyor, testler geçiyor (`.venv/bin/python -m pytest -q`).
- `docs/CALISMA_GUNLUGU.md` güncellenmiş ("Şu anki durum", "Sıradaki adımlar", tarihli kayıt).
- Ondan sonra sıradaki aşama Check (`docs/LITERATUR_TARAMASI.md` §6–8, 8a).
