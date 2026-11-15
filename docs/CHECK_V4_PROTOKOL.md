# Check v4 — özgün mantığa sadık iyileştirme: önceden yazılmış protokol (2026-11-14, Claude)

**Kullanıcı kararı:** Look–Check–Retell'in asıl mantığı korunur. Check = "captioner'ın Look ile seçilen bölgesine
ayrı, donmuş detektör (OWLv2) bakar; bölgede claim mi var, yoksa yerine geçen başka bir şey mi" (rapor §3,
`lcr/check.py: verdict`). Skoru artırmak için mantığı değiştiren sinyaller (dil önceliği, bütün görüntü VQA'sı,
captioner tutarlılığı; Check v2/v3) **ana Check olmaz**; yalnızca karşılaştırma satırıdır.
Bu dosya **koşmadan önce** yazıldı.

## 0. Denetim: nerede saptık?
- **Look:** harita captioner'ın kendi cross-attention başlıkları (v4: etiketsiz seçilmiş 4 başlık), bölge özgün
  r(b) kuralı (kütle/alan^α × objectness^γ). Gradyan ağırlıklı haritalar (Grad-CAM, Chefer) E3'te denendi, B'de
  kaybetti. Tek sapma: kişi için SAM kutusu; Check skoru en yakın OWLv2 önerisinden (ortanca IoU 0.88). Kişi
  yalnız varlık için kontrol ediliyor ve değerlendirme dışı; kayıt alanlarında açıkça yazılı. Değişiklik yok.
- **Check:** v1 kural mantığı doğru, ama B teşhisi (Look v4 kayıtları, özgün eşikler 0.2/0.05) şunu gösterdi:
  contradicted kesinliği 0.376, yakalama 0.215. Yanlış alarmların başında claim'i **içeren/taşıyan** nesneler
  geliyor: shirt→person, grass→dog, water→dog, ball→dog, collar→dog. Bunlar "yerine geçen" değil, "aynı yerde
  duran" nesneler. Kural metni de bunu istemiyor: alternatif, claim'in yerine söylenmiş olabilecek bir şey olmalı.

## 1. Bileşenler (hepsi Check'in kendi mantığı içinde)
| Kod | Bileşen | Seçenekler | Gerekçe |
|---|---|---|---|
| Q | Sorgu metni | Q1 tek şablon (mevcut) / Q2 7 şablon ortalaması (`23_owl_queries.py`) | CLIP/ViLD prompt ensembling; v2'de AUROC etkisi yoktu, kurala etkisi ölçülür |
| C | Skor ölçeği | C1 ham olasılık / C2 sınıf başına etiketsiz standartlaştırma: z = (logit − μ_c)/σ_c, μ_c, σ_c B görüntülerinde sınıfın görüntü başına en yüksek logit dağılımından | OWLv2 skorları sınıflar arası karşılaştırılamıyor (rapor, Guo+ 2017); etiket kullanmaz |
| A | Alternatif kümesi | A1 mevcut (bölgede detektörün ilk 5'i ∪ captioner'ın 3 adayı) / A2 aynı tür (thing↔thing, stuff↔stuff; kişi asla) / A3 yerine geçebilir grup (thing'de üst kategori, stuff'ta tür) | Teşhis: içeren/taşıyan nesne alternatif sayılıyor |
| R | Bölge | R1 Look'un seçtiği öneri / R2 Look'un r(b)'ye göre ilk 3 önerisinin r-ağırlıklı skoru | R4 yumuşak bölge (LITERATUR §8a); bölge yine attention'dan |
| G | Görüntü koruması | G0 yok / G1 contradicted için claim'in görüntüdeki en iyi skoru da τs'nin altında olmalı | Kuralın kendi gerekçesi: tek bölgede bulamamak yokluk kanıtı değil; s_image özgün kayıtta var |
| E | Attention kapısı | E0 yok / E1, E2: bölgenin excess kütlesi B'deki %10 / %20 dilimin altındaysa uncertain | Özgün τe kapısı (şu an 0, fiilen kapalı) |

A3 grupları (sabit, elle; `vocab.py` yorum başlıklarıyla aynı): hayvan, araç/tahta, oyun-spor nesnesi, giyim,
hayvan eşyası (collar/leash/harness/muzzle), elde taşınan/mobilya, yapı, bitki; stuff tek grup.

Karar kuralı biçimi **değişmez**: supported ⇔ s(c) ≥ τs; contradicted ⇔ s(c) < τc ve s(a) ≥ τs (+ G1); diğerleri
uncertain. Eşikler tür başına (thing, stuff).

## 2. Eşikler ve seçim (yalnız B)
- Eşik seçimi (her konfigürasyon, her tür): τs = supported kesinliği ≥ 0.95 olan en küçük eşik (v2/v3 ile aynı
  hedef); τc = contradicted kesinliği ≥ 0.5 (en az 20 karar) koşuluyla yanlış yakalamayı en büyüten eşik.
  Izgara: eğitim katı skorlarının %1'lik dilimleri.
- 2×2×3×2×2×3 = 144 konfigürasyon, **B'de görüntüye göre 5 katlı ÇD**: eşikler 4 katta, kararlar 5. katta;
  birleşik dış-kat metrikleri.
- Birincil ölçüt: birleşik yanlış yakalama (birleşik contradicted kesinliği ≥ 0.45 olan konfigürasyonlar
  arasında). En iyiden en fazla 1.0 puan geride olanlardan, varsayılandan (Q1 C1 A1 R1 G0 E0) en az bileşen
  değiştiren seçilir; eşitlikte daha yüksek kesinlik.
- Değerlendirme kişi dışı claim'ler; etiket otomatik referans proxy'si.

## 3. Validation (bir kez, kilitten sonra)
Seçilen konfigürasyon, eşikler tüm B'de yeniden seçilir. Karşılaştırmalar (görüntü düzeyinde eşleştirilmiş
bootstrap, 2.000): (i) canlı özgün kural (0.2/0.05), (ii) özgün bileşenler + B'de seçilmiş eşikler, (iii) Check v4.
Referans satırları (ana değil): Check v2 ve v3 öğrenilmiş doğrulayıcılar. Raporlanan: supported/contradicted
kesinliği, yanlış yakalama, J, karar oranları, gruplar (giyim/nesne/zemin), hedef 0.6/0.7 de; sürekli ölçek
olarak kalibre bölge skorunun AUROC'u.

## 4. Çıktı
`scripts/35_check_v4.py` → `results/check_v4_selection.json` (kilit), `results/check_v4.json`,
`data/cache/pipeline/A_{B,validation}_look4_check4.json`; tablo `docs/generated/CHECK_V4_METRICS.md`.

## 5. Değişiklik 1 (2026-11-14, seçim koşmadan önce)
**Ne gördük (B, yalnız varsayılan + 3 tek bileşen, duman testi):** §2'deki eşik politikası (τs = supported
kesinliği ≥ 0.95 olan en küçük eşik) tek bir bölge skorunda τs'yi çok yükseğe itiyor. Özgün kural alternatifi de
aynı τs ile sınadığı için contradicted neredeyse hiç oluşmuyor. Varsayılan bileşenlerle ÇD'de yanlış yakalama
0.001, J 0.203; canlı kural (0.2/0.05) 0.215 / 0.364. "B'de ayarlanmış özgün kural" canlı kuraldan kötü
çıkacağından, iyileştirme kıyası anlamsızlaşırdı.
**Değişiklik:**
- τs claim ve alternatif için **ortak varlık eşiği** olarak kalır (özgün biçim).
- (τs, τc) aynı ızgarada **birlikte** seçilir: üç yönlü kararın J'sini (calibrate.verdict_quality) en büyüten çift,
  koşul contradicted kesinliği ≥ hedef (en az 20 karar). Hiç contradicted vermemek her zaman bir seçenektir.
- Konfigürasyon seçiminde birincil ölçüt **ÇD J**'dir (contradicted kesinliği ≥ 0.45 olanlar arasında, en iyiden
  ≤ 1.0 puan geride olanlardan en az bileşen değiştiren). Gerekçe: kural iki yönlü karar veriyor, eşik amacı ile
  seçim amacı aynı olmalı. Yakalama ve kesinlik yine raporlanır.

Bileşen ızgarası, ÇD düzeni, validation kıyasları ve çıktılar değişmedi. Duman testinde görülen 4 konfigürasyon
dışında hiçbir sonuç görülmeden yazıldı. Validation hâlâ okunmadı.
