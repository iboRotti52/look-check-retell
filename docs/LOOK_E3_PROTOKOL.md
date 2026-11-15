# Look E3 — literatürde denenmemiş yöntemler: önceden yazılmış protokol (2026-11-14, Claude)

Neden: kullanıcı "vakit var, her adımı olabildiğince iyi hale getirelim, eksikleri deneyelim" dedi.
E1/E2 sonrası Look'ta hiç denenmemiş yöntemler kaldı (`docs/LOOK_V2_PLAN.md` §3, `docs/LITERATUR_TARAMASI.md` §5/5a).
Bu dosya **deney koşmadan önce** yazıldı; ızgara ve kabul kuralı buradan okunur.

## Taban çizgisi
- **v3** (Codex'in dondurduğu): arşiv sink haritası (son 4 katman × 12 başlık, caption ortalaması çıkarılıp
  kırpılmış) + E1 türe özel kutu kuralı (`config.LOOK_RULES_BY_KIND`).
- **S3-yeniden**: aynı sink haritası bu geçişte yeniden hesaplanır. E2'de arşivle küçük bir sapma görüldü
  (maks. 0.17); adaylar hem v3'e hem S3-yeniden'e karşı raporlanır, sapmanın kaynağı ayrıca incelenir.

## A aşaması: harita kaynağı (kutu kuralı sabit: v3 kuralları)
| Kod | Yöntem | Kaynak | Izgara |
|---|---|---|---|
| A1-G2 | Katman seçimi: tek katman ya da son k katman | ALBEF (Li+ NeurIPS'21): grounding belirli katmanda | l ∈ 1..12; k ∈ {1,2,4,6,8,12} |
| A1-G3 | Etiketsiz başlık seçimi: başlık başına kelimeye özgüllük (aynı görüntüdeki farklı sınıf claim'lerinin haritaları ne kadar ayrışıyor) × yoğunluk (Kang+ CVPR'25'in bağlı-bileşen entropisinin uyarlaması) | Kang+ CVPR'25, uyarlama | ilk m başlık, m ∈ {4,8,16,32,48} |
| A1-G4 | Denetimli başlık seçimi: B'de insan kutusuyla tek başlık pointing doğruluğu | bizim ekleme | m ∈ {4,8,16,32,48} |
| A2 | İsim öbeği: sıfat/sayı/isim değiştiricileri + baş isim, üretim adımlarının ortalaması | F30kE kutuları öbek düzeyinde (Plummer+ ICCV'15) | tek varyant |
| A3 | Oran tipi sink: ham / caption ortalaması | Zhang+ ICLR'25 (göreli attention), uyarlama | tek varyant |
| A4 | Gradyan-ağırlıklı relevance yayılımı (encoder–decoder kuralları; görüntü tarafında R^ii = I, çünkü ViT önbellekte) | Chefer+ ICCV'21 | tek varyant |
| A5 | Salience DropOut: en yüksek q yamayı özellik düzeyinde maskeleyip haritayı yeniden hesapla, iki haritayı birleştir | PnP-OVSS (Luo+ CVPR'24) | q ∈ {%5, %10} |
| A6 | Topluluk: S3 ile bir başka haritanın sıra-normalize ortalaması | LOOK_V2_PLAN §3.4 (bizim ekleme) | S3 + {S2 Grad-CAM, S4 kontrastif, A4, A5} |

## B aşaması: kutu (A'da seçilen haritanın üstüne, türe göre ayrı)
Codex'in tanısı: kişi hatalarının çoğunda tepe noktası doğru kişide, kutu ölçeği yanlış.
| Kod | Yöntem | Izgara |
|---|---|---|
| B1 | Haritanın tepe noktası SAM'e nokta istemi; SAM'in kendi IoU tahminiyle en iyi maskenin kutusu | – |
| B2 | Aynı, ama 3 maskeden attention kütlesi / alan^α skoru en yüksek olan | α ∈ {.25,.5,.75} |
| B3 | OWLv2 önerileri v3 skoru × IoU(öneri, SAM kutusu)^δ ile yeniden sıralanır | δ ∈ {.5,1,2} |

SAM (Kirillov+ ICCV'23) sınıf metni görmez; yalnızca görüntü ve nokta. İddia edilen sınıfın detektör skoru
yine kullanılmaz (döngüsellik ilkesi korunuyor).

## Seçim ve kabul
- Ölçüt: B'de insan kutusuyla doğru bölge (IoU ≥ 0.5, herhangi bir GT kutusu). Pointing ve enerji de raporlanır.
- Her aile için kazanç, **B'de görüntüye göre 5 katlı çapraz doğrulamayla** tahmin edilir: aile içinden yapı
  4 katta seçilir, 5. katta ölçülür. Böylece büyük ızgaralı aileler (başlık seçimi) küçüklerle adil kıyaslanır.
- A: en yüksek ÇD kazançlı aile, kazanç ≥ 1.0 puansa kabul edilir (E1 ile aynı eşik); yapı tüm B'de yeniden seçilir.
- B: tür başına, A sonrası haritaya göre ÇD kazancı ≥ 1.0 puansa kabul.
- Kabul edilen harita `14_look_sanity.py` tarzı kapılardan (transpoze, takas) geçmeli.
- Seçim kilitlenip dosyaya yazılmadan validation okunmaz. Validation'da bir kez: v3'e karşı görüntü düzeyinde
  eşleştirilmiş bootstrap (2.000). Reddedilen her aile de B ve validation sonucuyla raporlanır.
- Kabul olursa Look v4 dondurulur ve Check (script 21–22) v4 kayıtlarıyla yeniden koşulur.

## Kapsam dışı bırakılanlar (gerekçeyle)
- Attention rollout (Abnar & Zuidema ACL'20): self-attention için tanımlı; decoder cross-attention'da
  A4 (Chefer) bunun genelleştirmesi. Ayrıca denenmez.
- BLIP-ITM Grad-CAM: ayrı bir modelin haritası; captioner'ın kararını açıklamaz. Look'un tanımına uymuyor.
- R4 yumuşak bölge: Check'in parçası olarak ele alınacak.
- Denetimli grounding eğitimi (POS-SCAN): F30kE'yi eğitimde kullanmayı gerektirir, kural dışı.

## Uygulama açıklaması — Codex incelemesi, 2026-11-14

Bu ek, yukarıdaki önceden yazılmış ızgarayı veya kabul kuralını değiştirmez.
B2'nin uygulaması SAM maskesinin çevreleyen **dikdörtgenindeki** attention kütlesini
**dikdörtgen alanı**^α ile puanlar; gerçek maske içindeki kütle/maske alanı değildir.
A1-G3 başlık sıralaması etiketsizdir; m ve aile tercihi B insan kutularıyla yapılır.
E3 Salience DropOut, özgün PnP-OVSS'nin özellik düzeyindeki sınırlı uyarlamasıdır.
Aşamalı A→B seçimi bütün B'yi gördüğü için raporlanan aile ÇD değerleri tüm seçicinin tamamen
bağımsız dış-kat tahmini değildir. Aile/tür bootstrap aralıkları eşzamanlı güvence sağlamaz.
Kapanış, sayısal doğrulama ve canlı v4: `docs/LOOK_E3_INCELEME.md`.
