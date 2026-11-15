# Check düzeltmesi — ön kayıtlı protokol (2026-11-15, Claude)

Koşudan **önce** yazıldı. Kaynak: Fable incelemesi (`docs/FABLE_INCELEME.md`, C1a + C2 + C3 + C4 + C7).
Mantık değişmez: Look'un seçtiği bölgeye **yalnız donmuş OWLv2** bakar; caption modeli, dil önseli, referans ya
da ikinci bir doğrulayıcı çıkarımda yoktur. Değişen yalnız OWLv2 skorunun **nasıl okunduğu**.

## 1. Neden
- Mevcut varsayılan (`lcr_check_bundle.json`, `15fdb954e7de71c6`) skoru sınıf başına z'ye çeviriyor
  (μ_c, σ_c tüm görüntülerin max logit'inden). Yaygın sınıflarda (shirt, jacket…) μ_c "sınıf var" düzeyinde olduğu
  için gerçek nesne eşiği geçemiyor: referansta doğrulanmış giysi claim'lerinin %32'si contradicted (Fable, V1).
- `sa_any ≥ τs` koşulu boş (V2); ikame z ve max-3 kutu üzerinden seçiliyor (V4); GROUP birlikte görülen nesneleri
  topluyor, birbirini dışlayanları değil (V4); eşik proxy J'ye uyduruluyor (V5).

## 2. Yöntem (S3, `decision="positive_anchored"`)
**Ölçek:** ham OWLv2 logit'i, `(e·q + shift)·scale`. z yok (bundle'da μ=0, σ=1).

**Sınıf eşiği τ_c (C1a):** eğitim katındaki görüntülerde, referansları c'yi anan görüntülerin (temiz pozitif)
c için görüntü-max logit'inin α-kantili q_c. Nadir sınıf için büzme:
τ_c = (n_c·q_c + m·q_kind)/(n_c + m), m = 20; n_c < 5 ise τ_c = q_kind. q_kind: aynı türün (thing / stuff)
tüm sınıflarının pozitiflerinin birleşik α-kantili. Hiç "yanlış" etiketi kullanılmaz.

**Çekimser sınıflar (C4):** eğitim katındaki görüntülerde c'nin görüntü-max logit'i ile "referansta anılıyor"
arasındaki AUROC < 0.75 ya da n_c < 10 ise c **asla çelişkili ilan edilmez** (keep / keep_elsewhere / skip). Bu
AUROC bir otomatik proxy'dir; etiket gürültüsü onu aşağı çeker, bu yüzden yalnız çekimserlik için kullanılır.

**Kural (C2):** s_r = Look top-3 kutu üzerinde claim'in max ham logit'i; s_img = tüm önerilerde max.
1. Bölge yok → skip (no_region); "the camera" kalıbı → skip (non_visual_idiom), önceki gibi.
2. s_r ≥ τ_c → **keep** (supported).
3. değilse s_img ≥ τ_c → **keep_elsewhere** (uncertain).
4. değilse c çekimserse → **skip** (abstain_class).
5. değilse (baktığı yerde de, görüntünün hiçbir yerinde de yok) → contradicted:
   - Look **top-1** kutusunda, claim'in karışma kümesindeki (C3) çekimser olmayan bir j için
     logit_j ≥ τ_j **ve** logit_j > logit_c ise → **replace**, ikame = argmax_j (logit_j − τ_j);
   - yoksa → **remove**.
   `sa_any` koşulu kaldırıldı. Kayıtta `alt_any` = bölgede kendi eşiğini en çok aşan sınıf (bilgi amaçlı).

**Karışma kümeleri (C3):** vocab'dan elle, birbirini dışlayan adlar; verilere bakılarak ayarlanmadı.
Küme dışındaki sınıf için replace yok (yalnız remove).
- giysi: üst (shirt, jacket, sweater, vest); tüm-beden (dress, suit, uniform, costume, swimsuit, bikini);
  alt (pants, shorts, skirt); baş (hat, helmet, headband, bandanna); göz (glasses, goggles); ayak (shoe, boot);
  taşınan (backpack, bag)
- hayvan (dog, horse, bird, duck, cat, cow, sheep, goat, fish, camel, elephant)
- iki teker (bicycle, motorcycle, scooter, unicycle); yol aracı (car, truck, bus, train); tekne (boat, canoe,
  kayak, raft); kayma tahtası (skateboard, snowboard, surfboard, ski, sled)
- atılan (ball, frisbee, stick, bone, balloon); vuruş (racket, bat); oyun yapısı (swing, slide, trampoline, ramp)
- oturak (bench, chair, couch, bed); kap (cup, bottle, bucket); okunan (book, newspaper); sınır (fence, wall,
  railing); bitki (tree, bush, flower)
- stuff: su kütlesi (ocean, lake, river, pond, stream, puddle); zemin malzemesi (grass, sand, snow, dirt, mud, ice);
  yol yüzeyi (street, sidewalk)
- person: değişmedi (SAM bölgesi, yalnız varlık).

**Çalışma noktası (C7):** α ∈ {0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25}. Seçim: dış-kat **FCR ≤ 0.05** olan en büyük
α (FCR = referansta doğrulanmış thing+stuff claim'lerinden replace/remove alanların oranı; "camera" kalıbı hariç).
Seçimde başka ölçüt yok; swap ve proxy yalnız rapor.

**Çapraz uyum:** 5 kat, görüntü düzeyi (`calibrate.fold_of`), yalnız B. τ_c, çekimserlik ve q_kind her katta
yalnız eğitim katından. Son bundle seçilen α ile tüm B'de uydurulur.

## 3. Ölçüm (insan etiketi yok; hepsi B, dış-kat)
- **FCR**, gruba göre (giysi ayrı).
- **Swap testi** (Fable'ın kurgusu, aynen: referansta doğrulanmış thing claim'inin etiketi blip_alts'tan referansta
  olmayan bir thing ile, yoksa aynı GROUP'tan referansta olmayan rastgele sınıfla değiştirilir, seed 0): yakalama
  (replace+remove), ikame = gerçek sınıf oranı, ikame kesinliği; "yüksek taban" alt kümesi (değiştirilen sınıfın
  pozitif görüntü sayısı gerçek sınıfınkinden büyük) ayrıca — sınıf öncesinden gelen kazancı ayırmak için.
- **Proxy** (referans-anma) J / contradicted precision / recall: yalnız betimsel.
- **Karşılaştırma:** S2 (mevcut beş-yollu z kuralı, aynı katlarda yeniden uydurulur) vs S3.

## 4. Beklenti ve karar kuralı
- Beklenti (Fable'ın ön ölçümü): S3, FCR ≤ %5'te S2'den daha çok swap yakalar; giysi FCR'si belirgin düşer.
- S2 kendi (proxy J) çalışma noktasında daha yüksek FCR ile çalışıyor; farklı noktalardaki yakalamayı kıyaslamak
  yanıltır. Bu yüzden **eşleşmiş FCR'de** kıyas: S3'ün α ızgarasında dış-kat FCR'si S2'nin dış-kat FCR'sini
  aşmayan en büyük α'daki swap yakalaması.
- **Varsayılan yapılır** eğer (a) eşleşmiş FCR'de S3 yakalaması ≥ S2 yakalaması **ve** (b) seçilen α'da
  (FCR ≤ 0.05) S3 FCR'si S2'ninkinden düşük. Aksi halde S3 yalnız ablation olarak kalır, varsayılan değişmez.
  Sonuç ne olursa raporlanır.
- Validation ve test okunmaz. Sonuçlar automatic proxy / sentetik test olarak adlandırılır.

## 5. Çıktılar
`scripts/43_check_fix.py` → `results/lcr_check3_bundle.json`, `results/check_fix_B.json`,
`docs/generated/CHECK_FIX_B.md`, `data/cache/pipeline/A_B_lcr3.json`. Doğrulama: `scripts/39_verify_lcr.py`.
