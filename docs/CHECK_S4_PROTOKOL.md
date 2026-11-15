# Check S4: ön kayıtlı protokol (2026-11-15, Claude)

Koşudan **önce** yazıldı. Kaynak: Fable'ın ikinci incelemesi (`docs/FABLE_INCELEME_2.md`, I1 + I2 + I3 + I5).
**Önemli uyarı:** S4'ün bileşenleri Fable tarafından **aynı B katlarında keşfedildi**. Bu koşu bağımsız bir kanıt
değil. Yalnızca proje kodundaki uygulamanın keşif sonuçlarını tekrar ürettiğini ve önceden konmuş ölçütlerle
karşılaştırıldığını gösterir. Bağımsız doğrulama, yöntem dondurulduktan sonra validation'da bir kez yapılacak.

Mantık değişmez: Look'un seçtiği bölgeye yalnız donmuş OWLv2 bakar. Caption modeli, dil önseli ve başka bir
doğrulayıcı yoktur.

## 1. Yöntem (S4, `decision="rival_np"`)
**Rakip-farkında kanıt (I2).** Claim sınıfı c'nin (temizlenmiş) karışma kümesi S(c) ise her öneri kutusu b için:
v_c(b) = l_c(b) − 2·max(0, max_{j∈S(c)} l_j(b) − l_c(b)). Kümesiz sınıfta v_c(b) = l_c(b). Ham OWLv2 logit'i kullanılır.
- Bölge kanıtı: R_c = Look top-3 kutuda max v_c.
- Görüntü kanıtı: X_c = tüm önerilerde max v_c.

**Maliyete duyarlı sınıf eşiği (I1).** τ_c = argmax_τ [F_u,c(τ) − λ·F_p,c(τ)], tek bir λ tüm sınıflar için.
- F_p: X_c'nin, c'yi **en az 2 referansın** andığı görüntülerdeki dağılımı. Bu tür görüntü 5'ten azsa, c'yi anan tüm
  görüntüler kullanılır. Dağılım, türün (thing/stuff) birleşik merkezlenmiş şekliyle düzleştirilir (ağırlık m0 = 10).
- F_u: X_c'nin, c'yi hiçbir referansın anmadığı görüntülerdeki dağılımı. Bu bir **otomatik proxy**: anılmamak yok
  olmak demek değil (PU durumu).
- Aday eşikler: pozitif değerler ∪ F_u'nun %50–99.9 kantilleri (200 nokta).
- Pozitifi 2'den az olan sınıfın τ'su yok; sınıf çekimser kalır. Ayrıca elle yazılmış bir çekimserlik listesi yok.
- Sınıf öncesi kullanılmaz: λ herkese aynı; claim sıklığı ve caption olasılığı girmez.

**Kural.**
1. Bölge yok → skip.
2. Kalıp ya da birleşik kelime (I5) → skip. Kalıp: "at the camera". Birleşik kelimeler: "home plate" (plate),
   "hot dog" (dog), "toy car" (car), "dirt bike" / "motor bike" (bicycle).
3. R_c ≥ τ_c → keep.
4. Değilse X_c ≥ τ_c → keep_elsewhere.
5. Değilse, τ yoksa → skip.
6. Değilse contradicted, iki dala ayrılır:
   - **replace (I3):** Look **top-1** kutusunda S(c)'den bir j için l_j ≥ τ_j ve l_j > l_c ise. İkame,
     argmax_j (l_j − merkez_j). merkez_j = j'nin ≥ 2 referansla anıldığı görüntülerdeki ham görüntü-max'ının medyanı,
     türe doğru büzülmüş (`fit_positive_anchored`, α = 0.5).
   - Aksi halde **remove**.
- Kişi claim'leri değişmedi.

**Temizlenmiş karışma kümeleri (I5).** S3 kümelerinden şu çıkarmalar yapılır:
- `boat` çıkar; canoe, kayak ve raft'ın üst kavramı.
- `vest` üst-beden kümesinden çıkar ("life jacket" ile "life vest" aynı nesne).
- `swimsuit` ve `bikini` tüm-beden kümesinden çıkar. "wetsuit" swimsuit'e eşleniyor ve "suit" ile çakışıyor;
  bikini de bir swimsuit.
- Geri kalan kümeler aynı.

**Çalışma noktası.** λ ∈ {1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 100}. Seçilen λ*:
**dış-kat thing-FCR ≤ 0.03** olan noktalardan FCR'si en yüksek olan. İkincil rapor: ≤ 0.05.

**Çapraz uyum.** 5 görüntü katı (`calibrate.fold_of`), yalnız B. τ, merkez ve düzleştirme şekli her katta yalnız
eğitim görüntülerinden hesaplanır. Son bundle, λ* ile tüm B'de uydurulur.

## 2. Ölçüm (B, dış-kat; Fable'ın önerdiği tabakalar)
- **FCR:** thing, giysi, köpek, stuff, tümü; ayrıca ≥ 2 referansla doğrulanmış thing claim'lerinde.
- **Swap (aynı 2401):** üç tabakada yakalama (replace + remove).
  - *Temiz dışlayan:* değiştirilen etiket, gerçek sınıfın temizlenmiş kümesinde ve giysi değil.
  - *Aynı küme giysi.*
  - *Kümeler arası:* yalnız betimsel.
- **İkame kesinliği:** replace edilen swap'larda ikamenin gerçek sınıf olma oranı. Ayrıca, yakalanan aynı-küme
  swap'larda gerçek sınıfın geri kazanılma oranı.
- **Karşılaştırma satırları:** S2 (varsayılan), S3 (ön kayıtlı, α* = 0.075) ve iki ablation: S4a (yalnız I1,
  rakipsiz) ve S4 (S3'ün ikame kuralıyla).
- **Proxy J ve contradicted kesinliği:** yalnız betimsel.

## 3. Karar kuralı
λ*'da üç koşul **birlikte** sağlanırsa S4 varsayılan olur:
- (a) Temiz dışlayan swap'larda yakalama ≥ S2'nin aynı tabakadaki yakalaması.
- (b) Giysi FCR'si ≤ 0.05.
- (c) İkame kesinliği ≥ S2'nin ikame kesinliği.

S2 bu yanlış alarm düzeyine inemiyor (thing-FCR 0.117). Bu yüzden düşük FCR'de bu üçünü tutmak S2'ye üstünlük
demektir. Koşullar tutmazsa varsayılan değişmez. Sonuç ne olursa raporlanır. Validation ve test okunmaz.

## 4. Çıktılar
- `scripts/45_check_s4.py` → `results/lcr_check4_bundle.json`, `results/check_s4_B.json`,
  `docs/generated/CHECK_S4_B.md`, `data/cache/pipeline/A_B_lcr4.json`.
- Doğrulama: `scripts/39_verify_lcr.py --mode check4`. S4 varsayılan olursa `--live` da koşulur.
