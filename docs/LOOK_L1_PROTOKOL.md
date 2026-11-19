# Look L1 — referans kanıt sinyalleri: ön kayıtlı protokol (2026-11-18, Claude)

Koşudan **önce** yazıldı (repo kuralı: `AGENTS.md`). Kapsam: yalnız Look. Check, Retell, validation ve test
split'i bu çalışmanın hiçbir adımında değişmez veya okunmaz.

## 0. Neden

Öneri metninin III-B bölümü Look için bir çekimserlik kuralı tanımlıyordu: "If its excess mass is below a
threshold set on validation, the claim is marked as having no suitable region." Dondurulmuş **Look v4**
(`docs/LCR.md`) bu kuralı kaldırdı: küçük kutu filtresi B'de kaldırıldığından beri Look her claim için en
yüksek fazla-kütle adayını seçiyor, eşiksiz. Bu, önerinin taahhüt ettiği bir davranıştan sapma; L1 bu farkı
ölçüp, verirse geri getirmeyi hedefliyor — **kör bir eşik koyarak değil, önce sinyalleri ölçerek.**

İkinci amaç: mevcut Look ölçümü (pointing game, bölge IoU — `docs/generated/LCR_DEVELOPMENT.md`,
`scripts/13_look_benchmark.py`) claim'in kendisi yanlış olduğu için mi yoksa Look doğru claim için yanlış
bölge seçtiği için mi hata olduğunu ayırt etmiyor. L1 bu ikisini insan etiketiyle ayırır.

## 1. Sorular

- **L1a (sinyal ayrımı, B, otomatik proxy):** Üç referans sinyali (güç, yoğunluk, boş-hipotez karşılaştırması),
  referansta desteklenen/desteklenmeyen thing/stuff claim'lerini ne kadar ayırt ediyor?
- **L1b (zorlanmış halüsinasyon, B):** Claim kasten yanlış bir kelimeyle değiştirildiğinde, v4'ün aynı kod
  yoluyla yeniden hesapladığı attention hâlâ claim'in **gerçek** referansını mı işaret ediyor, yoksa B3
  (detektörün kendi kutusu) kadar mı kötü?
- **L1c (H6 düzeltmesi, B):** Look top-1'in B3'e karşı bölge doğruluğu, eşleştirilmiş görüntü-düzeyi bootstrap
  ile, aynı claim kümesinde ne kadar farklı?
- **L1d (çekimserlik, yalnız 4/5 sonuçlanırsa):** Sinyaller birleştirildiğinde, doğru claim'leri gereksiz yere
  çekimser bırakmadan hangi claim'lerde "bölge yok" demek haklı çıkar?

## 2. Sinyaller (`lcr/look_referent.py`, saf numpy)

Üç sinyal, her claim için, seçilen bölgeden **bağımsız** olarak hesaplanır (hangi kutunun seçildiğini
değiştirmezler):

1. **strength:** sink-düzeltilmiş haritanın toplamda-1'e normalize edilmeden önceki pozitif kütlesi
   (`look_v4.residual_strength`, adım 3 — `look_v4.selected_map`'in kendisi değişmez).
2. **concentration:** haritanın Shannon entropisi (bit) + top-1 adayın kutusundaki kütle payı.
3. **null karşılaştırması:** top-1 kutunun bu claim'in haritasındaki fazla-kütlesi, aynı başlıktaki **diğer**
   claim'lerin haritalarındaki fazla-kütlesine karşı — marj ve sıra olarak.

`lcr/look_referent.py` kasıtlı olarak `lcr.look`'u (ve dolayısıyla torch'u) içe aktarmaz; script 46'nın CPU'da,
yalnızca önbellekteki E3 işaretli başlıklar üzerinde çalışabilmesi için bağımsız tutulur. Kutu-kütlesi
geometrisi `lcr.look.attention_mass`/`excess_mass` ile aynı tanımı tekrar eder; `tests/test_look_referent.py`
bunu sentetik haritalarla, `lcr.look`'a karşı da doğrular.

## 3. Script 46 — B'de ucuz analiz (yalnız CPU)

`scripts/46_look_referent.py`: `data/cache/look_e3/A_B_look2_heads.f16.npy` + dondurulmuş `A_B_look4.json`
üzerinden üç sinyali hesaplar. **validation veya test split'i verilirse script hatayla durur** (assert).

- AUROC: referansta destekli vs desteksiz thing/stuff claim'lerini ayırma — **otomatik proxy** olarak
  etiketlenir (referansta geçmemek yanlışlık kanıtı değildir, bkz. `docs/LCR.md`).
- Aynı AUROC, Flickr30k Entities kutu alanına göre nesne boyutuyla tabakalanmış olarak da raporlanır: bu,
  sinyalin "küçük nesne" ile "yanlış nesne"yi karıştırıp karıştırmadığını gösterir.
- Çıktı: `results/look_referent_B.json`, `docs/generated/LOOK_REFERENT_B.md`.

## 4. Script 47 — zorlanmış halüsinasyon testi (decoder_A gerekir; CPU birkaç yüz claim için yeterli, yoksa Colab)

`scripts/47_look_forced_hallucination.py`, B'de, `lcr/corrupt.py`'deki karışma tablosunu kullanarak iki tür
değiştirilmiş başlık kurar:

- **Yanlış tanımlanmış (misidentified):** Flickr30k Entities kutusu olan bir nesnenin kelimesi değiştirilir;
  gerçek referans o kutudur.
- **Uydurulmuş (invented):** hiçbir referansın anmadığı ve OWLv2'nin görüntünün hiçbir yerinde bulmadığı
  (`s_image` eşik altı — `lcr/check.py`) bir sınıf eklenir. Bu **yalnızca proxy yokluk**; çıktıda böyle
  etiketlenir.

Her başlık `decoder_A` ile teacher-force edilir; değiştirilen kelimenin ilk alt-parçasında attention v4 ile
**birebir aynı kod yolundan** çıkarılır (aynı 4 başlık, aynı float16 nicemleme, aynı `region_evidence.candidates`
sıralaması) — bu yol zaten `scripts/review_look_e3.py`'nin `live()` fonksiyonunda donmuş drafts için kanıtlanmış
durumda; burada yalnız girdi başlığı değişir.

Ölçümler:
- Yanlış tanımlanmış: top-1/top-3 isabet (IoU ≥ 0.5) gerçek referans kutusunda, B3'e (yanlış kelimenin
  detektör kutusu) karşı aynı claim'lerde eşleştirilmiş görüntü-düzeyi bootstrap (`lcr.metrics.bootstrap_ci`).
- Uydurulmuş: referans sinyallerinin gerçek doğru claim'lere karşı AUROC'u.
- Takas testinin ne kadar iyimser olduğu: gerçek kelimenin bölgesini yeniden kullanmak vs. yeniden hesaplamak,
  ayrıca raporlanır.
- Sabit seed, sabit örneklem.

## 5. Split kuralları

- Geliştirme, seçim ve bu protokoldeki tüm ölçümler **yalnız B** (Flickr8k train'in ikinci yarısı).
- **Validation bir kez, yalnız ekip onayıyla**, yöntem (varsa referans sinyali eşiği) B'de dondurulduktan sonra.
- **Test split'i bu çalışmanın hiçbir aşamasında okunmaz.**
- Script 46 ve 47, split argümanı `train` dışındaysa çalışmayı reddeder (assert).

## 6. Başarı — ne sayılır

- L1a/L1b: AUROC'un şans düzeyinden (0.5) anlamlı şekilde yüksek olması, **tek başına eşik getirmek için
  yeterli değildir**; adım 6'nın (çekimserlik) koşulu.
- L1d (adım 6, yalnız 4 ve 5 sonuçlandıktan sonra): birleşik `referent_score` + eşik, B'de 5 katlı görüntü
  çapraz doğrulamasıyla seçilir; doğru claim'leri yanlışlıkla çekimser bırakmanın açık bir maliyeti olmalı.
  Karar kuralı ve sayısal eşik, script 46/47'nin **gerçek B sonuçları elde edildikten sonra** bu dosyaya ek
  bir bölüm olarak yazılır — şu an hiçbir eşik veya sayı burada sabitlenmez.
- Hiçbir adım, seçilen bölgeyi veya Check/Retell'in okuduğu alanları değiştirmez; `referent_score`/`no_referent`
  yalnız kayda **eklenecek** yeni alanlardır ve varsayılan olarak kapalı bir paket (bundle) bayrağının arkasında
  olacaktır (adım 6, henüz uygulanmadı).

## 7. Kapsam dışı / ertelenen

- **Adım 7 (Check için misidentified/invented ayrımı):** adım 6'nın gerçek sonuçları olmadan anlamsız;
  ertelendi.
- Script 46 ve 47'nin gerçek koşumu, `data/cache` (`bash scripts/fetch_data.sh`) ve proje `.venv`'i
  (python3.12 + `requirements.txt`) gerektirir; bu protokolün yazıldığı oturumda mevcut değildi. Kod ve testler
  hazır; sayılar bu dosyaya veya `docs/generated/LOOK_REFERENT_B.md`'ye yalnız gerçek bir koşudan sonra girer.

## 8. İnsan kontrolü (ayrı, adım 9)

`scripts/07_make_annotation.py` / `annotate.html`'e "yanlış" etiketlenen her claim için bir hata türü alanı
(doğru / yanlış tanımlanmış / uydurulmuş) ve yanlış tanımlanmış claim'ler için gerçek referansı işaretleyen bir
kutu çizim aracı eklenir. `scripts/10_score_labels.py` bu alanları okur. Bu, pilotu L1'in altın-standart test
kümesine çevirir; ayrı bir ikinci pilot koşusuna gerek kalmaması için, bu alanlar ekip onayıyla **sıradaki
insan pilotu başlamadan önce** araca eklenmelidir (bkz. `docs/CALISMA_GUNLUGU.md` "Sıradaki adımlar" madde 0).

## Revizyonlar (2026-11-19, PR #1 incelemesinden sonra; Claude)

İbrahim ve Astra (GPT-6.2) PR #1'i B önbelleğinde inceledi; İbrahim script 46'yı, Astra script 47'yi gerçek
B örnekleriyle CPU'da koştu. Aşağıdaki değişiklikler o incelemenin bulduğu **tasarım/uygulama hatalarına**
dayanıyor, hiçbir sinyalin AUROC değerine değil. Script 46'nın ilk koşusunun genel AUROC tablosu görüldü;
boyut tablosu kullanılamaz durumdaydı (aşağıda R2). Bu yüzden 46'nın ilk koşusu bağımsız bir sonuç değil,
yeniden koşulacak bir ön denemedir. Script 47'nin gerçek bir ölçüm koşusu yoktu.

**R1 — iki attention okuması (script 47).** v4 kelimenin ilk parçasını *üreten* adımı (p−1) okur. Decoder
nedensel olduğu için kelimeyi değiştirmek bu adımın ham attention'ını değiştirmez; yalnız başlık ortalaması
(sink düzeltmesi) değişir. Astra bunu dog→cat ile gösterdi; harness'te de doğrulandı (aynı uzunlukta takasta
fark tam 0, farklı uzunlukta ~1e-9; okuyan adımda ~4e-3). Bu yüzden 47 artık her claim için iki harita raporlar:
- *predict* (p−1, v4/LCR'nin kullandığı): "model kelimeyi söylemeden önce gerçek nesneye bakıyor muydu?"
- *read* (kelimenin son parçası, kelime okunduktan sonra): "yanlış kelime bağlama girince attention nereye gidiyor?"
İki harita da `look_v4.signed_heads` ile çıkar; predict haritasının `look_v4.attention_map` ile bit düzeyinde
aynı olduğu script içinde assert edilir (`tests/test_look_v4.py`).

**R2 — boyut kontrolü (script 46).** F30kE kutuları referans öbeklerine bağlı; referansta desteklenmeyen bir
claim'in neredeyse hiç kutusu yok. İlk B koşusunda F30kE-alanı bölünmesi küçük grupta 1, büyük grupta 0 negatif
bıraktı. Yeni boyut değişkeni Look'un kendi top-1 kutusunun alanı (her claim'de var); üçte birlik hücreler,
hücre başına en az 25 pozitif ve 25 negatif yoksa AUROC verilmez. `top1_area` ayrıca kendi başına bir sinyal
olarak skorlanır: bunu geçmeyen bir sinyal yalnız boyutu izliyor olabilir. F30kE kapsamı tabloda gösterilir.

**R3 — tek sıralama (46 ve 47).** 47 top-1'i v4'ün `rec_att` kutusundan (2-patch filtreli `look.select_region`),
top-3'ü filtresiz `region_evidence.candidates`'ten alıyordu; Astra B'de 4.594 kişi dışı claim'de top-1'in 292
kez değiştiğini, 46 örnekte top-3'e hiç girmediğini buldu. Artık ikisi de `candidates(..., S4 bundle'ın "look"
kuralı, n=3)`'ten gelir; top-1 her zaman top-3'ün ilkidir. B3 aynı geometrik öneri havuzunda (küçük kutu filtresi
yok) S4'ün kendi sorgularıyla yanlış kelimenin en yüksek detektör kutusudur.

**R4 — uydurulmuş claim'ler için yokluk eşiği (script 47).** Eski kod en düşük skorlu adayı alıyordu ama bir
eşik altında olmasını şart koşmuyordu. Artık bir sınıf yalnız S4'ün ham görüntü-maksimum logit'i, S4'ün sınıf
eşiği τ_c'nin altındaysa (`--absence-margin`, varsayılan 0) uydurulmuş olarak kullanılabilir; ham maksimum S4'ün
rakip-farkında X_c'sinden asla küçük olmadığı için S4 de bu sınıfı "yok" sayar. τ'su olmayan sınıflar kullanılmaz;
uygun sınıflar arasından rastgele seçilir (en düşük skorlu değil). Hâlâ **proxy yokluk**.

**R5 — uydurulmuş vs doğru AUROC (script 47, §4'te vaat edilen).** Artık hesaplanıyor: pozitif = aynı
görüntülerin referans destekli gerçek claim'leri (otomatik proxy), negatif = uydurulmuş claim; dört sinyal,
iki okuma, görüntü düzeyi bootstrap GA. Yanlış tanımlanmış vs doğru da betimsel olarak raporlanır. Takas
testinin iyimserliği = orijinal kelimenin (predict) top-1 isabeti − takastan sonra yeniden hesaplanan isabet.

**R6 — insan etiketi birleştirme (script 10, `lcr/pilot.py`).** Hata türünde anlaşmazlıkta hakem kullanılmıyordu;
iki farklı kutuda ilk CSV'nin kutusu alınıyordu (Astra sırayı değiştirerek kutuyu taşıdı). Kural artık ana
etiketle aynı: uzlaşma, yoksa hakem, yoksa belirsiz. Kutular: ≥2 kutu ve her çift IoU ≥ 0.5 ise ortalama; yoksa
hakemin kutusu; tek kutu varsa o; yoksa altın kümeye girmez ve "hakem gerekli" listesine yazılır. Sonuç CSV
sırasından bağımsızdır (`tests/test_pilot.py`).

Bu revizyonlar §6'nın başarı ölçütlerini değiştirmez. Adım 6 (eşik) hâlâ yeni koşunun sonuçlarını bekliyor.
