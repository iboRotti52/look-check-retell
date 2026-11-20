# B insan pilotu (S4/S2 + Look L1): ön kayıtlı protokol (2026-11-20, Claude)

Etiketlemeden **önce** yazıldı. Sorumlu: insan değerlendirme protokolü (Damla, öneri "Team and Responsibilities").
Tek bir etiketleme turu hem Check S4/S2'yi hem Look L1'i (`docs/LOOK_L1_PROTOKOL.md` §8) ölçer.
Yalnız B; validation ve test okunmaz.

## 1. Sorular
- **P1 — S4'ün müdahaleleri ne kadar isabetli?** `replace` ve `remove` kararlarının gerçek kesinliği
  (önceki 0,912 ikame kesinliği sentetik swap testinden; doğal taslaklarda bilinmiyor).
- **P2 — İkame doğru mu?** `replace` için önerilen adın o nesneyi doğru adlandırma oranı.
- **P3 — Silme zarar veriyor mu?** `remove`, referansı olan bir kelimeyi (doğru ya da yanlış adlandırılmış) ne sıklıkla siliyor?
- **P4 — S4 neyi kaçırıyor?** S4'ün `keep` dediği claim'lerde hata oranı; S2'nin çelişki dediği ama S4'ün tuttuğu
  claim'lerde (özellikle giysi) kim haklı?
- **P5 — Doğal oranlar:** rastgele örneklemde hata oranı ve hata türleri (yanlış tanımlanmış / uydurulmuş).
- **P6 — Kutu kıyası (Astra önerisi):** referansı olan claim'lerde insan kutusu, Look'un S4 kutusu ve detektörün kutusu;
  aynı donmuş OWLv2 ile hangisi doğru adı buluyor. Bu turda yalnız **veri toplanır**; analiz veri önbelleği gerektirir
  ve ayrı bir script olarak yazılacak.

## 2. Örneklem (`lcr/pilot.pilot_sample`, sabit seed 415)
Havuz: `A_B_lcr4.json` (S4) ve `A_B_lcr2.json` (S2) içindeki bütün kişi dışı B claim'leri. Bir claim ilk aldığı gruba girer:

| grup | ne | varsayılan |
|---|---|---|
| `random` | bütün havuzdan düzgün rastgele (doğal oranları yalnız bu grup tahmin eder) | 60 |
| `s4_replace` | S4 replace'lerin hepsi | hepsi |
| `s4_remove` | S4 remove'ların hepsi | hepsi |
| `s2_contradicted_s4_keep` | S2 replace/remove, S4 keep; önce giysi | 80 |
| `s4_keep_elsewhere` | S4 keep_elsewhere | 60 |
| `s4_keep` | kalan S4 keep (kaçırılan hata taban oranı) | 80 |

Grup büyüklükleri `scripts/07_make_annotation.py --n-random/--n-s2-keep/--n-elsewhere/--n-keep` ile değişir; her grubun
havuzdaki sayısı ve örneklenen sayısı `key.json`'a yazılır. Toplam ~300–400 claim hedeflenir (S4 replace/remove sayısına bağlı).

**Önkoşul:** `A_B_lcr4.json`, PR #2'nin `scripts/48_rewrite_lcr4_records.py --overwrite` ile yeniden yazılmış hâli olmalı
(kanıt kutusu düzeltmesi); script 07 aksi halde açık bir bayrak olmadan reddeder.

## 3. Etiketleme ve körleme
- Her görüntü bir kart; etiketleyici her ismi her zamanki gibi etiketler (doğru / yanlış / emin değil / nesne değil;
  yanlışsa yanlış tanımlanmış / uydurulmuş). Hangi ismin örneklendiği ve S4/S2 kararı gösterilmez.
- Ek sorular **yalnız örneklenmiş bir isim etiketlendikten sonra** açılır, etiketi yönlendirmesin diye:
  - S4 bu ismi değiştirmek istiyorsa: "Önerilen ad: X. Bu nesne için doğru mu?" (evet / hayır / emin değil).
  - İsmin referansı varsa (doğru ya da yanlış tanımlanmış): kelimenin gösterdiği nesnenin kutusu.
- Silme için ayrı soru yok: silme, kelime doğruysa ya da yanlış tanımlanmışsa (nesne var) zarar; uydurulmuşsa doğru.
- İki etiketleyici her şeyi bağımsız etiketler; anlaşmazlıkları üçüncü kişi (hakem) çözer. Kural her alan için aynı:
  uzlaşma, yoksa hakem, yoksa belirsiz; kutular CSV sırasından bağımsız birleştirilir (`lcr/pilot.py`).

## 4. Ölçümler (`scripts/10_score_labels.py` → `results/labels_<ad>.json` "pilot")
Her grup için: hata oranı = yanlış / (doğru + yanlış), Wilson %95 aralığı; hata türleri; belirsiz sayısı.
- `s4_replace`, `s4_remove`: hata oranı = S4'ün kesinliği (P1). `s4_replace`: `substitute_right` (P2).
  `s4_remove`: `removal_harm` (P3).
- `s4_keep`, `s4_keep_elsewhere`, `s2_contradicted_s4_keep`: hata oranı = S4'ün kaçırdığı (P4).
- `random`: doğal hata oranı ve hata türleri (P5). Hedefli grupların oranları doğal oran değildir.
- Kutular: `results/pilot_referent_boxes_<ad>.json` (insan kutusu, kaynağı, iki etiketleyicinin IoU'su, Look kutusu) (P6).
- Etiketleyici uyumu: Cohen's κ (etiket, hata türü), kutu IoU ortalaması.

## 5. Ne yapılmaz
- Pilot sonucuna bakarak S4'ün eşikleri yeniden seçilmez; bu ayrı ve ön kayıtlı bir karar olur.
- Hedefli grupların oranları birleştirilip "genel hata oranı" diye raporlanmaz; ağırlıklı tahmin gerekirse grup
  büyüklükleri `key.json`'da.

## 6. Koşu
```
python scripts/07_make_annotation.py --pilot lcr --name pilot_b
# iki etiketleyici annotation/pilot_b/annotate.html'i doldurur, CSV indirir; hakem yalnız anlaşmazlıkları
python scripts/10_score_labels.py --name pilot_b labels_pilot_b_A.csv labels_pilot_b_B.csv --adjudicator labels_pilot_b_H.csv
```

## Ek (2026-11-20, etiketleme başlamadan önce): P6 için doğru ad ve analiz
- "Yanlış tanımlanmış" seçildiğinde isteğe bağlı bir **doğru adı** alanı açılır (serbest metin). Script 10 bunu claim
  çıkarmayla aynı sözlükle bir sınıfa eşler (`grounding.phrase_class`) ve diğer alanlar gibi birleştirir (uzlaşma, hakem,
  belirsiz). Sözlükte olmayan adlar sayılmaz. Bu, P6'nın "doğru adı buluyor mu" sorusu için gerekli; olmadan yalnız
  "yanlış adı reddediyor mu" ölçülebilirdi.
- Analiz: `scripts/52_pilot_box_comparison.py --name <ad>`. Aynı önbellek öneri havuzundan üç öneri: yanlış kelimenin en iyi
  detektör kutusu (B3), S4 kaydının gösterdiği Look kutusu, insan kutusuna en iyi uyan öneri. Her birinde S4'ün kurallarıyla:
  doğru claim'i destekliyor mu (v ≥ τ), yanlış tanımlanmışta yanlış adı reddediyor mu, doğru adı buluyor mu (S4'ün replace
  koşulu), doğru ad S4'ün ikame kümesinde mi. Öneri kapsaması (insan kutusuna IoU ≥ 0,5 bir öneri var mı) ayrıca raporlanır;
  hiçbir kaynak onu geçemez. Farklar aynı claim'lerde, görüntü düzeyi bootstrap. Çıktı `results/pilot_box_comparison_<ad>.json`,
  `docs/generated/PILOT_BOX_COMPARISON_<ad>.md`.
- Karar okuması (Astra): insan kutusu doğru adı bulup Look bulamıyorsa Look'u iyileştir; insan kutusu da bulamıyorsa önce
  detektörün önerileri ve sınıf ayrımı.
