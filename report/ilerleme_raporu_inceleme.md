# Progress Report 1 — inceleme ve okunabilirlik düzeltmeleri

İncelenen dosya: `CS415_LCR_Progress_Report_1.pdf` (11 sayfa). Her sayı repodaki script çıktılarıyla
(`report/generated/*.tex`, `results/*.json`, `docs/generated/*.md`) karşılaştırıldı.

## Hocanın kriterlerine göre durum

| Hocanın istediği | Raporda | Durum |
|---|---|---|
| Problem ve motivasyonu hatırlatma | Bölüm I: örnek, problem tanımı, araştırma sorusu, H1–H6, kapsam | Tamam |
| Saygın dergi/konferanslardan kapsamlı literatür | Bölüm II: 6 alt başlık, 55 kaynak, karşılaştırma tablosu (Tablo I) | Tamam, güçlü |
| a) Tasarım kararları + gerekçeleri | Bölüm III: her karar için "ne / alternatif / neden", özet tablo (Tablo II) | Tamam, güçlü |
| b) Deneyler: görseller, şekiller, sayısal tablolar | Bölüm IV: 8 şekil, 6 tablo, güven aralıkları | Tamam, güçlü |
| Özet, sonuçlar, sonraki adımlar | Bölüm V: yapılanlar, ana sonuçlar, öneriden sapmalar, iş paketleri, riskler | Tamam; takvim (hafta) eksik |
| Biçim: IEEE çift sütun, LaTeX, PDF | Evet | Tamam |

Sayı kontrolü: rapordaki sayıların hepsi repo çıktılarıyla aynı (CIDEr 0.669 → 0.979; Look 12.9 → 80.4;
yanlış kelimede 81.7 / 91.0 / 36.3 / 22.3; S4 3.0 / 4.1 / 80.8 / 55.5 / 91.2; küçük nesne 4.9 → 55.6;
güç tablosu 7,469 / 52 / 85 / 50 / 82; S4 eylem sayıları 4,024 / 287 / 64 / 167 / 52). Astra incelemesinde
düzeltilen iki sayı (küçük nesne, H1 gücü) raporda zaten düzeltilmiş halleriyle duruyor.

## Düzeltilmesi gerekenler (önem sırasıyla)

### 1. Validation kuralıyla çelişki (hoca yakalayabilir)
III-A ve IV-A "validation yalnız raporlama için" diyor, ama IV-C'de epoch validation CIDEr'e bakılarak seçilmiş.
Gizlemek yerine açıkça yazmak daha iyi. IV-C'deki "so we froze the epoch-1 model as our draft captioner B0"
cümlesinden sonra:

```latex
This checkpoint choice used validation CIDEr and is the one exception to our rule that validation is
used only for reporting. All later checkpoints (the reversed-split captioner and the Retell runs)
will be selected on a held-out part of the training images.
```

### 2. Özet (abstract) iki sayıyı bağlamsız veriyor
- 81.7%: half B'deki 300 **sentetik** kelime değişimi üzerinde.
- 11.7% → 3.0%: **thing** claim'lerinde ve **referans proxy'siyle** ölçülmüş.

Önerilen cümleler:

```latex
On 300 synthetic word swaps, Look's region hits the real object in 81.7\% of cases, compared with
22.3\% for the detector's box for the wrong word. Our current Check (S4) lowers false alarms on
correct thing claims from 11.7\% to 3.0\% (automatic reference proxy) while catching more of the
clean synthetic errors.
```

### 3. Retell sonucu biraz fazla kesin
Rapor "evidence ignored" diyor ama aynı koşuda kanıtı tamamen maskelemek altyazıların %10.9'unu değiştiriyor
(`report/generated/intervention_table.tex`). Doğru ifade: model kanıtın **varlığına** tepki veriyor, **içeriğine**
vermiyor. IV-H'deki "When all evidence is masked..." cümlesinin yerine:

```latex
Masking all evidence changes 10.9\% of the captions, but the adoption of alternatives stays the same
(41.0\%), and replacing the suggested alternative with a wrong label changes none of them. The
pathway therefore reacts to the presence of evidence, not to its content.
```

### 4. Tanımlanmamış terim: "centred substitute rule"
Tablo VI ve metinde geçiyor ama Bölüm III-E'de hiç anlatılmıyor. "Five actions" listesindeki *replace*
maddesinden sonra:

```latex
When several rivals qualify, we pick the one whose score is highest relative to its own class centre
(the median score of that class on likely-present images), so that classes the detector always
scores high do not win by default.
```

### 5. Takvim eksik
Hoca "steps to be taken from now on" istiyor; öneride 12 haftalık plan vardı. Tablo VIII'e bir "Week" sütunu
ekleyin veya altına şu tabloyu koyun (repodaki `report/sections/5_conclusion.tex` ile aynı plan):

```latex
\begin{table}[t]
\caption{Remaining plan (weeks of the proposal's 12-week schedule).}
\centering\footnotesize
\begin{tabular}{@{}p{0.9cm}p{6.8cm}@{}}
\toprule
Week & Task \\
\midrule
8  & Human pilot on half B (two annotators, $\kappa$); real draft error rate, real precision of S4; fix test size $N$. \\
9  & Retrain Retell, B0+, B2, B3 with Look v4 and Check S4 records; $\rho$ ablation; re-run B1 with S4. \\
10 & Reversed split; COCO Karpathy-test inference and CHAIR for all systems (H1b); METEOR and SPICE. \\
11 & Blinded human evaluation on the test sample; H4 evidence interventions; H6 localisation. \\
12 & Statistics (paired bootstrap), limitations, final report. \\
\bottomrule
\end{tabular}
\end{table}
```

## Okunabilirlik (presentation/clarity puanı için)

En büyük sorun: okuyucu çok fazla kısa adla karşılaşıyor (B0, B0+, B1, B2, B3, S2, S3, S4, v1–v4, E1, E3, RUOR,
HICR) ve bunlar farklı yerlerde tanımlanıyor. Bölüm III'ün başına (veya IV-A'ya) tek bir isim tablosu koymak
raporu en çok kolaylaştıracak değişiklik:

```latex
\begin{table}[t]
\caption{Names used in this report.}
\centering\footnotesize
\begin{tabular}{@{}p{1.3cm}p{6.4cm}@{}}
\toprule
Name & Meaning \\
\midrule
B0     & Draft captioner: BLIP with its decoder fine-tuned on half A, then frozen. \\
B0+    & Retell architecture with no evidence (null token only); measures the effect of extra training data. \\
B1     & Rule baseline: replace every contradicted word with Check's alternative. No training. \\
B2     & Retell with image-level detector tags instead of claim records. \\
B3     & Retell with records from the detector's best box for the word instead of Look's box. \\
LCR    & Full system: Look regions, Check records, Retell. \\
v1--v4 & Look versions; v1 is the proposal's rule, v4 is the current one. \\
S2--S4 & Check versions; S4 is the current one. \\
RUOR   & Share of claims no reference caption mentions (proxy, not an error rate). \\
HICR   & Human-verified incorrect claim rate (main measure, not yet available). \\
\bottomrule
\end{tabular}
\end{table}
```

Daha küçük öneriler:
- **Şekil 4'teki çubuk adları** ("Raw +v1", "Sink +v2", "v3 (E1)") altyazıya bakmadan anlaşılmıyor.
  "raw attention", "+ sink correction", "+ new ranking", "+ smoothing (v3)", "+ head selection, SAM (v4)" gibi
  adlar daha açık.
- **E1/E3** yalnız bir kez geçiyor; okuyucuya bir şey katmıyor. "two tuning rounds on half B" demek yeterli.
- **Bölüm III'teki "Change from the proposal" paragrafları** uzun. Her birini 2–3 cümleye indirip ayrıntıyı
  Tablo II'ye bırakmak akışı hızlandırır.
- **Tablo II, Check satırı:** alternatif sütunu S2'yi (ortak eşik) ve öneriyi (üç karar) karıştırıyor.
  "Proposal: three verdicts; S2: shared threshold" diye ayırın.
- **74.6% ve 74.7%** (IV-E, "Does the attention map really matter?") iki farklı alt kümeden geliyor; okuyucu
  aynı sayının yanlış yazıldığını sanabilir. Birine bir yarım cümle ekleyin ("on the same claims with a
  matching swap partner").

## İsteğe bağlı dürüstlük notları
- Temiz sentetik değişimlerin çoğu (834'ün 590'ı) `dog` sınıfından geliyor (Astra incelemesi, 2026-11-18).
  S4'ün 80.8%'lik temiz yakalama oranı büyük ölçüde köpek örneklerini yansıtıyor. Rapora eklenecekse sayılar
  önce bir script ile `report/generated/` altına yazılmalı (elle sayı yok kuralı).
- S4 eşik kalibrasyonu, canlı çıkarımda atlanan kalıpları pozitif sayıyor (`docs/CHECK_S4.md`, bilinen hata).
  B'deki 4,594 claim'in yalnız ikisinde kararı değiştiriyor; bir cümleyle "known limitation" olarak yazılabilir.

## Sonuç
Rapor hocanın dört maddesini de karşılıyor; sayılar doğru, proxy'ler proxy diye yazılmış, tasarım gerekçeleri
ve sapmalar açık. Ana risk içerik değil, ilerleme miktarı: insan pilotu, COCO CHAIR ve yeni Retell eğitimi henüz
yok. Rapor bunu saklamıyor ve planı veriyor. Yukarıdaki 1–5 düzeltildiğinde teslim edilebilir.
