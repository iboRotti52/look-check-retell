# Progress Report – Yol Haritası

Hocanın istediği 4 bölüme göre ne yapmamız gerektiğinin kabaca planı. Rapor IEEE çift sütun LaTeX, PDF, sayfa sınırı yok.

## 0. Önce karar verilecekler
- Teslim tarihine kaç gün kaldığına göre hangi deneylerin yetişeceğini seçin (aşağıda "Minimum" ve "İyi olur" diye ayrıldı).
- GPU nerede? Mac (16 GB RAM) BLIP fine-tune için yetmiyor, swap'a düşüyor. **Colab / lab GPU** kullanın.
- Overleaf'te ortak proje açın (IEEE conference template).

## 1. Problem ve motivasyon (kolay, ~1 kişi, yarım gün)
- Proposal Bölüm I'den kısaltarak yazın: hallucination örneği, problem tanımı, araştırma sorusu, kapsam.
- Proposal'dan sonra değişen bir şey varsa burada söyleyin.

## 2. Related work ("comprehensive" diye notlanıyor, ~2 kişi, 1–2 gün)
- Proposal'daki 23 kaynağın her birini 1–3 cümleyle anlatın, alt başlıklara bölün:
  ölçüm (CHAIR, ALOHa, POPE) · detektörle düzeltme · öğrenilmiş düzenleme/feedback · frozen modeli koşullama · attention & grounding · araştırma boşluğu.
- Eklenebilecek bilinen çalışmalar: POPE, OPERA, VCD, BLIP-2, OWL-ViT, CIDEr, SPICE, Bottom-up/Top-down attention.
- Proposal'daki karşılaştırma tablosunu (Tablo 1) koruyun.

## 3a. Tasarım kararları + gerekçeler (notun en önemli kısmı, ~1–2 kişi, 1 gün)
Her karar için: **ne seçtik / alternatif neydi / neden bunu seçtik**. Bir özet tablo iyi durur.
- BLIP-base (tek GPU, açık cross-attention) · OWLv2 (open-vocab, önbelleğe alınabilir)
- Vocabulary V + person eşlemesi (cinsiyet/yaş kontrol edilmiyor, gerekçesiyle)
- Excess attention mass (ham kütle büyük kutuyu, alan-normalize küçük kutuyu kayırır)
- Asimetrik verdict kuralı (bir bölgede bulamamak = yok demek değil)
- Zero-init tanh gate + null token (eğitimden önce model = baseline)
- A/B held-out split · sentetik corruption · B0+ kontrolü · insan değerlendirmesi birincil metrik
- Uygularken değiştirdiğiniz her şeyi (ör. "sadece decoder'ı fine-tune ettik") burada açıkça yazın.

## 3b. Deneyler ve sonuçlar (hoca görsel + tablo + sayı istiyor)
Sadece **gerçekten çalıştırdığınız** sayıları koyun; yapılmayanlar "planned" olarak yazılsın.

**Minimum (mutlaka olsun):**
| # | Deney | Rapora giren | Sorumlu (proposal'a göre) |
|---|---|---|---|
| 1 | Veri/vocabulary istatistikleri | Tablo + sınıf frekans histogramı | Metin |
| 2 | Zero-shot BLIP vs A'da fine-tune BLIP (val) | BLEU-4/METEOR/CIDEr/SPICE tablosu | İbrahim G. |
| 3 | Attention + OWLv2 kutu overlay'leri (6–8 örnek) | Nitel figür | Metin |
| 4 | Pilot: 100 draft'ı elle etiketleme | Hata oranı, hata türleri, Cohen's κ | Damla |
| 5 | Zero-init kontrolü (eğitimsiz Retell = B0) | Max fark ≈ 0 cümlesi/tablosu | İbrahim B. |

**İyi olur:**
| # | Deney | Rapora giren | Sorumlu |
|---|---|---|---|
| 6 | OWLv2 verdict vs insan etiketi | Confusion matrix + threshold seçimi | Metin / Damla |
| 7 | k ∈ {1,4,12} localization, attention vs B3 | Tablo/çizgi grafik | Metin |
| 8 | B1 kelime değiştirme baseline'ı (val) | Fix/break oranı | Metin |
| 9 | COCO'da B0 için CHAIR | CHAIRi / CHAIRs | Zeynep |
| 10 | Retell ilk eğitim denemesi (B0+ ve Ours, val) | Loss eğrisi + ilk metrikler | İbrahim G. / B. |

## 4. Sonuç ve sonraki adımlar (~yarım gün)
- Ne yapıldı, hangi sayılar elde edildi (1 paragraf).
- Proposal takvimine göre neredeyiz, sapma var mı.
- Kalan haftalar için güncel tablo: Retell/B0+/B2/B3 eğitimi, ρ ablation, ters split, COCO CHAIR, kör insan değerlendirmesi, H4 testi, istatistik.
- Riskler ve durumları.

## Önerilen sıra
1. Gün 1: Overleaf + iskelet, 1 ve 2. bölüm taslağı başlasın; GPU'da deney 2 başlasın.
2. Gün 2–3: Deney 1, 3, 5; Damla pilot etiketlemeye başlasın (deney 4).
3. Gün 3–4: 3a bölümü; vakit varsa 6–10.
4. Son gün: sonuçları tabloya koy, 4. bölüm, okuma/düzeltme, PDF.

## Elde hazır olanlar (Fable'ın denemesinden kalan)
Klasörde Fable'ın yarıda bıraktığı şeyler var, isterseniz kullanırsınız, istemezseniz silinebilir:
- `data/`: Flickr8k indirilmiş + BLIP/OWLv2 önbellekleri (~9.4 GB)
- `lcr/`, `scripts/`, `tests/`: pipeline kodu (28 test geçiyor)
- `report/sections/`: giriş, related work, tasarım kararları taslakları (İngilizce)
- Gerçek ölçülen sayılar: 169 sınıf / 316 kelime, caption başına 2.10 claim; zero-shot BLIP val'de BLEU-4 25.6, CIDEr 0.669
