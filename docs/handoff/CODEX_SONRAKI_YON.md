# Codex için yön: Look'u kapat, Check'e geç (2026-11-14)

> **Not (kullanıcı isteği):** Buradakiler öneri, kısıt değil. Codex kendi incelemesine göre farklı bir yol seçebilir.
> Bağlayıcı olan sadece AGENTS.md'deki genel kurallar. Check'i bu arada Claude kurdu: `docs/CHECK_V2.md`.

Claude, Codex'in incelemesini (`docs/LOOK_V2_INCELEME.md`) ve düzeltmelerini kontrol etti: 43 test geçiyor,
v2 kutuları ve skorları değişmemiş (`results/look_consistency_fix.json`), validation benchmark aynı (%75.3).
İnceleme doğru ve faydalı. Özellikle şu üç bulgu: attention'ın objectness'a ek katkısı, kişi hatalarının
kutu ölçeğinden gelmesi, Kang CVPR'25 başlık seçiminin BLIP'e doğrudan taşınamaması.

Öneri: Look için iki deney paketi kaldı. İkisi bitince, sonuç ne
olursa olsun Look kapanır ve Check başlar. Kullanıcının isteği de bu: Look'ta oyalanmadan zinciri ilerletmek.

---

## E1. Türe özel kutu kuralı + yumuşatma (CPU, kayıtlı haritalar, ~yarım gün)

Önceden belirlenmiş ızgara önerisi. B'ye aşırı uymayı önlemek için ızgarayı koşmadan önce sabitlemek iyi bir pratik:
- Tür başına (person / thing / stuff) ayrı α ∈ {0.25, 0.5, 0.75, 1.0}, γ ∈ {0.25, 0.5, 1.0}.
- Kutu skorundan önce Gauss yumuşatma σ ∈ {0, 0.5, 1.0} yama. Tüm türler için ortak, kırpılmış sink haritası üzerinde.
- Toplam: 3 σ × (tür başına 12) = tür başına 36 konfigürasyon. Türler bağımsız seçilir.

Seçim B'de (`A_B_look2` haritaları + `data/cache/owl/train.npz`):
- Bir türün kuralı ancak o türde B doğruluğunu **≥ 1.0 puan** artırırsa değişir. Yoksa v2 kuralı (α=0.5, γ=0.5,
  σ=0) kalır.
- Rapor: validation'da v2'ye karşı, görüntü düzeyinde eşleştirilmiş bootstrap ile %95 GA (Codex'in önerisi;
  aynı görüntünün claim'leri bağımsız sayılmaz). Tür bazında thing/stuff kaybı da raporlanır.
- Kapı: yumuşatma seçilirse `14_look_sanity.py` yeni haritayla tekrar koşulur.

Uygulama önerisi: `scripts/16_look_regions.py`'ye `--per-kind` ve `--sigma` eklemek, ya da yeni bir `18_look_e1.py`.
Çıktı `results/look_e1.json`. Seçim, sonuç ve bootstrap aynı dosyada.

## E2. Gradyan haritaları + nedensellik kontrolü (decoder geçişi, B alt kümesi, ~1 gün)

Sadece **B'den sabit tohumla 500 görüntü** (~1.100 claim). Yerel CPU yeterli olmalı, `data/cache/ckpt/decoder_A.pt` var.
Uzun sürerse Colab.

Tek geçişte, claim başına 24×24 haritalar halinde sakla:
1. S1: ham, p−1 (mevcut referans; v2 ile aynı çıkmalı, bu bir doğrulama).
2. S3-prefix: sink düzeltmesinde bütün caption ortalaması yerine **sadece önek ortalaması** (konum < p).
   Bu, Codex'in "geriye dönük harita" itirazını test eder. Üretim anında da hesaplanabilir olduğu için daha temiz.
3. S2: Grad-CAM, ReLU(∂log p(w)/∂A ⊙ A), son 4 katman ortalaması.
4. S4: kontrastif, ReLU(∂[log p(w) − log p(w_alt)]/∂A ⊙ A), w_alt = `blip_alts[0]`.
5. 5a: okuma adımı (p) ve alt-kelime ortalaması varyantları (S3 ile).

Her harita için alt kümede pointing, enerji ve v2 kutu kuralıyla doğru kutu. Kabul kuralı: bir kaynak v2
haritasını (S3) aynı alt kümede doğru kutuda **≥ 2 puan** geçmezse bırakılır. Negatif sonuç da raporlanır.

**Nedensellik kontrolü** (aynı geçişte, ucuz, rapor için değerli): sabit caption öneki altında hedef kelimenin
log p(w) değeri ölçülür. Üç koşul var: (a) Look kutusuna düşen görüntü token'ları maskelenir (attention mask ile),
(b) aynı sayıda rastgele token maskelenir, (c) detektör kutusu maskelenir. Look kutusunu maskelemek rastgeleden
belirgin şekilde fazla düşürmeli. Bu bir **özellik düzeyinde** müdahale: ViT token'ları başka bölgelerden de
bilgi taşıdığı için piksel düzeyinde değil. Raporda tam olarak böyle adlandır.

T4 (başlık seçimi): öncelikli değil. E2'de gradyan haritaları katkı gösterirse "future work" olarak not edilir.
Gerekçe: Codex'in kendi bulgusu, Kang CVPR'25 ölçütünün BLIP cross-attention'a birebir uymaması. Uyarlamak
ayrı bir araştırma işi.

## Look'u kapatma

1. E1 ve E2 sonuçlarına göre ya v2 kalır ya da v3 (yalnızca kabul kuralını geçen bileşenlerle) dondurulur.
   `config` + `lcr/look.py` + kayıtlar `*_look3` (script 17 yolu).
2. `docs/LOOK_V2_PLAN.md` §7'ye tek bir ablation tablosu: R1 → sink → kutu kuralı → (E1) → (E2) satırları;
   B seçimi, validation raporu, bootstrap GA.
3. Rapor tablosu script'ten (`scripts/look_report.py`). Günlük güncellenir.

---

## Sonra: Check paketi (sıra ve kabul kuralları)

Kaynak: `docs/LITERATUR_TARAMASI.md` §6, 7, 8, 8a. Hepsi CPU, mevcut kayıtlarla (`*_look2` ya da `*_look3`).

- **C0. Kalibrasyon alt kümesi.** B'yi sabit tohumla ikiye böl: `B_fit` (%80) ve `B_cal` (%20). Eşik ve
  normalizasyon `B_cal`'da seçilir. Retell ileride `B_fit`'te eğitilir, epoch seçimi `B_cal`'da yapılır
  (validation'da değil; 12a'daki kural ihlali böyle kapanır).
- **C1. Sınıf başına skor normalizasyonu.** Ham olasılık / sınıf başına yüzdelik / sınıf başına logistic
  (Platt, az örnekli sınıflar için ortak parametreye büzülme). Ölçüt: B_cal'da Youden J ve risk–kapsam eğrisi
  (AURC). Validation'da raporla.
- **C2. Alternatif seçimi.** Mevcut (detektörün ilk 6 sınıfı ∪ captioner alternatifleri, en yüksek skor) yerine:
  (i) aynı türle sınırlı, (ii) normalize skorla seçim, (iii) sadece captioner alternatifleri. Ölçüt:
  "contradicted" kesinliği ve 5b ölçüsü (önerilen alternatif referansta mı).
- **C3. Karar kuralı.** Eşikli üç yollu kural ve risk–kapsam eğrisi. supported kesinliği, contradicted kesinliği
  ve uncertain oranı ayrı ayrı. Hedef: contradicted kesinliği taban oranın (0.24) belirgin üstünde. Şu an 0.37.
- **C4. İkinci proxy.** `in_f30k_boxes` alanı ve proxy anlaşmazlık raporu (LITERATUR_TARAMASI §4).
- **C5.** `s_image` alanı kayıtta kalsın ama Retell'e girip girmeyeceği bir ablation bayrağı olsun (H6 karıştırıcısı).

Check kapanış kapısı: validation'da C1 + C2 + C3 ile contradicted kesinliği ve J, v2'nin mevcut değerlerinden
(J 0.461 B'de, contradicted kesinliği 0.37) iyi. Ya da iyileşmenin olmadığı bir ablation tablosuyla gösterilmiş
olmalı. Sonra Retell (12a: gate weight decay, oracle kanıt deneyi, constrained-beam baseline).
