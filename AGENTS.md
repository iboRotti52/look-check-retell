# AGENTS.md — bu projede çalışan her ajan (Codex, Claude, …) için

Bu dosya her oturumun başında okunur. Kısa tutulur; ayrıntı aşağıdaki dosyalardadır.

## Önce bunları oku
0. **Yeni katılan için:** `docs/handoff/YOLCULUK.md` (projenin hikâyesi, modeller, kararlar, sıradaki adımlar) ve
   `docs/handoff/sohbetler/` (Claude oturum dökümleri). Repo: github.com/iboRotti52/look-check-retell (gizli).
1. `docs/CALISMA_GUNLUGU.md` — **ne oldu, şu an ne yapıyoruz, sıradaki adım ne.** Her oturumun sonunda güncellenir.
2. **Güncel tek yöntem: `docs/LCR.md`. Kapsam yalnız Look → Check; Retell kullanıcı kararıyla bekliyor.**
   B protokolü `docs/LCR_BIRLESTIRME_PROTOKOL.md`; sonuç `docs/generated/LCR_DEVELOPMENT.md`.
   GPU sınırına devir: `docs/handoff/CLAUDE_LOOK_CHECK_GPU.md`. Ağır modeli kullanıcı Claude'a Colab'da çalıştıracak.
   Look haritası eski v4'ten; küçük kutu filtresi B'de kaldırıldı. Check aynı attention adaylarını aynı donmuş
   OWLv2 ile ayrı skorlar. Eski v4 ve öğrenilmiş Check v2/v3 tarihsel/kıyas satırlarıdır; ana yol değiller.
2a. **Güncel Check: `docs/CHECK_S4.md` (S4: rakip-farkında kanıt + maliyete duyarlı eşik; varsayılan
    `results/lcr_check4_bundle.json`).** Kıyas satırları: beş-yollu z Check `docs/CHECK_TAMAMLAMA.md`, pozitif-çapalı
    `docs/CHECK_DUZELTME.md`. Fable incelemeleri: `docs/FABLE_INCELEME.md`, `docs/FABLE_INCELEME_2.md`.
2b. **İlk ilkeler değerlendirmesi ve Codex LCR incelemesi (Claude): `docs/ILK_ILKELER_TASARIM.md`.**
3. `docs/LITERATUR_TARAMASI.md` — pipeline'daki her işlem için literatür, çıkarımlar ve **kapılar**.
   Bir işlemi değiştirmeden önce ilgili bölümünü oku.
4. `README.md` — klasör yapısı, scriptler, Colab'da çalıştırma.
5. Proje önerisi: "Look–Check–Retell" (CS415, Sabancı). Rapor kaynağı `report/`.

## Değişmez kurallar
- **Sonuç uydurma yok.** Rapordaki her sayı bir script tarafından üretilir (`report/generated/*.tex`, `results/*.json`). Elle sayı yazılmaz.
- **Test split'i geliştirmede asla okunmaz.** Seçim/ayar: eğitim yarısı **B** (Flickr8k train'in ikinci yarısı) üzerinde. Raporlama: **validation**. Test yalnızca final değerlendirmede, bir kez.
- **Proxy'leri proxy diye adlandır.** "Referansta geçiyor" ≠ doğru; "detektör kutusu" ≠ insan kutusu. İnsan etiketi olmayan her hata oranı "automatic proxy" diye yazılır.
- **Veri seti Flickr8k.** Flickr30k Entities yalnızca Flickr8k görüntülerimizin insan çizimi kutularını almak için, sadece Look'u **ölçmek** amacıyla kullanılır (eğitimde kullanılmaz).
- **Sıra: Look → Check → Retell → hipotez testleri.** Bir önceki halka ölçülüp oturmadan sonrakine geçilmez (ekip kararı, 2026-11-13).
- **Güncel kullanıcı sınırı:** Look ve Check'i ağır model hesaplama noktasına getir; GPU işini başlatma,
  Retell'e geçme. Hazır sabit B pilotu ve Colab defteri devir notunda. Ana LCR ile script 06 eğitim başlatmaz.
- **Her ara aşama bağımsız bir kaynağa karşı ölçülür ("kapı").** "Script hatasız koştu" ya da "uçtan uca metrik
  makul" doğrulama sayılmaz. Kapılar: `docs/LITERATUR_TARAMASI.md`.
- Her yöntem seçimi bilimsel bir gerekçeye ve mümkünse literatüre dayanır; literatürdeki yöntemler birleştirilebilir, kendi eklemelerimiz olabilir — ama her bileşen ablation ile ölçülür.
- Atıflar gerçek olmalı; `report/refs.bib`'e eklemeden önce makalenin varlığını doğrula.
- Kullanıcının kişisel bilgileri (e-posta vb.) dış servislere gönderilmez.

## Ortam
- Yerel: Mac M4 16 GB. `.venv` (Python 3.12, torch, transformers 5.23.0). CPU'da çalıştırmak için `LCR_DEVICE=cpu`.
  MPS ile BLIP fine-tune belleği taşırıyor — GPU işleri Colab'da (A100) yapılır.
- Testler: `.venv/bin/python -m pytest -q` (hepsi CPU'da).
- Rapor: `cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main` (TinyTeX).
- Dil: ekip içi dokümanlar Türkçe, rapor İngilizce.

## Oturum sonunda
`docs/CALISMA_GUNLUGU.md` içindeki "Şu anki durum" ve "Sıradaki adımlar" bölümlerini güncelle ve
"Günlük" bölümüne tarihli kısa bir kayıt ekle (ne yapıldı, hangi dosyalar değişti, hangi sonuçlar çıktı).
