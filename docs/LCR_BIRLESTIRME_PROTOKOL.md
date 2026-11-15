# Tek Look–Check–Retell hattı: B geliştirme protokolü (2026-11-14)

Kullanıcı: attention verilen yerlerde nesne kanıtı ara; kelime destekleniyorsa koru, desteklenmiyorsa
kanıtı yeniden anlatıma ver. Literatür bu işlemleri uygulamak için kullanılır. VQA, captioner tutarlılığı,
dil önceliği veya ikinci bir doğrulayıcı ana Check'e eklenmez.

Bu plan yeni B karşılaştırmalarından önce yazıldı. Eski v4 ve validation dosyaları korunur.
Validation geliştirme/seçim için yeniden açılmaz, test kullanılmaz. Sonuçlar otomatik proxy diye adlandırılır.

## Sabitler ve seçim

- Captioner ve seçili dört attention başlığı korunur. Yeni attention/head araması yok.
- B deterministik görüntü hash'iyle B_fit (%80 civarı) ve B_cal (%20 civarı) ayrılır. B_cal Check'in
  son skor ölçeği/eşikleri içindir; Retell bunun üzerinde eğitilmez. Epoch sayısı önceden sabittir.
- Look: non-person için dört seçenek, minimum kenar 2 patch veya yalnız pozitif alan; alan kuvveti
  0.5 veya 1.0. Objectness kuvveti mevcut tür kuralında kalır. İlk üç **sonlu ve pozitif attention
  kanıtlı** öneri alınır; geçersiz kutu sıfır ağırlıkla bile kanıta sokulmaz. Kişi SAM tercihi korunur.
- Look seçimi yalnız B_fit insan kutularına karşı ilk üçten herhangi birinin IoU≥0.5 doğruluğuyla yapılır.
  0.5 puan içindekilerde küçük kutuyu eleyen sabit taban olmaması, sonra mevcut alpha=0.5 tercih edilir.
  İlk-kutu doğruluğu, küçük GT kutularındaki kapsama ve kişi sonuçları ayrıca raporlanır.
  İnsan kutuları Check etiketi veya Retell eğitim hedefi olarak kullanılmaz.
- Check: seçili Look üzerinde max veya r-ağırlıklı logit birleştirme; alternatifler tüm sınıflar
  veya mevcut sabit ikame grupları (v4 A3). Toplam dört seçenek. Her kutunun kendi skor ve kararı
  her iki seçenekte de saklanır; Retell her kutuyu ayrı kanıt olarak alır.
- Etiketsiz sınıf z ölçeği görüntü başına maksimum OWLv2 logit dağılımından; B_fit 5 katlı ÇD'de
  yalnız eğitim görüntülerinden hesaplanır. Eşikler de yalnız o eğitim katının referans proxy etiketleriyle
  seçilir. Seçilen Look daha önce B_fit'i gördüğünden bu, tüm zincirin bağımsız dış-kat tahmini değildir.
- Kural korunur: claim≥τs → supported; claim<τc ve alternatif≥τs → contradicted; diğerleri uncertain.
  Yalnız düşük skor yokluk kanıtı sayılmaz; sebebi kayıtta açıkça verilir.
  Eşikler tür başına J'yi en büyütür, contradicted kesinliği≥0.5 ve en az 20 karar kısıtı vardır.
- Check seçimi: B_fit ÇD contradicted kesinliği≥0.45 olanlarda en yüksek J; 0.01 içinde max ve
  ikame grubu öncelikli. Hiçbiri koşulu geçmezse durum açıkça raporlanır; sonuç gizlenmez.
  Bu küçük seçim tablosu geliştirme sonucudur; yeni genel başarı iddiası değildir.
- Son μ/σ ve eşikler yalnız B_cal'da fit edilir; kimlikli tek bundle olarak kaydedilir.
  B_fit eğitim kayıtları bu bundle'dan üretilir. B_cal kayıtları eğitim dışında kalır.

## Ortak uygulama ve Retell

Canlı çıkarım, önbellekten yeniden üretim ve sentetik yanlış kelimenin yeniden skorlanması aynı Python
uygulamasını kullanır. Kayıt; kutu indeksi/geometrisi, skor ölçeği, alternatif, eşikler ve bundle kimliğini
taşır. Eski skorla yeni alternatif adı eşleştirilmez; karar Retell girişinde eski kuralla yeniden yazılmaz.

Ana Retell yalnız attention ile seçilmiş bölge kanıtlarını alır; bütün görüntü maksimumu kanıt
token'ından çıkarılır. Gated BLIP mimarisi korunur. Gate/bias/LayerNorm weight decay'den çıkarılır;
eğitim epoch'u validation'da seçilmez. Eski eğitimler açıkça tarihsel kontrol olarak kalır.
Doğrulanmamış alternatif otomatik kelime değiştirme emri değildir. Son caption'ın doğruluğu ayrıca
ölçülmelidir; yeni Check skorundan Retell'in düzeldiği çıkarılmaz.

## Kapanış kontrolleri

1. B insan kutularıyla Look katkı tablosu; Check B_fit ÇD ve B_cal kalibrasyon sonuçları, alternatif proxy'si.
2. Küçük frizbi adaylarının hangi aşamada kaldığının yeni tanısı.
3. Canlı/önbellek/sentetik bozma eşitliği ve Retell'e giren karar–skor–kutu–alternatif tutarlılığı testleri.
4. Mevcut CPU testleri. Yeni ağır Retell eğitimi Colab'da; eğitilmeden iyileşmiş caption sonucu iddia edilmez.

Kaynaklar: `docs/LITERATUR_TARAMASI.md` §5–12a;
[OWLv2 model kartı](https://huggingface.co/google/owlv2-base-patch16-ensemble),
[OWLv2](https://arxiv.org/abs/2306.09683), [Flamingo](https://arxiv.org/abs/2204.14198).
İlk üç kutu, alan kuvveti ve max/ağırlıklı birleştirme burada sınanan mühendislik tercihleridir;
bu kaynakların birebir yöntemi veya garantisi diye sunulmaz.

## Kullanıcı kapsam düzeltmesi ve GPU sınırı (aynı gün, B kıyasından sonra)

Kullanıcı Retell'e geçilmemesini ve ağır model hesaplamasının Claude'a bırakılmasını istedi.
Bu nedenle Retell eğitimine hazırlık yapılmış olsa da ana lcr girdisiyle eğitim devre dışıdır.
Yeni GPU modeli çalıştırılmadı. Önceki B seçimleri değiştirilmedi.

Sonraki hesap için sabit kör pilot hazırlanır: B_fit ve B_cal'dan etiketsiz SHA256 sırasıyla 64'er
görüntü; mevcut Look adayları, her kenarda %10 kutu boyu bağlamıyla crop'a çevrilir. Aynı donmuş
OWLv2 yalnız özellik çıkarır; referanslar, eşik fit'i ve Retell bu işte yoktur. Bu, henüz başarı sonucu
olmayan sonraki Check deneyidir. Detay `docs/handoff/CLAUDE_LOOK_CHECK_GPU.md`.
