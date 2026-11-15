# Retell: kaynaklar ve düşünce notları

15 Kasım 2026 — Codex; Claude'un okuyabileceği araştırma notu.

Kullanıcının kapsamı: Retell hakkında ilgili çalışmalar bulmak ve düşünmek. Bu belge bir yol haritası,
uygulama talimatı, deney protokolü veya yöntem seçimi değildir. Eğitim/çıkarım yapılmadı; validation/test
verileri açılmadı. Buradaki örnek cümleler açıklama amacıyla yazıldı, veri setinden ölçülmüş örnekler değil.

**Ana düşünce:** Check'in verdiği bilginin doğru olması, captioner'ın bu bilgiyi doğru kullanması ve
düzeltirken cümlenin diğer doğru kısımlarını koruması ayrı meseleler. Retell'in problemi bu üçünün
buluştuğu yerde. Literatür, “kanıtı verince kendiliğinden düzelir” varsayımını destekleyecek kadar basit değil.

Okuma düzeyi: HALC, Woodpecker, LURE ve Volcano için yöntem metinleri de incelendi. Diğer kaynaklarda
aşağıda belirtilen resmi yayın sayfası/özet veya kaynak kodu esas alındı; tüm makaleler baştan sona okunmuş
ve sonuçları yeniden üretilmiş gibi yorumlanmamalı. “Çıkarım” paragrafları bizim projeye ilişkin yorumdur.

## Bizim Retell'in gerçekte yaptığı şey

`lcr/retell.py`, `lcr/evidence.py`, `lcr/captioner.py`, `scripts/06_train_retell.py` ve son Check belgesi
`docs/CHECK_TAMAMLAMA.md` okundu. Eski tasarım notlarının bazıları güncel koddan geride kalmış:

| Konu | 15 Kasım'da kaynak kodunda görülen durum | Düşündürdüğü |
|---|---|---|
| Taslak cümle | Retell'e doğrudan verilmez; görüntü ve kanıtla BOS'tan üretim yapılır. | Mevcut mimari, kelime düzenleyen bir editörden çok kanıtla koşullu captioner. |
| Yeni Check | `keep / keep_elsewhere / replace / remove / skip`, ayrıca `substitute` ve `alt_any` var. | Nesne hakkında gözlem ile cümleye uygulanacak işlem artık ayrı alanlar. |
| Kanıt kodlama | `record_features()` sınıf, alternatif, bölge, 9 skaler ve eski `verdict` tipini kodlar. `action`, `substitute` ve claim'in `pos` alanı kodlanmaz. | `remove` ve `replace`, aynı `contradicted` tipine düşüyor. Yeni sözleşmenin tüm anlamı eski kanalda korunmuyor. |
| Bölge token'ları | LCR yolunda her bölgenin skoru/alternatifi korunur, üst karar bütün bölgelere aktarılır. | Üstteki silme/ikame niyeti ile yerel alternatif etiketleri aynı şey değil. Bir `remove` kaydının alternatif etiketi başka nesnenin orada olduğunu anlatabilir; onu cümleye ekle demek değildir. |
| Görüntü skoru | Yeni LCR token'larında `s_image=0`; görüntü düzeyi veto Check içinde ayrı. | Eski “Retell tüm görüntü maksimumunu da serbestçe görüyor” eleştirisi yeni LCR için aynen geçerli değil. |
| Gate decay | `optimiser_groups()` gate, bias ve tek boyutlu parametreleri weight decay dışında bırakıyor. | Eski belgelerdeki “gate decay'den çıkarılmalı” maddesi artık yapılmamış iş diye sunulmamalı. |
| Epoch | Script sabit son epoch'u kullanıyor; validation raporu açık bayrakla. | Eski validation-CIDEr ile epoch seçimi eleştirisi güncel kodun davranışı değil. |
| Eğitim sınırı | Script ana `--attn lcr` yolunu açık hatayla durduruyor. | Bu notlar eğitimin açıldığı veya yeni Check'le öğrenilmiş Retell'in çalıştığı anlamına gelmiyor. |

Ek ayrım: script 42'nin **OOF değerlendirme kararları** ile yazdığı `A_B_lcr2.json` aynı şey değil.
`crossfit()` değerlendirmeyi kat dışı yapıyor; dosyaya yazılan son kayıtlar bütün B'de fit edilmiş bundle'dan
geliyor ve metadata bunu açıkça söylüyor. Bu kayıtları ileride öğrenilmiş Retell'e vermek, eğitim kanıtlarının
da OOF olduğu anlamına gelmez. Bu bir deney sonucu değil, scriptin veri akışının okunmasıdır.

## İlgili çalışmalarda ne var?

### HALC — bölgesel bilgiyle o anki kelime dağılımını değiştirmek

Chen ve arkadaşları, **ICML 2024**, *HALC: Object Hallucination Reduction via Adaptive Focal-Contrast
Decoding*. HALC nesneyle ilişkili üretim adımlarında detektörden başlangıç bölgesi alıyor, farklı görüş
alanları üretiyor ve bunların token dağılımlarını karşılaştırıyor. Sonra görsel eşleşmeye dayalı beam
seçimiyle yerel müdahaleyi bütün cümleyle ilişkilendiriyor. Detektör kutusu son sözü söyleyen doğru cevap
değil, görsel bağlam aramasının başlangıcı. Yöntem eğitim gerektirmiyor, fakat ek görsel geçişler ve arama
maliyeti var. [Resmi yayın](https://proceedings.mlr.press/v235/chen24bi.html),
[yöntem metni §4](https://arxiv.org/html/2403.00425v2),
[yazar kodu](https://github.com/BillChan226/HALC).

**Çıkarım:** Bizim fikre en yakın bağ, doğrulanan yerin üretim kararına geri dönmesi. Ancak HALC'in
bölgesini detektör arıyor; bizde Look'un captioner attention'ından gelmesi ayrı bir iddia. “HALC'i
uyarladık” demek, yalnız yanlış kelimeyi yasaklamaktan daha fazlasını gerektirir. Kırpımı OWLv2'ye
tekrar göstermek Check'i; kırpımı BLIP'e tekrar göstermek Retell'in görsel koşulunu değiştirir. Bunlar
aynı müdahale değil.

### Woodpecker ve VisualFactChecker — doğrulama sonucunu yeniden yazara vermek

Yin ve arkadaşları, **Science China Information Sciences 2024**, *Woodpecker*. Ana kavramları çıkarıyor,
sorular oluşturuyor, uzman modellerle görsel bilgiyi doğruluyor, bu bilgiden iddialar oluşturuyor ve
orijinal yanıtla birlikte bir LLM'e vererek düzelttiriyor. Düzeltici orijinal metni ve doğrulanmış bilgiyi
birlikte görüyor. Kullanılan bileşenler arasında Grounding DINO, VQA için BLIP-2-FlanT5-XXL ve metin
işleri için GPT-3.5-turbo var. “Eğitimsiz” olması, tek küçük captioner ile çalıştığı anlamına gelmiyor.
[Makale §3–4](https://arxiv.org/html/2310.16045v2),
[yayın DOI](https://doi.org/10.1007/s11432-024-4251-x).

Ge ve arkadaşları, **CVPR 2024**, *Visual Fact Checker: Enabling High-Fidelity Detailed Caption
Generation*. Birden çok caption önerisi, araçlarla doğrulama ve doğrulama sonuçlarını kullanan son
caption üretimi var. Son üretici hem önerileri hem doğrulama bilgisini kullanıyor.
[Resmi yayın](https://openaccess.thecvf.com/content/CVPR2024/html/Ge_Visual_Fact_Checker_Enabling_High-Fidelity_Detailed_Caption_Generation_CVPR_2024_paper.html).

**Çıkarım:** Bu iki çalışma, “neyi yanlış bulduk?” bilgisinin son üretime açıkça taşınmasına örnek.
BLIP-base bir talimat takip eden sohbet modeli olmadığı için “ball yanlış, frisbee yaz” cümlesini önüne
koymak aynı mekanizmayı elde etmez. Kanıtın metin olması ile modelin bu metni geri bildirim olarak
yorumlayabilmesi farklı özellikler. Bu çalışmalar ayrıca “aynı captioner kendisini düzeltiyor” iddiamızı
doğrudan kanıtlamaz; son yazarda farklı ve güçlü bir dil modeli bulunabilir.

### LURE — düzeltmeyi öğrenmek, sıradan captioning öğrenmekten farklı

Zhou ve arkadaşları, **ICLR 2024**, *Analyzing and Mitigating Object Hallucination in Large
Vision-Language Models*. LURE, görüntü ve hatalı açıklamadan daha doğru açıklamaya giden bir revisor
öğreniyor. Eğitim için doğru açıklamaları GPT-3.5 ile değiştirerek hatalı açıklamalar oluşturuyor.
Sentetik örneklerin gerekçesi nesnelerin birlikte görülmesi, üretim belirsizliği ve açıklamadaki konum
üzerine analizler. Model hatalı açıklamayı gerçekten girdi olarak görüyor; bu ayrıntı bizim mevcut
gated-captioner yolundan farklı.
[Resmi yayın](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fc625e831361cfcc82cb74224fdc66cb-Abstract-Conference.html),
[yöntem §3](https://arxiv.org/html/2310.00754).

**Çıkarım:** Bizde kanıtı değiştirip yine beş serbest referanstan birini hedeflemek, belirli bir taslak
hatasını düzeltmeyi öğrenmekle eşdeğer değil. Model görüntüden iyi caption üretip kanıtı yok sayarak da
kaybı azaltabilir. Ayrıca bir referansta bulunan doğru nesne, başka bir referansta hiç söylenmeyebilir;
bu hedef çeşitliliği hatalı etiketi silmek ile doğru ama gereksiz detayı atmak arasındaki ayrımı bulanıklaştırır.

### Show, Edit and Tell — doğru metni kopyalama fikri

Sammani ve Melas-Kyriazi, **CVPR 2020**, *Show, Edit and Tell: A Framework for Editing Image
Captions*. EditNet'in uyarlanabilir kopyalama mekanizması ve DCNet'in denoising bileşeniyle mevcut
caption'ı düzenliyor. Caption yapısını her seferinde yeniden öğrenmek yerine mevcut doğru içerikten
yararlanma fikri merkezi. Bu bir hallucination doğrulayıcısı değil; caption düzenleme mimarisi.
[Resmi yayın](https://openaccess.thecvf.com/content_CVPR_2020/html/Sammani_Show_Edit_and_Tell_A_Framework_for_Editing_Image_Captions_CVPR_2020_paper.html).

**Çıkarım:** “Düzelteceği kelimeyi bilmek” kadar “dokunmaması gereken kısımları bilmek” de Retell'e
yararlı olabilir. Taslağı hiç görmeyen bir modelin korunması gereken ifadeyi birebir korumasını beklemek
daha güçlü bir beklenti. Bununla birlikte eski LSTM mimarisini BLIP'e doğrudan taşımak ayrı bir çalışma olur.

### Volcano ve HELPD — geri bildirim ile onu kabul etme ayrı kararlar

Lee ve arkadaşları, **NAACL 2024**, *Volcano*. Görüntüye dayalı doğal dil eleştirisi, revizyon ve hangi
yanıtın kabul edileceğine karar verme işlemleri bulunuyor. Model bu davranış için eğitiliyor; salt
“kendine tekrar sor” yöntemi değil. [Resmi yayın](https://aclanthology.org/2024.naacl-long.23/),
[makale §2–3](https://aclanthology.org/2024.naacl-long.23.pdf).

Yuan ve arkadaşları, **EMNLP 2024**, *HELPD*. Nesne düzeyi ve cümle anlamı düzeyinde geri bildirim
öğrenmesini, görsel attention penceresine göre logit cezasıyla birleştiriyor. İncelenen resmi özet,
nesnenin varlığına bakmanın nesne ile cümle anlamı ilişkisini tek başına kapsamadığını vurguluyor.
[Resmi yayın](https://aclanthology.org/2024.emnlp-main.105/).

**Çıkarım:** Yeni caption üretildi diye eskisinden daha doğru olduğu sonucu çıkmıyor. Ayrıca “köpek var”
bilgisi, “çocuğun tuttuğu şey köpek” iddiasını doğrulamıyor. Bizim nesne-sözlüğü Check'i, cümlenin
eylemini, rengini ve kime ait olduğunu ayrıca doğrulamış sayılmaz. Retell bu boşlukları akıcı dille
doldurduğunda yanlış ilişkiyi koruyabilir veya yenisini ekleyebilir.

### Constrained beam search ve VCD — öğrenmeden dağılıma müdahale

Anderson ve arkadaşları, **EMNLP 2017**, *Guided Open Vocabulary Image Captioning with Constrained
Beam Search*. Mevcut captioning modeline test sırasında etiket sözcüklerini zorunlu kullandırabiliyor;
amaç özellikle yeni nesneler ve alan dışı captioning. Bu çalışma, yanlış etiketin zorla kullanılmasının
güvenli olduğunu göstermiyor. [Resmi yayın](https://aclanthology.org/D17-1098/).

Leng ve arkadaşları, **CVPR 2024**, *Visual Contrastive Decoding*. Orijinal ve bozulmuş görsel girdilerden
gelen çıktı dağılımlarını karşılaştırarak görsel bilgiye dayanmayan dil önceliğini azaltıyor. Yeni eğitim
ve harici detektör gerektirmeyen bir decoding müdahalesi.
[Resmi yayın](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html).

**Çıkarım:** Sözcüğü zorlamak, olasılığını artırmak ve yanlış sözcüğü yasaklamak farklı müdahaleler.
Yumuşak bir alternatif teşviki, yanlış etiketi zorlamanın riskini azaltabilir; fakat yanlış asıl kelime
sert biçimde yasaklanıyorsa bütün işlem “yumuşak” olmuş olmaz. VCD de doğrudan bir Check kaydını
kullanmıyor; genel görsel bağlılığı güçlendirme ile belirli geri bildirime uyma aynı başarı ölçüsü değil.

### Flamingo ve SmallCap — yeni kanalı öğretmek için mimari dayanak

Alayrac ve arkadaşlarının **NeurIPS 2022 Flamingo** çalışmasındaki sıfır başlangıçlı gated
cross-attention, donmuş dil modeline yeni görsel koşullama katmanın dayanaklarından.
[Makale](https://arxiv.org/abs/2204.14198).

Ramos ve arkadaşları, **CVPR 2023 SmallCap**, donmuş CLIP ve GPT-2 arasında yeni cross-attention
katmanları öğrenerek görüntü ve getirilen caption'larla koşullama yapıyor.
[Resmi yayın](https://openaccess.thecvf.com/content/CVPR2023/html/Ramos_SmallCap_Lightweight_Image_Captioning_Prompted_With_Retrieval_Augmentation_CVPR_2023_paper.html).

**Çıkarım:** İki çalışma da donmuş modele yeni bilgi kanalı eklemenin makul olduğunu gösteriyor;
bizim gürültülü doğrulama kayıtlarımızı az veriyle kullanacağını göstermiyor. Gated yolun mimari
gerekçesi ile geri bildirimin doğru kullanılmasının kanıtı ayrı tutulmalı.

### Ters yöndeki bulgu: grounding tek başına yetmeyebiliyor

Geigle, Timofte ve Glavaš, **EMNLP 2024**, *Does Object Grounding Really Reduce Hallucination of
Large Vision-Language Models?* Soru-cevap yerine açık caption üretimini de ele alan değerlendirmelerinde,
üç backbone üzerinde grounding hedeflerinin nesne halüsinasyonuna etkisini sınırlı veya yok buluyorlar.
Bu, belirli modeller ve deney koşulları için bir bulgu; bütün grounding yöntemlerinin işlemediği teoremi
değil. [Resmi yayın](https://aclanthology.org/2024.emnlp-main.159/).

**Çıkarım:** İyi kutu, iyi attention veya iyi yes/no cevabı, düzeltilmiş caption başarısının yerine
geçmiyor. Bizim farkımız bağımsız Check geri bildirimini tekrar üretime vermek olabilir; bu farkın
işe yaradığı mevcut Look ölçümlerinden tek başına çıkarılamaz.

### Daha yeni iki bağlantı: müdahalenin içeriği ve maliyeti

Zhuang ve arkadaşları, **CVPR 2025 VASparse**, görsel farkındalığı koruyan token seyrekleştirmesiyle
halüsinasyon azaltma ve verimliliği birlikte ele alıyor. Resmi özet, ikinci decoding ve rollback
gerektiren yaklaşımların hız maliyetini tartışıyor.
[Resmi yayın](https://openaccess.thecvf.com/content/CVPR2025/html/Zhuang_VASparse_Towards_Efficient_Visual_Hallucination_Mitigation_via_Visual-Aware_Token_Sparsification_CVPR_2025_paper.html).

Ji ve arkadaşları, **CVPR 2026 CausalLens**, görsel/text/system yollarını ayırıp seçili attention
başlıklarının hidden-state katkılarına müdahale eden, eğitim gerektirmeyen bir yaklaşım sunuyor.
[Resmi makale](https://openaccess.thecvf.com/content/CVPR2026/papers/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.pdf),
[yazar deposu](https://github.com/jijy20/CausalLens).

**Çıkarım:** Bu çalışmalar Retell'in yerine seçilmiş yöntemler değil. Düşündürdükleri, birden çok
yeniden üretimin ücretsiz olmadığı ve tüm görsel kanalı büyütmekle belirli dış kanıtı kullanmanın farklı
mekanizmalar olduğu. Tek akışta görsel/text token'ları işleyen LVLM müdahaleleri, ayrı görsel
cross-attention'ı olan BLIP'e aynı katman numaralarıyla taşınamaz. Bu ikisinin tam sonuç tabloları ve
bağımsız tekrar üretimi bu araştırmanın kapsamında incelenmedi.

## Bunları yan yana koyunca akla gelenler

### “Yanlış isim” bazen yanlış cümlenin yalnız görünen ucu

Hayali örnek: `a boy holding a red surfboard` içinde yanlış şey yalnız `surfboard` olmayabilir.
Gerçekte bir frizbi görünüyorsa, `red` ve `holding` de ayrıca yanlış olabilir. Sadece isim değişimi bu
bağları onarmaz. Tersine tüm cümleyi baştan üretmek doğru özneyi ve sahne bilgisini de kaybedebilir.

Hatadan sonrasını üretme fikrinin de sınırı var: hatalı nesneden önce `holding a red` çoktan sabitlenmiş
olabilir. “İlk hatalı noun'dan başlatmak”, “ilk hatalı anlam biriminden başlatmak” ile eşdeğer değil.
Doğru koruma birimi token, isim öbeği veya bütün yan cümle olabilir; burada bunlardan biri seçilmiyor.

Bu kaygının literatürde akrabası **multimodal hallucination snowballing**. Zhong ve arkadaşları,
**ACL 2024**, önceki hatalı içeriğin sonraki görsel cevapları da sürükleyebildiğini inceliyor. Onların
deneyi çok turlu konuşma; kısa BLIP caption'ında aynı büyüklükte etki olduğunu göstermiyor.
[Resmi yayın](https://aclanthology.org/2024.acl-long.648/).

### “Burada köpek var” ile “bu cat kelimesini dog yap” arasındaki eksik bağ

Aynı görüntüde gerçek bir kedi ve köpek bulunabilir. Sınıfı bütün caption boyunca yasaklamak, bir
instance hatasını doğru diğer instance'a yayabilir. Aynı kelime iki kez geçtiğinde sınıf etiketi ikisini
ayıramaz. `suit`, `uniform`, `shirt` gibi örtüşebilen etiketlerde alternatif skorunun yüksekliği de
mantıksal dışlama değildir.

Bu yüzden bölge kanıtının **hangi kelime kullanımına** bağlandığı anlamsal bilgi taşır. Mevcut kayıtta
`pos` bulunması, bu bilginin Retell token'ına girdiği anlamına gelmiyor. Benzer biçimde `alt_any`
“orada gördüğüm başka nesne”, `substitute` ise “önerilen ikame” rolünde; ikisinin aynı metin alanına
indirgenmesi modelin ne yapması gerektiğini belirsizleştirebilir.

### Kanıtı kullanmamak ve kanıtı körlemesine izlemek iki ayrı başarısızlık

Bir Retell yanlış Check'i izleyerek de bozulabilir; hiçbir Check'i izlemeyerek de. Caption'ın kanıt
değişince değişmesi yalnız duyarlılık gösterir. Değişimin doğru claim'de ve doğru yönde olup olmadığı
başka bir bilgi. Özellikle mevcut `false_evidence()` alternatifin adını değiştirip skorları/bölgeyi
koruyor. Bu, etiket kanalına bağımlılığı inceleyen yararlı bir müdahale; kendi içinde tutarlı gerçek
bir yanlış detektör kaydının birebir simülasyonu değil. “Absence” adayları da mevcut kodda referans
ve detektör proxy'sinden geliyor, insan tarafından kesin yok etiketi değil.

Gate büyüklüğü de tek başına çözüm değil. Kodda `gate_ff`, kanıt olmayan null durumda bile gizli
temsili değiştirebiliyor. Dolayısıyla gate açılması, dış kanıtın kullanıldığının doğrudan kanıtı değil.
Sıfır başlangıçta attention dalının bazı iç parametrelerine ilk adımda gradyan ulaşmaması beklenebilir;
gate'in kendisi gradyan alabilir. “Sıfır başladığı için hiç öğrenemez” çıkarımı da doğru değil. Mevcut
test dosyasında gate gradyanı ve açılmış gate'le kanıt etkisi için kontroller var; bu oturumda çalıştırılmadı.

### Belirsiz doğrulamada sert silmenin maliyeti

Kavramsal olarak düzeltmenin beklenen kazancı `p × yarar − (1−p) × zarar` diye düşünülebilir.
Burada `p`, Check müdahalesinin gerçekten gerekli olma olasılığı; OWLv2 z skoru doğrudan bu olasılık
değil. Doğru bir nesneyi silmenin zararı, yanlış detayı kaldırmanın yararına yakınsa, az miktar güven
sert müdahaleyi haklı çıkarmaz. Bu ifade bir model/ölçüm sonucu değil, karar probleminin düşünsel çerçevesi.

“İkameye yumuşak teşvik verelim” bunun yalnız yarısını çözer: doğru kelime ayrıca sert yasaklanmışsa
yanlış alarm hâlâ caption'ı bozabilir. Tam tersine hiçbir sert kısıt yoksa model geri bildirimi tamamen
yok sayabilir. Literatür bu dengeyi bizim Check dağılımımız için hazır belirlemiyor.

### Daha az şey söylemek ile daha doğru şey söylemek

İsimleri silerek veya erken EOS üreterek hata oranı düşebilir; caption'ın taşıdığı doğru bilgi de azalabilir.
CHAIR çalışması zaten standart cümle benzerliği metriklerinin görsel doğruluğu tek başına açıklamadığını
gösteriyor. Bizim referansta-geçme proxy'miz de “görüntüde var mı?” sorusunun tam etiketi değil.
[Rohrbach ve arkadaşları, EMNLP 2018](https://aclanthology.org/D18-1437/).

Bu nedenle benim çıkardığım kavramsal ayrım şu: gerçek hatayı düzeltmek, doğru içeriği korumak, yeni
hata eklememek ve yeterli bilgi vermek farklı nitelikler. Bir çıktı bu boyutlardan birinde iyileşip
diğerinde kötüleşebilir. Burada yeni metrik veya eşik seçilmiyor.

## Kütüphane tarafında araştırırken görülen somut ayrıntılar

Yerelde kurulu `transformers` kaynak metninde sürüm **5.23.0**. Model yüklenmedi; şu dosyalar okundu:

- `.venv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py`
- `.venv/lib/python3.12/site-packages/transformers/generation/logits_process.py`
- `.venv/lib/python3.12/site-packages/transformers/generation/utils.py`

`bad_words_ids` ve `sequence_bias` için yerleşik processor yolları var. Bunların BLIP ile bu projede
istenen davranışı verdiği bu oturumda çalıştırılarak gösterilmedi. Özellikle:

- Yasaklar token **dizisine** uygulanır; bir WordPiece parçasını tek başına yasaklamak ilgisiz kelimelere
  de dokunabilir. Tekil/çoğul ve eşanlamlılar kavramla birebir eşleşmeyebilir.
- Çok token'lı `sequence_bias`, dizi tamamlanmak üzereyken son token'a uygulanır; alternatifin bütün
  parçalarını baştan otomatik desteklediği varsayılmaz. Tek token bias ise her adımda uygulanabilir;
  gereksiz tekrar etkisi doğurabilir.
- `force_words_ids`, constrained-beam moduna gider; yerel mapping bu modu
  `transformers-community/constrained-beam-search` custom-generation deposuna yönlendiriyor. Basit bir
  yerleşik bayrakla kesin çalışır diye yazmak doğru olmaz. Kod, bu yönlendirmede remote-code güveni
  arıyor. Herhangi bir uzak kod indirilmedi veya çalıştırılmadı.
- `Captioner.generate()` şu anda kendi BOS `input_ids` değerini oluşturuyor. Kısmi taslak prefix'ini
  mevcut wrapper'a bir override gibi geçirmek, hazır bir prefix-devam arayüzü kullanmakla aynı şey değil.
- Eldeki 577 BLIP patch gömmesinin bir kısmını almak/masklemek ile gerçek piksel kırpımını yeniden
  vision encoder'dan geçirmek aynı görsel koşul değil; kırpım ölçek ve bağlamı da değiştirir.

Bu maddeler uygulanmış düzeltmeler değil; eski “R0 kolayca şu bayraklarla olur” fikrinin arkasındaki
varsayımları görünür kılan kod gözlemleri.

## Claude için notun yorumu

Buradaki kaynaklar ortak bir yönteme indirgenmedi. HALC dağılıma bölgesel müdahaleyi, Woodpecker/VFC
doğrulanmış bilgiyi yazara taşımayı, LURE/Volcano düzeltme davranışını öğrenmeyi, Show–Edit–Tell doğru
metni korumayı düşündürüyor. Grounding üzerine olumsuz bulgu ise bu bileşenlerin tek başına uçtan uca
başarı demek olmadığını hatırlatıyor.

Açık kalan esas düşünce: **Retell neyi öğrensin veya neyle koşullansın?** Görüntüde hangi nesnenin
bulunduğunu yeniden tahmin etmek, belirli Check geri bildirimini uygulamak ve doğru cümleyi az değiştirerek
onarmak aynı iş değil. Mevcut mimari birincisini zaten yapabiliyor; ikinci ve üçüncüye hangi bilginin
eksik geldiğini yukarıdaki kod gözlemleri gösteriyor. Hangi yöntemin kazanacağına ilişkin yeni deney,
sonuç, sıralama veya uygulama kararı bu notta yok.
