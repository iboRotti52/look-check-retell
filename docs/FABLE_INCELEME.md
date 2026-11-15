# Independent method review: Fable (2026-11-15)

**Scope.** The owner asked "is the system correct, and how can it be improved without leaving our logic?". The
review was done by Claude Fable 5.2 as an independent subagent:

- It used B only. It did not read validation or test, ran no GPU jobs, did not modify the project, and looked at no
  images.
- The scripts are in `scripts/review_fable_B/` (features in `data/cache/review_fable_B_feat.pkl`). These are
  exploratory scripts, not report numbers.

## Claude's verification
I recomputed V1 myself from `results/lcr_check_bundle.json` and `A_B_lcr2.json`. It matches exactly.

**Share of images reaching τs (1.339, z-scale):**

| Class | References mention it | References do not mention it |
|---|---|---|
| shirt (n=533) | 0.017 | 0.004 |
| pants (n=136) | 0.044 | 0.005 |
| jacket (n=220) | 0.382 | 0.044 |
| frisbee (n=34) | 0.941 | 0.075 |
| dog (n=761) | 0.913 | 0.000 |

**Reference-confirmed claims, in-sample:**
- All things (n=2401): 11.6% contradicted, 76.6% supported.
- Clothing (n=621): **32.5% contradicted**, only 40.7% supported.

**Cause:** the z-scale computes each class's mean and spread from the "per-image maximum". That only works if most
images lack the class. Classes such as shirt, which almost every picture of a person contains, have a high mean, so
real shirts can never exceed the threshold. This scale was my choice in Check v4 (C2), and it is the main cause of the
clothing failures.

## Turkish summary (main findings)
1. **The z-scale is wrong; it is the biggest loss** (V1).
   - The "swap test" catches a known wrong word inserted into a correct claim's region. On it, the current z-scale is
     no better than the image-free class prior (AUROC 0.761 vs 0.767). Raw logits give 0.900, per-class Platt 0.921.
   - Proposed fix (C1a): a per-class threshold anchored on clean positives (claims confirmed by the references). This
     needs no "wrong" labels and directly bounds the false-contradiction rate per class.
2. **The "something else is here" condition is nearly always true** (V2). 83% of object claims pass it. With max over
   168 classes and 3 boxes, something always clears the threshold.
3. **With the veto, Look adds nothing to error detection** (V3).
   - The region score and the image score are about equally informative; an image-only rule gives almost the same
     result.
   - Look's real job is choosing the substitute, and it has never been measured in that role.
4. **The `replace` groups are wrong** (V4).
   - The groups hold things that occur together, not mutually exclusive alternatives: shirt/pants/hat/backpack all sit
     in the same group.
   - On raw scores using only the box Look ranks first, the substitute is right 0.90–0.93 of the time; the current
     setup reaches 0.69.
5. **Look is close to its ceiling** (pointing 0.923, top-3 0.944, proposal-pool ceiling 0.988). The coordinate,
   time-step and padding geometry was verified correct.
6. **Thresholds are being fit to the proxy** (V5). The suggestion is to choose the operating point by a target
   false-contradiction rate (2–5%), not by J.

**Top three next steps (Fable):**
1. Per-class thresholds anchored on positives, plus a restructured rule:
   - absence from the image → contradiction;
   - the box Look ranks first → substitute;
   - mutually exclusive confusion sets.
2. Measure Look on wrong words (insert the wrong word and check where the attention goes).
3. Label 400 stratified claims (by action) with humans before choosing any threshold.

## Fable's original report

### Scope and caveats
All numbers below are mine, computed on half B only (3000 `train_*` images, 4594 non-person claims) from the cached
OWLv2 proposals and `A_B_look4` maps. I read no validation or test data, ran no GPU or model job, changed nothing in
the project, and viewed no images, so nothing here is a visual judgement.

Two label-noise-free instruments carry most of the argument:
- **FCR**: the share of reference-confirmed claims (`in_refs=True`, a clean positive) that Check contradicts.
- **Swap test**: 2401 reference-confirmed thing claims keep their Look region, but the label is replaced by the
  captioner's own alternative (`blip_alts`, 2057 cases) or a random same-group class not in the references (344).
  This gives wrong-by-construction claims with a known true class.

The pooled swap numbers are confounded by class frequency: a class-prior-only score with no image gets AUROC 0.767,
because swapped classes are rarer than true ones. I report within-class figures where that matters.

### 1. Correctness issues

#### Verified

**V1. The label-free z-scale is the wrong normalisation, and it is the main cause of the clothing failures.**
- Where: `scripts/42_check_complete.py:52` (`mu, sd = image_max.mean(0), image_max.std(0)`), applied in
  `lcr/region_evidence.py` `RegionChecker.record` (`row = (raw - self.mu) / self.sd`).
- Why it is wrong: mean/std of per-image max logits is a null reference only if most images lack the class. A fixed z
  threshold then flags roughly the top 9% of images for every class, whatever its prevalence.
- Evidence at τs = 1.339, image level:

| Class | Images mentioning it that reach τs | Images not mentioning it that reach τs |
|---|---|---|
| shirt (533 mentions) | 1.7% | 0.4% |
| pants | 4.4% | 0.5% |
| suit | 27% | — |
| jacket | 38% | 4.4% |
| elephant | 100% | 8.1% (raw p ≥ 0.3: 0.1%) |
| frisbee | 94% | 7.5% |
| fence | 75% | 8.7% |

- Consequence: on reference-confirmed claims the bundle keeps only 42% of clothing and contradicts 32% (all things:
  11.9% contradicted).
- Scale comparison on the swap test, image level:

| Scale | Swap AUROC | Detection at 5% FCR | Animals AUROC | Proxy AUROC (things) |
|---|---|---|---|---|
| Current z | 0.761 | 0.380 | 0.765 | 0.687 |
| Raw logit | 0.900 | 0.535 | 0.992 | 0.770 |
| Platt, per-class offset fit on image-level mention (out-of-fold) | 0.921 | 0.612 | 0.998 | 0.796 |

- Honest limit: where the swapped class has the higher base logit (n=852), z detects more at 5% FCR (0.474 vs 0.366
  raw, 0.435 Platt). Raw and Platt gain partly from class prevalence. The current z is no better than the image-free
  prior (0.761 vs 0.767).
- Fix: see C1.

**V2. The "something else is there" condition is nearly vacuous.**
- Where: `decide5`, `sa_any >= ts`.
- Evidence: 83% of thing claims pass it (76% of those with claim < τc). The median region has 3 other classes above
  τs; this is max over 168 classes × 3 boxes on a scale that guarantees tail hits.
- Side effect of V1: `person` never reaches τs in any region (0.0%), so "a person is there" can never be evidence.

**V3. With the veto, Look does not affect error detection at all.**
- Evidence: every contradicted claim has image-level z < τs. The image-only rule `s_img < τc` gives proxy precision
  0.499 and recall 0.432; the bundle gives 0.513 and 0.358.
- Region and image scores are equally informative: swap AUROC 0.908 vs 0.900 raw; proxy AUROC 0.777 vs 0.778.
- Half of all claims have a top-3 box covering more than 50% of the image, so max over 3 is close to the image max.
- This is not a bug, since absence is an image-level fact. But Look's only causal role today is choosing the
  substitute, and Look has never been measured for that role.

**V4. `replace` is computed in a way that damages it.**
- Where: `region_evidence.py:26-34` (`GROUP`), and `record` (`sj = argmax(sub_mask, row)` on the max-pooled z row).
- Group problem: groups contain co-occurring items, not mutually exclusive ones.
  - clothing: shirt, pants, hat, glasses, backpack, bag, shoe
  - handheld: camera, bench, table, bed, tent, sign
  - play: ball, swing, slide, ramp, ladder, tire
  - structure: pool, log
  - all stuff is one group
- Scale problem: the substitute is ranked on z, which favours rare classes.
- Evidence on the swap test (true class known): when the bundle replaces, the substitute is the true class 69% of the
  time (0.291 of 0.419). The top same-group class is the true one 52% of the time on z with max over 3 boxes, 63% on
  raw logits with the top-1 Look box.
- A raw rule (contradict if image max < a; replace if a same-group class has raw ≥ −0.85 on the top-1 box):

| Rule | FCR | Detected | Replaced with true class | Substitute precision |
|---|---|---|---|---|
| Current bundle | 11.9% | 56.1% | 29.1% | 0.69 |
| Raw, a = −2.18 | 5% | 53.5% | 33.4% | 0.93 |
| Raw, a = −1.49 | 10% | 69.9% | 40.9% | 0.90 |

**V5. The threshold objective is fitted to the wrong target.**
- Where: `fit_thresholds`.
- J-max with proxy precision ≥ 0.5 treats "unmentioned" as wrong. The result is τc ≈ τs (1.128 vs 1.339), leaving
  only 161 claims in the uncertain band, and 24% of swapped claims are still kept.

#### Checked and found correct
- Token step: attention at p−1 for first piece p (`look_sources.py:105`, `look_v4.py:51`).
- Caption-mean subtraction: BOS replaced, SEP excluded.
- BLIP squash grid vs OWLv2 bottom/right pad: `look.owl_to_image`.
- Partial-patch mass: `attention_mass`.
- Empirical confirmation of the geometry on Flickr30k Entities boxes (2334 thing claims): pointing 0.923, top-1 0.848,
  top-3 0.944, proposal oracle 0.988. A coordinate bug would not allow this.
- Cross-fitting: scale and thresholds are fitted on image folds; no leakage. The shipped bundle is in-sample on all B,
  as declared.
- `RegionChecker.record` agrees with script 42 (asserted in the script).
- Prompt ensemble vs single template: no difference (0.901 vs 0.900).

#### Suspected, not verified
- **Ambiguous single-word queries.** "a photo of a bat / slide / swing / track / court / pool / glasses / tube / net".
  OWLv2 may score the wrong sense (`lcr/vocab.py`, `config.QUERY_TEMPLATE`).
- **Stuff.** Proposals are the top 100 by objectness after NMS (`check.py:62`). Amorphous regions have low objectness,
  so both region and veto see a biased box set.
- **Look survivorship.** Look is only measured on correct claims that have human boxes. Where attention goes for a
  wrong word is unmeasured.
- **Small extraction losses.** `reference_classes` matches verbs ("park", "rock", "wave", "steps"). The compound rule
  drops real claims ("bike rider" loses bike).

### 2. Improvements, ranked

#### Look
Look is near its ceiling for locating correct objects (top-3 0.944 vs oracle 0.988). Grad-CAM and Chefer already lost.
More map work will not pay; the gains are in how the region is used.

**L1. Measure Look on wrong words.**
- What: teacher-force the swapped word into the draft, extract the same 4-head map, and score it against the human box
  of the true object.
- Why: this is the only case where Look matters (V3).
- Cost: CPU decoder passes with cached features; no training.
- Within the logic: it only measures the captioner's own attention.

**L2. Use the top-1 attention box for "what is there"; use top-3 only for "is the claim here".**
- Evidence: substitute accuracy 0.634 (top-1, raw) vs 0.521 (current).
- For the claim's witness box, letting the detector pick within the top 3 raises the human-box hit from 0.848 to
  0.877.

**L3. Suppress near-whole-image candidates when a smaller one has comparable r(b).**
- Evidence: 50% of claims have a >0.5-area box in the top 3.
- Untested for hit rate. Measure with human boxes and substitute recovery.

**L4. Keep the masking intervention (`look_sources.intervention`) as a per-claim reliability signal.**
- Use it to abstain from `replace`, not from detection.

Weakest groups at top-1: structure 0.70, play 0.76, clothing 0.80, handheld 0.80. Top-5 reaches 0.95–0.97, so these
are ranking problems, not map problems.

#### Check
**C1. Replace the z-scale (largest gain).** Two options, both keeping frozen OWLv2 as the only verifier:
- (a) Per-class thresholds from clean positives. Set τ_c to the α-quantile of the class's image-max logit over images
  whose references mention it, shrunk to a global quantile for rare classes. It needs no "wrong" labels and bounds FCR
  per class. Measured out-of-fold: α = 0.10 gives FCR 5.4% and detection 48% (current z: 38% at 5%). It carries no
  class prior.
- (b) Platt scaling with a per-class offset on image-level mention, over all image × class pairs. Detection 61% at 5%
  FCR. It carries class prevalence, so report it with the hi-base-swap control. Elkan & Noto (KDD 2008) justify
  treating mention as positive-unlabelled data.
- I would make (a) the main method and (b) an ablation.

**C2. Restructure the rule to match what each signal can do.**
- Contradiction: image-level absence test.
- Substitute: top-1 Look box, raw or calibrated scores, with a margin over the claim on that same box.
- Drop the vacuous `sa_any` condition.
- Evidence: table in V4.

**C3. Replace `GROUP` with mutually exclusive confusion sets.**
- Examples: upper-body garments; lower-body; headwear; animals; boards; boats.
- No `replace` outside a set.
- Evaluate by substitute recovery on swaps. My quick version used a non-comparable denominator, so I report no number.

**C4. Per-class abstention list from within-class swap AUROC (scale-invariant).**
- Measured: suit 0.57, shirt 0.71, wall 0.72, ramp 0.76, tree 0.78; dog 1.00, horse 0.98, dress 0.94, hat 0.92.
  Weighted mean 0.85.
- Classes below about 0.75 should never be contradicted, as with stuff now.

**C5. Crop re-detection on the attended box for clothing and small objects.**
- This is the existing GPU pilot. It is the only route to better clothing evidence; rescaling cannot fix AUROC 0.78.
- Measure with FCR and swaps.

**C6. Sense-specific prompts for ambiguous classes.**
- Cheap: text embeddings only. Measure with within-class AUROC.

**C7. Choose the operating point by an FCR target (2–5%), not J on the proxy.**

#### Retell (brief)
- The true error rate is probably low. The tail of proxy-wrong scores suggests roughly 15–25% of them behave like
  clean negatives, but the estimate is not identifiable (it ranges 0.14–0.82 by threshold).
- At 7% prevalence, 5% FCR and 55% detection, contradicted precision is about 0.45 at best. Retell must therefore be
  cheap when wrong.
- Design consequences:
  - Ban the word and regenerate from the prefix, rather than forcing the substitute.
  - Re-run Look and Check on the new caption.
  - Accept only if the new claim is supported and nothing new is contradicted.
- This closed loop stays inside the logic and bounds the damage from false alarms.

### 3. The measurement problem

**Measurable now, without humans:**
- FCR on reference-confirmed claims, per group.
- Swap detection and substitute recovery, reported with within-class AUROC and the prior-only baseline (0.767).
- Image-level detector vs mention: mean per-class AUROC 0.912; worst are camera 0.57, suit 0.67, shirt 0.78.
- Look on human boxes, including L1.

**Minimal human labelling: about 400 B claims**, three-way (present / absent / unclear), plus "is the substitute the
right name for the attended object" on replaces:

| Stratum | Claims | Settles |
|---|---|---|
| replace, clothing | 60 | clothing replace |
| replace, other things | 60 | substitute quality |
| remove | 100 | contradicted precision |
| keep_elsewhere | 60 | the veto |
| skip, score-stratified | 60 | thresholds |
| keep | 60 | supported precision |

- Double-annotate 100 for agreement.
- Reweight by stratum size for population rates.
- This gives contradicted precision to about ±7 points and the true base rate, which fixes the FCR target.

### 4. Bottom line
Look is close to as good as it gets for correct objects. Check is not: its scale, alternative condition and substitute
logic lose roughly a third of achievable detection at equal false-contradiction rate, and they cause the clothing
failures.

Top three next steps:
1. Replace the z-scale with positive-anchored per-class thresholds (C1a) and restructure the rule (C2, C3).
2. Run the swapped-word Look measurement (L1), since that is where Look earns its place.
3. Label the 400 stratified claims before choosing any threshold.

---

## Takip: C1a+C2+C3+C4+C7 uygulandı (2026-11-15, Claude)
Ön kayıtlı protokolle uygulandı ve B'de ölçüldü: `docs/CHECK_DUZELTME.md`. Kısaca: FCR yarıya indi ve giysi sorunu
çözüldü (0.32 → 0.05). Ama C4'ün "az pozitif" çekimserliği, swap'ların nadir sınıflara düşen kısmını görünmez yaptı.
Eşleşmiş FCR'de yakalama S2'nin altında kaldı, bu yüzden varsayılan değişmedi. Çekimserlik olmadan (sonradan) S3
toplamda S2 düzeyinde; giyside iyi, yüksek-taban swap'larda kötü. "z, sınıf önselinden iyi değil" bulgusu havuzlanmış
AUROC için doğru; giysi dışı thing'lerde çalışma noktasında z ile pozitif-çapalı eşik arasında fark yok.

