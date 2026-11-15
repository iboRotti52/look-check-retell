# Second independent review of Check: Fable (2026-11-15)

**Question.** After the positive-anchored fix (S3) missed its pre-registered criterion: what actually caps Check, and
how can it be made substantially better without leaving the Look → Check logic (captioner attention picks the place;
only the frozen OWLv2 judges)?

**Scope and rules followed.**
- B only (3000 `train_*` images, 4594 non-person claims, 2401 swaps: the same swap set as `scripts/43_check_fix.py`).
  Validation and test were not read. No GPU job. No project file, bundle, result or doc was changed.
- Scripts: `scripts/review_fable2_B/` (see its README). Caches: `data/cache/review_fable2_B_*`.
- All numbers are out-of-fold (5 image folds) unless marked in-sample. They are exploratory, not report numbers.
- FCR = share of reference-confirmed claims that get contradicted (reference mention is an automatic proxy for
  "correct"). Swap detection is a synthetic test. Visual judgements are mine (Claude), on small samples, and are not
  human labels.
- My harness reproduces the project's numbers exactly: S3 without abstention at α=0.15 gives FCR 0.086, detection
  0.585, high-base 0.388; S2 through the project's own functions gives FCR 0.085, detection 0.559.

## 1. Short answer

1. **Half of the "missed half" is the swap test, not Check.** 52% of swaps relabel the claim with a class that is
   *not* mutually exclusive with the true one (toy, shirt, wall, building, leash, uniform …). In a 16-image sample of
   such missed swaps, the "wrong" label was visibly present in about 10. On the swaps that are wrong by construction
   and visually distinct (mutually exclusive, non-clothing; n=826), a better-read OWLv2 detects **0.87 at 4.5% FCR**.
2. **Two things in S3 lose real performance, and both are fixable inside the logic.**
   - One α for every class spends the false-contradiction budget on easy classes. At thing-FCR 0.05, 49 of S3's 116
     false contradictions are true dogs, although a wrong "dog" almost never scores near a real one.
   - The claim is scored alone. OWLv2 separates mutually exclusive names much better by *comparison at the same box*
     than by an absolute threshold.
3. **Fixing both (S4 below) gives, at thing-FCR ≤ 0.05: detection 0.46 → 0.60 overall, 0.70 → 0.87 on clean
   exclusive swaps, 0.28 → 0.58 on same-set clothing.** S2 gets 0.56 overall at thing-FCR 0.117.
4. **Clothing is capped by the detector, and crop re-detection does not lift the cap.** I ran the crop idea on CPU
   with the same frozen OWLv2 on 714 swaps: it is slightly worse than the cached box (clothing pairwise accuracy 0.747
   vs 0.785). Prompt variants do nothing. The GPU crop pilot is not worth running in its current form.
5. **Look does earn its place, in naming.** The substitute read at Look's top-1 box is the true class 0.89 of the
   time; read from the image-level best box it is 0.71.

## 2. Diagnosis: what caps Check

### 2.1 The swap test mixes three different things
Composition of the 2401 swaps (`e0_baseline.py`, `e1_anatomy.py`):

| Part of the swap set | n | S3 (no abstention, α=0.15) detection |
|---|---|---|
| Swap label in the same confusion set as the true class | 1161 | 0.688 |
| Swap label outside the set (can co-occur with the true object) | 1240 | 0.489 |

- 31% of swapped claims are `dog`, mostly relabelled as another animal (dog→cat 228, →horse 85, →sheep 82). These are
  easy for OWLv2 and inflate the headline.
- The largest miss contributors are labels that are plausibly true: toy (ball→toy 59, detection 0.44), shirt
  (uniform→shirt 0.11, dress→shirt 0.37), wall, building, leash, uniform.
- 69% of missed swaps are "supported at the Look region", not "found elsewhere": the detector sees the swapped label
  at the very place.

**Visual look at missed swaps (16 random each; my observation, not a human label; `e10_sheets.py`):**

| Sample | Swap label visibly present / defensible | Ambiguous | Really wrong (true miss) |
|---|---|---|---|
| Cross-set (dock, shirt on a uniform, tree, window, cup, helmet …) | about 10 | about 3 | about 3 |
| Same-set clothing (life jacket→vest, coat→shirt underneath, wetsuit→suit …) | about 3 | about 7 | about 6 |
| Same-set non-clothing (canoe→boat, dirt bike→motorcycle …) | about 4 | about 4 | about 8 |

Consequences:
- "Not in the references" is not "absent". The cross-set half of the swap test mostly measures whether Check accepts
  things that are there, and penalises it for doing so.
- The hand-made `CONFUSION` sets contain non-exclusive pairs: `boat` is a hypernym of canoe/kayak/raft; `suit` collides
  with wetsuit→swimsuit; "life jacket" is a vest. In the vocabulary, "bike" maps to bicycle, so dirt bikes become
  false positives of bicycle (bicycle is the detector's most "confused" vehicle: within-set naming 0.82, 14 of 16
  errors say motorcycle).
- The **high-base subset** (n=627) is dominated by these cases: its top swap labels are shirt 102, toy 63, wall 35.
  S2's advantage there (0.59 vs 0.39) is largely S2 contradicting labels that are true; it is the same behaviour that
  gives S2 its 0.32 clothing FCR.

### 2.2 FCR is also contaminated, in the other direction
16 random non-clothing false contradictions of S3 at α=0.05 (my observation):
- 9 were real dogs (in water, snow, close-up, racing): true detector-threshold errors, caused by the uniform α.
- 3 were not false at all: "home plate" → plate, "hot dog" → dog, "toy car" → car. Claim extraction turns compounds
  into object claims and the references contain the same compound, so the claim counts as reference-confirmed.
- 4 were genuine detector misses on hard objects (flower field, painted wall, skate rail, tiny football).

Reference noise is measurable (`e5_strict.py`): for non-clothing claims, FCR is 0.093 when one reference mentions the
class and 0.046 when two or more do (S3, α=0.10). At image level, single-mention "positives" of rare classes look like
negatives (horse: 10% quantile −4.80 with one mention, −1.02 with ≥ 2; unmentioned median −4.49).

### 2.3 Where the detector is and is not the limit
Threshold-free separability on same-set swaps (`e8_naming.py`): P(true class scores above swapped class).

| Evidence | Same-set clothing (n=312) | Same-set non-clothing (n=849) |
|---|---|---|
| Image-max, raw | 0.766 | 0.948 |
| Look top-1 box, centred per class | 0.804 | 0.940 |
| Look top-3 max, centred | 0.817 | 0.945 |

Within-class AUROC (true claims of class c vs swaps relabelled c; `e1_anatomy.py`): dog 1.00, horse 0.98, motorcycle
0.99, dress 0.94, hat 0.92 — but shirt 0.71, wall 0.72, suit 0.57, sweater 0.53.

- **Non-clothing exclusive objects:** the detector is not the limit (≈0.95–0.99). Misses there come from how the score
  is read.
- **Upper-body garments, suit, wall-like surfaces:** the detector (and the label itself) is the limit. No rescaling
  can fix an AUROC of 0.53–0.71.

### 2.4 What S3 loses by how it reads the score
- **Uniform α.** Score distributions (`e4_dist.py`): dog positives have 5% quantile −0.18, while images without a dog
  have 99% quantile −1.25. A threshold anywhere in between costs nothing, yet S3 puts τ_dog at the 9% quantile and
  discards 6.6% of real dogs at thing-FCR 0.05. For rare classes the opposite happens: shrinkage pulls τ down to a
  kind-level raw quantile dominated by clothing, so horse/goat/truck swaps pass (detection 0.56/0.57/0.52 at α=0.09, thresholds fitted in-sample for this table).
- **Oracle check (in-sample, optimistic, `e3_calib.py`):** with the best per-class thresholds on the same image-max
  score, detection would be 0.80 at FCR 0.041 (S3: 0.47 at 0.05). The oracle exploits the swap label mix, so this is
  an upper bound, but it shows the loss is in threshold allocation, not in the evidence.
- **Absolute instead of relative.** For dog→{cat, horse, sheep, cow, goat} swaps, `dog` beats the swapped label at the
  swapped label's own best box in 90% of cases, yet 24% pass the absolute threshold.

### 2.5 The base rate sets what "good enough" means
The earlier visual audit of 24 proxy-wrong claims (`docs/generated/CHECK_FIX_B.md`) found 2 wrong, 15 correct, 5
unclear, 2 non-visual. 841 of 3242 thing claims are proxy-wrong, so the true error rate of thing claims is probably a
few percent. Precision of a contradiction = p·det / (p·det + (1−p)·FCR):

| True error rate p | S2-like (FCR 0.085, det 0.56) | FCR 0.05, det 0.60 | FCR 0.03, det 0.78 | FCR 0.02, det 0.85 |
|---|---|---|---|---|
| 0.03 | 0.17 | 0.27 | 0.45 | 0.57 |
| 0.05 | 0.26 | 0.39 | 0.58 | 0.69 |
| 0.10 | 0.42 | 0.57 | 0.74 | 0.83 |

At a low base rate, **FCR matters more than detection**. A Check with FCR ≈ 0.085 is mostly wrong when it contradicts,
whatever its detection. This is arithmetic, not a measurement; p is unknown until humans label.

## 3. Improvements, ranked (all inside the logic)

All rows: thing-FCR is the constraint; detection on all swaps, on the clean exclusive subset (same set, non-clothing,
boat set excluded, n=826), on same-set clothing (n=312), cross-set (n=1240). Source: `e11_final.py`.

| System | thing-FCR | clothing FCR | dog FCR | det all | det clean exclusive | det same-set clothing | det cross-set |
|---|---|---|---|---|---|---|---|
| S2 five-way z (default) | 0.117 | 0.320 | 0.028 | 0.559 | 0.557 | 0.606 | 0.559 |
| S3 no abstention, α=0.06 | 0.029 | 0.038 | 0.042 | 0.380 | 0.609 | 0.192 | 0.281 |
| S4a: cost-sensitive thresholds | 0.021 | 0.032 | 0.005 | 0.434 | 0.617 | 0.253 | 0.366 |
| S4b: + rival-aware evidence | 0.028 | 0.030 | 0.013 | 0.491 | 0.731 | 0.420 | 0.356 |
| S4c: + exemplar-refined queries | 0.025 | 0.019 | 0.015 | 0.523 | 0.783 | 0.494 | 0.364 |
| S3 no abstention, α=0.09 | 0.049 | 0.060 | 0.066 | 0.461 | 0.695 | 0.276 | 0.361 |
| S4a | 0.048 | 0.103 | 0.005 | 0.533 | 0.669 | 0.484 | 0.464 |
| S4b | 0.042 | 0.058 | 0.013 | 0.589 | 0.843 | 0.561 | 0.435 |
| S4c | 0.045 | 0.058 | 0.018 | 0.603 | 0.866 | 0.580 | 0.440 |

### I1. Cost-sensitive per-class thresholds instead of one α (largest, cheapest)
- **What.** τ_c = argmax_τ [ F_unmentioned,c(τ) − λ · F_positive,c(τ) ], one λ for all classes; λ is the single
  operating-point knob, chosen by the FCR target. F_positive is the image-max distribution over images where ≥ 2
  references mention c (falls back to any mention under 5 images), smoothed with a pooled centred shape (weight 10).
  F_unmentioned is the same score over images whose references do not mention c. `e6_np.np_tau`.
- **Why it is inside the logic.** Only OWLv2 score distributions are used. There is no class prior: λ is the same for
  every class, and neither claim frequency nor caption probability enters. Where a class cannot be separated (shirt),
  the rule automatically becomes conservative; where it can (dog), it stops sacrificing true claims. It replaces both
  the uniform α and the hand-set abstention list.
- **Evidence.** S3 → S4a at thing-FCR ≈ 0.05: detection 0.461 → 0.533, high-base 0.268 → 0.384; dog FCR 0.066 →
  0.005. Alone it raises clothing FCR (0.103) because it shifts budget to clothing; I2 corrects that.
- **Part of it:** using ≥ 2-mention positives alone gives +3 to +5 points at equal FCR (`e5_strict.py`).
- **Tried and rejected:** shrinking in a z-normalised space (worse: 0.39 at FCR 0.05), an LDA log-likelihood ratio
  with a global threshold (0.51 but clothing FCR 0.12), no shrinkage (0.46).

### I2. Rival-aware evidence: the claim only counts where OWLv2 does not prefer an exclusive name
- **What.** For a claim c with confusion set S: X_c = max over boxes b of [ l_c(b) − 2·max(0, max_{j∈S} l_j(b) − l_c(b)) ].
  The absence test uses X_c instead of the plain image-max. Classes without a set are unchanged. `e6_np.rival_X`.
- **Why.** "A horse is here" is weak evidence if the same box says "dog" more strongly. This is exactly the project's
  rule ("what the object at that place is, the detector decides"), applied to the evidence itself.
- **Evidence.** At thing-FCR ≈ 0.05: same-set swaps 0.57 → 0.72–0.74 with PA thresholds (`e2_rival.py`); with I1,
  clean exclusive 0.669 → 0.843 and clothing FCR back to 0.058. The penalty weight is not sensitive (1, 2, 4 give
  0.69–0.74 on same-set).
- **Needs:** the confusion sets cleaned (I5).

### I3. Substitute = centred argmax within the confusion set at Look's top-1 box
- **What.** Among set members above their own τ at Look's top-1 box, pick the one with the largest (logit − class
  median of positives), not (logit − τ) with the "beats the claim" condition.
- **Evidence** (`e11_final.py`, S4b at thing-FCR ≤ 0.05, 1414 detected swaps, 874 of them same-set):

| Substitute rule | Substitute = true class | True class recovered among detected same-set swaps |
|---|---|---|
| S3 rule (Look top-1, ≥ own τ, > claim) | 0.851 | 0.786 |
| Look top-3 max | 0.823 | 0.811 |
| **Look top-1, centred argmax** | **0.888** | **0.887** |
| Image-level best box of the claim (no Look) | 0.677 | 0.565 |
| Image-max (no Look) | 0.708 | 0.613 |

- **Why it matters for Look.** This is the first direct evidence that Look adds something Check cannot get from the
  image alone: +18 points of substitute precision. Caveat: in the swap test the region comes from the *correct* word,
  so this is favourable to Look (my earlier L1 is still open).
- **Measurement note.** "0.27 of swaps replaced with the true class" has a ceiling of 0.48, because only 48% of swaps
  have the true class in the swapped label's confusion set. Report recovery among detected same-set swaps instead.

### I4. Exemplar-refined queries from the same frozen detector (small, optional)
- **What.** Per class, the mean OWLv2 box embedding of the best box in ≥ 2-mention training images (a prototype);
  query = normalise(text query + 0.5·(prototype − mean prototype of its set rivals)). `e7b_queries.py`.
- **Evidence.** At thing-FCR ≈ 0.03: detection 0.52 → 0.56, same-set 0.67 → 0.74; at 0.05 the gain is ≈ +1–2 points.
  Mean within-class swap AUROC only 0.844 → 0.855. Pure prototypes without text are worse (0.774).
- **Judgement.** It is OWLv2's own image-conditioned mode, so it is inside the logic, but the exemplars are selected
  by reference mention. Keep it as an ablation row, not in the main method.

### I5. Clean the confusion sets and the claim vocabulary (zero-cost correctness)
- Remove hypernyms from exclusive sets: `boat` out of {canoe, kayak, raft}; decide suit/swimsuit ("wetsuit");
  jacket/vest ("life jacket"/"life vest").
- Stop "dirt bike / motorbike" surface forms from landing in `bicycle`.
- Block compounds in claim extraction: "home plate", "hot dog", "toy car", "tank top".
- These were found by looking at false contradictions; they affect both FCR and the swap test.

### Tested, no gain — do not spend more time here
| Idea | Result | Script |
|---|---|---|
| Text prompt variants (grammatical, 7-template ensemble, sense-specific names, "a person wearing …") | Within-class AUROC 0.844–0.845 for all; "wearing" is worse (0.830) | `e7b_queries.py` |
| **Crop re-detection of the attended box** (same OWLv2, +10% pad, 714 same-set swaps, CPU) | Pairwise accuracy, clothing: cached Look box 0.785, crop 0.747, mean of both 0.772. Non-clothing 0.930 vs 0.928. Crop right/box wrong 20, box right/crop wrong 32 (clothing) | `e9_crop.py`, `e13_crop_eval.py` |
| Region-level contradiction ("claim exists elsewhere but the attended box is a rival") | +1–2 points detection for +0.2–0.5 points FCR | `e11_final.py` |
| More Look boxes for the substitute (top-3) | Lower precision than top-1 (0.823 vs 0.888) | `e11_final.py` |

The crop result is a pilot with one geometry; it does not prove no crop scheme can help. But it removes the main
reason to run the GPU job: the cached box embeddings already carry what the detector knows about the garment.

## 4. What is realistically reachable with a frozen OWLv2

- **Mutually exclusive, visually distinct objects** (animals, vehicles, sports objects, furniture): about 0.85–0.9
  detection at 3–5% FCR, substitutes right about 0.9 of the time. Measured: 0.78–0.87 on the clean subset, and part
  of the remainder is label noise (hypernyms, "bike").
- **Clothing type** (shirt/jacket/sweater/vest/suit): about 0.5–0.6 detection at 5% FCR, and I see no route to much
  more with this detector. A third of the test cases are themselves debatable. The honest design is to treat upper-body
  garment type as low-confidence: contradict only on a large margin, never "remove".
- **Co-occurring / generic labels** (toy, wall, building, leash, shirt on a clothed person): Check will and should
  keep most of them, because they are usually there. This is not a failure.
- **Overall headline on the current swap test:** about 0.60 at 5% FCR. It cannot go far beyond that, because roughly
  half of the remaining "misses" are not errors.
- **False contradictions:** a floor of roughly 2–3% from small/atypical true objects, reference noise and extraction
  errors. With I1+I2 the measured thing-FCR at λ=4 is 0.028 with detection 0.49 (clean exclusive 0.73).

## 5. Are the instruments measuring the right thing?

| Instrument | Verdict | Fix |
|---|---|---|
| FCR on reference-confirmed claims | Right quantity, and the one that matters most at a low base rate. Contaminated by compound-word claims and single-annotator mentions | Report it on ≥ 2-mention claims as well; fix extraction (I5) |
| Swap test, all 2401 | **Misleading as a headline.** 52% of labels are not exclusive with the true class and are often present; 31% are easy dog→animal swaps; the swap direction (common true class → rarer label) is the reverse of how a captioner hallucinates | Report three rows: clean exclusive (n=826), same-set clothing (n=312), cross-set (descriptive only) |
| "High-base" subset | Does not isolate class-prior gain; it is mostly shirt/toy/wall labels that are true | Drop as a criterion |
| "Replaced with true class / all swaps" | Ceiling 0.48 by construction | Substitute precision and recovery among detected same-set swaps |
| Proxy contradicted precision / J | Most proxy-wrong claims are correct; cannot rank systems | Descriptive only |
| Matched-FCR criterion against S2 | S2's FCR is 0.32 on clothing; matching the pooled 0.085 rewards contradicting true clothing | Compare at a fixed target (0.03 and 0.05), per stratum |

Still unmeasured, and only humans can measure it: the true error rate, and whether real hallucinations look like the
clean exclusive swaps (dog vs cat) or like the clothing ones. The class mix of proxy-wrong claims (shirt 173, jacket
55, toy 43, uniform 30 of 841) correlates far better with claim frequency (Spearman 0.82) than with swap labels
(0.36), which suggests real disputes are concentrated in clothing — exactly where the detector is weakest and where
most proxy-wrong claims are in fact correct.

## 6. Recommended next step

1. **Pre-register and run S4 = I1 + I2 + I3 with I5's set/vocabulary cleaning** (CPU only, cached features, no new
   model). Criterion at a fixed thing-FCR target of 0.03 (secondary 0.05): detection on the clean exclusive subset,
   clothing FCR, substitute precision. Exemplar queries (I4) and S2/S3 as ablation rows. The numbers in this review
   were explored on the same B folds, so the pre-registered run should state that and keep validation for the single
   confirmation.
2. **Do not run the GPU crop pilot** unless someone has a different crop design; the CPU pilot shows no gain.
3. **Human labels, about 300 B claims,** stratified: S4 contradicted (all), S2 contradicted / S4 keep in clothing,
   and a random sample of kept claims for the base rate. This fixes p in the table of §2.5 and therefore the FCR
   target. Without it, no further threshold choice is justified.
4. Look: only the open L1 measurement (attention on a wrong word) matters for Check now; map work will not pay.
