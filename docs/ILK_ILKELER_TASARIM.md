# Look–Check–Retell from first principles, and a review of Codex's unified pipeline (2026-11-14, Claude)

**User's request:** "Review what Codex did; it doesn't have to be right. Then think from first principles: what do
we want? We look at the attention, then check whether the object there is right; if it's wrong, we pass that
information back and have the model regenerate or fix the caption."

**Numbers** come from `docs/generated/FIRST_PRINCIPLES_B.md` (`scripts/review_first_principles_B.py`) and
`docs/generated/LCR_DEVELOPMENT.md` (Codex). B only; validation and test were not read.

**Visual audit** (`results/claude_visual_audit_B.json`): these are my own observations, **not human labels**, and
the samples are small.

## 0. Summary
1. **The biggest problem is not in Look/Check/Retell; it's the label.**
   - The "wrong" label from the reference proxy ("did any of the 5 annotators mention it?") is mostly wrong in the
     audited samples: of 12 random proxy-wrong claims, 6 are clearly correct, 2 are the "looking at the camera"
     idiom, and 4 are unclear.
   - Every selection we made (v2/v3/v4 and Codex's LCR) maximised J against this label, which in practice means
     **predicting what gets mentioned (salience)**. The learned verifiers win for exactly this reason (language prior,
     captioner consistency).
   - The human pilot is not an extra gate. Without it, it is not possible to measure whether the Check
     question ("is the object there?") is answered correctly.
2. **Codex's fixes are correct and important:**
   - the small-box filter: top-3 accuracy on small objects 0.395 → 0.833 on B_fit;
   - the bug in my v4 → Retell connection: the verdicts were being rewritten;
   - a single shared implementation.

   I disagree on three points (§1).
3. **From first principles, Check must answer three questions, not one:**
   - Is the object at the attended place?
   - If not, is it anywhere in the image?
   - What is at the attended place?

   The original rule only says "contradicted" when an alternative exists. But in the B diagnostic, **only half of the
   wrong claims have a same-group substitute in the references; the other half need deletion.**
4. **The training-free version of Retell is "pass the information back and regenerate" in its most literal form:**
   - the same captioner, with the wrong word banned;
   - only the part after the error is regenerated;
   - a soft push toward the substitute where there is one.

   Learned Retell should come after that and has to beat it.

## 1. Review of Codex's work

### Correct and important
- **Small-box filter.**
  - The two-patch minimum side length was dropping candidates before attention ranking, so small objects never
    entered the pool.
  - On B_fit human boxes, after removing it: first box 0.805 → 0.830, any of top 3 0.900 → 0.933, small objects
    0.395 → 0.833.
  - This is a genuine Look drift that my audit missed. The fix stays within the logic.
- **Contract bug (my mistake).**
  - `35_check_v4.py` wrote the new alternative name next to the old `s_alt`.
  - `finalise` recomputed the v4 verdicts with the old raw rule: 669 contradicted → 292 preserved.
  - Codex fixed it with a single `RegionChecker` class. Live, cache and synthetic-corruption paths share the same
    code, with a provenance bundle and tests. Its final verify log matches the current bundle (`4c0481274067b130`).
- **Max pooling.** Support needs strong evidence in one box; contradiction needs the claim to be low in all boxes.
  This matches the logic that "failing to find it in one place is not evidence of absence".
- **Separate calibration images.** It is right that verdicts used to train Retell should not be fit with those same
  images' labels.
- **Look time step.** I checked: the attention is taken at step p−1, the step that produces the word
  (`lcr/look_sources.py`). There is no off-by-one shift.

### Disagreements and open problems
1. **The threshold point is aggressive and was fit on little data.**
   - Thing thresholds are τc = τs = 1.31 (z), so for objects there is **no uncertain band**: any low claim plus any
     strong alternative becomes contradicted.
   - B_cal is only 635 images. On B_fit, which was not used to fit these thresholds, contradicted precision is
     0.465, below the 0.5 target.
   - Better: cross-fitting across all of B. Thresholds for each fold come from the other folds, so it avoids leakage
     just like holding out B_cal, but uses five times as much data.
2. **Dropping the image-level score entirely.**
   - The concern for H6's cleanliness is justified as far as Retell's free feature goes.
   - But from first principles, the image score is the right **veto** inside Check's decision: of correct claims,
     13–16% are not in Look's region but are detected elsewhere in the image (§3 of the table).
   - Proposal: use the image score only to block "delete/replace", never as a feature that says "keep".
3. **Contradicted still requires an alternative.** About half of wrong claims have no substitute (§2.2). Insertion
   (co-occurrence) errors never reach an actionable verdict and stay uncertain.
4. **Still fit against the proxy.** This is my error too (v4 selection). Given §2.1, the J ordering may partly
   reflect salience.
5. **GPU crop pilot (re-showing attention crops to OWLv2):** reasonable, but low priority (§3, Look).
6. **Report sections.** Codex updated `report/sections/3_design.tex`, `4_experiments.tex` and `5_conclusion.tex`.
   The numbers come from macros (`report/generated/lcr_numbers.tex`). I did not review the text line by line; that
   should happen before submission.

## 2. What the data say (B)

### 2.1 The label
- **Detector-confident sample:** of 9 randomly chosen proxy-wrong claims with OWLv2 image probability ≥ 0.5, I judged
  8 correct (shorts, ball, pole, field, newspaper, house, bat) and 1 wrong (surfboard, a jet ski).
- **Random sample:** of 12 random proxy-wrong claims: 6 correct, 2 "at the camera" idioms, 4 unclear.
- **Detector view:** at probability ≥ 0.3, OWLv2 detects the object somewhere in the image for 42% of proxy-wrong
  claims (82% at ≥ 0.1).
- **Reading:** this fits proxy noise better than detector error. Annotators leave out clothing and backgrounds;
  clothing makes up a large share of the wrong claims (`dress, uniform, suit, bikini…`).
- **Consequence:** the precision ceiling for "contradicted" is set by the proxy. Even a perfect presence detector
  would score low against this label, which explains why stuff and clothing never reach precision 0.5.

### 2.2 Action
- **Substitute or delete:** for proxy-wrong claims, a same-group class is in the references for 0.506 overall (objects
  0.452, ground 0.683, clothing 0.500). The other half has no substitute, so the right action is deletion or
  rephrasing.
- **Absent everywhere:** of 192 claims found nowhere in the image (prob < 0.1), only 0.401 have a possible
  substitute. These are typical insertion (co-occurrence) errors, the kind LURE reports.

### 2.3 Alternative quality
- **Whether the suggestion appears in the references:** A1 0.290, A3 0.177. Restricting to the same group lowers the
  rate.
- **When a substitute exists:** the gap narrows (0.386 vs 0.350).
- **What the region holds:** the attention region often contains something other than the substitute (the context
  or the container that triggered the word).

## 3. From first principles: what should each piece be?

**Goal:** for every object word, ask whether it was said because the model saw something real. If yes, keep it. If
not, tell the captioner and have it write a caption without the error.

### Look: "what was the model looking at when it said this word?"
- **What it is uniquely for:**
  - The **substitute**: if the model looked at a dog and said "cat", the thing it looked at is the fix.
  - **Instance specificity**: telling apart which object or which person the word refers to.
- **What it is not needed for:** presence alone, which the detector can search for across the whole image. That is
  why the v3 crop features added nothing to verification. Look's value comes into play in Retell's correction.
- **Keep:** the 4 heads, the p−1 step, the r(b) rule, top 3 without the size filter (Codex's fix).
- **Add (soft, not a hard gate):** a Look reliability signal (excess mass, peakedness). As a hard gate it lowered J
  (v4 E1/E2), but Retell can use it as "how far to trust the region".
- **Crop re-detection:** do it only if, after human labels, small-object misses still remain. OWLv2 proposal
  embeddings are already region-specific, and the failure we found was the filter.

### Check: three questions, five outcomes
Every score is the frozen OWLv2 on calibrated class scores. No language prior or second verifier.

| Question | Signal | Outcome |
|---|---|---|
| Is the claim at the attended place? | s_here = max over Look top 3 | strong → **supported** |
| If not, is it elsewhere in the image? | s_img = image max | strong → **present_elsewhere**: keep the word; record it as a Look miss |
| If it's nowhere, what is at the attended place? | same-group class in the region (A3) | strong → **replace(alt)** |
| | none compatible, claim absent in the image | → **remove** (insertion error) |
| Other | weak evidence | → **uncertain**: Retell does not touch it |

- **Same rule form:** "the claim is there / something else is there instead". The original rule covered
  supported/contradicted/uncertain.
- **Two additions:**
  - The image-level veto. It was in the original record (`s_image`) and follows the rule's own reasoning.
  - An explicit remove outcome. In the original it was hidden inside uncertain.
- **Principle (user, 2026-11-14): what is at the attended place is decided by the object detector, not by the
  caption.**
  - The caption only tells us which word to check.
  - The object in the region is OWLv2's highest-scoring class across the whole vocabulary, independent of the claim.
  - Same-group compatibility only decides whether the detector's answer is offered as a "substitute" or the word is
    "deleted". It never changes what the detector reports.
  - The old live rule (`pipeline.region_record`) added the captioner's 3 alternatives (`blip_alts`) to the candidate
    set. That leak is gone in v4 and in Codex's `region_evidence`: candidates are all classes, scored by the
    detector.
  - The remaining use is the synthetic corruption (`lcr/evidence.py`), which draws the wrong word from `blip_alts`.
    That generates an error, not evidence about the region.
- **Two roles for the alternative:**
  - "Something else is here" is contradiction evidence (any class, A1).
  - "What should replace it" is the suggestion (same group, A3).

  Codex tested A3 for both roles at once and A3 lost on J. Separating them is my suggestion.
- **Thresholds:** fit against human labels. Until labels exist, use cross-fitting against the proxy, and do not let
  any J ordering below a few points drive the decision.
- **Small fix:** "at/to/from the camera" is not an object claim (21 of the proxy-wrong claims on B).

### Retell: "pass the information back and regenerate"

**R0 (main, training-free): the same captioner (decoder_A), constrained regeneration.**
- **remove:** ban the word and its vocabulary synonyms with `bad_words_ids`.
- **replace:** ban the wrong word and add a soft logit bias toward the substitute's tokens. Forcing it outright
  (Anderson 2017, constrained beam) risks phrases like "a dog with a dog", so the soft push is safer.
- **Edit, not rewrite:** keep the draft up to the first flagged word and regenerate only the rest. This keeps the
  correct parts and limits drift. The current Retell rewrites from scratch (LITERATÜR §12a).
- **supported / present_elsewhere / uncertain:** do nothing.

**Literature, all already in refs.bib:**
- Anderson et al. 2017 (constrained beam).
- HALC (`chen2024halc`): corrects object tokens during decoding using detector-localised visual context. The closest
  analogue to our idea.
- Woodpecker (`yin2024woodpecker`): a pipeline of key concept → detector validation → correction.
- LURE (`zhou2024analyzing`): hallucinations come from co-occurrence and uncertainty, which supports the delete
  action.
- Our difference: verification is tied to **the place the captioner itself attended**, and the correction is made by
  **the captioner itself**.

**R1 (learned, later):** gated BLIP + per-box evidence tokens (Codex's interface is ready). It only makes sense if it
beats R0. Its training must cover:
- deletion, as well as substitution;
- insertion errors in the synthetic corruption, not only label swaps: for example, adding a co-occurring object.

### Measurement
- **Main label:** human claim-level presence labels.
  - Development: about 300–500 claims on B, stratified by thing/stuff/clothing and by detector confidence.
  - Reporting: on validation.
  - The tool: `annotation/pilot/annotate.html`.
- **Proxy:** a secondary row only.
- **Check:** precision/recall of the remove and replace decisions against human labels.
- **Retell:** human hallucination rate (CHAIR-like) before and after correction, plus CIDEr to make sure fluency
  holds.

## 4. Proposed order
1. **Human pilot on B.**
   - Labelling presence per claim is enough; the question is "is it in the image?"
   - Until it is done, the proxy stays secondary; no Check change is chosen by proxy J alone.
2. **Check outcomes.**
   - Add the five outcomes on top of Codex's `RegionChecker` (the image veto and remove).
   - Switch the thresholds to cross-fitting.
   - Add the camera idiom filter.
   - All of this is light CPU work.
3. **R0 Retell pilot.** Constrained regeneration with decoder_A on a few hundred B drafts. CPU is possible; for the
   full run, a Colab T4.
4. **Learned Retell (R1)** only if it beats R0. **The crop GPU pilot** only if human labels show small-object misses.

## 5. Response to Codex's Retell research note (2026-11-15)

Note: `docs/handoff/RETELL_ARASTIRMA_NOTLARI.md`. The literature summary and code observations hold up. I checked
these directly:

- **The five Check actions do not reach the learned Retell.** `record_features` encodes only the 3-way verdict, so
  replace and remove are both `contradicted`. `action`, `substitute` and `pos` are not encoded.
- **`gate_ff` acts even when there is no evidence**, so an open gate is not proof that the evidence is being used.
- **`force_words_ids` does not work out of the box in transformers 5.23.** Constrained beam is routed to a remote
  community repository that needs `trust_remote_code`. We are not using it.
- **`Captioner.generate` builds its own BOS start**, so prefix continuation needs a small wrapper.
- **The CausalLens (CVPR 2026) citation was verified** on the CVF proceedings page. The author repository was not
  verified.

**What this changes in the R0 design** (§3 "Retell" is updated accordingly):

1. **Our own LogitsProcessor instead of `bad_words_ids` / constrained beam.**
   - Ban: block all WordPiece sequences of the word's forms (singular/plural), only from the restart point onward.
     A second instance earlier in the caption is not touched.
   - Replace: a small per-step bias on the substitute's first token. No forcing, and no remote code.
2. **Restart point: the start of the noun phrase, not the noun.** Back up to the closest article or preposition
   before the wrong noun, so the adjectives and the "holding" part get regenerated too. The verb/relation problem
   (HELPD, snowballing) is not fully solved; that will be written down as a limitation.
3. **Hard intervention only where Check is confident.**
   - On the proxy, about half of `remove` decisions are proxy-correct claims.
   - Following Codex's expected-value framing (p × benefit − (1−p) × harm), the confidence threshold for `remove` and
     `replace` is set with the human pilot. Until then they are run only as a B pilot, and the outputs are inspected
     by eye.
4. **Measurement has four dimensions:** the error is fixed / the correct content is kept / no new error is added /
   enough information is left (Codex's split; no CHAIR-style single number).
5. **Prerequisites for learned R1:**
   - add `action`, `substitute`/`alt_any` and `pos` to the evidence token;
   - feed out-of-fold Check decisions into training. `A_B_lcr2.json` holds in-sample decisions; script 42 already
     computes OOF actions but does not write them to the records.
