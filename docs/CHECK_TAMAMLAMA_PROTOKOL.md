# Completing Check: pre-registered protocol (2026-11-15, Claude)

**Basis:** `docs/ILK_ILKELER_TASARIM.md` §3. The user's instruction was "complete Check".

This file was written **before the code was run**. Only B is used; validation and test are not read. There is **no
configuration search**: the design below is fixed in advance, and the B numbers are descriptive.

## Fixed parts (carried over from Codex's LCR)
- **Look:** the 4 heads, step p−1, the r(b) rule. Top 3 with no minimum-side filter (min_patches 0, alpha 0.5).
  This is the selection in Codex's `results/lcr_bundle.json`.
- **Pooling:** the max over the three boxes, for each class.
- **Score scale:** the per-class label-free z-scale, from the per-image max OWLv2 logit.
- **What is in the region is decided by the detector:** OWLv2's highest-scoring class across the whole vocabulary.
  The caption only says which word to check.

## New parts
1. **Image-level veto.**
   - s_img is the claim's z-score at the best proposal in the whole image.
   - If s_img ≥ τs, the claim is **never contradicted**. Its verdict is `uncertain` and its action `keep_elsewhere`
     (the word is kept and the case is logged as a Look miss).
   - "Supported" stays region-only (s_here ≥ τs). The veto only blocks the delete/replace decisions.
   - The same presence standard (τs) applies to the claim here, the claim in the image, and the alternative.
2. **Splitting the contradiction into two actions.** The original condition stays: s_here < τc and the strongest other
   class in the region (any class, A1) ≥ τs.
   - The highest-scoring **same-group** class in the region (person excluded) reaches ≥ τs → `replace(substitute)`.
   - Otherwise → `remove`: something else is in the region and the claim is nowhere in the image.
3. **Five actions:**

   | Action | 3-way verdict |
   |---|---|
   | keep | supported |
   | keep_elsewhere | uncertain |
   | replace | contradicted |
   | remove | contradicted |
   | skip | uncertain |

   The 3-way `verdict` field is kept so existing consumers don't break.
4. **"Camera" idiom.**
   - A `camera` claim whose draft contains `(at|to|toward|towards|from|for|into) the camera` is not an object claim.
   - It gets verdict `uncertain`, action `skip`, reason `non_visual_idiom`, and is left out of threshold fitting.
5. **Thresholds and cross-fitting.**
   - Per kind (thing/stuff), choose (τs, τc) to maximise J, subject to contradicted precision ≥ 0.5 (at least 20
     verdicts). The veto is part of the rule during fitting.
   - Evaluation uses all of B with 5-fold image cross-fitting (`calibrate.fold_of`): μ/σ and thresholds come from
     the training folds, verdicts are made on the held-out fold.
   - The final (live) bundle fits μ/σ and thresholds on all of B.
   - The only things the labels feed into the final bundle are 4 thresholds (the μ/σ are label-free). This label
     leak into Retell training is negligible, which is why Codex's B_cal holdout (20% of the data) is dropped.

## Reported (descriptive only; there is no selection)
- **Systems:**
  - S0: Codex's stored LCR verdicts, with thresholds from B_cal.
  - S1: Codex's rule with cross-fitted thresholds.
  - S2: completed Check.
- **Main comparison: on B_fit images,** because S0 was fit on B_cal. All of B is also reported for S1 and S2.
- **Metrics:** proxy precision/recall/J, and the decision rates.
- **S2 action breakdown:**
  - for replace, how often the substitute appears in the references;
  - for remove, how often the references contain no same-group class (deletion fits).
- **Audited items:** S2's actions on the 24 claims in Claude's visual audit, next to my visual judgments. This is
  descriptive and not human-labelled.

**Expected and stated in advance:** the veto may lower proxy J. In the visual audit, the proxy labelled as wrong
claims that were present but unmentioned, and the veto keeps exactly those. Whether the veto stays is decided not by
proxy J but by the human pilot. Until then the veto is the design choice (first principles: failing to find the object
in one region is not evidence of absence).

## Outputs
- `scripts/42_check_complete.py`
- `results/lcr_check_bundle.json` (the new default bundle)
- `results/check_complete_B.json`, `docs/generated/CHECK_COMPLETE_B.md`
- `data/cache/pipeline/A_B_lcr2.json`
- Historical files stay: `results/lcr_bundle.json`, `A_B_lcr.json`.
