# Check v4: the original rule, improved within its own logic. Results and interpretation (2026-11-14, Claude)

> Historical frozen experiment. The current attention-object path is documented in `docs/LCR.md`.
> Subsequent B-only review found a small-box selection failure and fixed the score/record plumbing;
> the old validation numbers below are preserved and do not describe the new path. The gap to learned
> verifiers is not an established upper bound on region evidence. Retell is currently deferred.

**This was the Check v4 experiment.** It keeps the project's definition: the frozen OWLv2 detector looks at the region the
captioner attended to (Look v4). It asks whether the claimed object is there, or whether something that could
replace it is there. There is no language prior, no captioner signal and no second verifier. Check v2/v3 (learned
verifiers) are only comparison rows.

- **Protocol** (written before running): `docs/CHECK_V4_PROTOKOL.md`. §5 holds the threshold-policy amendment,
  made before selection.
- **Numbers:** all from the automatic table `docs/generated/CHECK_V4_METRICS.md` (`scripts/36_check_v4_table.py`).
- **Sources:**
  - `results/check_v4_selection.json`: B, locked.
  - `results/check_v4.json`: validation, read once.
- **Label:** automatic reference proxy, not a human label. Evaluation covers non-person claims only.

## 1. What was done
- **Script:** `scripts/35_check_v4.py`, CPU, about 2 minutes.
- **Grid:** 144 configurations, built from six components that all stay inside the rule:
  - Q: query text
  - C: label-free per-class score scale
  - A: alternative set
  - R: region (single box / r-weighted top 3)
  - G: image guard
  - E: attention gate
- **Rule form unchanged:**
  - supported ⇔ s(c) ≥ τs
  - contradicted ⇔ s(c) < τc and s(a) ≥ τs
  - everything else is uncertain
- **Thresholds:** per kind, on B. The (τs, τc) pair maximises three-way J, subject to contradicted precision ≥ 0.5.
- **Selection:** 5-fold image cross-validation on B.
- **Validation:** read once. Thresholds were refit on all of B.
- **Comparisons:** paired image bootstrap (2,000 draws) against the live original rule (0.2/0.05).
- **Region index check:** the region indices recomputed from the Look v4 maps matched the records exactly (0
  mismatches on B and validation). Look was not touched.

## 2. Amendment (protocol §5)
- **The problem:** the first threshold policy took τs as the threshold reaching 0.95 supported precision. Because
  the original rule tests the alternative against the same τs, contradicted almost never fired. With default
  components on B, wrong-claim recall was 0.001.
- **Why it mattered:** the "tuned original rule" baseline would have come out worse than the live one, so the
  improvement comparison would have meant nothing.
- **The fix:** choose (τs, τc) jointly to maximise J, and make the selection criterion J as well.
- **Disclosure:** this was seen in a smoke test of 4 configurations, before selection, and is written into the
  protocol.

## 3. Selection on B
- **Chosen: Q1 C2 A1 R2 G0 E0.** Two components change from the default:
  - per-class label-free z-scale: z = (logit − μ_c)/σ_c, using per-image max logits on B;
  - soft region: the r(b)-weighted score of Look's top 3 proposals.
- **Cross-validated result:** J 0.473 (default 0.341). Contradicted precision 0.480, recall 0.292.
- **The two components only help together.**
  - C2 alone: 0.312.
  - R2 alone: 0.389.
  - Together: 0.473.
  - Per-class scaling makes the region score comparable across classes. The soft region removes the noise of a
    single box. A one-at-a-time ablation would have missed this, which is why the full grid was run.
- **Other single-component effects:**
  - G1 (image guard) is the strongest single change: 0.434.
  - A3 (substitute group): 0.360.
  - Q2 (prompt ensemble): 0.348, no effect, as in CHECK_V2 §2c.
  - E1/E2 (attention gate): both lower J.
- **My diagnosis that containing objects (shirt→person, grass→dog) count as alternatives was right, but restricting
  the set did not win.** Once the scores are calibrated, narrowing the alternative set (A2/A3) costs recall.

## 4. Validation (once)
| | live original | v4 | v4 − live (95% CI) |
|---|---|---|---|
| contradicted precision | 0.440 | 0.523 | +0.083 [+0.007, +0.157] |
| wrong-claim recall | 0.219 | 0.287 | +0.068 [+0.015, +0.120] |
| J | 0.481 | 0.553 | +0.072 [−0.021, +0.159] |

**By group:**
- **Objects:** recall 0.242 → 0.358, precision 0.483 → 0.551.
- **Clothing:** recall 0.239 → 0.420, precision about the same (0.516 → 0.504).
- **Ground (stuff):** v4 gives **no contradicted verdicts**, because no threshold on B reaches 0.5 precision. The
  live rule did contradict stuff, at only 0.294 precision; those cases are now uncertain. This follows the rule's
  own reasoning: "failing to find it in a single region is not evidence of absence."

**At other targets:**
- At 0.6 and 0.7 precision, region evidence alone cannot reach the target, so no contradicted verdicts are given.
- The tuned original rule (raw scores) gives no contradictions even at 0.5.

**Continuous scale:** AUROC of the calibrated region score is 0.715.

## 5. Interpretation and caveats
1. **The improvement inside the logic is real but limited.** Precision and recall gains have intervals above zero;
   the J gain's interval includes zero.
2. **Learned verifiers are much stronger** (J 0.959 / 1.131 vs 0.553; AUROC 0.826 / 0.862 vs 0.715). The gap
   reflects information the region evidence cannot carry: the language prior and captioner consistency. It is an
   H6-type finding: the region-based detector check is weaker than verifiers that do not look at the region. It
   should be written into the report as a negative result.
3. **What goes to Retell:**
   - The main Check (v4) gives Retell verdicts plus the alternative.
   - Stuff claims are never contradicted, so Retell will not change ground words.
   - Whether to use a learned verifier for Retell is a separate decision, recorded as a comparison/ablation row.
4. **Records:** `data/cache/pipeline/A_{B,validation}_look4_check4.json`.
   - On B: per-fold cross-validated verdicts.
   - On validation: thresholds from all of B.
   - Person claims keep the live record (presence only).
5. **Still to do:**
   - Move v4 into the live pipeline (`lcr/pipeline.py`). This includes persisting the per-class μ/σ
     calibration file.
   - Measure how often the alternative appears in the references.
   - Check against the human pilot.
