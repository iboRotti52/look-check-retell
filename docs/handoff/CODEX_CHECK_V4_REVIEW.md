# Codex: review Check v4 (the main Check) (2026-11-14, Claude)

**User decision:** Check stays within its original logic. The frozen OWLv2 detector looks at the region the
captioner attended to (Look). It asks whether the claim is there, or whether something that could replace it is
there. Improvements are welcome within that logic; replacing it with a different checker is not.

You looked at Look; now look at Check. Review scope is open: the list below is directions and suspicions, not
limits. If needed, try other methods within the logic, fix things, rerun. Only the AGENTS.md rules are binding.
If a fix breaks the lock or the "validation read once" rule, write it down openly.

## Result (validation, automatic reference proxy)
Table: `docs/generated/CHECK_V4_METRICS.md`.
- **Chosen config:** Q1 C2 A1 R2 G0 E0, meaning a per-class label-free z-scale plus the r-weighted top 3 of Look's
  proposals.
- **Against the live original rule (0.2/0.05):**
  - contradicted precision: 0.440 → 0.523
  - wrong-claim recall: 0.219 → 0.287
  - J: the difference's CI includes zero
- **Stuff is never contradicted.**
- **Learned verifiers (v2/v3, comparison only) are much stronger.**
- **Details:** `docs/CHECK_V4.md`. Protocol: `docs/CHECK_V4_PROTOKOL.md`. §5 is the amendment made before selection.

## Files
- `scripts/35_check_v4.py`: features, rule, threshold fitting, CV, lock, report and records.
- `scripts/36_check_v4_table.py`: the table.
- `tests/test_check_v4.py`: tests.
- `results/check_v4_selection.json` (locked), `results/check_v4.json`.
- `data/cache/pipeline/A_{B,validation}_look4_check4.json`: records.

## What I especially want looked at
1. **The §5 amendment.** Joint J-max thresholds, τs shared by claim and alternative. Is it fair and in keeping with
   the original rule?
   - Would a separate τalt be more correct? If the alternative must be "present", its threshold could differ from
     the claim's.
   - Does choosing the J criterion together with the precision constraint ≥ 0.5 bias anything?
2. **C2 calibration.** μ_c and σ_c come from per-image max logits on half-B images. Same split as the selection,
   with no labels.
   - Is per-image max the right reference distribution?
   - Literature: standardising open-vocabulary detector scores per class.
3. **R2 soft region.** Top 3 from `look.select_frozen`'s r(b) with clipped weights. Are m=3 and the weighting
   reasonable? Do the C2+R2 synergy findings (each weak alone) hold up?
4. **Alternative set.** A1 (original, person allowed) won. Did the containing-object false alarms in my diagnosis
   (shirt→person, grass→dog) drop with C2+R2? For Retell's sake, measure how often the alternative appears in the
   references (`*_check4.json`).
5. **Stuff.** It is never contradicted. Is that the right outcome, or is it a weakness of the stuff region/score (box
   proposals vs. regions)? Can it be improved within the logic, for example with a mask/segment region?
6. **The gap to learned verifiers.** AUROC is 0.715 vs 0.826/0.862. What does it mean for H6, and how should it be
   written into the report?
7. **Live pipeline.** `lcr/pipeline.py` still produces the original rule (0.2/0.05). It must be moved over before
   the test evaluation: the per-class μ/σ file, the soft region, the thresholds.
8. **Look.** My audit found it faithful to the original logic (person SAM box + nearest OWLv2 proposal is the only
   drift). Do you agree?
