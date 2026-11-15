# Completing Check: results and interpretation (2026-11-15, Claude)

- **Protocol** (written before running): `docs/CHECK_TAMAMLAMA_PROTOKOL.md`.
- **Numbers:** `docs/generated/CHECK_COMPLETE_B.md` (`scripts/42_check_complete.py`) and
  `results/check_complete_B.json`.
- **Scope:** B only; no selection was made, validation and test were not read.
- **Labels:** automatic reference proxy. The audit columns are my visual observations, not human labels.

## What changed
- **Code:** `lcr/region_evidence.py` gains a five-way decision (`decide5`) alongside Codex's shared `RegionChecker`.
- **Behaviour of old bundles:** unchanged. The `decision` key is missing in them, so they fall back to the original
  3-way rule.
- **New default bundle:** `results/lcr_check_bundle.json` (`15fdb954e7de71c6`). Live inference (`pipeline.run`,
  `region_rule="lcr"`), synthetic corruption and the verify script now use it.
  - Historical bundle: `results/lcr_bundle.json`, Codex's, with B_cal thresholds (script 37 writes only that).
- **Records:** `data/cache/pipeline/A_B_lcr2.json`. Each record adds these fields:
  - `action`: keep / keep_elsewhere / replace / remove / skip;
  - `substitute`;
  - `alt_any`: what the detector sees in the region;
  - `s_image_veto`.

  The 3-way `verdict` stays. The image-level score is not passed to Retell as a feature (`s_image = 0`); it is used
  only for the veto.
- **Camera idiom:** a "camera" claim with the "at/to/from the camera" idiom is `skip` (34 claims on B). Helper:
  `claims.non_visual`.
- **Thresholds:** cross-fitted on all of B for evaluation. The final bundle is fit on all of B, so the B_cal holdout
  is no longer needed.
- **Verification:**
  - `scripts/39_verify_lcr.py --live`: across 3,000 B images, every region score was recomputed independently, the
    stored verdicts and actions are preserved, and the fixed-draft live replay gives identical results (drift 0).
    Output: `results/lcr2_contract.json`.
  - New tests: `tests/test_check_complete.py`.

## Results (B, proxy)
**Systems (B_fit images):**

| System | J | Contradicted precision | Recall |
|---|---|---|---|
| S0 Codex (B_cal thresholds) | 0.512 | 0.465 | 0.342 |
| S1 Codex rule, cross-fitted | 0.482 | 0.484 | 0.308 |
| S2 completed | 0.458 | 0.488 | 0.257 |

- **S0 vs S1:** S0's higher J comes from its operating point. With τc = τs it gives more contradictions and drops
  below the 0.5 precision target (0.465). The cross-fitted thresholds meet the target.
- **S2 vs S1:** as stated in advance, the veto lowers proxy J; recall drops by 5 points.

**S2 actions (all of B, out-of-fold):**
- **keep:** 62.5%.
- **keep_elsewhere:** 3.6% (164 claims). The proxy calls 35% of these wrong; given the audit, many of them are
  probably present but unmentioned.
- **replace:** 7.1%.
  - Half of these are proxy-correct claims.
  - The substitute appears in the references only 0.241 of the time.
- **remove:** 5.6%. In 0.654 of these the references have no same-group class, so deletion is plausible.
- **skip:** 21.3%.

**The 24 audited claims:**
- `replace` misbehaves on clothing: sweater → suit, shirt → uniform, wall → stairs. The claims are correct by my visual
  judgment, but the detector picked another class from the same group.
- The one claim I judged wrong (surfboard) gets `keep`: a detector false positive.

## Interpretation
1. **Completing the logic worked: the five outcomes are clear and the code is unified.**
   - The image veto keeps the "present elsewhere" cases from being destroyed.
   - The remove/replace split gives Retell what to do.
2. **`replace` is not reliable yet, especially for clothing.**
   - OWLv2 confuses classes inside the same clothing group (suit/uniform/scarf/shirt).
   - So for now Retell should treat `replace` as a soft suggestion (a logit push toward the substitute, never forcing
     it). Or clothing replace could be reduced to `skip`, but that is decided with the human pilot; no changes were
     made by looking at the proxy.
3. **Is Check now at its best?**
   - Within the logic, yes: there are no missing outcomes, and the thresholds, veto and idiom are in place.
   - Measurement-wise, unknown. Proxy J cannot measure exactly the behaviour we want: it punishes keeping present
     objects. The final call (whether to keep the veto, clothing replace, the threshold point) depends on the human
     pilot.
4. **Not done:**
   - Validation was not read (once the method is frozen, it is reported once).
   - The GPU crop pilot was not run (low priority).
   - Retell training is still disabled in the code.

## Next step
- **Human pilot:** about 300–500 B claims, stratified by action (especially keep_elsewhere, replace, remove). The tool:
  `annotation/pilot/annotate.html`.
- **In parallel, Retell R0 pilot:** remove → ban the word; replace → soft push; regenerate from the error point.
