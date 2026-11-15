# Fable second review — B diagnostics (2026-11-15)

Exploratory scripts behind `docs/FABLE_INCELEME_2.md`. B only (`train_*` ids of `A_B_look4`); validation/test are never
read. Not part of the method and not report numbers. Run from this directory, one at a time:
`OMP_NUM_THREADS=2 ../../.venv/bin/python <script>`.

- `common.py` harness (features → `data/cache/review_fable2_B_feat.pkl`; swap set identical to `scripts/43_check_fix.py`)
- `e0_baseline.py` reproduces S3 without abstention; composition of the swap set
- `e1_anatomy.py` where swap misses come from; per-class separability; class mix of proxy-wrong claims vs swap labels
- `e2_rival.py` rival-aware evidence · `e3_calib.py` shrink targets, LDA LLR, oracle thresholds · `e4_dist.py` score distributions
- `e5_strict.py` positives mentioned by >= 2 references · `e6_np.py` cost-sensitive per-class thresholds
- `e7a_text.py`, `e7b_queries.py` text prompt variants and exemplar-refined queries
- `e8_naming.py` pairwise separability and naming at the attended box
- `e9_crop.py` CPU pilot of crop re-detection (same frozen OWLv2) · `e13_crop_eval.py` its evaluation
- `e10_sheets.py` contact sheets for a visual look (Claude's observation, not a human label)
- `e12_s2.py` S2 flags through the project's own script-43 functions · `e11_final.py` candidate S4 vs S3 vs S2, substitutes
