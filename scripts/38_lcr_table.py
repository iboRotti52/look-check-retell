"""Render frozen B development results for the report, without dataset access."""
import _common
from lcr import cache, config

r = cache.load_json(config.RESULTS / "lcr_development.json")
# Small-object coverage: script 37's first count credited a hit on any box of the class (2026-11-19). Use the
# corrected count from scripts/49 when it exists; otherwise print "--" rather than the old number.
small_path = config.RESULTS / "look_small_object_B.json"
small = cache.load_json(small_path)["rates"] if small_path.exists() else None
look_rows = r["look_options"]
baseline = next(s for s in look_rows if s["rule"] == dict(min_patches=2, alpha=.5))
selected = next(s for s in look_rows if s["rule"] == r["look_choice"])
_common.macros("lcr_numbers", dict(
    LCRLookBefore=round(100 * baseline["top3"] / baseline["n"], 2),
    LCRLookAfter=round(100 * selected["top3"] / selected["n"], 2),
    LCRSmallBefore=small["with_filter"]["small_top3"] if small else "--",
    LCRSmallAfter=small["without_filter"]["small_top3"] if small else "--",
    LCRFitImages=r["n_fit_images"], LCRCalImages=r["n_cal_images"],
    LCRCheckJ=f"{r['check_choice']['quality']['J']:.3f}",
    LCRCheckPrecision=f"{r['check_choice']['quality']['contradicted_precision']:.3f}",
    LCRCheckRecall=f"{r['check_choice']['quality']['recall_wrong']:.3f}",
))
