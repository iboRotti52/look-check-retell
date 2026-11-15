"""Render frozen B development results for the report, without dataset access."""
import _common
from lcr import cache, config

r = cache.load_json(config.RESULTS / "lcr_development.json")
look_rows = r["look_options"]
baseline = next(s for s in look_rows if s["rule"] == dict(min_patches=2, alpha=.5))
selected = next(s for s in look_rows if s["rule"] == r["look_choice"])
_common.macros("lcr_numbers", dict(
    LCRLookBefore=round(100 * baseline["top3"] / baseline["n"], 2),
    LCRLookAfter=round(100 * selected["top3"] / selected["n"], 2),
    LCRSmallBefore=round(100 * baseline["small_top3"] / baseline["small_n"], 2),
    LCRSmallAfter=round(100 * selected["small_top3"] / selected["small_n"], 2),
    LCRFitImages=r["n_fit_images"], LCRCalImages=r["n_cal_images"],
    LCRCheckJ=f"{r['check_choice']['quality']['J']:.3f}",
    LCRCheckPrecision=f"{r['check_choice']['quality']['contradicted_precision']:.3f}",
    LCRCheckRecall=f"{r['check_choice']['quality']['recall_wrong']:.3f}",
))
