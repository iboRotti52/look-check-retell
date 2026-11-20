"""Report macros for the default Check (S4) from its frozen B results. CPU only, no dataset access.

Reads results/check_s4_B.json (scripts/45, out-of-fold on B) and results/lcr_check4_bundle.json and writes
report/generated/check_s4_macros.tex, so the report's Check section quotes no hand-typed number.
FCR = share of reference-supported claims that Check contradicts (automatic proxy for false alarms);
detection = share of synthetic label swaps caught; substitute precision = share of replaced swaps whose
substitute is the original label.

    python scripts/51_check_s4_report.py
"""
import _common
from lcr import cache, config

r = cache.load_json(config.RESULTS / "check_s4_B.json")
b = cache.load_json(config.RESULTS / "lcr_check4_bundle.json")
assert r["bundle_id"] == b["id"], "S4 results and bundle disagree"
lam = r["decision"]["lam_star"]
s4 = r["rows"]["S4@fcr03"]
s2 = r["rows"]["S2_five_way_z"]
no_rival = r["rows"]["S4a_no_rival@fcr03"]
old_sub = r["rows"]["S4_S3_substitute@fcr03"]
assert s4["lam"] == lam
pct = lambda x: f"{100 * x:.1f}"
acts = r["final"]["in_sample_actions"]
_common.macros("check_s4_macros", dict(
    CSLambda=f"{lam:g}", CSRivalWeight=f"{b['rival_weight']:g}", CSTarget=pct(r["target"]),
    CSNoTau=r["final"]["n_classes_without_tau"], CSClaims=f"{r['n_claims']:,}".replace(",", "{,}"),
    CSSwaps=f"{r['strata_n']['all']:,}".replace(",", "{,}"), CSCleanSwaps=r["strata_n"]["clean_exclusive"],
    CSKeep=f"{acts['keep']:,}".replace(",", "{,}"), CSElsewhere=acts["keep_elsewhere"], CSSkip=acts["skip"],
    CSRemove=acts["remove"], CSReplace=acts["replace"],
    CSFcrThing=pct(s4["FCR_thing"]), CSTwoFcrThing=pct(s2["FCR_thing"]),
    CSFcrClothing=pct(s4["FCR_clothing"]), CSTwoFcrClothing=pct(s2["FCR_clothing"]),
    CSFcrStuff=pct(s4["FCR_stuff"]), CSTwoFcrStuff=pct(s2["FCR_stuff"]),
    CSCleanDetect=pct(s4["detection"]["clean_exclusive"]), CSTwoCleanDetect=pct(s2["detection"]["clean_exclusive"]),
    CSAllDetect=pct(s4["detection"]["all"]), CSTwoAllDetect=pct(s2["detection"]["all"]),
    CSClothDetect=pct(s4["detection"]["same_set_clothing"]), CSTwoClothDetect=pct(s2["detection"]["same_set_clothing"]),
    CSSubPrec=pct(s4["substitute_precision"]), CSTwoSubPrec=pct(s2["substitute_precision"]),
    CSNoRivalClean=pct(no_rival["detection"]["clean_exclusive"]), CSUncentredSub=pct(old_sub["substitute_precision"]),
))
print("check_s4_macros written; lambda", lam)
