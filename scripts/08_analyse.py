"""Turn the Colab outputs into the report's tables, figures and number macros.

    python scripts/08_analyse.py

Reads (all produced by colab/run_all.sh):
  results/finetune_A.json, results/full_metrics.json, results/retell_*.json
  data/cache/pipeline/A_validation.json (+ .maps.npz), data/cache/owl/validation.npz
Writes report/generated/*.tex, report/figures/*.pdf and results/analysis.json.
Every number here is a validation-split number; the test split is never read.
Claims are compared with the five references only as a PROXY for correctness.
"""
import glob
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _common import C, COL_W, PAGE_W, macros, plt, savefig, write_tex
from lcr import cache, claims, config, data, look, metrics, pipeline
from lcr.vocab import KIND

R = config.RESULTS
P = config.CACHE / "pipeline"
out: dict = {}
mac: dict = {}


def fmt(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


# ------------------------------------------------------------------ misc constants
th = pipeline.thresholds()
mac.update(ThrSupport=th["t_support"], ThrContra=th["t_contra"], ThrExcess=th["t_excess"], LastK=th["last_k"])
try:
    col = subprocess.run([str(config.ROOT / ".venv/bin/python"), "-m", "pytest", "--collect-only", "-q"],
                         cwd=config.ROOT, capture_output=True, text=True).stdout
    mac["NumTests"] = int(re.search(r"(\d+) tests? collected", col).group(1))
except Exception:
    mac["NumTests"] = len(re.findall(r"::", col))

# ------------------------------------------------------------------ power analysis
# Moved to scripts/50_h1_power.py (2026-11-19): it needs no Colab outputs, and the effect is now held at
# H1's 15% instead of a fixed 1/3 fix rate. That script is the only writer of power_table.tex.

# ------------------------------------------------------------------ captioner quality
ft = cache.load_json(R / "finetune_A.json")
fm = cache.load_json(R / "full_metrics.json") if (R / "full_metrics.json").exists() else {}
out["finetune"] = ft
mac.update(FtMinutes=fmt(ft.get("minutes"), 0), FtBestEpoch=ft.get("best_epoch"), FtDevice=ft.get("device", "?"))
cols = ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"]


def cap_row(name, m):
    return name + " & " + " & ".join(fmt(m.get(c), 3 if c == "CIDEr" else 1) for c in cols)


best = ft["epochs"][ft["best_epoch"] - 1]
zs = fm.get("draft_zero", ft["zero_shot"])
fine = fm.get("draft_A", best)
lines = [cap_row("BLIP zero-shot", zs) + f" & {fmt(ft['zero_shot']['RUOR'])}" + r" \\",
         cap_row("BLIP fine-tuned on A (B0)", fine) + f" & {fmt(best['RUOR'])}" + r" \\"]
toolkit_note = ("Full COCO toolkit with PTB tokenisation. " if "draft_zero" in fm and "draft_A" in fm else
                "Full COCO outputs are unavailable locally; BLEU-4 and CIDEr use the stored alphabetic-tokenisation scores. "
                "A dash marks an unavailable metric. ")
write_tex("caption_table", r"""\begin{table}[t]
\caption{Captioning quality on the Flickr8k validation split (1{,}000 images). """ + toolkit_note + r"""
BLEU-4, METEOR, ROUGE-L, SPICE $\times100$. RUOR: share of generated vocabulary claims that
no reference mentions (reference agreement, \emph{not} a hallucination rate).}
\label{tab:caption}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Captioner & B-4 & M & R-L & CIDEr & SPICE & RUOR \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
mac.update(ZsCIDEr=fmt(zs["CIDEr"], 3), FtCIDEr=fmt(fine["CIDEr"], 3),
           ZsSPICE=fmt(zs.get("SPICE")), FtSPICE=fmt(fine.get("SPICE")),
           ZsBLEU=fmt(zs["BLEU-4"]), FtBLEU=fmt(fine["BLEU-4"]))
mac.update(ZsRUOR=fmt(ft["zero_shot"]["RUOR"]), FtRUOR=fmt(ft["epochs"][ft["best_epoch"] - 1]["RUOR"]))

# training curve
fig, ax = plt.subplots(figsize=(COL_W, 1.6))
lc = np.array(ft["loss_curve"])
ax.plot(lc[:, 0], lc[:, 1], color=C["blue"], lw=1)
ax.set_xlabel("update step"); ax.set_ylabel("train loss")
ax2 = ax.twinx(); ax2.grid(False)
steps_per_epoch = lc[-1, 0] / len(ft["epochs"])
ax2.plot([steps_per_epoch * e["epoch"] for e in ft["epochs"]], [e["CIDEr"] for e in ft["epochs"]],
         "o-", color=C["orange"], ms=3, lw=1)
ax2.axhline(ft["zero_shot"]["CIDEr"], ls="--", color=C["grey"], lw=0.8)
ax2.set_ylabel("val CIDEr", color=C["orange"])
ax2.spines["right"].set_visible(True)
savefig(fig, "finetune_curve")

# ------------------------------------------------------------------ pipeline (Look / Check) on validation
rows_val = cache.load_json(P / "A_validation.json")
owl = cache.OwlCache("validation")
allc = [(r, c) for r in rows_val for c in r["claims"]]
mac["ValClaims"] = f"{len(allc):,}".replace(",", "{,}")
mac["ValClaimsPerImage"] = fmt(len(allc) / len(rows_val), 2)
kinds = ["person", "thing", "stuff"]
kind_n = Counter(c["kind"] for _, c in allc)
inref = defaultdict(list)
for _, c in allc:
    inref[c["kind"]].append(c["in_refs"])
mac["RefSupportedAll"] = fmt(100 * np.mean([c["in_refs"] for _, c in allc]))
for k in kinds:
    mac[f"RefSupported{k.capitalize()}"] = fmt(100 * np.mean(inref[k]))

# verdicts (provisional thresholds) x kind x reference agreement
fin = [(c, pipeline.finalise(c["rec_att"], th)) for _, c in allc]
vt = {k: {ref: Counter() for ref in (True, False)} for k in kinds}
for c, r in fin:
    vt[c["kind"]][c["in_refs"]][r["verdict"]] += 1
out["verdicts"] = {k: {str(ref): dict(v) for ref, v in d.items()} for k, d in vt.items()}
nores = sum(r["region"] < 0 for _, r in fin)
mac["NoRegionPct"] = fmt(100 * nores / len(fin))
vc = Counter(r["verdict"] for _, r in fin)
for v in ("supported", "contradicted", "uncertain"):
    mac[f"Verdict{v.capitalize()}Pct"] = fmt(100 * vc[v] / len(fin))

fig, axes = plt.subplots(1, 3, figsize=(PAGE_W * 0.62, 1.7), sharey=True)
colors = dict(supported=C["green"], uncertain=C["grey"], contradicted=C["red"])
for ax, k in zip(axes, kinds):
    for xi, ref in enumerate((True, False)):
        cnt = vt[k][ref]; n = sum(cnt.values()) or 1
        bottom = 0
        for v in ("supported", "uncertain", "contradicted"):
            h = 100 * cnt[v] / n
            ax.bar(xi, h, bottom=bottom, color=colors[v], width=0.6, label=v if (k == "person" and xi == 0) else None)
            bottom += h
        ax.text(xi, 102, f"n={n}", ha="center", fontsize=6)
    ax.set_xticks([0, 1], ["in refs", "not in refs"]); ax.set_title(k); ax.set_ylim(0, 112)
axes[0].set_ylabel("% of claims")
fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), frameon=False)
savefig(fig, "verdict_by_kind")

# detector score separates reference-supported from unsupported claims?
aur = {}
for k in kinds:
    pos = [c["rec_att"]["s_image"] for _, c in allc if c["kind"] == k and c["in_refs"]]
    neg = [c["rec_att"]["s_image"] for _, c in allc if c["kind"] == k and not c["in_refs"]]
    aur[k] = metrics.auroc(np.array(pos), np.array(neg))
    mac[f"Auroc{k.capitalize()}"] = fmt(aur[k], 2)
out["auroc_s_image"] = aur
# reference incompleteness: unsupported-by-refs claims the detector finds confidently in the image
unsup = [c for _, c in allc if not c["in_refs"] and c["kind"] == "thing"]
mac["UnsupThingHighDet"] = fmt(100 * np.mean([c["rec_att"]["s_image"] >= th["t_support"] for c in unsup]))
mac["UnsupThingN"] = len(unsup)

# Look: attention vs detector region agreement, per k
agree = {}
for k in ("1", "4", "12"):
    vals = []
    for r, c in allc:
        ra, rd = c["regions_by_k"][k], c["rec_det"]["region"]
        if ra < 0 or rd < 0:
            continue
        b = owl[data.parse_id(r["id"])[1]]["boxes"].astype(np.float32)
        vals.append(float(look.iou(b[ra], b[rd:rd + 1])[0]) >= 0.5)
    agree[k] = 100 * np.mean(vals)
    mac[f"AgreeK{ {'1': 'One', '4': 'Four', '12': 'Twelve'}[k] }".replace(" ", "")] = fmt(agree[k])
out["att_det_agreement_iou50"] = agree
ex = np.array([c["excess_by_k"]["4"] for _, c in allc])
mac["ExcessMedian"] = fmt(float(np.median(ex)), 3)
mac["ExcessPositivePct"] = fmt(100 * float(np.mean(ex > 0)))
ent = np.array([c["att_entropy"] for _, c in allc])
mac["EntropyMedian"] = fmt(float(np.median(ent)), 2)

fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.5))
axes[0].bar(["1", "4", "12"], [agree[k] for k in ("1", "4", "12")], color=C["blue"], width=0.55)
axes[0].set_xlabel("last $k$ decoder layers"); axes[0].set_ylabel("IoU$\\geq$0.5 with B3 (%)")
for kk in kinds:
    axes[1].hist([c["excess_by_k"]["4"] for _, c in allc if c["kind"] == kk], bins=30, histtype="step",
                 lw=1, color=dict(person=C["purple"], thing=C["blue"], stuff=C["orange"])[kk], label=kk)
axes[1].set_xlabel("excess attention mass $e$"); axes[1].legend(frameon=False)
fig.tight_layout()
savefig(fig, "look_stats")

# ---- attention diagnosis (scripts/09_attention_variants.py)
av_path = R / "attention_variants.json"
if av_path.exists():
    av = cache.load_json(av_path)
    v = av["variants"]
    mac.update(AttnN=v["raw_k4"]["n"], AttnImages=av["n_images"],
               AttnDecoder="zero-shot" if av["decoder"] == "zero" else "fine-tuned",
               AttnChance=fmt(v["raw_k4"]["chance"]), AttnRawPoint=fmt(v["raw_k4"]["pointing"]),
               AttnSinkPoint=fmt(v["sink_k4"]["pointing"]), AttnRawTop=fmt(v["raw_k4"]["top5_mass"], 2),
               AttnSinkTop=fmt(v["sink_k4"]["top5_mass"], 2), AttnRawIoU=fmt(v["raw_k4"]["agree_iou50"]),
               AttnSinkIoU=fmt(v["sink_k4"]["agree_iou50"]))

# ---- sink-corrected attention (second Colab stage), same drafts and claims
sink_path = P / "A_validation_sink.json"
rows_sink = cache.load_json(sink_path) if sink_path.exists() else None
if rows_sink:
    alls = [(r, c) for r in rows_sink for c in r["claims"]]
    assert len(alls) == len(allc), "raw and sink pipelines must contain the same claims"
    agree_s = {}
    for k in ("1", "4", "12"):
        vals = []
        for r, c in alls:
            ra, rd = c["regions_by_k"][k], c["rec_det"]["region"]
            if ra < 0 or rd < 0:
                continue
            b = owl[data.parse_id(r["id"])[1]]["boxes"].astype(np.float32)
            vals.append(float(look.iou(b[ra], b[rd:rd + 1])[0]) >= 0.5)
        agree_s[k] = 100 * np.mean(vals)
        mac[f"SinkAgreeK{ {'1': 'One', '4': 'Four', '12': 'Twelve'}[k] }".replace(" ", "")] = fmt(agree_s[k])
    out["att_det_agreement_iou50_sink"] = agree_s
    fs = [pipeline.finalise(c["rec_att"], th) for _, c in alls]
    vcs = Counter(r["verdict"] for r in fs)
    for v in ("supported", "contradicted", "uncertain"):
        mac[f"SinkVerdict{v.capitalize()}Pct"] = fmt(100 * vcs[v] / len(fs))
    out["verdicts_sink"] = dict(vcs)

# qualitative overlays: the same thing claims under raw (top) and sink-corrected (bottom) attention
def load_maps(name):
    z = np.load(P / name)
    return z["maps"], {(int(a), int(b)): i for i, (a, b) in enumerate(z["owner"])}


maps_raw, idx_raw = load_maps("A_validation.maps.npz")
variants = [("raw", rows_val, maps_raw, idx_raw)]
if rows_sink:
    ms, isx = load_maps("A_validation_sink.maps.npz")
    variants.append(("sink-corrected", rows_sink, ms, isx))
rng = np.random.default_rng(config.SEED + 1)
pool = [(i, j) for i, r in enumerate(rows_val) for j, c in enumerate(r["claims"])
        if c["kind"] == "thing" and c["in_refs"] and c["rec_det"]["region"] >= 0]
picks = [pool[t] for t in rng.permutation(len(pool))[:4]]
fig, axes = plt.subplots(len(variants), 4, figsize=(PAGE_W, 1.75 * len(variants) + 0.3), squeeze=False)
for row_i, (vname, rows_v, maps_v, idx_v) in enumerate(variants):
    for col, (i, j) in enumerate(picks):
        ax = axes[row_i, col]
        r = rows_v[i]; c = r["claims"][j]
        im = data.load_image(r["id"]); W, H = im.size
        ax.imshow(im); ax.axis("off")
        att = maps_v[idx_v[(i, j)]].astype(np.float32)
        ax.imshow(att / max(att.max(), 1e-9), extent=(0, W, H, 0), cmap="inferno", alpha=0.5, interpolation="bilinear")
        rec = pipeline.finalise(c["rec_att"], th)
        if c["rec_det"]["region"] >= 0:
            x0, y0, x1, y1 = c["rec_det"]["box"]
            ax.add_patch(plt.Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H, fill=False, ec="#FFFFFF", lw=1, ls="--"))
        if rec["region"] >= 0:
            x0, y0, x1, y1 = rec["box"]
            ax.add_patch(plt.Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H, fill=False, ec="#00E5FF", lw=1.4))
        title = f"{vname}: claim \u201c{c['cls']}\u201d \u2192 {rec['verdict']}"
        if row_i == 0:
            title = f"\u201c{r['draft']}\u201d\n" + title
        ax.set_title(title, fontsize=5.2)
fig.tight_layout()
savefig(fig, "attention_overlays")

# ------------------------------------------------------------------ B1: word replacement on validation
def replace_word(text, word, new):
    return re.sub(rf"\b{re.escape(word)}\b", new, text, count=1)


refs_val = {r["id"]: data.references(r["id"]) for r in rows_val}
b1 = {}
n_rep = 0
for r in rows_val:
    t = r["draft"]
    for c in r["claims"]:
        rec = pipeline.finalise(c["rec_att"], th)
        if rec["verdict"] == "contradicted" and rec["alt"]:
            t = replace_word(t, c["word"], rec["alt"]); n_rep += 1
    b1[r["id"]] = t
draft = {r["id"]: r["draft"] for r in rows_val}
out["b1"] = {"replacements": n_rep, **metrics.caption_scores(b1, refs_val), **metrics.ruor(b1, refs_val)}
out["b0_local"] = {**metrics.caption_scores(draft, refs_val), **metrics.ruor(draft, refs_val)}
# reference-proxy repair: a contradicted claim whose class is not in refs but whose alternative is
fix = brk = 0
for _, c in allc:
    rec = pipeline.finalise(c["rec_att"], th)
    if rec["verdict"] == "contradicted" and rec["alt"]:
        present = claims.reference_classes(refs_val[_["id"]])
        fix += (not c["in_refs"]) and rec["alt"] in present
        brk += c["in_refs"] and rec["alt"] not in present
out["b1"]["proxy_fix"] = fix; out["b1"]["proxy_break"] = brk
mac.update(BOneN=n_rep, BOneFix=fix, BOneBreak=brk)

# ------------------------------------------------------------------ Check: region source x thresholds (validation)
# Thresholds are calibrated on half B (scripts/11_proxy_thresholds.py); here they are applied to held-out validation.
def check_quality(pairs, thr):
    v = [(pipeline.finalise(r, thr)["verdict"], ok) for r, ok in pairs]
    pos = [x for x, ok in v if ok]; neg = [x for x, ok in v if not ok]
    rate = lambda xs, k: float(np.mean([x == k for x in xs])) if xs else float("nan")
    allv = [x for x, _ in v]
    n_con = sum(x == "contradicted" for x in allv)
    return dict(sup=100 * rate(allv, "supported"), con=100 * rate(allv, "contradicted"), unc=100 * rate(allv, "uncertain"),
                J=rate(pos, "supported") - rate(neg, "supported") + rate(neg, "contradicted") - rate(pos, "contradicted"),
                prec=100 * sum(x == "contradicted" and not ok for x, ok in v) / n_con if n_con else float("nan"))


def thr_file(tag):
    f = R / f"thresholds_proxy_{tag}.json"
    return {**pipeline.DEFAULT_THRESHOLDS, **cache.load_json(f)} if f.exists() else None


TAGNAME = {"att": "Raw", "att_sink": "Sink", "det": "Det"}
prov = dict(pipeline.DEFAULT_THRESHOLDS)
src = [("Attention (raw)", rows_val, "rec_att", "att")]
if rows_sink:
    src.append(("Attention (sink-corr.)", rows_sink, "rec_att", "att_sink"))
src.append(("Detector-selected", rows_val, "rec_det", "det"))
cq_lines, out["check_quality"] = [], {}
for lab, rws, key, tag in src:
    pairs = [(c[key], bool(c["in_refs"])) for r in rws for c in r["claims"] if KIND[c["cls"]] != "person"]
    for tl, thr in (("provisional", prov), ("calibrated on B", thr_file(tag))):
        if thr is None:
            continue
        q = check_quality(pairs, thr)
        out["check_quality"][f"{tag}/{tl}"] = dict(thresholds=[thr["t_support"], thr["t_contra"]], **q)
        cq_lines.append(f"{lab} & {tl} & {thr['t_support']:.2f}/{thr['t_contra']:.3f} & {fmt(q['sup'])} & {fmt(q['con'])} & "
                        f"{fmt(q['unc'])} & {fmt(q['J'], 2)} & {fmt(q['prec'])} \\\\")
        nm = TAGNAME[tag] + ("Prov" if tl == "provisional" else "Cal")
        mac[f"J{nm}"] = fmt(q["J"], 2); mac[f"Prec{nm}"] = fmt(q["prec"]); mac[f"Con{nm}"] = fmt(q["con"])
        if tl != "provisional":
            mac[f"Ts{nm}"] = f"{thr['t_support']:.2f}"; mac[f"Tc{nm}"] = f"{thr['t_contra']:.3f}"
    cq_lines.append(r"\addlinespace")
mac["NumCheckClaims"] = f"{sum(KIND[c['cls']] != 'person' for _, c in allc):,}".replace(",", "{,}")
mac["UnsupBaseRate"] = fmt(100 * np.mean([not c["in_refs"] for _, c in allc if KIND[c["cls"]] != "person"]))
write_tex("check_table", r"""\begin{table*}[t]
\caption{Quality of the Check verdict on validation thing/stuff claims (\NumCheckClaims{} claims), measured against the
reference proxy (claim class mentioned in a reference). Calibrated thresholds were chosen on training half B
and are applied here unchanged. $J$: informedness of the two verdicts,
$[P(\mathrm{sup}\mid\mathrm{ref})-P(\mathrm{sup}\mid\neg\mathrm{ref})]+[P(\mathrm{con}\mid\neg\mathrm{ref})-P(\mathrm{con}\mid\mathrm{ref})]$,
where 0 is chance. Prec.: share of \emph{contradicted} claims that no reference mentions (base rate \UnsupBaseRate\%).}
\label{tab:check}
\centering\footnotesize
\begin{tabular}{@{}llcccccc@{}}
\toprule
Region & Thresholds & $\tau_s/\tau_c$ & Supp.\ (\%) & Contra.\ (\%) & Uncert.\ (\%) & $J$ & Prec.\ (\%) \\
\midrule
""" + "\n".join(cq_lines[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def b1_proxy(rws, key, thr):
    n = fx = bk = 0
    for r in rws:
        present = claims.reference_classes(refs_val[r["id"]])
        for c in r["claims"]:
            rec = pipeline.finalise(c[key], thr)
            if rec["verdict"] == "contradicted" and rec["alt"]:
                n += 1; fx += (not c["in_refs"]) and rec["alt"] in present; bk += c["in_refs"] and rec["alt"] not in present
    return n, fx, bk


out["b1_variants"] = {}
for lab, rws, key, tag in src:
    thr = thr_file(tag)
    if thr is None:
        continue
    n, fx, bk = b1_proxy(rws, key, thr)
    out["b1_variants"][tag] = dict(n=n, fix=fx, brk=bk)
    mac.update(**{f"BOne{TAGNAME[tag]}N": n, f"BOne{TAGNAME[tag]}Fix": fx, f"BOne{TAGNAME[tag]}Break": bk})

# ------------------------------------------------------------------ Retell systems
ROUND1 = dict(b0plus="B0+ (no evidence)", ours="LCR, raw attention", ours_rho0=r"LCR raw, $\rho=0$",
              ours_sink="LCR, sink-corr. attention", b3="B3 (detector regions)", b2="B2 (image tags)")
ROUND2 = dict(b0plus_fix_sink="B0+ (no evidence)", ours_thr_sink="LCR, calibrated thr.\\ only",
              ours_fix_sink="LCR", ours_fixr30_sink=r"LCR, $\rho=0.3$",
              b3_fix_sink="B3 (detector regions)", b2_fix_sink="B2 (image tags)")
sysnames = {**ROUND1, **ROUND2}
SHORT = {**{k: v + " [R1]" for k, v in ROUND1.items()}, **{k: v + " [R2]" for k, v in ROUND2.items()}}
ret = {}
for name in sysnames:
    p = R / f"retell_{name}.json"
    if p.exists():
        ret[name] = cache.load_json(p)
out["retell"] = {k: dict(conditions=v["conditions"], gates=v["gates"], init=v["init_identical_to_draft"],
                         params=v["trainable_params"], evidence=v["evidence_stats"], best_epoch=v.get("best_epoch"),
                         minutes=v.get("minutes")) for k, v in ret.items()}
if ret:
    any_ = next(iter(ret.values()))
    mac["ParamsM"] = fmt(any_["trainable_params"] / 1e6, 1)
    mac["InitIdentical"] = "all" if all(v["init_identical_to_draft"] for v in ret.values()) else "NOT all"
    d = any_["draft"]
    lines = [f"B0 (draft) & {fmt(d['BLEU-4'])} & {fmt(d['CIDEr'], 3)} & {fmt(d['RUOR'])} & -- & -- & -- & -- \\\\",
             f"B1 (word repl., provisional thr.) & {fmt(out['b1']['BLEU-4'])} & {fmt(out['b1']['CIDEr'], 3)} & {fmt(out['b1']['RUOR'])} & "
             f"{fmt(100 * sum(b1[i] != draft[i] for i in draft) / len(draft))} & -- & -- & -- \\\\"]

    def sens(name, cond):
        se = ret[name].get("sensitivity") or {}
        if cond not in se:            # runs from before the metric existed: recompute from the saved outputs
            f = P / f"retell_{name}_validation_outputs.json"
            o = cache.load_json(f) if f.exists() else {}
            if cond in o and "real" in o:
                se[cond] = 100 * float(np.mean([o[cond][i] != o["real"][i] for i in o["real"]]))
                ret[name]["sensitivity"] = se
        return fmt(se.get(cond))

    for title, group in ((r"\emph{Round 1: provisional thresholds, no patch dropout}", ROUND1),
                         (r"\emph{Round 2: sink-corr.\ attention, thresholds calibrated on B, patch dropout 0.5 (except `calibrated thr.\ only')}", ROUND2)):
        names = [n for n in group if n in ret]
        if not names:
            continue
        lines.append(r"\midrule" + "\n" + r"\multicolumn{8}{@{}l}{" + title + r"} \\")
        for name in names:
            cr = ret[name]["conditions"]["real"]
            lines.append(f"{group[name]} & {fmt(cr['BLEU-4'])} & {fmt(cr['CIDEr'], 3)} & {fmt(cr['RUOR'])} & "
                         f"{fmt(cr['changed_captions'])} & {fmt(cr['alternative_adopted'])} & "
                         f"{fmt(cr['contradicted_claim_retained'])} & {sens(name, 'masked')} \\\\")
    write_tex("retell_table", r"""\begin{table*}[t]
\caption{Retell development runs on the validation split (decoder fine-tuned on A, pathway trained on B, 3 epochs,
$\rho=0.15$ unless stated). Each round has its own matched no-evidence control B0+. BLEU-4 and CIDEr use our
simplified tokeniser (compare only with scores using the same tokenisation). \emph{Changed}: captions that
differ from the draft. \emph{Alt.\ adopt.} / \emph{Con.\ kept}: of the draft claims whose record is
\emph{contradicted}, the share whose alternative label appears in the output / whose claimed label is still in
the output (the set of contradicted records depends on the thresholds, so these two columns are comparable only
within a round). \emph{Sens.}: captions that change when all evidence is masked. All columns are automatic;
none is a human-verified error rate.}
\label{tab:retell}
\centering\footnotesize
\begin{tabular}{@{}lccccccc@{}}
\toprule
System & BLEU-4 & CIDEr & RUOR & Changed (\%) & Alt.\ adopt.\ (\%) & Con.\ kept (\%) & Sens.\ (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    # evidence interventions for ours / b3
    il = []
    for name in ("ours", "ours_sink", "b3", "ours_fix_sink", "ours_fixr30_sink", "b3_fix_sink"):
        if name not in ret:
            continue
        for cond in ("real", "masked", "false"):
            cr = ret[name]["conditions"].get(cond)
            if cr:
                lab = SHORT[name] if cond == "real" else ""
                il.append(f"{lab} & {cond} & {fmt(cr['CIDEr'], 3)} & {fmt(cr['RUOR'])} & "
                          f"{fmt(cr['alternative_adopted'])} & {sens(name, cond) if cond != 'real' else '--'} \\\\")
        il.append(r"\addlinespace")
    write_tex("intervention_table", r"""\begin{table}[t]
\caption{Evidence interventions on validation: the same trained model with real records, all records masked
(null token only), or false records (alternative label of each contradicted record replaced by a class the
detector does not see in the image). \emph{Sens.}: captions that differ from the real-evidence output.
[R1]/[R2]: development round.}
\label{tab:interv}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{2.3cm}lcccc@{}}
\toprule
System & Evidence & CIDEr & RUOR & Alt.\ adopt. & Sens.\ (\%) \\
\midrule
""" + "\n".join(il[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    if "ours" in ret:
        o = ret["ours"]
        for cond in ("real", "masked", "false"):
            if cond in o["conditions"]:
                mac[f"Ours{cond.capitalize()}Alt"] = fmt(o["conditions"][cond]["alternative_adopted"])
                mac[f"Ours{cond.capitalize()}Changed"] = fmt(o["conditions"][cond]["changed_captions"])
        es = o["evidence_stats"]
        mac["OursCorruptPct"] = fmt(100 * es["corrupted"] / max(es["tokens"], 1))
        tot = sum(es["verdicts"].values()) or 1
        mac["OursTrainContraPct"] = fmt(100 * es["verdicts"].get("contradicted", 0) / tot)
        # gate curves
        pair = ("ours", "ours_fix_sink") if "ours_fix_sink" in ret else ("ours", "b0plus")
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.55), sharey=True)
        for ax, name in zip(axes, pair):
            if name not in ret:
                continue
            gc = ret[name]["gate_curve"]
            steps = [s for s, _ in gc]
            for key in gc[0][1]:
                if key.endswith(".attn"):
                    ax.plot(steps, [g[key] for _, g in gc], lw=0.8, label=key.split(".")[0])
            ax.set_title(SHORT[name], fontsize=7); ax.set_xlabel("step")
        axes[0].set_ylabel(r"$\tanh(\alpha_{\mathrm{attn}})$")
        axes[1].legend(fontsize=5, frameon=False, ncol=2)
        fig.tight_layout()
        savefig(fig, "gate_curves")
    # round-2 macros
    for name, nm in (("ours_fix_sink", "Fix"), ("ours_fixr30_sink", "FixRThirty"), ("b3_fix_sink", "BThreeFix"),
                     ("ours_thr_sink", "Thr"), ("b0plus_fix_sink", "BZeroFix"), ("ours", "One")):
        if name not in ret:
            continue
        r_ = ret[name]
        g = r_["gates"]
        mac[f"Gate{nm}Attn"] = fmt(max(abs(v) for k, v in g.items() if k.endswith(".attn")), 3)
        mac[f"Gate{nm}Ff"] = fmt(max(abs(v) for k, v in g.items() if k.endswith(".ff")), 3)
        for cond in ("masked", "false"):
            mac[f"Sens{nm}{cond.capitalize()}"] = sens(name, cond)
        for cond, crd in r_["conditions"].items():
            mac[f"Cider{nm}{cond.capitalize()}"] = fmt(crd["CIDEr"], 3)
            mac[f"Ruor{nm}{cond.capitalize()}"] = fmt(crd["RUOR"])
            mac[f"Alt{nm}{cond.capitalize()}"] = fmt(crd["alternative_adopted"])
            mac[f"Kept{nm}{cond.capitalize()}"] = fmt(crd["contradicted_claim_retained"])
            mac[f"Changed{nm}{cond.capitalize()}"] = fmt(crd["changed_captions"])

cache.save_json(out, R / "analysis.json")
macros("result_macros", {k: v for k, v in mac.items()})
print(json.dumps({k: mac[k] for k in sorted(mac)}, indent=0)[:3000])
