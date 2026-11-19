"""Look L1 (docs/LOOK_L1_PROTOKOL.md): forced-hallucination test, B only.

Each sampled B draft is teacher-forced through decoder_A three times -- as written, with one claim word
swapped (misidentified), and with an absent class appended (invented) -- and Look is re-read through the
v4/LCR code path: same 4 frozen heads, same float16 signed-head quantisation (look_v4.signed_heads, which
reproduces look_v4.attention_map exactly and is asserted to here), same smoothing, and ONE ranking for
top-1 and top-3: region_evidence.candidates with the S4 bundle's own Look rule. B3 is the claimed word's
best detector box in the same geometric proposal pool (no attention, no small-box filter that the Look
pool does not have either), on the S4 bundle's own detector queries.

Two attention readings per claim (docs/LOOK_L1_PROTOKOL.md, Revizyon 1):
* predict -- the step that produces the word's first piece (what v4/LCR uses). The decoder is causal, so
  a swapped word cannot change this step's raw attention, only the caption mean. It answers: "before
  saying the word, was the model looking at the true object?"
* read    -- the step that has just read the whole word. It answers: "once the wrong word is in context,
  where does attention go?"

Corruptions (B references only):
* misidentified -- a non-person claim whose class has a Flickr30k Entities box gets the most plausible
  wrong same-kind word from lcr/corrupt.py's confusion table; the true referent is that box.
* invented -- a class no reference mentions and that S4's own detector scores CERTIFY absent: the raw
  image-wide maximum over all proposals is below the class threshold tau_c minus --absence-margin. The raw
  maximum is never below S4's rival-aware X_c, so S4 itself would call it absent. Classes without tau are
  never used. This is still PROXY absence (not detected != not there), and the output says so.

Reported: misidentified top-1/top-3 hit on the true referent vs B3, the optimism of reusing the correct
word's region (what the swap test in Check does) against recomputing it; invented-vs-correct AUROC of the
referent signals (correct = reference-supported real claims of the same images, an automatic proxy);
misidentified-vs-correct AUROC as a descriptive contrast. Confidence intervals: image-level bootstrap
(lcr.metrics.bootstrap_ci); every unit is one image. Fixed seed, fixed sample. Refuses validation/test.

    python scripts/47_look_forced_hallucination.py --n 300 --seed 415
"""
import argparse
from collections import Counter

import numpy as np
import torch

import _common  # noqa: F401
from lcr import cache, claims, config, corrupt, data, grounding, look, look_referent, look_v4, metrics
from lcr import region_evidence as re_
from lcr.captioner import Captioner
from lcr.look_sources import LookDecoder
from lcr.vocab import CLASS_INDEX, CLASSES, KIND, STUFF, THINGS

P = config.CACHE / "pipeline"
MODES = ("predict", "read")
SIGNALS = ("strength", "top1_concentration", "neg_entropy", "null_margin")
SURFACE = {c: (THINGS.get(c) or STUFF.get(c) or [c])[0] for c in CLASSES}


class TeacherForcing:
    """Duck-typed captioner exactly like scripts/review_look_e3.py's ReplayCap: only tok/bos/pad are read
    by the Captioner methods it borrows."""
    encode_captions = Captioner.encode_captions
    word_alignment = Captioner.word_alignment

    def __init__(self, dec: LookDecoder):
        self.dec, self.tok, self.bos, self.pad = dec, dec.tok, dec.bos, 0

    def cross_attention(self, emb, ids):
        with torch.no_grad(), self.dec.capture() as att:
            out = self.dec.forward(ids, emb)
        return torch.stack(att, 1), out.logits


def geometric_pool(prop, rule):
    """The proposals region_evidence.candidates may rank, before its attention-dependent filter."""
    boxes = prop["boxes"].astype(np.float32)
    wh = boxes[:, 2:] - boxes[:, :2]
    ok = np.isfinite(boxes).all(1) & (wh > 0).all(1)
    if rule["min_patches"]:
        ok &= (wh >= rule["min_patches"] / config.BLIP_GRID).all(1)
    return ok


def best_iou(gt, box) -> float:
    return float(look.iou(np.asarray(box, np.float32), np.asarray(gt, np.float32)).max())


class ImageData:
    """Everything one B image contributes: features, proposals, S4 detector scores, references, F30kE boxes."""

    def __init__(self, row, s4, mm, owl, queries):
        self.id = row["id"]
        idx = data.parse_id(self.id)[1]
        self.emb = torch.from_numpy(np.array(mm[idx]))[None].float()
        self.prop = owl[idx]
        self.boxes = self.prop["boxes"].astype(np.float32)
        self.pool = geometric_pool(self.prop, s4["look"])
        self.z = (re_.proposal_logits(self.prop, queries) - np.asarray(s4["mu"])) / np.asarray(s4["sd"])
        self.present = claims.reference_classes(data.references(self.id))
        self.gt = grounding.class_boxes(self.id)

    def b3_box(self, cls):
        s = np.where(self.pool, self.z[:, CLASS_INDEX[cls]], -np.inf)
        j = int(np.argmax(s)) if len(s) else 0
        return self.boxes[j] if len(s) and np.isfinite(s[j]) else None

    def absence_margin(self, cls, tau, margin):
        """tau_c - raw image-wide max if it certifies absence (by at least `margin`), else None."""
        t = tau[CLASS_INDEX[cls]]
        if not np.isfinite(t) or not len(self.z):
            return None
        x = float(self.z[:, CLASS_INDEX[cls]].max())
        return float(t - x) if x < t - margin else None


def run_pass(ctx, image, text):
    """Teacher-force `text` the way pipeline.run_v4 does and read both maps for every claim word."""
    ids, mask = ctx.cap.encode_captions([text])
    attention, _ = ctx.cap.cross_attention(image.emb, ids)
    attention, valid_t = attention[0], int(mask.sum()) - 1
    ws, positions = ctx.cap.word_alignment(ids[0])
    every = {c.index: c for c in claims.extract_from_words(ws, dedupe=False)}
    maps = {}
    for i, c in every.items():
        p = positions[i]
        sigma = config.LOOK_RULES_BY_KIND[c.kind]["sigma"]
        maps[i] = {}
        for mode, query in (("predict", p - 1), ("read", ctx.dec.word_end(ids, p) - 1)):
            signed = look_v4.signed_heads(attention, query, valid_t)
            raw = look_v4.selected_map(signed, ctx.heads)
            if mode == "predict":
                assert np.array_equal(raw, look_v4.attention_map(attention, p, valid_t, ctx.heads)), "left the v4 path"
            maps[i][mode] = (look.smooth_map(raw, sigma), look_v4.residual_strength(signed, ctx.heads))
    return dict(ws=ws, every=every, pipeline=[c.index for c in claims.extract_from_words(ws)], maps=maps)


def score_claim(ctx, image, passed, index, gt=None):
    """Top-1/top-3 (one ranking) and referent signals of one claim, for both readings."""
    c = passed["every"][index]
    others = [i for i in passed["pipeline"] if i != index]
    out = {}
    for mode in MODES:
        att, strength = passed["maps"][index][mode]
        top, _ = re_.candidates(att, image.prop, c.kind, ctx.rule, n=3)
        r = dict(n_candidates=int(len(top)), top1=int(top[0]) if len(top) else -1)
        if len(top):
            r.update(look_referent.signals(strength, att, image.boxes[top[0]],
                                           [passed["maps"][i][mode][0] for i in others]))
        else:                                   # no positive-attention box at all: the plainest "no referent"
            r.update(strength=float(strength), entropy=look_referent.entropy(att), top1_concentration=0.,
                     null_margin=float("nan"), null_rank=-1, null_n_other_claims=len(others))
        r["neg_entropy"] = -r["entropy"]
        if gt is not None:
            ious = [best_iou(gt, image.boxes[t]) for t in top]
            r.update(hit1=bool(ious and ious[0] >= .5), hit3=bool(any(v >= .5 for v in ious)))
        out[mode] = r
    return out


def real_claims(ctx, image, base):
    """Real thing/stuff claims of the unmodified draft, labelled by reference support (automatic proxy)."""
    return [dict(cls=base["every"][i].cls, in_refs=base["every"][i].cls in image.present,
                 **score_claim(ctx, image, base, i))
            for i in base["pipeline"] if base["every"][i].kind != "person"]


def article(cls, word):
    return "the" if KIND[cls] == "stuff" else ("an" if word[0] in "aeiou" else "a")


def try_misidentified(ctx, image, base, table, rng, dropped):
    cands = [i for i in base["pipeline"] if base["every"][i].kind != "person" and base["every"][i].cls in image.gt]
    if not cands:
        return None
    i = cands[int(rng.integers(len(cands)))]
    true_cls = base["every"][i].cls
    new_cls = corrupt.corrupt_label(true_cls, image.present, table, rng)
    if new_cls is None:
        dropped["misidentified_no_confusion"] += 1
        return None
    swapped = run_pass(ctx, image, " ".join(base["ws"][:i] + [SURFACE[new_cls]] + base["ws"][i + 1:]))
    if i not in swapped["every"] or swapped["every"][i].cls != new_cls:
        dropped["misidentified_not_reextracted"] += 1
        return None
    gt = image.gt[true_cls]
    original, recomputed = score_claim(ctx, image, base, i, gt), score_claim(ctx, image, swapped, i, gt)
    b3 = image.b3_box(new_cls)
    return dict(id=image.id, true_cls=true_cls, new_cls=new_cls, original=original, swapped=recomputed,
                b3_hit=bool(b3 is not None and best_iou(gt, b3) >= .5),
                top1_changed={m: original[m]["top1"] != recomputed[m]["top1"] for m in MODES})


def try_invented(ctx, image, base, rng, margin, dropped):
    claimed = {c.cls for c in base["every"].values()}
    certified = {c: m for c in CLASSES
                 if KIND[c] != "person" and c not in image.present and c not in claimed
                 and (m := image.absence_margin(c, ctx.tau, margin)) is not None}
    if not certified:
        dropped["invented_nothing_certified_absent"] += 1
        return None
    new_cls = str(rng.choice(sorted(certified)))
    word = SURFACE[new_cls]
    inserted = run_pass(ctx, image, " ".join(base["ws"] + ["and", article(new_cls, word), word]))
    index = len(base["ws"]) + 2
    if index not in inserted["every"] or inserted["every"][index].cls != new_cls:
        dropped["invented_not_reextracted"] += 1
        return None
    return dict(id=image.id, cls=new_cls, absence_margin=certified[new_cls], **score_claim(ctx, image, inserted, index))


# ------------------------------------------------------------------------------------------------ statistics
def rate(units, get):
    return metrics.bootstrap_ci(units, lambda us: 100 * float(np.mean([get(u) for u in us])))


def misidentified_summary(mis):
    out = dict(n=len(mis))
    if not mis:
        return out
    out["b3_hit1"] = rate(mis, lambda u: u["b3_hit"])
    for m in MODES:
        out[m] = dict(
            hit1=rate(mis, lambda u: u["swapped"][m]["hit1"]), hit3=rate(mis, lambda u: u["swapped"][m]["hit3"]),
            hit1_minus_b3=metrics.bootstrap_ci(mis, lambda us: 100 * float(np.mean(
                [u["swapped"][m]["hit1"] - u["b3_hit"] for u in us]))),
            reuse_minus_recomputed=metrics.bootstrap_ci(mis, lambda us: 100 * float(np.mean(
                [u["original"]["predict"]["hit1"] - u["swapped"][m]["hit1"] for u in us]))),
            top1_changed=100 * float(np.mean([u["top1_changed"][m] for u in mis])),
            no_candidate=sum(u["swapped"][m]["n_candidates"] == 0 for u in mis))
    return out


def auroc_vs_correct(units, negative):
    """Positives: reference-supported real claims; negatives: the units' corrupted claims. One unit = one image."""
    usable = [u for u in units if any(r["in_refs"] for r in u["real"])]
    out = dict(n_images=len(usable), n_correct=sum(sum(r["in_refs"] for r in u["real"]) for u in usable),
               n_negative=len(usable))
    for m in MODES:
        out[m] = {}
        for s in SIGNALS:
            def stat(us, m=m, s=s):
                pos = [r[m][s] for u in us for r in u["real"] if r["in_refs"] and np.isfinite(r[m][s])]
                neg = [negative(u)[m][s] for u in us if np.isfinite(negative(u)[m][s])]
                return metrics.auroc(np.array(pos), np.array(neg)) if pos and neg else float("nan")
            v, lo, hi = metrics.bootstrap_ci(usable, stat) if usable else (float("nan"),) * 3
            out[m][s] = dict(auroc=v, lo95=lo, hi95=hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="images per corruption kind")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--absence-margin", type=float, default=0.,
                    help="invented class: raw image-wide max must be below tau_c minus this (S4 logit scale)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows = cache.load_json(P / "A_B_lcr4.json")
    assert all(r["id"].startswith("train_") for r in rows), "Look L1 is B-only; refuses validation/test"
    s4 = cache.load_json(re_.BUNDLE_PATH)
    if rows[0].get("bundle_id") != s4["id"]:
        raise ValueError("A_B_lcr4.json was not produced by the current default Check bundle")
    queries = np.load(config.CACHE / "owl" / s4["query_file"])
    dec = LookDecoder()

    class Ctx:
        pass
    ctx = Ctx()
    ctx.dec, ctx.cap, ctx.rule = dec, TeacherForcing(dec), s4["look"]
    ctx.heads = look_v4.selection()[0]["heads"]
    ctx.tau = np.array([np.nan if t is None else t for t in s4["tau"]], np.float64)
    table = corrupt.build_confusions([data.references(r["id"]) for r in rows])
    mm, owl = cache.open_blip("train"), cache.OwlCache("train")

    mis, inv, dropped = [], [], Counter()
    for ri in rng.permutation(len(rows)):
        if len(mis) >= a.n and len(inv) >= a.n:
            break
        image = ImageData(rows[int(ri)], s4, mm, owl, queries)
        base = run_pass(ctx, image, " ".join(claims.words(rows[int(ri)]["draft"])))
        real = None
        if len(mis) < a.n and (u := try_misidentified(ctx, image, base, table, rng, dropped)) is not None:
            real = real_claims(ctx, image, base)
            mis.append(dict(u, real=real))
        if len(inv) < a.n and (u := try_invented(ctx, image, base, rng, a.absence_margin, dropped)) is not None:
            inv.append(dict(id=image.id, invented=u, real=real if real is not None else real_claims(ctx, image, base)))

    out = dict(protocol="docs/LOOK_L1_PROTOKOL.md", bundle_id=s4["id"], look_rule=s4["look"], heads=ctx.heads,
               seed=a.seed, absence_margin=a.absence_margin, dropped=dict(dropped),
               labels="correct = reference-supported real claim (automatic proxy); invented = S4-certified absent (proxy)",
               misidentified=misidentified_summary(mis),
               invented=dict(n=len(inv), mean_absence_margin=float(np.mean([u["invented"]["absence_margin"] for u in inv]))
                             if inv else None,
                             no_candidate={m: sum(u["invented"][m]["n_candidates"] == 0 for u in inv) for m in MODES}),
               invented_vs_correct=auroc_vs_correct(inv, lambda u: u["invented"]),
               misidentified_vs_correct=auroc_vs_correct(mis, lambda u: u["swapped"]),
               records=dict(misidentified=mis, invented=inv))
    cache.save_json(out, config.RESULTS / "look_forced_hallucination_B.json")
    write_markdown(out, config.ROOT / "docs" / "generated" / "LOOK_FORCED_HALLUCINATION_B.md")
    print(f"misidentified {len(mis)}, invented {len(inv)}, dropped {dict(dropped)}")


def fmt(ci, digits=1):
    v, lo, hi = ci
    return "–" if v != v else f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write_markdown(out, path):
    m, inv = out["misidentified"], out["invented"]
    L = ["# Look L1 — zorlanmış halüsinasyon testi (B)", "",
         "Üreten: `scripts/47_look_forced_hallucination.py`. Protokol: `docs/LOOK_L1_PROTOKOL.md`. "
         "GA: görüntü düzeyi bootstrap %95. *predict* = kelimeyi üreten adım (v4/LCR); *read* = kelimeyi okuyan adım.",
         f"Bundle `{out['bundle_id']}`, Look kuralı `{out['look_rule']}`, seed {out['seed']}, "
         f"yokluk marjı {out['absence_margin']}. Atılan örnekler: {out['dropped']}.", "",
         f"## Yanlış tanımlanmış (n = {m['n']})", ""]
    if m["n"]:
        L += [f"B3 (yanlış kelimenin detektör kutusu) top-1 isabet: {fmt(m['b3_hit1'])}", "",
              "| okuma | top-1 | top-3 | top-1 − B3 | yeniden kullanma − yeniden hesaplama | top-1 değişti % | aday yok |",
              "|---|---|---|---|---|---|---|"]
        for mode in MODES:
            r = m[mode]
            L.append(f"| {mode} | {fmt(r['hit1'])} | {fmt(r['hit3'])} | {fmt(r['hit1_minus_b3'])} | "
                     f"{fmt(r['reuse_minus_recomputed'])} | {r['top1_changed']:.1f} | {r['no_candidate']} |")
    L += ["", f"## Uydurulmuş (n = {inv['n']}; yokluk yalnız S4 detektör proxy'si)", ""]
    for key, title in (("invented_vs_correct", "Uydurulmuş vs doğru (ana ölçüm)"),
                       ("misidentified_vs_correct", "Yanlış tanımlanmış vs doğru (betimsel)")):
        r = out[key]
        L += [f"### {title}", "",
              f"{r['n_images']} görüntü, {r['n_correct']} referans destekli gerçek claim (proxy), {r['n_negative']} bozulmuş claim.", "",
              "| sinyal | AUROC predict | AUROC read |", "|---|---|---|"]
        for s in SIGNALS:
            p, q = r["predict"][s], r["read"][s]
            L.append(f"| {s} | {fmt((p['auroc'], p['lo95'], p['hi95']), 3)} | {fmt((q['auroc'], q['lo95'], q['hi95']), 3)} |")
        L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
