"""Look, map -> region (docs/LOOK_V2_PLAN.md section 3.5): choose the configuration on half B, report on validation.

    python scripts/16_look_regions.py --select A_B_sink --report A_validation_sink

Every variant scores the OWLv2 proposals from the claim's attention map plus class-agnostic box properties only
(area, OWLv2 objectness head, whether the box contains the map peak). The claimed class's detector score is never
used (principle 2: otherwise Check would verify the detector against itself).

Families (all restricted to proposals with sides >= MIN_BOX_PATCHES):
  excess   mass - beta * area                       (beta = 1 is round 1, R1)
  ratio    mass / area^alpha
  obj      either of the above times objectness^gamma (only boxes with positive base score)  (R3)
  peak     only proposals that contain the map's argmax patch (fallback: all)
  cam      threshold the map at tau * max, take the bounding box of the connected component with the most mass
           (CAM, Zhou+ CVPR'16), snap to the proposal with the highest IoU (x objectness^gamma)       (R2)
Target: Flickr30k Entities human boxes of the claim's class; metric: region accuracy (IoU >= 0.5), mean IoU.
Downstream (automatic proxy): AUROC of the chosen region's detector score for the claim vs 'a reference mentions it',
over ALL thing/stuff claims (claims with F30kE boxes are almost always mentioned by a reference, since F30kE
phrases come from the same sentences, so that subset has almost no negatives).
"""
import argparse
import itertools

import numpy as np
from scipy import ndimage

import _common  # noqa: F401
from lcr import cache, check, config, data, grounding, look, metrics, pipeline
from lcr.vocab import CLASS_INDEX

g = config.BLIP_GRID


def cam_box(att, tau):
    lab, n = ndimage.label(att >= tau * att.max())
    if n == 0:
        return None
    k = 1 + int(np.argmax(ndimage.sum(att, lab, index=np.arange(1, n + 1))))
    ys, xs = np.nonzero(lab == k)
    return np.array([xs.min() / g, ys.min() / g, (xs.max() + 1) / g, (ys.max() + 1) / g], np.float32)


def load(stem):
    """Per-claim arrays over the proposals; gtiou is None for claims whose class has no F30kE boxes."""
    Q = pipeline.class_queries()
    P = config.CACHE / "pipeline"
    rows = cache.load_json(P / f"{stem}.json")
    z = np.load(P / f"{stem}.maps.npz")
    midx = {(int(r), int(c)): i for i, (r, c) in enumerate(z["owner"])}
    owl = cache.OwlCache(data.parse_id(rows[0]["id"])[0])
    out = []
    for ri, r in enumerate(rows):
        gtd = grounding.class_boxes(r["id"])
        claims_gt = list(enumerate(r["claims"]))
        prop = owl[data.parse_id(r["id"])[1]]
        boxes = prop["boxes"].astype(np.float32)
        obj = prop["objectness"].astype(np.float32)
        ok = look.size_ok(boxes)
        area = look.box_area(boxes)
        scores = check.score(prop, Q)
        for ci, c in claims_gt:
            gt = gtd.get(c["cls"])
            gt = gt if gt is not None and len(gt) else None
            att = z["maps"][midx[(ri, ci)]].astype(np.float32)
            att = att / max(att.sum(), 1e-12)
            i, j = np.unravel_index(int(np.argmax(att)), att.shape)
            py, px = (i + 0.5) / g, (j + 0.5) / g
            peak = (boxes[:, 0] <= px) & (px <= boxes[:, 2]) & (boxes[:, 1] <= py) & (py <= boxes[:, 3])
            cams = {}
            for tau in (0.1, 0.2, 0.3, 0.5):
                cb = cam_box(att, tau)
                cams[tau] = look.iou(cb, boxes) if cb is not None else np.zeros(len(boxes))
            out.append(dict(kind=c["kind"], in_refs=c["in_refs"], ok=ok, area=area, obj=obj, peak=peak & ok,
                            mass=look.attention_mass(att, boxes), cams=cams,
                            gtiou=None if gt is None else np.max([look.iou(b, boxes) for b in gt], axis=0),
                            s_cls=scores[:, CLASS_INDEX[c["cls"]]]))
    return out


def configs():
    for beta, gamma, peak in itertools.product((0.5, 1.0, 1.5, 2.0), (0, 0.5, 1, 2), (False, True)):
        yield ("excess", beta, gamma, peak)
    for alpha, gamma, peak in itertools.product((0.25, 0.5, 0.75), (0, 0.5, 1, 2), (False, True)):
        yield ("ratio", alpha, gamma, peak)
    for tau, gamma in itertools.product((0.1, 0.2, 0.3, 0.5), (0, 0.5, 1, 2)):
        yield ("cam", tau, gamma, False)


def name(cfg):
    fam, p, gamma, peak = cfg
    sym = {"excess": "beta", "ratio": "alpha", "cam": "tau"}[fam]
    return f"{fam}({sym}={p}, gamma={gamma}{', peak' if peak else ''})"


def choose(x, cfg):
    if not x["ok"].any():
        return None
    fam, p, gamma, peak = cfg
    if fam == "cam":
        base = x["cams"][p]
    elif fam == "excess":
        base = x["mass"] - p * x["area"]
    else:
        base = x["mass"] / np.clip(x["area"], 1e-6, None) ** p
    s = np.where(base > 0, base * x["obj"] ** gamma, base) if gamma else base
    valid = x["peak"] if (peak and x["peak"].any()) else x["ok"]
    s = np.where(valid, s, -np.inf)
    return int(np.argmax(s))


auroc = metrics.auroc


def evaluate(allx, cfg):
    picks_all = [choose(x, cfg[x["kind"]] if isinstance(cfg, dict) else cfg) for x in allx]
    xs = [x for x in allx if x["gtiou"] is not None]
    picks = [k for x, k in zip(allx, picks_all) if x["gtiou"] is not None]
    iou_ = np.array([x["gtiou"][k] if k is not None else 0.0 for x, k in zip(xs, picks)])
    res = dict(n=len(xs), region=100 * float((iou_ >= 0.5).mean()), miou=float(iou_.mean()))
    for kind in ("person", "thing", "stuff"):
        m = np.array([x["kind"] == kind for x in xs])
        res[f"region_{kind}"] = 100 * float((iou_[m] >= 0.5).mean()) if m.any() else float("nan")
    ts = [(x["s_cls"][k] if k is not None else 0.0, x["in_refs"])
          for x, k in zip(allx, picks_all) if x["kind"] != "person"]
    res["auroc_thing_stuff"] = auroc([s for s, r in ts if r], [s for s, r in ts if not r])
    return res


def references(allx):
    xs = [x for x in allx if x["gtiou"] is not None]
    det = [int(np.argmax(np.where(x["ok"], x["s_cls"], -np.inf))) if x["ok"].any() else None for x in xs]
    iou_det = np.array([x["gtiou"][k] if k is not None else 0.0 for x, k in zip(xs, det)])
    oracle = np.array([x["gtiou"][x["ok"]].max() if x["ok"].any() else 0.0 for x in xs])
    ts = [(x["s_cls"][x["ok"]].max(), x["in_refs"]) for x in allx if x["kind"] != "person" and x["ok"].any()]
    return dict(detector_B3=100 * float((iou_det >= 0.5).mean()), oracle=100 * float((oracle >= 0.5).mean()),
                auroc_detector_B3=auroc([s for s, r in ts if r], [s for s, r in ts if not r]))



def run_by_kind(sel, args):
    baseline = ("ratio", .5, .5, False)
    grid = [baseline] + [c for c in configs() if c[0] == "ratio" and c != baseline]
    chosen, tables = {}, {}
    for kind in ("person", "thing", "stuff"):
        xs = [x for x in sel if x["kind"] == kind]
        table = sorted(((evaluate(xs, c), c) for c in grid), key=lambda t: -t[0]["region"])
        chosen[kind] = table[0][1]
        tables[kind] = [dict(cfg=list(c), **res) for res, c in table]
    out = dict(select=args.select, stage="T1: by-kind region rule", chosen_cfg=chosen,
               baseline=evaluate(sel, baseline), selected=evaluate(sel, chosen), grid_by_kind=tables)
    if not args.select_only:
        rep = load(args.report)
        out.update(report=args.report, baseline_report=evaluate(rep, baseline), selected_report=evaluate(rep, chosen))
    cache.save_json(out, config.RESULTS / f"look_regions_by_kind_{args.select}.json")
    print(out["stage"], chosen, "baseline", out["baseline"], "selected", out["selected"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select", default="A_B_sink")
    ap.add_argument("--report", default="A_validation_sink")
    ap.add_argument("--by-kind", action="store_true", help="select separate alpha/gamma/peak rules on B")
    ap.add_argument("--select-only", action="store_true", help="do not access validation")
    a = ap.parse_args()
    if a.select != "A_B_sink":
        ap.error("configuration selection is restricted to the existing half-B sink artifact")
    if "test" in a.report:
        ap.error("the test split is never used during development")
    sel = load(a.select)
    if a.by_kind:
        run_by_kind(sel, a)
        return
    if a.select_only:
        ap.error("--select-only currently requires --by-kind")
    table = sorted(((evaluate(sel, c), c) for c in configs()), key=lambda t: -t[0]["region"])
    r1 = ("excess", 1.0, 0, False)
    best = table[0][1]
    best_by_family = {}
    for res, c in table:
        best_by_family.setdefault((c[0], c[2] > 0, c[3]), (res, c))
    print(f"select on {a.select}: {sum(x['gtiou'] is not None for x in sel)}/{len(sel)} claims with F30kE boxes; references {references(sel)}")
    for res, c in table[:12]:
        print(f"  {res['region']:5.1f}  mIoU {res['miou']:.3f}  {name(c)}")

    rep = load(a.report)
    rows = [("R1 (round 1): excess mass", r1)]
    for (fam, use_obj, peak), (res, c) in sorted(best_by_family.items(), key=lambda t: -t[1][0]["region"]):
        if c != r1:
            rows.append((f"best {fam}{' + objectness' if use_obj else ''}{' + peak' if peak else ''} (on B)", c))
    out = dict(select=a.select, report=a.report, metric_version="auroc_average_ties_v2", chosen=name(best), chosen_cfg=list(best),
               references_select=references(sel), references_report=references(rep),
               grid_on_select=[dict(cfg=name(c), **res) for res, c in table],
               report_rows=[dict(label=lbl, cfg=name(c), select=evaluate(sel, c), report=evaluate(rep, c)) for lbl, c in rows])
    cache.save_json(out, config.RESULTS / f"look_regions_{a.report}.json")
    print(f"\nchosen on B: {name(best)}")
    print(f"report on {a.report} ({sum(x['gtiou'] is not None for x in rep)}/{len(rep)} claims with boxes); references {references(rep)}")
    print(f"  {'':52s} {'B':>6s} {'val':>6s} {'mIoU':>6s} {'pers':>6s} {'thing':>6s} {'stuff':>6s} {'AUROC':>6s}")
    for row in out["report_rows"]:
        s, v = row["select"], row["report"]
        print(f"  {row['label'][:52]:52s} {s['region']:6.1f} {v['region']:6.1f} {v['miou']:6.3f} {v['region_person']:6.1f} "
              f"{v['region_thing']:6.1f} {v['region_stuff']:6.1f} {v['auroc_thing_stuff']:6.3f}")


if __name__ == "__main__":
    main()
