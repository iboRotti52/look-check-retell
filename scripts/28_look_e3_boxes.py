"""Look E3 stage B (docs/LOOK_E3_PROTOKOL.md): SAM-based boxes per kind, CV on half B, one validation report.

    python scripts/28_look_e3_boxes.py            # select on B, lock -> results/look_e3_boxes_selection.json
    python scripts/28_look_e3_boxes.py --report   # validation once -> results/look_e3.json
"""
import importlib
import sys

import numpy as np


def plain(o):
    """numpy scalars/arrays -> JSON types."""
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o

import _common  # noqa: F401
from lcr import cache, config, look, look_eval as ev
from lcr.look_e3_eval import KINDS, Bank, record_list

FOLDER = config.CACHE / "look_e3"
STEMS = {"B": "A_B_look2", "val": "A_validation_look2"}
ALPHAS = (.25, .5, .75)
DELTAS = (.5, 1., 2.)
THRESHOLD = 1.0
SEL = config.RESULTS / "look_e3_boxes_selection.json"


def stage_a(bank, stem):
    sel = cache.load_json(config.RESULTS / "look_e3_selection.json")
    w = sel.get("winner")
    if not w:
        return bank.archived
    return importlib.import_module("26_look_e3_select").final_maps(w, stem)


def gt_iou(e, box):
    return float(look.iou(box, e["gt"]).max()) if e["gt"] is not None and len(e["gt"]) else 0.


def variant_hits(bank, maps, sam):
    """Per-claim hits for: base (v3 rule on stage-A map), B1, B2(alpha), B3(delta)."""
    rules = config.LOOK_RULES_BY_KIND
    att = bank.prepare(maps)
    base_k = bank.regions(att, prepared=True)
    out = {"base": np.zeros(bank.n, bool), "B1": np.zeros(bank.n, bool),
           **{f"B2_a{a}": np.zeros(bank.n, bool) for a in ALPHAS},
           **{f"B3_d{d}": np.zeros(bank.n, bool) for d in DELTAS}}
    any_mask = np.zeros(bank.n, bool)
    for j, e in enumerate(bank.e):
        if e["gt"] is None:
            continue
        k = base_k[j]
        out["base"][j] = k >= 0 and e["gtiou"][k] >= .5
        boxes = sam["boxes"][j]
        b1 = boxes[int(np.argmax(sam["iou"][j]))]
        out["B1"][j] = gt_iou(e, b1) >= .5
        any_mask[j] = max(gt_iou(e, b) for b in boxes) >= .5
        mass = look.attention_mass(att[j], boxes)
        for a in ALPHAS:
            pick = boxes[int(np.argmax(mass / np.clip(look.box_area(boxes), 1e-6, None) ** a))]
            out[f"B2_a{a}"][j] = gt_iou(e, pick) >= .5
        r = rules[e["kind"]]
        pm = np.einsum("ni,ij,nj->n", bank.wy[j], att[j], bank.wx[j])
        score = pm / np.clip(e["area"], 1e-6, None) ** r["alpha"] * np.clip(e["obj"], 1e-6, None) ** r["gamma"]
        overlap = look.iou(b1, e["boxes"])
        for d in DELTAS:
            s = np.where(e["valid"], score * np.clip(overlap, 1e-6, None) ** d, -np.inf)
            kk = int(np.argmax(s)) if e["valid"].any() else -1
            out[f"B3_d{d}"][j] = kk >= 0 and e["gtiou"][kk] >= .5
    return out, any_mask


def load_sam(bank, stem):
    z = np.load(FOLDER / f"{stem}_sam.npz")
    assert (z["owner"] == bank.owner).all()
    return {k: z[k] for k in z.files}


def main():
    if SEL.exists():
        raise FileExistsError("stage B locked")
    bank = Bank(STEMS["B"]); sam = load_sam(bank, STEMS["B"])
    hits, any_mask = variant_hits(bank, stage_a(bank, STEMS["B"]), sam)
    names = [n for n in hits if n != "base"]
    res, rule = {}, {}
    for kind in KINDS:
        m = bank.has_gt & (bank.kind == kind)
        cvh = np.zeros(bank.n, bool); per_fold = []
        for f in range(5):
            tr, te = m & (bank.fold != f), m & (bank.fold == f)
            k = int(np.argmax([hits[n][tr].mean() for n in names])); per_fold.append(names[k])
            cvh[te] = hits[names[k]][te]
        base = 100 * hits["base"][m].mean(); cv = 100 * cvh[m].mean()
        full = names[int(np.argmax([hits[n][m].mean() for n in names]))]
        accepted = cv - base >= THRESHOLD - 1e-10
        rule[kind] = full if accepted else "base"
        res[kind] = dict(n=int(m.sum()), base=base, cv=cv, cv_gain_pp=cv - base, per_fold=per_fold,
                         full_B_best=full, accepted=accepted, sam_any_mask_upper_bound=100 * any_mask[m].mean(),
                         B={n: 100 * hits[n][m].mean() for n in hits})
        print(kind, {k: v for k, v in res[kind].items() if k != "B"}, flush=True)
    cache.save_json(plain(dict(protocol="docs/LOOK_E3_PROTOKOL.md", threshold_pp=THRESHOLD, kinds=res, rule=rule,
                         stage_a_source=str(sam["source"]))), SEL)
    print("E3 stage B locked", rule)


def report():
    sel = cache.load_json(SEL)
    out_path = config.RESULTS / "look_e3.json"
    if out_path.exists():
        raise FileExistsError("E3 validation already reported")
    bank = Bank(STEMS["val"]); sam = load_sam(bank, STEMS["val"])
    stage_a_maps = stage_a(bank, STEMS["val"])
    hits, any_mask = variant_hits(bank, stage_a_maps, sam)
    final = np.array([hits[sel["rule"][k]][j] if bank.has_gt[j] else False for j, k in enumerate(bank.kind)])
    v3 = bank.hits(bank.archived)
    rows = [r["id"] for r in bank.rows]
    summ = lambda h: {k: dict(n=int((bank.has_gt & ((bank.kind == k) if k != "all" else True)).sum()),
                              region=100 * float(h[bank.has_gt & ((bank.kind == k) if k != "all" else True)].mean()))
                      for k in ("all", *KINDS)}
    out = dict(rule=sel["rule"], stage_a_source=str(sam["source"]), v3=summ(v3), stage_a=summ(hits["base"]),
               final=summ(final), variants={n: summ(h) for n, h in hits.items()},
               sam_any_mask_upper_bound=summ(any_mask),
               bootstrap_final_vs_v3=ev.paired_bootstrap(record_list(bank, v3), record_list(bank, final), rows),
               bootstrap_stage_a_vs_v3=ev.paired_bootstrap(record_list(bank, v3), record_list(bank, hits["base"]), rows))
    cache.save_json(plain(out), out_path)
    print("v3", out["v3"]["all"], "stage A", out["stage_a"]["all"], "final", out["final"], out["bootstrap_final_vs_v3"]["all"])


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
