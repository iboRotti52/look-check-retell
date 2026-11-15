"""Look E3 stage A (docs/LOOK_E3_PROTOKOL.md): map families, 5-fold image CV on half B, one validation report.

Needs scripts/25_look_e3_extract.py for both splits. Selection is written to results/look_e3_selection.json
before any validation human box is read; the validation report goes to results/look_e3.json.
"""
import itertools
import json

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
from lcr import cache, config, look_eval as ev
from lcr.look_e3_eval import Bank, component_entropy, rank_maps, record_list, sanity, unit_maps

FOLDER = config.CACHE / "look_e3"
STEMS = {"B": "A_B_look2", "val": "A_validation_look2"}
M_GRID = (4, 8, 16, 32, 48)
LAST_K = (1, 2, 4, 6, 8, 12)
THRESHOLD = 1.0


def load(stem):
    info = cache.load_json(FOLDER / f"{stem}_info.json")
    heads = np.load(FOLDER / f"{stem}_heads.f16.npy", mmap_mode="r")
    maps = dict(np.load(FOLDER / f"{stem}_maps.npz"))
    return info, heads, {k: v.reshape(len(v), -1).astype(np.float32) for k, v in maps.items()}


def head_mean(heads, sel, chunk=512):
    """clamp(mean over the selected (layer, head) pairs of the signed sink maps)."""
    flat_sel = np.asarray(sel)
    out = np.empty((heads.shape[0], 576), np.float32)
    for s in range(0, heads.shape[0], chunk):
        h = np.asarray(heads[s:s + chunk], np.float32).reshape(-1, 144, 576)
        out[s:s + chunk] = h[:, flat_sel].mean(1)
    return unit_maps(out)


def head_stats(bank, heads, chunk=256):
    """Per head: pointing hit per claim (labels), and label-free specificity / component entropy."""
    point = np.zeros((bank.n, 144), bool)
    entropy = np.zeros(144); ent_n = 0
    peaks = np.zeros((bank.n, 144), np.int32)
    for s in range(0, bank.n, chunk):
        h = np.clip(np.asarray(heads[s:s + chunk], np.float32).reshape(-1, 144, 576), 0, None)
        peaks[s:s + chunk] = h.argmax(-1)
        point[s:s + chunk] = np.take_along_axis(bank.ptmask[s:s + chunk], peaks[s:s + chunk], 1)
        if s % (chunk * 4) == 0:  # entropy on a fixed systematic subsample (label-free)
            for row in h[::8]:
                entropy += [component_entropy(row[k].reshape(24, 24)) for k in range(144)]; ent_n += 1
    # specificity: same image, different class -> how different are the head's maps (1 - histogram intersection)
    spec = np.zeros(144); pairs = 0
    by_img = {}
    for j in range(bank.n):
        by_img.setdefault(bank.iid[j], []).append(j)
    for js in by_img.values():
        for a, b in itertools.combinations(js, 2):
            if bank.cls[a] == bank.cls[b]:
                continue
            ha = unit_maps(np.clip(np.asarray(heads[a], np.float32).reshape(144, 576), 0, None))
            hb = unit_maps(np.clip(np.asarray(heads[b], np.float32).reshape(144, 576), 0, None))
            spec += 1 - np.minimum(ha, hb).sum(1); pairs += 1
    spec /= max(pairs, 1); entropy /= max(ent_n, 1)
    return point, dict(specificity=spec.tolist(), entropy=entropy.tolist(), pairs=pairs,
                       label_free_score=(spec / (1 + entropy)).tolist())


def cv(bank, family_hits, baseline_hits):
    """family_hits: fold -> list over configs of per-claim hits (fold-specific for supervised families)."""
    gt = bank.has_gt
    total, chosen = np.zeros(bank.n, bool), []
    for f in range(5):
        configs = family_hits(f)
        train = gt & (bank.fold != f); test = gt & (bank.fold == f)
        k = int(np.argmax([h[train].mean() for h in configs]))
        chosen.append(k); total[test] = configs[k][test]
    acc, base = 100 * total[gt].mean(), 100 * baseline_hits[gt].mean()
    return dict(cv_region=acc, baseline=base, cv_gain_pp=acc - base, chosen_per_fold=chosen)


def config_maps(family, cfg, heads_sel, maps, heads):
    """Maps of one family configuration for any split."""
    if family.startswith("A1_"):
        return head_mean(heads, heads_sel)
    if family == "A6_ensemble":
        partner, how = cfg.split("+")[1].split(":")
        if how == "rank":
            return unit_maps(rank_maps(maps["S3"]) + rank_maps(maps[partner]))
        return unit_maps(maps["S3"] + maps[partner])
    return unit_maps(maps[cfg])


def final_maps(w, stem):
    _, heads, maps = load(stem)
    return config_maps(w["family"], w["config"], w.get("heads"), maps, heads)


def report():
    """Validation, once, after the stage-A lock."""
    sel = cache.load_json(config.RESULTS / "look_e3_selection.json")
    out_path = config.RESULTS / "look_e3_stageA.json"
    if out_path.exists():
        raise FileExistsError("stage-A validation report already written")
    bank = Bank(STEMS["val"])
    info, heads, maps = load(STEMS["val"])
    assert [tuple(o) for o in info["owner"]] == [tuple(o) for o in bank.owner]
    v3 = bank.hits(bank.archived)
    rows = [r["id"] for r in bank.rows]
    rep = dict(v3=bank.summary(bank.archived), S3_recomputed=bank.summary(maps["S3"]), families={})
    for fam, r in sel["refit"].items():
        m = config_maps(fam, r["config"], r.get("heads"), maps, heads)
        h = bank.hits(m)
        rep["families"][fam] = dict(config=r["config"], summary=bank.summary(m),
            paired_bootstrap=ev.paired_bootstrap(record_list(bank, v3), record_list(bank, h), rows))
    w = sel.get("winner")
    if w:
        bmaps = final_maps(w, STEMS["B"])
        rep["winner"] = w["family"]
        rep["sanity_B"] = sanity(Bank(STEMS["B"]), bmaps)
        rep["sanity_val"] = sanity(bank, config_maps(w["family"], w["config"], w.get("heads"), maps, heads))
    cache.save_json(plain(rep), out_path)
    print(json.dumps({f: (v["config"], round(v["summary"]["all"]["region"], 2), v["paired_bootstrap"]["all"])
                      for f, v in rep["families"].items()}, indent=1))
    print("v3", rep["v3"]["all"], "winner", w and w["family"], rep.get("sanity_B"), rep.get("sanity_val"))


def main():
    if (config.RESULTS / "look_e3_selection.json").exists():
        raise FileExistsError("E3 stage A is locked; do not reselect after seeing validation")
    bank = Bank(STEMS["B"])
    info, heads, maps = load(STEMS["B"])
    assert [tuple(o) for o in info["owner"]] == [tuple(o) for o in bank.owner]
    v3 = bank.hits(bank.archived)
    s3 = bank.hits(maps["S3"])
    print("B v3", 100 * v3[bank.has_gt].mean(), "S3 recomputed", 100 * s3[bank.has_gt].mean(), flush=True)

    point, label_free = head_stats(bank, heads)
    lf_rank = np.argsort(-np.asarray(label_free["label_free_score"]))
    layer_sel = [list(range(l * 12, l * 12 + 12)) for l in range(12)] + \
                [list(range((12 - k) * 12, 144)) for k in LAST_K]
    names_g2 = [f"layer{l+1}" for l in range(12)] + [f"last{k}" for k in LAST_K]
    cache_hits = {}

    def hits_of(key, make):
        if key not in cache_hits:
            cache_hits[key] = bank.hits(make())
        return cache_hits[key]

    families = {}
    g2 = [hits_of(n, lambda s=s: head_mean(heads, s)) for n, s in zip(names_g2, layer_sel)]
    families["A1_G2_layers"] = (lambda f: g2, names_g2)
    g3 = [hits_of(f"lf{m}", lambda m=m: head_mean(heads, lf_rank[:m])) for m in M_GRID]
    families["A1_G3_labelfree_heads"] = (lambda f: g3, [f"top{m}" for m in M_GRID])

    def g4(f):
        train = bank.has_gt & (bank.fold != f)
        rank = np.argsort(-point[train].mean(0))
        return [hits_of(f"sup{f}_{m}", lambda m=m, r=rank: head_mean(heads, r[:m])) for m in M_GRID]
    families["A1_G4_supervised_heads"] = (g4, [f"top{m}" for m in M_GRID])
    for fam, names in (("A0_sink_all_positions", ["S3_all"]), ("A2_phrase", ["A2_phrase"]),
                       ("A3_ratio", ["A3_ratio"]), ("A4_chefer", ["A4_chefer"]), ("A5_salience_dropout", ["A5_q5", "A5_q10"])):
        hh = [hits_of(n, lambda n=n: maps[n]) for n in names]
        families[fam] = (lambda f, hh=hh: hh, names)
    ens_names, ens = [], []
    for partner in ("S2_gradcam", "S4_contrast", "A4_chefer", "A5_q10"):
        for how in ("rank", "prob"):
            key = f"S3+{partner}:{how}"
            make = (lambda p=partner: unit_maps(rank_maps(maps["S3"]) + rank_maps(maps[p]))) if how == "rank" \
                else (lambda p=partner: unit_maps(maps["S3"] + maps[p]))
            ens.append(hits_of(key, make)); ens_names.append(key)
    families["A6_ensemble"] = (lambda f: ens, ens_names)

    results = {}
    for fam, (fh, names) in families.items():
        r = cv(bank, fh, v3); r["vs_S3_recomputed_pp"] = r["cv_region"] - 100 * s3[bank.has_gt].mean()
        r["configs"] = names; results[fam] = r
        print(fam, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "configs"}, flush=True)
    # full-B refit of every family (for reporting) and the winner
    refit = {}
    for fam, (fh, names) in families.items():
        configs = fh(None) if fam != "A1_G4_supervised_heads" else None
        if fam == "A1_G4_supervised_heads":
            rank = np.argsort(-point[bank.has_gt].mean(0))
            configs = [hits_of(f"supall_{m}", lambda m=m: head_mean(heads, rank[:m])) for m in M_GRID]
            refit_heads = rank
        k = int(np.argmax([h[bank.has_gt].mean() for h in configs]))
        refit[fam] = dict(config=names[k], B_region=100 * configs[k][bank.has_gt].mean())
        if fam == "A1_G4_supervised_heads":
            refit[fam]["heads"] = refit_heads[:M_GRID[k]].tolist()
        if fam == "A1_G3_labelfree_heads":
            refit[fam]["heads"] = lf_rank[:M_GRID[k]].tolist()
        if fam == "A1_G2_layers":
            refit[fam]["heads"] = layer_sel[k]
    best = max(results, key=lambda f: results[f]["cv_gain_pp"])
    accepted = results[best]["cv_gain_pp"] >= THRESHOLD - 1e-10
    winner = dict(family=best, accepted=accepted, **refit[best]) if accepted else None
    out = dict(protocol="docs/LOOK_E3_PROTOKOL.md", threshold_pp=THRESHOLD, n_B=int(bank.has_gt.sum()),
               B_v3=100 * v3[bank.has_gt].mean(), B_S3_recomputed=100 * s3[bank.has_gt].mean(),
               families=results, refit=refit, winner=winner, head_label_free=label_free,
               head_pointing_B=(100 * point[bank.has_gt].mean(0)).tolist())
    cache.save_json(plain(out), config.RESULTS / "look_e3_selection.json")
    print("E3 stage A locked:", winner, flush=True)


if __name__ == "__main__":
    import sys
    report() if "--report" in sys.argv else main()
