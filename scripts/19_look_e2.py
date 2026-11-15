"""Bounded E2: 500 fixed B images, the prescribed maps and feature-level interventions."""
from collections import Counter
import time

import numpy as np

import _common  # noqa: F401
from lcr import cache, config, data, look_eval as ev
from lcr.look_sources import CANDIDATES, METHODS, LookDecoder


def main():
    if not cache.load_json(config.RESULTS / "look_e1.json").get("sanity_passed"):
        raise RuntimeError("Complete E1 and its gates before E2")
    if (config.RESULTS / "look_e2.json").exists():
        raise FileExistsError("E2 already completed; do not resample/reselect")
    rows = cache.load_json(config.CACHE / "pipeline" / "A_B_look2.json")
    chosen_indices = sorted(np.random.default_rng(config.SEED).choice(len(rows), 500, replace=False).tolist())
    chosen_ids = [rows[i]["id"] for i in chosen_indices]
    protocol = dict(seed=config.SEED, image_ids=chosen_ids, n_images=500, methods=list(METHODS),
        threshold_pp=2., source_selection="B subset only; fixed v2 box rule",
        contrast_target="highest-scoring whole-word token in blip_alts[0]'s class",
        missing_alternative="S3 fallback, availability reported", target="first_wordpiece",
        intervention="feature-level mask, fixed caption prefix; matched random token count; CLS retained",
        causal_acceptance="image-paired 95% interval for drop_look - drop_random strictly above zero",
        head_selection="excluded by direction document")
    cache.save_json(protocol, config.RESULTS / "look_e2_protocol.json")
    folder = config.CACHE / "look_e2"; folder.mkdir(exist_ok=True)
    saved, owners, diagnostics, causal = {m: [] for m in METHODS}, [], [], []
    progress = folder / "progress.json"; maps_path = folder / "sources.npz"
    completed = 0
    if progress.exists():
        old = cache.load_json(progress)
        assert old["image_ids"] == chosen_ids
        completed = old["completed_images"]
        z = np.load(maps_path)
        saved = {m: list(z[m]) for m in METHODS}; owners = z["owner"].tolist()
        diagnostics, causal = old["diagnostics"], old["causal"]
    decoder = LookDecoder()
    features = np.load(config.CACHE / "blip" / "train.f16.npy", mmap_mode="r")
    t0 = time.monotonic()
    if completed == 500 and old.get("intervention_version") != 2:
        # Preserve extracted maps/sample; refresh only the invalid mask-routing
        # intervention. This is the same E2 condition, not an expanded experiment.
        causal = []
        for sample_index, ri in enumerate(chosen_indices):
            row = rows[ri]; feat = features[data.parse_id(row["id"])[1]]; ids = decoder.tokens(row)
            for ci, claim in enumerate(row["claims"]):
                causal.append(dict(iid=row["id"], kind=claim["kind"],
                    **decoder.intervention(ids, claim, feat, config.SEED + 64*ri + ci)))
            if (sample_index+1) % 50 == 0:
                print(f"E2 corrected feature masks {sample_index+1}/500", flush=True)
        cache.save_json(dict(image_ids=chosen_ids, completed_images=500, intervention_version=2,
                             diagnostics=diagnostics, causal=causal), progress)
    for sample_index, ri in enumerate(chosen_indices):
        if sample_index < completed:
            continue
        row = rows[ri]; feat = features[data.parse_id(row["id"])[1]]
        ids, maps, infos = decoder.extract(row, feat)
        for ci, (claim, mm, info) in enumerate(zip(row["claims"], maps, infos)):
            owners.append([ri, ci])
            for method in METHODS:
                saved[method].append(mm[method])
            diagnostics.append(dict(iid=row["id"], ci=ci, **info))
            causal.append(dict(iid=row["id"], kind=claim["kind"],
                **decoder.intervention(ids, claim, feat, config.SEED + 64*ri + ci)))
        if (sample_index+1) % 25 == 0 or sample_index == 499:
            np.savez_compressed(maps_path, owner=np.asarray(owners, np.int32),
                                **{m: np.asarray(v, np.float16) for m, v in saved.items()})
            cache.save_json(dict(image_ids=chosen_ids, completed_images=sample_index+1,
                                 intervention_version=2, diagnostics=diagnostics, causal=causal), progress)
            print(f"E2 {sample_index+1}/500 images, {len(owners)} claims, {(time.monotonic()-t0)/60:.1f} min", flush=True)
    del decoder, features
    _, entries = ev.load_bank("A_B_look2", set(chosen_ids))
    assert [[e["ri"], e["ci"]] for e in entries] == owners
    baseline, records_base = ev.measure(entries)
    results, gate_results = {}, {}
    for method in METHODS:
        results[method], rr = ev.measure(entries, saved[method])
        gate_results[method] = sanity(entries, saved[method])
        results[method]["paired_bootstrap"] = ev.paired_bootstrap(records_base, rr, chosen_ids)
    # Mandatory reproduction of the existing caption-wide map, not a new audit.
    differences = [float(np.max(np.abs(ev.normalise(m)-e["att"])))
                   for m, e in zip(saved["S3_caption"], entries)]
    raw_cache = np.load(config.CACHE / "pipeline" / "A_B.maps.npz")
    raw_values = raw_cache["maps"]
    raw_index = {tuple(map(int, k)): j for j, k in enumerate(raw_cache["owner"])}
    raw_differences = [float(np.max(np.abs(ev.normalise(m)-ev.normalise(raw_values[raw_index[tuple(owner)]]))))
                       for m, owner in zip(saved["S1_raw"], owners)]
    if max(raw_differences) > .001:
        raise RuntimeError("Required S1 raw reference reproduction failed")
    accept = {m: dict(gain_pp=results[m]["all"]["region"]-baseline["all"]["region"],
                      passes_map_gates=gate_results[m]["passed"]) for m in CANDIDATES}
    eligible = [m for m in CANDIDATES if accept[m]["gain_pp"] >= 2. and accept[m]["passes_map_gates"]]
    winner = max(eligible, key=lambda m: results[m]["all"]["region"]) if eligible else None
    reference_compatible = max(differences) <= .001
    # The acceptance comparison always uses the fixed archived S3 control.
    source_promotion = winner
    causal_result = causal_summary(causal, chosen_ids)
    out = dict(protocol=protocol, n_claims=len(entries), baseline=baseline, sources=results,
               map_gates=gate_results, acceptance=accept, threshold_winner=winner, accepted_source=source_promotion,
               reproduction=dict(max_abs_S1=max(raw_differences), S1_passed=True,
                   max_abs_S3=max(differences), mean_abs_S3=float(np.mean(differences)),
                   S3_reference_compatible=reference_compatible,
                   note="archived S3 is the fixed control; recomputed caption-reference drift is reported"),
               contrast_available=sum(d["contrast_available"] for d in diagnostics),
               multi_piece_claims=sum(d["pieces"] > 1 for d in diagnostics),
               degenerate_maps=dict(Counter(m for d in diagnostics for m, flag in d["degenerate"].items() if flag)),
               nonzero_gradients=sum(d.get("gradient_abs_sum", 0) > 0 for d in diagnostics),
               causal=causal_result, intervention_version=2, maps_path=str(maps_path.relative_to(config.ROOT)))
    cache.save_json(out, config.RESULTS / "look_e2.json")
    print("E2 acceptance", accept, "threshold winner", winner, "promoted", source_promotion,
          "causal", causal_result, flush=True)


def sanity(entries, maps):
    from lcr import grounding
    groups = {}
    for i, e in enumerate(entries):
        if e["gt"] is not None:
            groups.setdefault(e["iid"], []).append(i)
    own, transposed, other, matched_own = [], [], [], []
    def point(att, gt):
        y, x = np.unravel_index(np.asarray(att).argmax(), (24, 24))
        return grounding.point_in_boxes((y+.5)/24, (x+.5)/24, gt)
    for indices in groups.values():
        for i in indices:
            e = entries[i]
            own.append(point(maps[i], e["gt"])); transposed.append(point(np.asarray(maps[i]).T, e["gt"]))
            alternatives = [j for j in indices if entries[j]["cls"] != e["cls"]]
            if alternatives:
                matched_own.append(point(maps[i], e["gt"]))
                other.append(point(maps[alternatives[0]], e["gt"]))
    avg = lambda xs: 100*float(np.mean(xs))
    return dict(pointing=avg(own), transposed=avg(transposed), swap_own=avg(matched_own),
                swap_other=avg(other), passed=bool(avg(own)>avg(transposed) and avg(matched_own)>avg(other)))


def causal_summary(causal, image_ids):
    idx = {iid: i for i, iid in enumerate(image_ids)}
    n = np.zeros(len(idx)); sums = np.zeros((len(idx), 3))
    for c in causal:
        i = idx[c["iid"]]; n[i] += 1
        sums[i] += [c["drop_look"], c["drop_random"], c["drop_detector"]]
    rng = np.random.default_rng(config.SEED); delta = []
    for _ in range(2000):
        draw = rng.integers(0, len(idx), len(idx)); den = n[draw].sum()
        if den:
            totals = sums[draw].sum(axis=0)/den; delta.append(totals[0]-totals[1])
    means = sums.sum(axis=0)/n.sum(); interval = np.quantile(delta, [.025, .975]).tolist()
    return dict(n_claims=len(causal), mean_logp_drop=dict(zip(("look", "random", "detector"), means.tolist())),
                look_minus_random=float(means[0]-means[1]), ci95=interval, bootstrap_unit="image", n_boot=2000,
                feature_level_gate_passed=bool(interval[0]>0),
                mean_masked_look_tokens=float(np.mean([c["look_tokens"] for c in causal])),
                mean_masked_detector_tokens=float(np.mean([c["detector_tokens"] for c in causal])))


if __name__ == "__main__":
    main()
