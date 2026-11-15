"""E1: the fixed grid in CODEX_SONRAKI_YON.md, B selection then one validation report."""
import itertools

import numpy as np

import _common  # noqa: F401
from lcr import cache, config, look, look_eval as ev

ALPHAS = (.25, .5, .75, 1.)
GAMMAS = (.25, .5, 1.)
SIGMAS = (0., .5, 1.)
BASE = dict(alpha=.5, gamma=.5, sigma=0.)


def main():
    outpath = config.RESULTS / "look_e1.json"
    if outpath.exists():
        raise FileExistsError("E1 already exists; do not select again after seeing validation")
    protocol = dict(alpha=list(ALPHAS), gamma=list(GAMMAS), sigma=list(SIGMAS),
        configurations_per_kind=36, threshold_pp=1., seed=config.SEED,
        sigma_interpretation="same fixed sigma grid for every kind; triples independently selected as instructed",
        selection="half B only", report="validation once", bootstrap_unit="image", bootstrap_replicates=2000)
    cache.save_json(protocol, config.RESULTS / "look_e1_protocol.json")
    rows_b, entries = ev.load_bank("A_B_look2")
    baseline, _ = ev.measure(entries)
    grid = {kind: [] for kind in ev.KINDS}
    # No class-conditioned detector scores, peak constraints, or expanded grid.
    for sigma in SIGMAS:
        print("E1 B sigma", sigma, flush=True)
        masses = [look.attention_mass(ev.transformed(e["att"], sigma), e["boxes"]) for e in entries]
        for alpha, gamma in itertools.product(ALPHAS, GAMMAS):
            grouped = {kind: [] for kind in ev.KINDS}
            for e, mass in zip(entries, masses):
                if e["gtiou"] is None:
                    continue
                score = mass / np.clip(e["area"], 1e-6, None)**alpha * np.clip(e["obj"], 1e-6, None)**gamma
                k = int(np.argmax(np.where(e["valid"], score, -np.inf))) if e["valid"].any() else None
                grouped[e["kind"]].append(bool(k is not None and e["gtiou"][k] >= .5))
            for kind, hits in grouped.items():
                grid[kind].append(dict(alpha=alpha, gamma=gamma, sigma=sigma,
                    n=len(hits), region=100*float(np.mean(hits))))
    chosen, acceptance = {}, {}
    for kind in ev.KINDS:
        # Prefer baseline parameters when scores tie.
        ranked = sorted(grid[kind], key=lambda x: (-x["region"], x["sigma"],
                         abs(x["alpha"]-.5)+abs(x["gamma"]-.5)))
        best = ranked[0]; gain = best["region"] - baseline[kind]["region"]
        accepted = gain >= protocol["threshold_pp"] - 1e-10
        chosen[kind] = {k: best[k] for k in BASE} if accepted else dict(BASE)
        acceptance[kind] = dict(best=best, gain_pp=gain, accepted=accepted)
    bmaps = [ev.transformed(e["att"], chosen[e["kind"]]["sigma"]) for e in entries]
    selected_b, _ = ev.measure(entries, bmaps, chosen)
    # Persist the choice before any validation annotation is read.
    out = dict(protocol=protocol, chosen=chosen, acceptance=acceptance,
               grid_on_B=grid, baseline_B=baseline, selected_B=selected_b)
    cache.save_json(out, config.RESULTS / "look_e1_selection.json")
    print("E1 choices locked", chosen, acceptance, flush=True)
    rows_v, val = ev.load_bank("A_validation_look2")
    vbase, rbase = ev.measure(val)
    vmaps = [ev.transformed(e["att"], chosen[e["kind"]]["sigma"]) for e in val]
    vselected, rselected = ev.measure(val, vmaps, chosen)
    out.update(baseline_validation=vbase, selected_validation=vselected,
        paired_bootstrap=ev.paired_bootstrap(rbase, rselected, [r["id"] for r in rows_v]))
    # Save the exact maps for script 14's orientation/specificity gates.
    p = config.CACHE / "pipeline"
    for stem, rows, xs, maps in (("A_B_e1", rows_b, entries, bmaps),
                                  ("A_validation_e1", rows_v, val, vmaps)):
        gate_rows = [dict(id=r["id"], draft=r["draft"], map_only=True,
                         artifact_purpose="E1 map sanity; use look3 for frozen region records",
                         claims=[{k: c[k] for k in ("cls", "kind", "word", "pos", "in_refs")} for c in r["claims"]])
                     for r in rows]
        cache.save_json(gate_rows, p / f"{stem}.json")
        np.savez_compressed(p / f"{stem}.maps.npz", maps=np.array(maps, np.float16),
                            owner=np.array([[e["ri"], e["ci"]] for e in xs], np.int32))
    cache.save_json(out, outpath)
    print("E1 validation", vselected, "paired bootstrap", out["paired_bootstrap"], flush=True)


if __name__ == "__main__":
    main()
