"""OWLv2 text queries: single template vs a template ensemble (choose on B, report on validation).

    python scripts/23_owl_queries.py

Round 1 queried every class as "a photo of a {}" — ungrammatical for mass nouns ("a photo of a grass") and
plural-only nouns ("a photo of a pants"). CLIP-style prompt ensembling averages the L2-normalised text embeddings
of several templates; ViLD (Gu+ ICLR'22) used CLIP's ensemble plus detection-specific "... in the scene" prompts,
and OWL-ViT-based open-world work reports averaging the seven CLIP prompts per class.
The cached OWLv2 proposal embeddings make any new query exact without re-running the detector (lcr/check.py).
Look regions are untouched (Look never uses class scores); only the class score inside them changes.
Writes data/cache/owl/class_queries_<set>.npy and results/owl_queries.json.
"""
import numpy as np
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor

import _common  # noqa: F401
from lcr import cache, calibrate, check, config, data
from lcr.vocab import CLASS_INDEX, CLASSES, KIND

DISPLAY = {"tshirt": "t-shirt"}
MASS = {c for c in CLASSES if KIND[c] == "stuff"} | {"pants", "shorts", "glasses", "goggles", "stairs"}


def art(name):
    """'a dog' / 'grass' / 'pants' (no article for mass and plural-only nouns)."""
    n = DISPLAY.get(name, name)
    return n if name in MASS else ("an " if n[0] in "aeiou" else "a ") + n


SETS = {
    "single": [lambda n: f"a photo of a {DISPLAY.get(n, n)}"],                    # round 1, exactly
    "grammatical": [lambda n: f"a photo of {art(n)}"],
    "ensemble": [lambda n: f"a photo of {art(n)}", lambda n: f"a photo of the {DISPLAY.get(n, n)}",
                 lambda n: art(n), lambda n: DISPLAY.get(n, n), lambda n: f"a close-up photo of {art(n)}",
                 lambda n: f"a cropped photo of the {DISPLAY.get(n, n)}", lambda n: f"there is {art(n)} in the scene"],
}

proc = Owlv2Processor.from_pretrained(config.OWL_NAME)
model = Owlv2ForObjectDetection.from_pretrained(config.OWL_NAME).eval()


@torch.no_grad()
def embed(prompts):
    enc = proc(text=prompts, return_tensors="pt", padding="max_length")
    out = model.owlv2.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    q = model.owlv2.text_projection(out.pooler_output)
    return (q / (q.norm(dim=-1, keepdim=True) + 1e-6)).numpy()


queries = {}
for name, temps in SETS.items():
    e = np.mean([embed([t(c) for c in CLASSES]) for t in temps], axis=0)
    queries[name] = e / np.linalg.norm(e, axis=1, keepdims=True)
    np.save(config.CACHE / "owl" / f"class_queries_{name}.npy", queries[name].astype(np.float32))
orig = np.load(config.CACHE / "owl" / "class_queries.npy")
print("single template reproduces the cached queries: max |diff| =", float(np.abs(queries["single"] - orig).max()))


def collect(stem):
    rows = cache.load_json(config.CACHE / "pipeline" / f"{stem}.json")
    owl = cache.OwlCache(data.parse_id(rows[0]["id"])[0])
    out = {k: [] for k in queries}; meta = []
    for r in rows:
        prop = owl[data.parse_id(r["id"])[1]]
        sc = {k: check.score(prop, q) for k, q in queries.items()}
        for c in r["claims"]:
            reg = c["rec_att"]["region"]
            if reg < 0 or c["kind"] == "person":
                continue
            j = CLASS_INDEX[c["cls"]]
            for k in queries:
                out[k].append((sc[k][reg, j], sc[k][:, j].max()))
            meta.append((j, bool(c["in_refs"]), calibrate.fold_of(r["id"])))
    return {k: np.array(v) for k, v in out.items()}, np.array([m[0] for m in meta]), \
        np.array([m[1] for m in meta]), np.array([m[2] for m in meta])


SB, cB, yB, fB = collect("A_B_look2")
SV, cV, yV, _ = collect("A_validation_look2")
res = {}
for k in queries:
    oof = calibrate.cross_fit("offset_slope", SB[k][:, 0], cB, yB, fB)
    cal = calibrate.make("offset_slope").fit(SB[k][:, 0], cB, yB)
    res[k] = dict(B_auroc_region_raw=calibrate.auroc(SB[k][:, 0], yB), B_auroc_region_cal=calibrate.auroc(oof, yB),
                  B_auroc_image_raw=calibrate.auroc(SB[k][:, 1], yB),
                  val_auroc_region_raw=calibrate.auroc(SV[k][:, 0], yV),
                  val_auroc_region_cal=calibrate.auroc(cal(SV[k][:, 0], cV), yV),
                  val_auroc_image_raw=calibrate.auroc(SV[k][:, 1], yV))
    print(k, {m: round(v, 3) for m, v in res[k].items()})
chosen = max(res, key=lambda k: res[k]["B_auroc_region_cal"])
print("chosen on B (calibrated region AUROC):", chosen)
cache.save_json(dict(sets={k: [t("dog") for t in v] + [t("grass") for t in v] for k, v in SETS.items()},
                     results=res, chosen=chosen, label="reference proxy; thing/stuff claims at the Look v2 region"),
                config.RESULTS / "owl_queries.json")
