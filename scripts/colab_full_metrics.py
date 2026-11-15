"""Full COCO caption toolkit (PTB tokenizer, BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE) on validation.

Needs Java (available on Colab). Scores every caption set that exists:
  * zero-shot drafts and decoder-A drafts (data/cache/pipeline/<decoder>_validation.json)
  * every Retell system's outputs under each evidence condition
Writes results/full_metrics.json. BLEU-4, METEOR, ROUGE-L and SPICE are x100.
"""
import glob
import json
from pathlib import Path

import _common  # noqa: F401
from lcr import cache, config, data
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def score(gens: dict[str, str]) -> dict[str, float]:
    tok = PTBTokenizer()
    res = tok.tokenize({k: [{"caption": v}] for k, v in gens.items()})
    gts = tok.tokenize({k: [{"caption": r} for r in data.references(k)] for k in gens})
    out = {}
    b, _ = Bleu(4).compute_score(gts, res, verbose=0)
    out["BLEU-4"] = 100 * b[3]
    out["METEOR"] = 100 * Meteor().compute_score(gts, res)[0]
    out["ROUGE-L"] = 100 * Rouge().compute_score(gts, res)[0]
    out["CIDEr"] = float(Cider().compute_score(gts, res)[0])
    try:
        out["SPICE"] = 100 * Spice().compute_score(gts, res)[0]
    except Exception as e:  # SPICE downloads CoreNLP on first use; report rather than crash
        out["SPICE_error"] = repr(e)
    return out


sets: dict[str, dict[str, str]] = {}
pdir = config.CACHE / "pipeline"
for dec in ("zero", "A"):
    p = pdir / f"{dec}_validation.json"
    if p.exists():
        sets[f"draft_{dec}"] = {r["id"]: r["draft"] for r in cache.load_json(p)}
for p in sorted(glob.glob(str(pdir / "retell_*_validation_outputs.json"))):
    name = Path(p).name.replace("retell_", "").replace("_validation_outputs.json", "")
    for cond, gens in cache.load_json(Path(p)).items():
        if cond != "draft":
            sets[f"{name}:{cond}"] = gens

results = {}
for name, gens in sets.items():
    results[name] = score(gens)
    print(name, json.dumps(results[name]), flush=True)
cache.save_json(results, config.RESULTS / "full_metrics.json")
