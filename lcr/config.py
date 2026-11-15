"""Central paths, model names and fixed protocol constants.

Everything that the proposal says must be "fixed in advance" lives here so that
no script silently uses a different value.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"            # large, re-computable artefacts (not submitted)
RESULTS = ROOT / "results"        # small JSON/CSV artefacts that feed the report
FIGURES = ROOT / "report" / "figures"
GENERATED = ROOT / "report" / "generated"   # LaTeX snippets written by scripts

os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))
os.environ.setdefault("NLTK_DATA", str(DATA / "nltk"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BLIP_NAME = "Salesforce/blip-image-captioning-base"
OWL_NAME = "google/owlv2-base-patch16-ensemble"

SEED = 415
SPLIT_SEED = 415          # A/B halves of the 6,000 training images
BLIP_GRID = 24            # 384 / 16 image patches per side
TOP_PROPOSALS = 100       # cached OWLv2 proposals per image
NMS_IOU = 0.5
MIN_BOX_PATCHES = 2       # minimum proposal side, in BLIP patches
# Look map -> region (chosen on half B by scripts/16_look_regions.py, docs/LOOK_V2_PLAN.md 7.4):
# score = attention mass / area**REGION_ALPHA * objectness**REGION_GAMMA
REGION_ALPHA = 0.5
REGION_GAMMA = 0.5
# Frozen after the prescribed E1/E2 acceptance rules. V2 constants above stay
# available for historical controls and E2's fixed region rule.
LOOK_VERSION = "lcr"              # one attention-object path; v1-v4 remain explicit historical controls
LOOK_MAP_SOURCE = "S3_caption"
LOOK_RULES_BY_KIND = {
    "person": dict(alpha=0.5, gamma=0.5, sigma=1.0),
    "thing": dict(alpha=0.5, gamma=0.5, sigma=0.0),
    "stuff": dict(alpha=0.5, gamma=0.25, sigma=0.0),
}
QUERY_TEMPLATE = "a photo of a {}"

GEN_KWARGS = dict(num_beams=3, max_new_tokens=30, min_new_tokens=5, do_sample=False)


def device() -> torch.device:
    if os.environ.get("LCR_DEVICE"):          # e.g. LCR_DEVICE=cpu for smoke tests
        return torch.device(os.environ["LCR_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
