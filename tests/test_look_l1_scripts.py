import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from lcr import region_evidence as re
from lcr.vocab import CLASSES, CLASS_INDEX

ROOT = Path(__file__).resolve().parent.parent


def _script47():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("s47", ROOT / "scripts" / "47_look_forced_hallucination.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_present_control_uses_s4_rival_aware_evidence():
    """Raw backpack logit clears tau, but bag is stronger on the same box: S4 would not find a backpack."""
    s47 = _script47()
    s4 = json.loads((ROOT / "results" / "lcr_check4_bundle.json").read_text())
    tau = np.array([np.nan if t is None else t for t in s4["tau"]])
    bp, bag = CLASS_INDEX["backpack"], CLASS_INDEX["bag"]
    img = s47.ImageData.__new__(s47.ImageData)
    img.z = np.full((1, len(CLASSES)), -10.); img.z[0, bp] = tau[bp] + .1; img.z[0, bag] = tau[bp] + 1.
    img.rival, img._v = (re.set_matrix(s4["confusion"]), float(s4["rival_weight"])), None
    assert img.presence_margin("backpack", tau) is None
    img.z[0, bag], img._v = -10., None
    assert np.isclose(img.presence_margin("backpack", tau), .1)
    assert img.absence_margin("backpack", tau, 0.) is None            # raw max above tau: never "absent"
