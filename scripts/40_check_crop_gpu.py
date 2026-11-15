"""Extract frozen OWLv2 features from fixed attended crops; CUDA only, no training.

This is a Check experiment, not a new verifier. It never computes attention,
chooses regions, reads references, or tunes thresholds. Use --dry-run locally.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, default=Path("check_crop_job/manifest.json"))
    ap.add_argument("--out", type=Path, default=Path("results/check_crop_pilot"))
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    job = json.loads(a.job.read_text())
    assert job["split"] == "B" and job["method"] == "frozen_owlv2_attended_crops"
    base = a.job.parent
    q = np.load(base / "queries.npy")
    assert hashlib.sha256(q.astype(np.float32).tobytes()).hexdigest() == job["query_sha256"]
    assert q.shape[0] == len(job["classes"])
    assert a.bs > 0
    for crop in job["crops"]:
        assert (base / crop["image_file"]).is_file()
        x0, y0, x1, y1 = crop["box"]
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    if a.dry_run:
        print(dict(status="ready", images=job["n_images"], crops=len(job["crops"]),
                   next_step="CUDA extraction only; no CPU or MPS model fallback"))
        return
    import torch
    if not torch.cuda.is_available():
        ap.error("CUDA GPU required. Do not run heavy model extraction on the M4; use Colab.")
    # Import the project's exact detector only after the GPU guard.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lcr.check import Detector
    from lcr import config
    assert config.OWL_NAME == job["detector"]
    torch.set_num_threads(2)
    detector = Detector(dev=torch.device("cuda"))
    a.out.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(a.job.read_bytes()).hexdigest()
    missing = []
    for i, c in enumerate(job["crops"]):
        path = a.out / f"crop_{i:05d}.npz"
        if path.exists():
            with np.load(path) as z:
                assert z["job_sha256"].item() == manifest_hash
        else:
            missing.append((i, c))
    start = time.time()
    for s in range(0, len(missing), a.bs):
        batch = missing[s:s+a.bs]
        images, bounds = [], []
        for i, crop in batch:
            with Image.open(base / crop["image_file"]) as im:
                im = im.convert("RGB"); w, h = im.size
                x0, y0, x1, y1 = crop["box"]
                # Record actual pixel-rounded geometry for exact coordinate mapping.
                px = (max(0, int(np.floor(x0*w))), max(0, int(np.floor(y0*h))),
                      min(w, int(np.ceil(x1*w))), min(h, int(np.ceil(y1*h))))
                images.append(im.crop(px))
                bounds.append(np.array(px, np.float32) / [w,h,w,h])
        props = detector.proposals(images)
        for (i, crop), prop, bound in zip(batch, props, bounds, strict=True):
            boxes = prop["boxes"].astype(np.float32).copy()
            boxes[:, [0,2]] = bound[0] + boxes[:, [0,2]] * (bound[2]-bound[0])
            boxes[:, [1,3]] = bound[1] + boxes[:, [1,3]] * (bound[3]-bound[1])
            np.savez_compressed(a.out / f"crop_{i:05d}.npz", **prop, image_boxes=boxes,
                                crop_box=bound, job_sha256=np.array(manifest_hash))
        print(f"Crops {min(s+a.bs,len(missing))}/{len(missing)}; {time.time()-start:.1f}s", flush=True)
    summary = dict(status="complete", job_sha256=manifest_hash, source_bundle=job["source_bundle"],
                   detector=job["detector"], device=torch.cuda.get_device_name(),
                   n_images=job["n_images"], n_crops=len(job["crops"]),
                   seconds_this_session=time.time()-start,
                   scope="fixed B pilot; no references, thresholds, training, validation, test, or Retell")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (a.out / "manifest.json").write_text(a.job.read_text())
    target = a.out.with_suffix(".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(a.out.iterdir()):
            z.write(path, path.name)
    print(summary)
    print("Return this result archive for CPU analysis:", target)


if __name__ == "__main__":
    main()
