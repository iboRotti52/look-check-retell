"""Create a blinded, fixed B pilot for the next GPU-only Check computation."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import _common  # noqa: F401
import numpy as np
import pyarrow.parquet as pq

from lcr import cache, config
from lcr.region_evidence import RegionChecker


def main():
    checker = RegionChecker.load()
    rows = cache.load_json(config.CACHE / "pipeline/A_B_lcr.json")
    chosen = []
    for partition in ("fit", "cal"):
        eligible = [r for r in rows if r["evidence_partition"] == partition and any(c["kind"] != "person" for c in r["claims"])]
        chosen += sorted(eligible, key=lambda r: hashlib.sha256(f"lcr-crop-pilot:{r['id']}".encode()).hexdigest())[:64]
    chosen.sort(key=lambda r: r["id"])
    root = config.ROOT / "output/check_gpu_pilot"
    if root.exists():
        raise FileExistsError("Pilot package exists; preserve the fixed pilot rather than resampling")
    jobdir = root / "check_crop_job"
    (jobdir / "images").mkdir(parents=True)
    wanted = {int(r["id"].rsplit("_",1)[1]) for r in chosen}
    found = set(); offset = 0
    # Stream encoded training images in small batches; no full-image table in RAM.
    for source in sorted((config.DATA / "flickr8k/data").glob("train-*.parquet")):
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16, columns=["image"]):
            for cell in batch.column(0).to_pylist():
                if offset in wanted:
                    (jobdir / "images" / f"train_{offset:04d}.jpg").write_bytes(cell["bytes"])
                    found.add(offset)
                offset += 1
    assert found == wanted
    crops = []
    for row in chosen:
        unique = {}
        for c in row["claims"]:
            if c["kind"] == "person":
                continue
            for region in c["rec_att"]["regions"]:
                x0,y0,x1,y1 = region["box"]
                dx,dy = .1*(x1-x0), .1*(y1-y0)
                box = [max(0.,x0-dx),max(0.,y0-dy),min(1.,x1+dx),min(1.,y1+dy)]
                key = tuple(box)
                if key not in unique:
                    unique[key] = dict(image_id=row["id"], image_file=f"images/{row['id']}.jpg", box=box, references=[])
                unique[key]["references"].append(dict(claim=c["cls"], proposal=region["region"]))
        crops.extend(unique.values())
    # 'references' here means source-region links, never reference captions or truth labels.
    for crop in crops:
        crop["source_regions"] = crop.pop("references")
    np.save(jobdir / "queries.npy", checker.queries)
    manifest = dict(schema_version=1, method="frozen_owlv2_attended_crops", split="B", n_images=len(chosen),
                    image_ids=[r["id"] for r in chosen], source_bundle=checker.bundle["id"],
                    detector=config.OWL_NAME, classes=checker.bundle["classes"],
                    query_sha256=checker.bundle["query_sha256"], context_margin_fraction=.1,
                    pilot_selection="64 B_fit + 64 B_cal, sha256(lcr-crop-pilot:id), no labels used",
                    crops=crops)
    (jobdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (root / "scripts").mkdir()
    shutil.copyfile(config.ROOT / "scripts/40_check_crop_gpu.py", root / "scripts/40_check_crop_gpu.py")
    (root / "lcr").mkdir()
    for path in (config.ROOT / "lcr").glob("*.py"):
        shutil.copyfile(path, root / "lcr" / path.name)
    shutil.copyfile(config.ROOT / "requirements.txt", root / "requirements.txt")
    archive = config.ROOT / "output/lcr_check_gpu_pilot.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(root))
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(accelerator="GPU", kernelspec=dict(name="python3", display_name="Python 3")), cells=[
        dict(cell_type="markdown", metadata={}, source=["# Look → Check: fixed OWLv2 crop pilot\n", "Look is frozen. Same OWLv2, no Retell, no reference labels, validation or test. Select a GPU runtime, then upload lcr_check_gpu_pilot.zip.\n"]),
        dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=["from google.colab import files\n", "uploaded = files.upload()\n", "import zipfile, pathlib\n", "assert 'lcr_check_gpu_pilot.zip' in uploaded\n", "with zipfile.ZipFile('lcr_check_gpu_pilot.zip') as z: z.extractall('/content/lcr_check_pilot')\n", "%cd /content/lcr_check_pilot\n"]),
        dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=["%pip -q install 'transformers==5.23.0' scipy pillow numpy torchvision\n", "import torch\n", "assert torch.cuda.is_available(), 'Choose a Colab GPU runtime; do not fall back to CPU/MPS'\n", "print(torch.cuda.get_device_name())\n"]),
        dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=["!python scripts/40_check_crop_gpu.py --dry-run\n", "!python scripts/40_check_crop_gpu.py --bs 4\n", "import json\n", "summary = json.load(open('results/check_crop_pilot/summary.json'))\n", "assert summary['status'] == 'complete'\n", "files.download('results/check_crop_pilot.zip')\n"]),
    ])
    (config.ROOT / "colab/LOOK_CHECK_GPU_PILOT.ipynb").write_text(json.dumps(notebook, indent=1))
    cache.save_json(dict(images=len(chosen), crops=len(crops), source_bundle=checker.bundle["id"],
                         archive_bytes=archive.stat().st_size), config.RESULTS / "check_gpu_pilot_manifest.json")
    print(dict(images=len(chosen), crops=len(crops), package=str(archive), bytes=archive.stat().st_size))


if __name__ == "__main__":
    main()
