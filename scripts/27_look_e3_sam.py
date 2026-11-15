"""Look E3 stage B input: SAM (Kirillov et al., ICCV'23) masks prompted by the peak of the stage-A map.

    python scripts/27_look_e3_sam.py --split train|validation

SAM sees only the image and one point (no class text, no detector score). For every claim we store the
three multimask boxes (normalised x0,y0,x1,y1), SAM's predicted IoUs and mask areas.
Output: data/cache/look_e3/{stem}_sam.npz. The stage-A map is read from results/look_e3_selection.json.
"""
import argparse
import time

import numpy as np
import torch

import _common  # noqa: F401
from lcr import cache, config, data
from lcr.look_e3_eval import Bank

FOLDER = config.CACHE / "look_e3"
STEM = {"train": "A_B_look2", "validation": "A_validation_look2"}


def stage_a_maps(bank, stem):
    """The map locked in stage A (archived v3 map if no family was accepted)."""
    import importlib
    sel = cache.load_json(config.RESULTS / "look_e3_selection.json")
    w = sel.get("winner")
    if not w:
        return bank.archived, "v3_archived"
    s26 = importlib.import_module("26_look_e3_select")
    return s26.final_maps(w, stem), w["family"] + ":" + w["config"]


def mask_box(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return np.zeros(4, np.float32), 0.
    h, w = mask.shape
    return np.array([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h], np.float32), float(mask.mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=list(STEM))
    a = ap.parse_args()
    from transformers import SamModel, SamProcessor
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    bank = Bank(STEM[a.split])
    maps, source = stage_a_maps(bank, STEM[a.split])
    prepared = bank.prepare(maps)
    peaks = prepared.reshape(bank.n, -1).argmax(1)
    py, px = (peaks // 24 + .5) / 24, (peaks % 24 + .5) / 24
    boxes = np.zeros((bank.n, 3, 4), np.float32); iou = np.zeros((bank.n, 3), np.float32)
    area = np.zeros((bank.n, 3), np.float32)
    by_img = {}
    for j in range(bank.n):
        by_img.setdefault(bank.iid[j], []).append(j)
    t0 = time.monotonic()
    for n_img, (iid, js) in enumerate(by_img.items()):
        img = data.load_image(iid); W, H = img.size
        pts = [[[float(px[j] * W), float(py[j] * H)]] for j in js]
        inp = proc(img, input_points=[pts], return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_embeddings(inp["pixel_values"].to(dev))
            out = model(image_embeddings=emb, input_points=inp["input_points"].float().to(dev), multimask_output=True)
        masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"],
                                                        inp["reshaped_input_sizes"])[0].numpy()
        scores = out.iou_scores[0].float().cpu().numpy()
        for k, j in enumerate(js):
            iou[j] = scores[k]
            for m in range(3):
                boxes[j, m], area[j, m] = mask_box(masks[k, m])
        if (n_img + 1) % 250 == 0:
            print(f"{a.split} SAM {n_img+1}/{len(by_img)} images, {(time.monotonic()-t0)/60:.1f} min", flush=True)
    np.savez_compressed(FOLDER / f"{STEM[a.split]}_sam.npz", boxes=boxes, iou=iou, area=area,
                        peak=peaks, owner=bank.owner, source=np.array(source))
    print("SAM done", a.split, source, flush=True)


if __name__ == "__main__":
    main()
