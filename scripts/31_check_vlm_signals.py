"""Check v3 signals from independent BLIP models (docs/CHECK_V3_PROTOKOL.md §1): VQA yes/no and ITM.

    python scripts/31_check_vlm_signals.py --stem A_B_look4
    python scripts/31_check_vlm_signals.py --stem A_validation_look4

Only COCO / VQA v2 fine-tuned checkpoints (Flickr30k-tuned ones would have seen our images, test included).
Writes data/cache/pipeline/<stem>_vlm.json: {image id: {"claims": [per-claim signals], "alts": {class: vqa_yes}}}.
"""
import argparse
import re
import time

import numpy as np
import torch

import _common  # noqa: F401
from lcr import cache, config, data

CLOTHING = {"shirt", "jacket", "sweater", "hat", "helmet", "dress", "shorts", "pants", "suit", "uniform",
            "costume", "vest", "skirt", "scarf", "bikini", "swimsuit", "glove", "boot", "shoe", "glasses",
            "goggles", "mask", "headband", "bandanna"}
P = config.CACHE / "pipeline"


def art(w):
    return ("an " if w[0] in "aeiou" else "a ") + w


def group(cls, kind):
    return "clothing" if cls in CLOTHING else kind


def questions(cls, kind):
    g = group(cls, kind)
    if g == "clothing":
        return [f"is anyone wearing {art(cls)}?", f"is someone wearing {art(cls)}?", f"does anyone have {art(cls)} on?"]
    if g == "stuff":
        return [f"is there {cls} in the picture?", f"is there {cls}?", f"does this image contain {cls}?"]
    return [f"is there {art(cls)} in the picture?", f"is there {art(cls)}?", f"does this image contain {art(cls)}?"]


def class_text(cls, kind):
    return f"a photo of {cls}" if kind == "stuff" else f"a photo of {art(cls)}"


def drop_word(draft, word):
    pat = r"t\s*-\s*shirt" if word == "tshirt" else r"\b" + re.escape(word) + r"\b"
    return re.sub(pat, "thing", draft, count=1, flags=re.I)


def crop(img, box, pad=.1):
    W, H = img.size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, x0 - pad * w), min(1, x1 + pad * w)
    y0, y1 = max(0, y0 - pad * h), min(1, y1 + pad * h)
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return img
    return img.crop((int(x0 * W), int(y0 * H), max(int(x1 * W), int(x0 * W) + 2), max(int(y1 * H), int(y0 * H) + 2)))


class VLM:
    def __init__(self, dev):
        from transformers import BlipForImageTextRetrieval, BlipForQuestionAnswering, BlipProcessor
        self.dev = dev
        self.pv = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.mv = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(dev).eval()
        self.pi = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.mi = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(dev).eval()
        tok = self.pv.tokenizer
        self.yes = tok("yes", add_special_tokens=False).input_ids[0]
        self.no = tok("no", add_special_tokens=False).input_ids[0]
        self.bos = self.mv.config.text_config.bos_token_id

    @torch.no_grad()
    def vqa(self, img, qs):
        px = self.pv(images=img, return_tensors="pt").pixel_values.to(self.dev)
        vis = self.mv.vision_model(pixel_values=px)[0]
        t = self.pv.tokenizer(qs, return_tensors="pt", padding=True).to(self.dev)
        v = vis.expand(len(qs), -1, -1)
        q = self.mv.text_encoder(input_ids=t.input_ids, attention_mask=t.attention_mask, encoder_hidden_states=v,
                                 encoder_attention_mask=torch.ones(v.shape[:2], dtype=torch.long, device=self.dev))[0]
        bos = torch.full((len(qs), 1), self.bos, device=self.dev)
        lg = self.mv.text_decoder(input_ids=bos, encoder_hidden_states=q, encoder_attention_mask=t.attention_mask).logits[:, -1]
        return torch.softmax(torch.stack([lg[:, self.yes], lg[:, self.no]], 1).float(), 1)[:, 0].cpu().numpy()

    @torch.no_grad()
    def itm(self, imgs, texts):
        """imgs: one image (shared) or a list aligned with texts."""
        if not isinstance(imgs, list):
            px = self.pi(images=imgs, return_tensors="pt").pixel_values.to(self.dev)
            vis = self.mi.vision_model(pixel_values=px)[0].expand(len(texts), -1, -1)
        else:
            px = self.pi(images=imgs, return_tensors="pt").pixel_values.to(self.dev)
            vis = self.mi.vision_model(pixel_values=px)[0]
        t = self.pi.tokenizer(texts, return_tensors="pt", padding=True).to(self.dev)
        h = self.mi.text_encoder(input_ids=t.input_ids, attention_mask=t.attention_mask, encoder_hidden_states=vis,
                                 encoder_attention_mask=torch.ones(vis.shape[:2], dtype=torch.long, device=self.dev))[0]
        return torch.softmax(self.mi.itm_head(h[:, 0]).float(), 1)[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stem", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    assert "test" not in a.stem
    rows = cache.load_json(P / f"{a.stem}.json")
    out_path = P / (f"{a.stem}_vlm.json" if not a.limit else f"{a.stem}_vlm_limit.json")
    if a.limit:
        rows = rows[:a.limit]
    out = cache.load_json(out_path) if out_path.exists() else {}
    dev = config.device()
    m = VLM(dev); t0 = time.monotonic()
    for n, r in enumerate(rows):
        if r["id"] in out or not r["claims"]:
            continue
        img = data.load_image(r["id"])
        cl = r["claims"]
        # VQA: three templates per claim + surface word + the captioner alternatives
        qs, idx = [], []
        for c in cl:
            q3 = questions(c["cls"], c["kind"]); qs += q3 + [f"is there {c['word']} in the picture?"]
        alts = sorted({k for c in cl for k in c["blip_alts"]} - {c["cls"] for c in cl})
        alt_kind = {k: "thing" for k in alts}
        try:
            from lcr.vocab import KIND
            alt_kind = {k: KIND[k] for k in alts}
        except Exception:
            pass
        aq = [questions(k, alt_kind[k])[0] for k in alts]
        p = m.vqa(img, qs + aq)
        texts = [class_text(c["cls"], c["kind"]) for c in cl] + [r["draft"]] + [drop_word(r["draft"], c["word"]) for c in cl]
        it = m.itm(img, texts)
        crops_att = [crop(img, c["rec_att"]["box"]) if len(c["rec_att"].get("box", [])) == 4 else img for c in cl]
        crops_det = [crop(img, c["rec_det"]["box"]) if len(c["rec_det"].get("box", [])) == 4 else img for c in cl]
        ca = m.itm(crops_att, [class_text(c["cls"], c["kind"]) for c in cl])
        cd = m.itm(crops_det, [class_text(c["cls"], c["kind"]) for c in cl])
        k = len(cl); d = it[k]
        out[r["id"]] = dict(
            claims=[dict(cls=c["cls"], vqa_yes=float(p[4*i]), vqa_ens=float(p[4*i:4*i+3].mean()), vqa_word=float(p[4*i+3]),
                         itm_full=float(it[i]), itm_drop=float(d - it[k+1+i]), itm_crop_att=float(ca[i]),
                         itm_crop_det=float(cd[i])) for i, c in enumerate(cl)],
            alts={kk: float(v) for kk, v in zip(alts, p[4*k:])})
        if (n + 1) % 200 == 0:
            cache.save_json(out, out_path)
            print(f"{a.stem} {n+1}/{len(rows)} {(time.monotonic()-t0)/60:.1f} min", flush=True)
    cache.save_json(out, out_path)
    print("done", a.stem, len(out), flush=True)


if __name__ == "__main__":
    main()
