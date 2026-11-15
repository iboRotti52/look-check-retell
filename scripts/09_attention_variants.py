"""Diagnose attention as a localisation signal and test a sink-corrected variant.

    python scripts/09_attention_variants.py --n 300 --decoder zero

For every thing/stuff claim we compare where different attention maps point with the
box the detector itself picks for the claimed class (B3's region), which serves as a
proxy target until the hand-drawn boxes exist. Variants:
  raw      : head-averaged cross-attention of the last k layers at the claim position
  sink     : raw minus the caption's mean attention map over all positions, clipped at 0.
             Patches that absorb attention for every token ("attention sinks",
             cf. Darcet et al.) cancel out; what remains is token-specific.
Metrics: pointing game (argmax patch inside the target box, vs. the box's area share
as the chance level), share of mass in the top 5% patches, and IoU>=0.5 between the
excess-mass region and the target box.
Writes results/attention_variants.json and report/generated/attn_table.tex.
"""
import argparse

import numpy as np
import torch

from _common import write_tex
from lcr import cache, check, claims, config, data, look, pipeline
from lcr.captioner import Captioner

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--decoder", default="zero")
a = ap.parse_args()

w = None if a.decoder == "zero" else str(config.CACHE / "ckpt" / f"decoder_{a.decoder}.pt")
cap = Captioner(decoder_weights=w)
q = pipeline.class_queries()
owl = cache.OwlCache("validation")
blip = cache.open_blip("validation")
ids = data.ids("validation")[: a.n]
res = {f"{v}_k{k}": dict(point=[], area=[], top5=[], agree=[]) for v in ("raw", "sink") for k in (1, 4, 12)}


def norm(x):
    x = np.clip(x, 0, None)
    s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1 / x.size)


for s in range(0, len(ids), 16):
    chunk = ids[s:s + 16]
    emb = torch.from_numpy(np.stack([np.asarray(blip[data.parse_id(i)[1]]) for i in chunk])).float().to(cap.dev)
    toks = cap.generate(emb)
    attn, _ = cap.cross_attention(emb, toks)                       # (B, L, H, T, 577)
    for b, iid in enumerate(chunk):
        prop = owl[data.parse_id(iid)[1]]
        boxes = prop["boxes"].astype(np.float32)
        scores = check.score(prop, q)
        ws, pos = cap.word_alignment(toks[b])
        valid_t = int((toks[b] != cap.pad).sum()) - 1
        for c in claims.extract_from_words(ws):
            if c.kind == "person":
                continue
            d = check.select_by_detector(scores, c.cls, boxes)
            if d is None:
                continue
            tgt = boxes[d]
            p = pos[c.index]
            for k in (1, 4, 12):
                A = attn[b, -k:, :, :valid_t, 1:].float().mean(dim=(0, 1)).cpu().numpy()   # (T, 576)
                raw = norm(A[p - 1])
                sink = norm(A[p - 1] - A.mean(axis=0))
                for name, m in (("raw", raw), ("sink", sink)):
                    g = m.reshape(24, 24)
                    iy, ix = np.unravel_index(g.argmax(), g.shape)
                    py, px = (iy + .5) / 24, (ix + .5) / 24
                    r = res[f"{name}_k{k}"]
                    r["point"].append(bool(tgt[0] <= px <= tgt[2] and tgt[1] <= py <= tgt[3]))
                    r["area"].append(float(look.box_area(tgt[None])[0]))
                    r["top5"].append(float(np.sort(m)[-29:].sum()))
                    sel, _ = look.select_by_attention(g, boxes, threshold=-np.inf)
                    r["agree"].append(bool(sel is not None and look.iou(boxes[sel], tgt[None])[0] >= 0.5))

summary = {}
for key, r in res.items():
    summary[key] = dict(n=len(r["point"]), pointing=100 * np.mean(r["point"]), chance=100 * np.mean(r["area"]),
                        top5_mass=float(np.mean(r["top5"])), agree_iou50=100 * np.mean(r["agree"]))
    print(key, summary[key])
cache.save_json(dict(decoder=a.decoder, n_images=len(ids), variants=summary), config.RESULTS / "attention_variants.json")
lines = []
for v, lab in (("raw", "raw"), ("sink", "sink-corrected")):
    for k in (1, 4, 12):
        sm = summary[f"{v}_k{k}"]
        lines.append(f"{lab} & {k} & {sm['pointing']:.1f} & {sm['chance']:.1f} & {sm['top5_mass']:.2f} & {sm['agree_iou50']:.1f} \\\\")
n = summary["raw_k4"]["n"]
write_tex("attn_table", r"""\begin{table}[t]
\caption{Attention as a localisation signal (""" + f"{n}" + r""" non-person claims, first """ + f"{len(ids)}" + r""" validation images,
""" + ("zero-shot" if a.decoder == "zero" else "fine-tuned") + r""" decoder). Target: the region the detector itself selects
for the claimed class (proxy until hand-drawn boxes exist). \emph{Point}: attention peak inside the target;
\emph{Chance}: mean target area. \emph{Top-5\%}: attention mass on the 5\% most attended patches.
\emph{IoU}: excess-mass region overlaps the target with IoU$\geq$0.5.}
\label{tab:attn}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
Map & $k$ & Point (\%) & Chance (\%) & Top-5\% & IoU (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
