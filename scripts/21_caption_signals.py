"""Captioner-side signals for every draft claim (inputs to the learned Check verifier, scripts/22_check_verifier.py).

    LCR_DEVICE=cpu python scripts/21_caption_signals.py --stems A_B_look2 A_validation_look2

Literature: token-level confidence predicts caption correctness (Petryk+ WACV'24); hallucinated objects are tied
to uncertainty, position and co-occurrence (LURE, ICLR'24) and to a weak dependence on the image (LURE's
image-removed comparison; contrastive decoding, VCD CVPR'24); MetaToken (arXiv 2405.19186) and HaloProbe
(ICML'26) combine such signals in a small classifier.

Per claim, teacher-forced on the draft with the A decoder (first word piece of the claim word):
  lp        log p(w | image, prefix)
  entropy   entropy of the next-token distribution at that step
  margin    lp minus the log-probability of the best other token
  lp_null   log p(w | 'average image', prefix): image embeddings replaced by the mean BLIP embedding of half B
  vis_gain  lp - lp_null  (how much the image raises the word; small = language-prior driven)
  rel_pos   token position / caption length
Writes data/cache/pipeline/<stem>_capsig.json: {image id: {class: signals}}.
"""
import argparse

import numpy as np
import torch

import _common  # noqa: F401
from lcr import cache, claims, config, data
from lcr.captioner import Captioner

ap = argparse.ArgumentParser()
ap.add_argument("--stems", nargs="+", default=["A_B_look2", "A_validation_look2"])
ap.add_argument("--decoder", default="A")
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
assert not any("test" in s for s in a.stems), "the test split is not used during development"
cap = Captioner(decoder_weights=str(config.CACHE / "ckpt" / f"decoder_{a.decoder}.pt"))
P = config.CACHE / "pipeline"

# 'average image' from half B (training images; never validation)
bmm = cache.open_blip("train")
b_idx = [data.parse_id(i)[1] for i in data.ids("B")]
acc = np.zeros(bmm.shape[1:], np.float64)
for s in range(0, len(b_idx), 256):
    acc += np.asarray(bmm[sorted(b_idx[s:s + 256])], np.float64).sum(0)
null_img = torch.from_numpy((acc / len(b_idx)).astype(np.float32)).to(cap.dev)


@torch.no_grad()
def step_logprobs(emb, ids, mask):
    return torch.log_softmax(cap.decoder_forward(ids, mask, emb).logits.float(), dim=-1)


for stem in a.stems:
    rows = cache.load_json(P / f"{stem}.json")
    split = data.parse_id(rows[0]["id"])[0]
    mm = cache.open_blip(split)
    out, n_claims, n_matched = {}, 0, 0
    for s in range(0, len(rows), a.bs):
        chunk = rows[s:s + a.bs]
        emb = torch.from_numpy(np.stack([np.asarray(mm[data.parse_id(r["id"])[1]]) for r in chunk])).float().to(cap.dev)
        ids, mask = cap.encode_captions([r["draft"] for r in chunk])
        ids, mask = ids.to(cap.dev), mask.to(cap.dev)
        lp_img = step_logprobs(emb, ids, mask)
        lp_nul = step_logprobs(null_img.expand_as(emb), ids, mask)
        for b, r in enumerate(chunk):
            ws, pos = cap.word_alignment(ids[b].cpu())
            length = int(mask[b].sum()) - 1
            found = {c.cls: (pos[c.index], c.word) for c in claims.extract_from_words(ws)}
            sig = {}
            for c in r["claims"]:
                n_claims += 1
                aligned = found.get(c["cls"])
                if aligned is None:
                    continue
                p, word = aligned
                if p != c["pos"] or word != c["word"]:
                    raise ValueError(f"claim alignment changed: {r['id']} {c['cls']}: "
                                     f"stored {(c['pos'], c['word'])}, re-tokenised {(p, word)}")
                n_matched += 1
                tok = int(ids[b, p])
                row = lp_img[b, p - 1]
                lp = float(row[tok])
                other = row.clone(); other[tok] = -float("inf")
                pr = row.exp()
                sig[c["cls"]] = dict(lp=lp, entropy=float(-(pr * row).sum()), margin=lp - float(other.max()),
                                     lp_null=float(lp_nul[b, p - 1, tok]), vis_gain=lp - float(lp_nul[b, p - 1, tok]),
                                     rel_pos=p / max(length, 1))
            out[r["id"]] = sig
        if s % (a.bs * 20) == 0:
            print(f"{stem}: {s}/{len(rows)}", flush=True)
    cache.save_json(out, P / f"{stem}_capsig.json")
    print(f"{stem}: {n_matched}/{n_claims} claims matched after re-tokenising the draft", flush=True)
