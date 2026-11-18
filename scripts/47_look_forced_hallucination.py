"""Look L1 (docs/LOOK_L1_PROTOKOL.md): forced-hallucination test, B only.

Teacher-forces decoder_A on a deliberately corrupted version of each B draft and re-extracts
attention through exactly the live Look v4 code path (pipeline.run_v4 / look_v4.rewrite_claim):
same 4 frozen heads, same float16 signed-head quantisation, same look.region_score ranking that
backs both look.select_region and region_evidence.candidates. This is the same replay harness
scripts/review_look_e3.py's `live()` already checks byte-for-byte against the frozen cache for
unmodified drafts; here only the input text changes.

Two corruptions, built from lcr/corrupt.py's confusion table (B references only):
* misidentified -- a claim whose object has a Flickr30k Entities box has its word swapped for a
  plausible wrong one; the true referent is that box.
* invented -- a class that no reference mentions and that OWLv2 does not detect anywhere in the
  image (lowest image-wide detector score among a random pool, lcr.check.score) is appended to
  the caption. This is only PROXY absence (not detected != not present); the output says so.

Needs decoder_A; CPU is fine for a few hundred claims (--n), otherwise run on Colab. B only;
refuses validation/test. Fixed seed, fixed sample.

TODO (not implemented here): referent-signal AUROC for invented claims against real correct
claims (docs/LOOK_L1_PROTOKOL.md Section 4) needs each invented claim's `strength`
(look_v4.residual_strength), which needs the pre-normalisation signed-head array that
pipeline.run_v4 does not expose -- only its final normalised map. Whoever runs this for real
should either add an optional raw-signed-head return to look_v4.attention_map, or score
entropy/top1_concentration/null_margin only (both already available from the returned map).

    python scripts/47_look_forced_hallucination.py --n 300 --seed 415
"""
import argparse
import re

import numpy as np
import torch

import _common  # noqa: F401
from lcr import cache, check, claims, config, corrupt, data, grounding, look, metrics, pipeline
from lcr import region_evidence as re_
from lcr.captioner import Captioner
from lcr.look_sources import LookDecoder
from lcr.vocab import CLASS_INDEX, CLASSES, KIND, STUFF, THINGS

P = config.CACHE / "pipeline"
SURFACE = {c: (THINGS.get(c) or STUFF.get(c) or [c])[0] for c in CLASSES}


class _SwappedTextCap:
    """Feeds one pre-built caption text through pipeline.run_v4 instead of generating one.

    Duck-typed exactly like scripts/review_look_e3.py's ReplayCap: only tok/bos/pad/dev are
    read by the Captioner methods it borrows.
    """
    dev = torch.device("cpu")
    encode_captions = Captioner.encode_captions
    word_alignment = Captioner.word_alignment
    decode = Captioner.decode

    def __init__(self, dec: LookDecoder):
        self.dec, self.tok, self.bos, self.pad = dec, dec.tok, dec.bos, 0

    def generate(self, emb):
        ids = self.tok(self.text, return_tensors="pt")["input_ids"]
        ids[:, 0] = self.bos
        return ids

    def cross_attention(self, emb, ids):
        with torch.no_grad(), self.dec.capture() as att:
            out = self.dec.forward(ids, emb)
        return torch.stack(att, 1), out.logits


def swap_word(text: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, text, count=1)


def best_iou(gt_boxes, box) -> float:
    """Best IoU of one predicted box (or [] for "no region") against one or more GT boxes."""
    if not len(box):
        return 0.0
    return float(look.iou(np.asarray(box, np.float32), np.asarray(gt_boxes, np.float32)).max())


def best_iou_multi(gt_boxes, boxes) -> float:
    if not len(boxes):
        return 0.0
    return max(best_iou(gt_boxes, b) for b in boxes)


def pick_misidentified(row, gtboxes, present, table, rng):
    cands = [c for c in row["claims"] if c["kind"] != "person" and c["cls"] in gtboxes]
    if not cands:
        return None
    c = cands[int(rng.integers(len(cands)))]
    new_cls = corrupt.corrupt_label(c["cls"], present, table, rng)
    if new_cls is None:
        return None
    return dict(true_cls=c["cls"], new_cls=new_cls, true_box=gtboxes[c["cls"]],
               text=swap_word(row["draft"], c["word"], SURFACE[new_cls]))


def pick_invented(present, claimed, prop, q, rng, n_try=12):
    """Lowest image-wide detector score among a random pool of unclaimed, unreferenced classes."""
    pool = [c for c in CLASSES if KIND[c] != "person" and c not in present and c not in claimed]
    if not pool:
        return None
    sample = rng.choice(pool, size=min(n_try, len(pool)), replace=False)
    scores = check.score(prop, q)
    best = min(sample, key=lambda c: float(scores[:, CLASS_INDEX[c]].max()))
    return dict(new_cls=best, s_image_proxy=float(scores[:, CLASS_INDEX[best]].max()))


def run_one(dec, cap, row, prop, text, new_cls):
    """Teacher-force `text` and return (claim_record, att_map, top3_boxes) for the `new_cls`
    claim, or None if it did not survive re-tokenisation/POS tagging/claim extraction."""
    cap.text = text
    out_row, maps = next(pipeline.run_v4(cap, [row["id"]], bs=1, sam=None))
    hits = [(i, c) for i, c in enumerate(out_row["claims"]) if c["cls"] == new_cls]
    if not hits:
        return None
    i, c = hits[0]
    att = maps[i]
    kind = KIND[new_cls]
    rule = dict(min_patches=0, alpha=config.LOOK_RULES_BY_KIND[kind]["alpha"])   # frozen v4: no small-box filter
    top, _ = re_.candidates(att, prop, kind, rule, n=3)
    return c, att, prop["boxes"].astype(np.float32)[top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="target number of claims per corruption kind")
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows = cache.load_json(P / "A_B_look4.json")
    assert all(r["id"].startswith("train_") for r in rows), "Look L1 is B-only; refuses validation/test"
    table = corrupt.build_confusions([data.references(r["id"]) for r in rows])
    q = pipeline.class_queries()
    owl = cache.OwlCache("train")
    dec = LookDecoder(); cap = _SwappedTextCap(dec)

    misid, invented = [], []
    for ri in rng.permutation(len(rows)):
        if len(misid) >= a.n and len(invented) >= a.n:
            break
        row = rows[int(ri)]
        _, idx = data.parse_id(row["id"])
        prop = owl[idx]
        present = claims.reference_classes(data.references(row["id"]))
        gtboxes = grounding.class_boxes(row["id"])

        if len(misid) < a.n and (picked := pick_misidentified(row, gtboxes, present, table, rng)) is not None:
            res = run_one(dec, cap, row, prop, picked["text"], picked["new_cls"])
            if res is not None:
                c, _, top3 = res
                misid.append(dict(
                    id=row["id"], true_cls=picked["true_cls"], new_cls=picked["new_cls"],
                    top1_hit=best_iou(picked["true_box"], c["rec_att"]["box"]) >= 0.5,
                    top3_hit=best_iou_multi(picked["true_box"], top3) >= 0.5,
                    b3_det_hit=best_iou(picked["true_box"], c["rec_det"]["box"]) >= 0.5))

        if len(invented) < a.n:
            claimed = {c["cls"] for c in row["claims"]}
            picked = pick_invented(present, claimed, prop, q, rng)
            if picked is not None:
                res = run_one(dec, cap, row, prop, row["draft"].rstrip(" .") + f" and a {SURFACE[picked['new_cls']]}.",
                             picked["new_cls"])
                if res is not None:
                    c, _, _ = res
                    invented.append(dict(id=row["id"], new_cls=picked["new_cls"],
                                         s_image_proxy=picked["s_image_proxy"], has_region=bool(c["rec_att"]["box"])))

    out = summarise(misid, invented)
    cache.save_json(out, config.RESULTS / "look_forced_hallucination_B.json")
    write_markdown(out, config.ROOT / "docs" / "generated" / "LOOK_FORCED_HALLUCINATION_B.md")
    print(f"misidentified: {len(misid)} claims, invented: {len(invented)} claims "
         f"-> results/look_forced_hallucination_B.json")


def summarise(misid, invented):
    out = dict(n_misidentified=len(misid), n_invented=len(invented))
    if misid:
        out["top1_hit_rate"] = metrics.bootstrap_ci(misid, lambda rs: 100 * np.mean([r["top1_hit"] for r in rs]))
        out["top3_hit_rate"] = metrics.bootstrap_ci(misid, lambda rs: 100 * np.mean([r["top3_hit"] for r in rs]))
        out["b3_det_hit_rate"] = metrics.bootstrap_ci(misid, lambda rs: 100 * np.mean([r["b3_det_hit"] for r in rs]))
        out["top1_minus_b3"] = metrics.bootstrap_ci(
            misid, lambda rs: 100 * np.mean([r["top1_hit"] for r in rs]) - 100 * np.mean([r["b3_det_hit"] for r in rs]))
    if invented:
        out["invented_has_region_rate"] = 100 * float(np.mean([v["has_region"] for v in invented]))
        out["invented_mean_s_image_proxy"] = float(np.mean([v["s_image_proxy"] for v in invented]))
    return out


def write_markdown(out, path):
    L = ["# Look L1 — zorlanmış halüsinasyon testi (B)", "",
         "Üreten: `scripts/47_look_forced_hallucination.py`. Protokol: `docs/LOOK_L1_PROTOKOL.md`.", "",
         f"Yanlış tanımlanmış claim: {out['n_misidentified']}, uydurulmuş claim: {out['n_invented']}.", ""]
    if "top1_hit_rate" in out:
        for k in ("top1_hit_rate", "top3_hit_rate", "b3_det_hit_rate", "top1_minus_b3"):
            v, lo, hi = out[k]
            L.append(f"- **{k}:** {v:.1f} [{lo:.1f}, {hi:.1f}] (görüntü düzeyi bootstrap, 95% GA)")
    if "invented_has_region_rate" in out:
        L += ["", f"Uydurulmuş claim'lerin bölge bulma oranı: {out['invented_has_region_rate']:.1f}% "
                  f"(ortalama proxy s_image: {out['invented_mean_s_image_proxy']:.3f}; bu yalnız proxy yokluktur)."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
