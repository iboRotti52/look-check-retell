"""Check: frozen OWLv2 proposals, cached once per image, scored against text queries.

Cached per image (fp16), for the top-100 proposals after NMS:
  boxes      (K, 4)   xyxy, normalised to the original image
  objectness (K,)     sigmoid of the objectness logit
  embeds     (K, 512) L2-normalised class embeddings
  shift      (K,)     per-proposal logit shift   } OWLv2 applies these to the
  scale      (K,)     per-proposal logit scale   } cosine similarity
With shift and scale cached, the exact OWLv2 logit for ANY later text query is
(embeds @ q + shift) * scale, so the detector never has to be run again.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from . import config, look
from .vocab import CLASSES, CLASS_INDEX, KIND  # noqa: F401

VERDICTS = ("supported", "contradicted", "uncertain")


class Detector:
    def __init__(self, dev: torch.device | None = None):
        self.dev = dev or config.device()
        self.processor = Owlv2Processor.from_pretrained(config.OWL_NAME)
        self.model = Owlv2ForObjectDetection.from_pretrained(config.OWL_NAME).to(self.dev).eval()

    @torch.no_grad()
    def text_embeddings(self, names: list[str]) -> np.ndarray:
        """L2-normalised query embeddings for `names` under the fixed template."""
        prompts = [config.QUERY_TEMPLATE.format(n) for n in names]
        enc = self.processor(text=prompts, return_tensors="pt", padding="max_length").to(self.dev)
        out = self.model.owlv2.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        q = self.model.owlv2.text_projection(out.pooler_output)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        return q.float().cpu().numpy()

    @torch.no_grad()
    def proposals(self, images) -> list[dict[str, np.ndarray]]:
        px = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.dev)
        fmap, _ = self.model.image_embedder(px)                       # (B, h, w, D)
        b, h, w, d = fmap.shape
        feats = fmap.reshape(b, h * w, d)
        boxes = self.model.box_predictor(feats, fmap)                 # (B, N, 4) cxcywh
        obj = self.model.objectness_predictor(feats).sigmoid()        # (B, N)
        head = self.model.class_head
        emb = head.dense0(feats)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)
        shift = head.logit_shift(feats).squeeze(-1)
        scale = head.elu(head.logit_scale(feats)).squeeze(-1) + 1
        out = []
        for i, im in enumerate(images):
            W, H = im.size
            bx = torch.from_numpy(look.owl_to_image(boxes[i].float().cpu().numpy(), W, H))
            sc = obj[i].float().cpu()
            # NMS in pixel units so that IoU is geometrically meaningful
            keep = nms(bx * torch.tensor([W, H, W, H]), sc, config.NMS_IOU)[: config.TOP_PROPOSALS]
            out.append(dict(
                boxes=bx[keep].numpy().astype(np.float16),
                objectness=sc[keep].numpy().astype(np.float16),
                embeds=emb[i].float().cpu()[keep].numpy().astype(np.float16),
                shift=shift[i].float().cpu()[keep].numpy().astype(np.float16),
                scale=scale[i].float().cpu()[keep].numpy().astype(np.float16),
            ))
        return out


def score(prop: dict[str, np.ndarray], queries: np.ndarray) -> np.ndarray:
    """OWLv2 probabilities, shape (K proposals, Q queries), from the cache alone."""
    e = prop["embeds"].astype(np.float32)
    logits = (e @ queries.T + prop["shift"].astype(np.float32)[:, None]) * prop["scale"].astype(np.float32)[:, None]
    return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class Record:
    """One verification record = one evidence token for Retell."""
    claim: str                 # claimed class
    alt: str                   # strongest alternative class for the region ("" if none)
    s_claim: float             # detector score of the claim in the region
    s_alt: float               # detector score of the alternative in the region
    s_image: float             # best score of the claim over all proposals (image level)
    box: list[float]           # selected region, xyxy normalised ([] if no region)
    region: int                # proposal index (-1 if no region)
    excess: float              # excess attention mass of the selected region
    verdict: str               # supported / contradicted / uncertain

    def to_dict(self):
        return asdict(self)


def verdict(s_claim: float, s_alt: float, has_region: bool, alt: str,
            t_support: float, t_contra: float) -> str:
    """Three-way verdict.

    supported    : the claim scores at least `t_support` in the region.
    contradicted : the claim scores below `t_contra` AND a different class
                   scores at least `t_support` in the same region.
    uncertain    : everything else, including 'no suitable region'. A low score
                   alone never yields 'contradicted': not finding an object in
                   one region is not proof that it is absent from the image.
    """
    if not has_region:
        return "uncertain"
    if s_claim >= t_support:
        return "supported"
    if alt and s_claim < t_contra and s_alt >= t_support:
        return "contradicted"
    return "uncertain"


def make_record(claim: str, region: int | None, excess: float, prop: dict[str, np.ndarray],
                scores: np.ndarray, extra_alternatives: list[str] | None = None,
                t_support: float = 0.2, t_contra: float = 0.05, n_det_alt: int = 5) -> Record:
    """Build the record for one claim.

    scores: (K, |V|) detector probabilities of every class for every proposal.
    Alternatives = the detector's `n_det_alt` best classes for the region plus
    `extra_alternatives` (the captioner's own next-best words at that position).
    Alternatives never include the claim itself; a 'person' alternative is only
    allowed for non-person claims and vice versa (no gender/age substitutions
    are possible because all person terms share one class).
    """
    ci = CLASS_INDEX[claim]
    s_image = float(scores[:, ci].max()) if len(scores) else 0.0
    if region is None:
        return Record(claim, "", 0.0, 0.0, s_image, [], -1, float(excess), "uncertain")
    row = scores[region]
    cand = [CLASSES[j] for j in np.argsort(-row)[: n_det_alt + 1]]
    cand += list(extra_alternatives or [])
    cand = [c for c in dict.fromkeys(cand) if c != claim]
    alt = max(cand, key=lambda c: row[CLASS_INDEX[c]]) if cand else ""
    s_alt = float(row[CLASS_INDEX[alt]]) if alt else 0.0
    s_claim = float(row[ci])
    v = verdict(s_claim, s_alt, True, alt, t_support, t_contra)
    return Record(claim, alt, s_claim, s_alt, s_image,
                  [float(x) for x in prop["boxes"][region]], int(region), float(excess), v)


def select_by_detector(scores: np.ndarray, claim: str, boxes: np.ndarray) -> int | None:
    """B3 region choice: the proposal with the highest detector score for the claim."""
    if len(scores) == 0:
        return None
    s = np.where(look.size_ok(boxes.astype(np.float32)), scores[:, CLASS_INDEX[claim]], -np.inf)
    j = int(np.argmax(s))
    return j if np.isfinite(s[j]) else None


__all__ = ["Detector", "Record", "score", "verdict", "make_record", "select_by_detector", "KIND"]
