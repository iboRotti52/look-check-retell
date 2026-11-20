"""Draft -> Look -> Check for a list of images, entirely from cached features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import cache, check, claims, config, data, look
from .captioner import Captioner
from .vocab import CLASSES, CLASS_INDEX

LAST_K = (1, 4, 12)
DEFAULT_THRESHOLDS = dict(last_k=4, t_excess=0.0, t_support=0.2, t_contra=0.05)


def output_stem(decoder: str, split: str, sink: bool, region_rule: str = "v2") -> str:
    """Keep the main LCR path and every historical method in distinct outputs."""
    if region_rule == "v1":
        suffix = "_sink" if sink else ""
    elif region_rule in ("v2", "v3", "v4", "lcr"):
        tag = lcr_tag() if region_rule == "lcr" else "look" + region_rule[1:]
        if region_rule in ("v4", "lcr") and not sink:
            raise ValueError("Frozen attention maps require sink attention")
        suffix = f"_{tag}" if sink else f"_raw_{tag}"
    else:
        raise ValueError(f"Unknown region rule: {region_rule}")
    return f"{decoder}_{split}{suffix}"


def lcr_tag() -> str:
    """Record name of the default Check bundle. "lcr" alone is the first unified bundle (A_B_lcr.json); the
    default has been S4 (A_B_lcr4.json) since 2026-11-15, and a live run must not write under the old name."""
    from . import region_evidence as re
    names = {re.HISTORICAL_BUNDLE_PATH: "lcr", re.FIVE_WAY_BUNDLE_PATH: "lcr2", re.CHECK3_BUNDLE_PATH: "lcr3",
             re.CHECK4_BUNDLE_PATH: "lcr4"}
    return next(t for path, t in names.items() if path.resolve() == re.BUNDLE_PATH.resolve())


def look_metadata(sink: bool, region_rule: str = "v2", available_last_k=LAST_K) -> dict:
    if region_rule == "lcr":
        from .region_evidence import RegionChecker
        if not sink:
            raise ValueError("LCR requires sink attention")
        from .look_v4 import metadata
        checker = RegionChecker.load()
        return dict(metadata(), region_rule="lcr", proposal_rule=checker.bundle["look"],
                    bundle_id=checker.bundle["id"])
    if region_rule == "v4":
        if not sink:
            raise ValueError("Look v4 requires sink attention")
        from .look_v4 import metadata
        return metadata()
    if region_rule not in ("v1", "v2", "v3"):
        raise ValueError(f"Unknown region rule: {region_rule}")
    meta = dict(schema_version=2, map_source="blip_decoder_cross_attention",
                map_correction="caption_mean_subtract_relu" if sink else "none",
                token_position="first_piece_prediction", last_k=4, head_pooling="mean",
                posthoc_caption_baseline=bool(sink), region_rule=region_rule,
                region_parameters=dict(alpha=config.REGION_ALPHA, gamma=config.REGION_GAMMA)
                if region_rule in ("v2", "v3") else dict(excess_threshold=0.0),
                **(dict(rules_by_kind=config.LOOK_RULES_BY_KIND, source=config.LOOK_MAP_SOURCE,
                        selection="results/look_e1.json and results/look_e2.json") if region_rule == "v3" else {}),
                available_last_k=list(available_last_k))
    if region_rule == "v3":
        meta.pop("region_parameters")
        meta["map_storage"] = "float32 after E1 quantisation and normalisation"
    return meta


def rewrite_look_claim(claim: dict, att: np.ndarray, prop, scores, *,
                       region_parameters=None, look_version="v2") -> dict:
    """Migrate a saved k=4 map; explicitly archive unavailable k=1/12 results."""
    att = np.asarray(att, np.float32)
    att = att / max(float(att.sum()), 1e-12)
    boxes = prop["boxes"].astype(np.float32)
    reg, _ = look.select_region(att, boxes, prop["objectness"].astype(np.float32), **(region_parameters or {}))
    exv = float(look.excess_mass(att, boxes[reg:reg + 1])[0]) if reg is not None else 0.0
    rec = region_record(claim["cls"], reg, exv, prop, scores, claim["blip_alts"])
    rec.update(by="attention", look=look_version)
    dreg = claim["rec_det"]["region"]
    agree = float(look.iou(boxes[reg], boxes[dreg:dreg + 1])[0]) if reg is not None and dreg >= 0 else 0.0
    legacy = claim.get("legacy_look")
    if legacy is None:
        legacy = dict(region_rule=claim["rec_att"].get("look", "v1"),
                      **{k: claim[k] for k in ("regions_by_k", "excess_by_k", "iou_att_det") if k in claim})
    det = dict(claim["rec_det"])
    if look_version in ("v3", "v4"):
        det["excess"] = float(look.excess_mass(att, boxes[dreg:dreg+1])[0]) if dreg >= 0 else 0.0
    return dict(claim, rec_att=rec, rec_det=det, iou_att_det=agree, regions_by_k={"4": reg if reg is not None else -1},
                excess_by_k={"4": exv}, available_last_k=[4], legacy_look=legacy,
                att_entropy=float(-(att * np.log(att + 1e-12)).sum() / np.log(att.size)))


def thresholds() -> dict:
    """Frozen operating point; results/thresholds.json overrides the provisional defaults."""
    import os
    p = Path(os.environ["LCR_THRESHOLDS"]) if os.environ.get("LCR_THRESHOLDS") else config.RESULTS / "thresholds.json"
    return {**DEFAULT_THRESHOLDS, **(cache.load_json(p) if p.exists() else {})}


def class_queries() -> np.ndarray:
    return np.load(config.CACHE / "owl" / "class_queries.npy")


def token_class_map(cap: Captioner) -> dict[int, str]:
    """Tokenizer ids that are whole words belonging to a vocabulary class."""
    out = {}
    for tok, i in cap.tok.get_vocab().items():
        if tok.isalpha():
            c = claims.word_class(tok)
            if c is not None:
                out[i] = c
    return out


def blip_alternatives(logits_row: torch.Tensor, claim: str, tok2cls: dict[int, str], n: int = 3, top: int = 30):
    """The captioner's own next-best vocabulary classes at the claim position."""
    alts: list[str] = []
    for t in torch.topk(logits_row, top).indices.tolist():
        c = tok2cls.get(t)
        if c is not None and c != claim and c not in alts:
            alts.append(c)
            if len(alts) == n:
                break
    return alts


def region_record(claim: str, region: int | None, excess: float, prop, scores, alts) -> dict:
    """Raw record (verdict filled in later by `finalise`)."""
    return check.make_record(claim, region, excess, prop, scores, extra_alternatives=alts).to_dict()


def finalise(rec: dict, th: dict) -> dict:
    """Apply the frozen thresholds: excess-mass gate, then the three-way verdict."""
    r = dict(rec)
    if r.get("check") in ("v2", "v3", "v4", "lcr"):
        # Check v2 (scripts/22_check_verifier.py): verdict already decided by the calibrated verifier
        return r
    # excess-mass gate: only for round-1 (v1) attention records; the v2 region rule already scores the box
    # and its gate is to be re-defined with Check (docs/LITERATUR_TARAMASI.md 8a)
    if (r["region"] >= 0 and r.get("by", "attention") == "attention" and r.get("look", "v1") == "v1"
            and r["excess"] < th["t_excess"]):
        r.update(region=-1, box=[], s_claim=0.0, s_alt=0.0, alt="")
    r["verdict"] = check.verdict(r["s_claim"], r["s_alt"], r["region"] >= 0, r["alt"],
                                 th["t_support"], th["t_contra"])
    return r


@torch.no_grad()
def run(cap: Captioner, image_ids: list[str], bs: int = 32, progress=None, sink: bool = True,
        region_rule: str | None = None, sam=None):
    """Yields one dict per image: draft, claims and raw records; plus attention maps."""
    region_rule = region_rule or config.LOOK_VERSION
    if region_rule == "lcr":
        if not sink:
            raise ValueError("LCR requires sink attention")
        yield from run_lcr(cap, image_ids, bs=bs, progress=progress, sam=sam)
        return
    metadata = look_metadata(sink, region_rule)
    if region_rule == "v4":
        yield from run_v4(cap, image_ids, bs=bs, progress=progress, sam=sam)
        return
    q = class_queries()
    tok2cls = token_class_map(cap)
    blip_mm: dict[str, np.memmap] = {}
    owl: dict[str, cache.OwlCache] = {}
    it = range(0, len(image_ids), bs)
    for s in (progress(it) if progress else it):
        chunk = image_ids[s:s + bs]
        rows = []
        for iid in chunk:
            sp, idx = data.parse_id(iid)
            if sp not in blip_mm:
                blip_mm[sp], owl[sp] = cache.open_blip(sp), cache.OwlCache(sp)
            rows.append(np.asarray(blip_mm[sp][idx]))
        emb = torch.from_numpy(np.stack(rows)).float().to(cap.dev)
        ids = cap.generate(emb)
        attn, logits = cap.cross_attention(emb, ids)
        texts = cap.decode(ids)
        for b, iid in enumerate(chunk):
            sp, idx = data.parse_id(iid)
            prop = owl[sp][idx]
            boxes = prop["boxes"].astype(np.float32)
            scores = check.score(prop, q)
            present = claims.reference_classes(data.references(iid))
            ws, pos = cap.word_alignment(ids[b])
            valid_t = int((ids[b] != cap.pad).sum()) - 1     # decoder positions of this caption
            out_claims, maps = [], []
            for c in claims.extract_from_words(ws):
                p = pos[c.index]
                alts = blip_alternatives(logits[b, p - 1], c.cls, tok2cls)
                per_k = {}
                for k in LAST_K:
                    att = look.attention_map(attn[b], p, last_k=k, sink=sink, valid_t=valid_t)
                    # Look v2 region rule (mass / sqrt(area) x objectness^0.5, chosen on half B)
                    if region_rule == "v3":
                        reg, _, att = look.select_frozen(att, boxes, prop["objectness"].astype(np.float32), c.kind)
                    elif region_rule == "v2":
                        reg, _ = look.select_region(att, boxes, prop["objectness"].astype(np.float32))
                    else:
                        reg, _ = look.select_by_attention(att, boxes)
                    exv = float(look.excess_mass(att, boxes[reg:reg + 1])[0]) if reg is not None else 0.0
                    per_k[k] = (reg, exv, att)
                reg, exv, att = per_k[4]
                rec_att = region_record(c.cls, reg, exv, prop, scores, alts)
                rec_att.update(by="attention", look=region_rule)
                dreg = check.select_by_detector(scores, c.cls, boxes)
                dex = float(look.excess_mass(att, boxes[dreg:dreg + 1])[0]) if dreg is not None else 0.0
                rec_det = region_record(c.cls, dreg, dex, prop, scores, alts); rec_det["by"] = "detector"
                agree = float(look.iou(boxes[reg], boxes[dreg:dreg + 1])[0]) if reg is not None and dreg is not None else 0.0
                a = att.ravel()
                out_claims.append(dict(
                    cls=c.cls, kind=c.kind, word=c.word, pos=int(p), in_refs=c.cls in present,
                    blip_alts=alts, rec_att=rec_att, rec_det=rec_det, iou_att_det=agree,
                    regions_by_k={str(k): int(v[0]) if v[0] is not None else -1 for k, v in per_k.items()},
                    excess_by_k={str(k): v[1] for k, v in per_k.items()},
                    att_entropy=float(-(a * np.log(a + 1e-12)).sum() / np.log(a.size)),
                ))
                maps.append(att.astype(np.float32 if region_rule == "v3" else np.float16))
            yield dict(id=iid, draft=texts[b], claims=out_claims, look_metadata=metadata), maps


def relabel(rec: dict, new_claim: str, prop, scores, alts: list[str]) -> dict:
    """Synthetic corruption: same region, new (wrong) claimed label, scores recomputed."""
    if rec.get("check") == "lcr":
        from .region_evidence import RegionChecker
        return RegionChecker.load().relabel(rec, new_claim, prop)
    if rec.get("check") in ("v2", "v3", "v4"):
        raise NotImplementedError("Historical calibrated records require their original rescorer; use the unified LCR path")
    region = rec["region"] if rec["region"] >= 0 else None
    out = region_record(new_claim, region, rec["excess"], prop, scores, alts)
    out["by"] = rec.get("by", "attention")
    if "look" in rec:
        out["look"] = rec["look"]
    if rec.get("box_source") == "sam":
        # Scores still refer to the matched proposal; retain the selected geometry.
        for key in ("box", "box_source", "score_source", "score_box", "sam_proposal_iou"):
            if key in rec:
                out[key] = rec[key]
    return out


@torch.no_grad()
def run_v4(cap, image_ids, bs=32, progress=None, sam=None, include_references=True):
    """Live E3 extraction without reading E3 maps or human boxes.

    Retokenise the finished draft exactly as scripts/25 did, and expose only k=4
    as a compatibility key. SAM is loaded only when the accepted rule needs it.
    """
    from . import look_v4 as v4
    selected, box_rules = v4.selection()
    meta = v4.metadata()
    q = class_queries(); tok2cls = token_class_map(cap)
    mm, owls = {}, {}
    it = range(0, len(image_ids), bs)
    for start in (progress(it) if progress else it):
        chunk = image_ids[start:start+bs]; features = []
        for iid in chunk:
            sp, idx = data.parse_id(iid)
            if sp not in mm:
                mm[sp], owls[sp] = cache.open_blip(sp), cache.OwlCache(sp)
            features.append(np.array(mm[sp][idx]))
        emb = torch.from_numpy(np.stack(features)).float().to(cap.dev)
        texts = cap.decode(cap.generate(emb))
        for b, (iid, text) in enumerate(zip(chunk, texts)):
            ids, mask = cap.encode_captions([text])
            ids = ids.to(cap.dev)
            attn, logits = cap.cross_attention(emb[b:b+1], ids)
            ws, positions = cap.word_alignment(ids[0])
            present = claims.reference_classes(data.references(iid)) if include_references else None
            sp, idx = data.parse_id(iid); prop = owls[sp][idx]
            scores = check.score(prop, q); boxes = prop['boxes'].astype(np.float32)
            out_claims, maps = [], []
            for c in claims.extract_from_words(ws):
                p = positions[c.index]
                alts = blip_alternatives(logits[0, p-1], c.cls, tok2cls)
                m = v4.attention_map(attn[0], p, int(mask.sum())-1, selected['heads'])
                m = look.smooth_map(m, config.LOOK_RULES_BY_KIND[c.kind]['sigma'])
                dreg = check.select_by_detector(scores, c.cls, boxes)
                det = region_record(c.cls, dreg, 0., prop, scores, alts); det['by'] = 'detector'
                out_claims.append(dict(cls=c.cls, kind=c.kind, word=c.word, pos=int(p),
                                       in_refs=c.cls in present if present is not None else None, blip_alts=alts, rec_det=det,
                                       rec_att=dict(look='v4')))
                maps.append(m)
            candidates = None
            if any(box_rules[c['kind']] != 'base' for c in out_claims):
                if sam is None:
                    sam = v4.SamBoxes()
                candidates = sam(data.load_image(iid), maps)
            rewritten = [v4.rewrite_claim(c, m, prop, scores, box_rules[c['kind']],
                         candidates[j] if candidates is not None else None)
                         for j, (c, m) in enumerate(zip(out_claims, maps))]
            # Fresh runs have no historical k=1/12 records to archive.
            for c in rewritten:
                c.pop('legacy_look', None)
            yield dict(id=iid, draft=text, claims=rewritten, look_metadata=meta), maps


@torch.no_grad()
def run_lcr(cap, image_ids, bs=32, progress=None, sam=None):
    """One live attention-object checker. References never enter this inference path."""
    from .region_evidence import RegionChecker, partition, proposal_logits
    checker = RegionChecker.load()
    owls = {}
    for row, maps in run_v4(cap, image_ids, bs=bs, progress=progress, sam=sam, include_references=False):
        sp, idx = data.parse_id(row["id"])
        if sp not in owls:
            owls[sp] = cache.OwlCache(sp)
        prop = owls[sp][idx]
        lg = proposal_logits(prop, checker.queries)
        for c, att in zip(row["claims"], maps):
            rec = checker.record(c["cls"], att, prop, logits=lg,
                                 person_record=c["rec_att"] if c["kind"] == "person" else None,
                                 non_visual=claims.non_visual(row["draft"], c["cls"],
                                                             compounds=checker.bundle.get("compounds", False)))
            c.update(rec_att=rec, regions_by_k={"4": rec["region"]}, excess_by_k={"4": rec["excess"]})
        row.update(bundle_id=checker.bundle["id"], evidence_partition=partition(row["id"]) if sp == "train" else "inference",
                   look_metadata=dict(row["look_metadata"], region_rule="lcr", proposal_rule=checker.bundle["look"]),
                   check_metadata=dict(method="lcr", bundle_id=checker.bundle["id"],
                                       calibration=checker.bundle.get("calibration", "B_cal only")))
        yield row, maps


__all__ = ["run", "finalise", "thresholds", "relabel", "class_queries", "CLASSES", "CLASS_INDEX"]
