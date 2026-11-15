"""Frozen E3 inference shared by live decoding and cache migration.

SAM geometry and OWLv2 score geometry are explicitly separate. The latter is a
nearest-proposal proxy, never a detector evaluation of the SAM crop itself.
"""
from __future__ import annotations

import numpy as np
import torch

from . import cache, config, look


def selection():
    a = cache.load_json(config.RESULTS / 'look_e3_selection.json')['winner']
    b = cache.load_json(config.RESULTS / 'look_e3_boxes_selection.json')
    if not a or a['family'] != 'A1_G3_labelfree_heads' or not a.get('accepted'):
        raise ValueError('Look v4 requires the frozen E3 label-free head selection')
    if any(r != 'base' and not r.startswith('B2_a') for r in b['rule'].values()):
        raise ValueError('Unsupported frozen E3 SAM rule')
    return a, b['rule']


def metadata():
    a, rules = selection()
    return dict(schema_version=3, look_version='v4', region_rule='v4',
                map_source='blip_decoder_cross_attention', map_correction='caption_mean_subtract_relu',
                head_pooling='selected_heads', heads=a['heads'],
                head_indexing='flat zero-based layer*12+head',
                head_names=[f'L{h//12+1}H{h%12+1}' for h in a['heads']],
                sam_selection_score='attention mass in bounding rectangle / rectangle area**alpha',
                head_selection=a['family'] + ':' + a['config'],
                token_position='first_piece_prediction', posthoc_caption_baseline=True,
                caption_tokens='retokenised draft; BOS replaced; SEP excluded from mean',
                signed_head_storage='float16 before selected-head mean',
                available_last_k=[4], last_k_role='compatibility key; selected heads, not last four layers',
                rules_by_kind=config.LOOK_RULES_BY_KIND, box_rule_by_kind=rules,
                score_source='nearest valid OWLv2 proposal; proxy for SAM box',
                selection='results/look_e3_selection.json, results/look_e3_boxes_selection.json')


def selected_map(signed_heads, heads):
    x = np.asarray(signed_heads, np.float32).reshape(144, 576)[heads].mean(0)
    x = np.clip(x, 0, None)
    return (x / x.sum() if x.sum() > 0 else np.full_like(x, 1 / x.size)).reshape(24, 24)


def attention_map(attention, position, valid_t, heads):
    """Match E3's signed-head cache quantisation, before pooling and ReLU."""
    a = attention.float()
    signed = a[:, :, position - 1, 1:] - a[:, :, :valid_t, 1:].mean(2)
    return selected_map(signed.detach().cpu().numpy().astype(np.float16), heads)


def mask_box(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return np.zeros(4, np.float32), 0.
    h, w = mask.shape
    return np.array([xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h], np.float32), float(mask.mean())


class SamBoxes:
    """Lazy SAM; image embedding reused for every claim's peak in that image."""
    def __init__(self, dev=None):
        from transformers import SamModel, SamProcessor
        self.dev = dev or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = SamModel.from_pretrained('facebook/sam-vit-base', local_files_only=True).to(self.dev).eval()
        self.proc = SamProcessor.from_pretrained('facebook/sam-vit-base', local_files_only=True)

    @torch.no_grad()
    def __call__(self, image, maps):
        if not len(maps):
            return np.empty((0, 3, 4), np.float32)
        w, h = image.size
        peaks = np.asarray(maps).reshape(len(maps), -1).argmax(1)
        pts = [[[(float(p % 24)+.5)/24*w, (float(p // 24)+.5)/24*h]] for p in peaks]
        inp = self.proc(image, input_points=[pts], return_tensors='pt')
        emb = self.model.get_image_embeddings(inp['pixel_values'].to(self.dev))
        out = self.model(image_embeddings=emb, input_points=inp['input_points'].float().to(self.dev),
                         multimask_output=True)
        masks = self.proc.image_processor.post_process_masks(out.pred_masks.cpu(), inp['original_sizes'],
                                                             inp['reshaped_input_sizes'])[0].numpy()
        return np.asarray([[mask_box(m)[0] for m in masks[j]] for j in range(len(maps))], np.float32)


def rewrite_claim(claim, att, prop, scores, box_rule, sam_boxes=None):
    """Use an already smoothed map; refresh every geometry-dependent field."""
    from . import pipeline
    rule = config.LOOK_RULES_BY_KIND[claim['kind']]
    out = pipeline.rewrite_look_claim(claim, att, prop, scores, look_version='v4',
                                     region_parameters=dict(alpha=rule['alpha'], gamma=rule['gamma']))
    boxes = prop['boxes'].astype(np.float32)
    det = dict(out['rec_det']); dreg = det['region']
    det['excess'] = float(look.excess_mass(att, boxes[dreg:dreg+1])[0]) if dreg >= 0 else 0.
    out['rec_det'] = det
    if box_rule == 'base':
        return out
    if sam_boxes is None:
        raise ValueError('SAM candidates required for the frozen person rule')
    candidates = np.asarray(sam_boxes, np.float32)
    areas = look.box_area(candidates)
    valid_sam = np.isfinite(candidates).all(1) & (areas > 0)
    if not valid_sam.any():
        # Explicit inference failure policy; not exercised by the frozen caches.
        out['rec_att']['sam_fallback'] = 'no nonempty finite mask'
        return out
    alpha = float(box_rule.removeprefix('B2_a'))
    values = look.attention_mass(att, candidates) / np.clip(areas, 1e-6, None)**alpha
    sbox = candidates[int(np.argmax(np.where(valid_sam, values, -np.inf)))]
    valid = look.size_ok(boxes)
    overlap = np.where(valid, look.iou(sbox, boxes), -1.)
    reg = int(overlap.argmax()) if valid.any() else None
    excess = float(look.excess_mass(att, sbox[None])[0])
    rec = pipeline.region_record(claim['cls'], reg, excess, prop, scores, claim['blip_alts'])
    rec.update(by='attention', look='v4', box=sbox.tolist(), box_source='sam',
               score_source='owl_nearest_proposal_proxy',
               score_box=boxes[reg].tolist() if reg is not None else [],
               sam_proposal_iou=float(overlap[reg]) if reg is not None else 0.)
    out.update(rec_att=rec, regions_by_k={'4': reg if reg is not None else -1},
               excess_by_k={'4': excess},
               iou_att_det=float(look.iou(sbox, boxes[dreg:dreg+1])[0]) if dreg >= 0 else 0.)
    return out
