"""Decoder-only extraction for the bounded E2 experiment; no encoder or training."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer, BlipConfig
from transformers.models.blip.modeling_blip_text import BlipTextLMHeadModel

from . import config, pipeline
from .captioner import Captioner

METHODS = ("S1_raw", "S3_caption", "S3_prefix", "S2_gradcam", "S4_contrast", "S3_read", "S3_pieces")
CANDIDATES = ("S3_prefix", "S2_gradcam", "S4_contrast", "S3_read", "S3_pieces")


def install_visual_mask_routing(model):
    """Route BLIP's prepared cross-mask to the argument read by its self module.

    In the installed transformers build BlipTextAttention forwards the mask as
    attention_mask, but cross-attention's inner module reads encoder_attention_mask.
    This local hook changes no weights and adds zero for the unmasked condition.
    """
    def route(_module, args, kwargs):
        if kwargs.get("encoder_hidden_states") is not None and kwargs.get("encoder_attention_mask") is None:
            kwargs["encoder_attention_mask"] = kwargs.get("attention_mask")
        return args, kwargs
    return [layer.crossattention.self.register_forward_pre_hook(route, with_kwargs=True)
            for layer in model.bert.encoder.layer]


def map_array(weights):
    a = weights.detach().float().clamp_min(0)
    degenerate = not bool(a.sum() > 0)
    if degenerate:
        a = torch.ones_like(a)
    a = (a / a.sum()).reshape(24, 24).cpu().numpy().astype(np.float16)
    return a, degenerate


class LookDecoder:
    def __init__(self):
        torch.set_num_threads(4)
        conf = BlipConfig.from_pretrained(config.BLIP_NAME, local_files_only=True)
        self.tok = AutoTokenizer.from_pretrained(config.BLIP_NAME, local_files_only=True)
        self.model = BlipTextLMHeadModel(conf.text_config).eval()
        self.model.load_state_dict(torch.load(config.CACHE / "ckpt" / "decoder_A.pt",
                                             map_location="cpu", weights_only=True))
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.mask_hooks = install_visual_mask_routing(self.model)
        self.bos = conf.text_config.bos_token_id
        self.class_tokens = {}
        for token, cls in pipeline.token_class_map(SimpleNamespace(tok=self.tok)).items():
            self.class_tokens.setdefault(cls, []).append(token)

    @contextmanager
    def capture(self):
        attention = []
        hooks = [layer.crossattention.self.register_forward_hook(
                 lambda _m, _i, out: attention.append(out[1])) for layer in self.model.bert.encoder.layer]
        try:
            yield attention
        finally:
            for hook in hooks:
                hook.remove()

    def forward(self, ids, image, image_mask=None):
        if image_mask is None:
            image_mask = torch.ones(image.shape[:2], dtype=torch.long)
        return self.model(input_ids=ids, attention_mask=torch.ones_like(ids), encoder_hidden_states=image,
                          encoder_attention_mask=image_mask, use_cache=False)

    def tokens(self, row):
        ids = self.tok(row["draft"], return_tensors="pt")["input_ids"]
        ids[:, 0] = self.bos
        words, positions = Captioner.word_alignment(SimpleNamespace(tok=self.tok), ids[0])
        alignment = dict(zip(positions, words))
        for c in row["claims"]:
            if alignment.get(c["pos"]) != c["word"]:
                raise ValueError(f"Draft retokenisation changed claim alignment: {row['id']} {c['word']}")
        return ids

    def word_end(self, ids, p):
        pieces = self.tok.convert_ids_to_tokens(ids[0].tolist())
        end = p + 1
        while end < len(pieces) and pieces[end].startswith("##"):
            end += 1
        if pieces[p:p+3] == ["t", "-", "shirt"]:
            end = p + 3
        return end

    def extract(self, row, image_features, gradients=True):
        ids = self.tokens(row)
        image = torch.tensor(np.array(image_features), dtype=torch.float32)[None]
        image.requires_grad_(gradients)
        result, diagnostics = [], []
        with torch.set_grad_enabled(gradients), self.capture() as attention:
            output = self.forward(ids, image)
            full = torch.stack([a[0] for a in attention])  # L,H,T,577
            valid_t = ids.shape[1] - 1
            mean = full[-4:, :, :valid_t, 1:].mean((0, 1, 2))
            for c in row["claims"]:
                p = c["pos"]; end = self.word_end(ids, p)
                raw = full[-4:, :, p-1, 1:].mean((0, 1))
                sources = dict(S1_raw=raw, S3_caption=(raw-mean).clamp_min(0),
                    S3_prefix=(raw-full[-4:, :, :p, 1:].mean((0, 1, 2))).clamp_min(0),
                    S3_read=(full[-4:, :, p, 1:].mean((0, 1))-mean).clamp_min(0),
                    S3_pieces=(full[-4:, :, p-1:end-1, 1:].mean((0, 1, 2))-mean).clamp_min(0))
                info = dict(word=c["word"], pos=p, pieces=end-p, contrast_available=False)
                if gradients:
                    lp = output.logits[0, p-1].log_softmax(-1)
                    target = lp[ids[0, p]]
                    grad = torch.autograd.grad(target, tuple(attention), retain_graph=True)
                    g = torch.stack([v[0] for v in grad])
                    local = full[-4:, :, p-1, 1:]
                    sources["S2_gradcam"] = (g[-4:, :, p-1, 1:] * local).clamp_min(0).mean((0, 1))
                    info["gradient_abs_sum"] = float(sum(v.abs().sum() for v in grad))
                    alternatives = self.class_tokens.get(c["blip_alts"][0], []) if c["blip_alts"] else []
                    if alternatives:
                        alt = alternatives[int(output.logits[0, p-1, alternatives].argmax())]
                        ga = torch.autograd.grad(lp[alt], tuple(attention), retain_graph=True)
                        contrast = torch.stack([v[0] for v in ga])
                        sources["S4_contrast"] = ((g[-4:, :, p-1, 1:]-contrast[-4:, :, p-1, 1:]) * local).clamp_min(0).mean((0, 1))
                        info.update(contrast_available=True, alt_class=c["blip_alts"][0],
                                    alt_token=self.tok.convert_ids_to_tokens(alt))
                    else:
                        # No candidate was generated: retain S3, and report availability.
                        sources["S4_contrast"] = sources["S3_caption"]
                maps = {}
                info["degenerate"] = {}
                for name, weights in sources.items():
                    maps[name], info["degenerate"][name] = map_array(weights)
                result.append(maps); diagnostics.append(info)
        return ids, result, diagnostics

    @torch.no_grad()
    def intervention(self, ids, claim, image_features, seed):
        """First-wordpiece log probability under four fixed-prefix feature masks."""
        p = claim["pos"]
        yy, xx = np.mgrid[:24, :24]
        xx, yy = (xx+.5)/24, (yy+.5)/24
        def in_box(box):
            return (((xx >= box[0]) & (xx <= box[2]) & (yy >= box[1]) & (yy <= box[3])).ravel()
                    if box else np.zeros(576, bool))
        look_mask, det_mask = in_box(claim["rec_att"]["box"]), in_box(claim["rec_det"]["box"])
        random_mask = np.zeros(576, bool)
        random_mask[np.random.default_rng(seed).choice(576, int(look_mask.sum()), replace=False)] = True
        masks = torch.ones((4, 577), dtype=torch.long)
        for j, hidden in enumerate((look_mask, random_mask, det_mask), 1):
            masks[j, 1:] = torch.from_numpy((~hidden).astype(np.int64))
        image = torch.tensor(np.array(image_features), dtype=torch.float32)[None].expand(4, -1, -1)
        out = self.forward(ids[:, :p].expand(4, -1), image, masks)
        scores = out.logits[:, -1].log_softmax(-1)[:, ids[0, p]].tolist()
        return dict(logp_base=scores[0], drop_look=scores[0]-scores[1], drop_random=scores[0]-scores[2],
                    drop_detector=scores[0]-scores[3], look_tokens=int(look_mask.sum()),
                    detector_tokens=int(det_mask.sum()), target="first_wordpiece",
                    level="contextualised_visual_features; CLS retained")
