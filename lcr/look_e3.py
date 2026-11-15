"""Look E3 map extraction: the literature gaps listed in docs/LOOK_E3_PROTOKOL.md.

One teacher-forced decoder pass per draft (frozen decoder_A, cached ViT features) gives, per claim:
- per-head signed sink maps  A_lh(p-1) - mean_t A_lh(t)   (12 x 12 x 576), for layer/head selection;
- S3 (recomputed v2 map), A3 ratio sink, A2 noun-phrase map;
- S2 Grad-CAM, S4 contrastive Grad-CAM (as in E2), A4 Chefer et al. (ICCV'21) relevance propagation;
- A5 Salience DropOut (PnP-OVSS): S3 recomputed with the top-q patches masked at feature level.
No training, no test split, no F30kE boxes here (boxes are only read by the selection script).
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import nltk
import numpy as np
import torch

from . import config
from .captioner import Captioner
from .look_sources import LookDecoder

nltk.data.path.insert(0, str(config.DATA / "nltk"))
MODIFIER_TAGS = {"JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "CD"}
MAPS = ("S3", "S3_all", "A3_ratio", "A2_phrase", "S2_gradcam", "S4_contrast", "A4_chefer", "A5_q5", "A5_q10")
DROPOUT_FRACTIONS = {"A5_q5": .05, "A5_q10": .10}


def unit(a: torch.Tensor) -> np.ndarray:
    """Clamp at zero and normalise to a 24x24 probability map (uniform if empty)."""
    a = a.detach().float().clamp_min(0)
    a = a / a.sum() if float(a.sum()) > 0 else torch.full_like(a, 1 / a.numel())
    return a.reshape(24, 24).numpy()


def chefer_normalise(r: torch.Tensor) -> torch.Tensor:
    """R_bar = (R - I) with rows normalised to sum 1, plus I (Chefer et al. 2021, eq. 8-9)."""
    eye = torch.eye(r.shape[0])
    hat = r - eye
    s = hat.sum(-1, keepdim=True)
    hat = torch.where(s > 0, hat / s.clamp_min(1e-12), hat)
    return hat + eye


class E3Decoder(LookDecoder):
    @contextmanager
    def capture_all(self):
        cross, self_att = [], []
        hooks = []
        for layer in self.model.bert.encoder.layer:
            hooks.append(layer.crossattention.self.register_forward_hook(
                lambda _m, _i, out: cross.append(out[1])))
            hooks.append(layer.attention.self.register_forward_hook(
                lambda _m, _i, out: self_att.append(out[1])))
        try:
            yield cross, self_att
        finally:
            for h in hooks:
                h.remove()

    def phrase_positions(self, ids, p: int) -> list[int]:
        """First-piece positions of the claim word and its contiguous pre-modifiers."""
        words, positions = Captioner.word_alignment(SimpleNamespace(tok=self.tok), ids[0])
        j = positions.index(p)
        tags = [t for _, t in nltk.pos_tag(words)]
        start = j
        while start > 0 and tags[start - 1] in MODIFIER_TAGS:
            start -= 1
        return positions[start:j + 1]

    def extract_e3(self, row, image_features):
        ids = self.tokens(row)
        valid_t = ids.shape[1] - 1
        image = torch.tensor(np.array(image_features), dtype=torch.float32)[None]
        image.requires_grad_(True)
        with torch.enable_grad(), self.capture_all() as (cross_l, self_l):
            out = self.forward(ids, image)
            cross = torch.stack([a[0] for a in cross_l])      # L,H,T,577
            selfa = torch.stack([a[0] for a in self_l])       # L,H,T,T
            capmean = cross[:, :, :valid_t, 1:].mean(2)        # L,H,576
            lp_all = out.logits[0].log_softmax(-1)
            heads, maps, info = [], [], []
            for c in row["claims"]:
                p = c["pos"]
                signed = (cross[:, :, p - 1, 1:] - capmean).detach()
                heads.append(signed.numpy().astype(np.float16))
                m = {"S3": unit(signed[-4:].mean((0, 1)))}
                # caption mean over every position incl. [SEP]: probes the archived-map drift seen in E2
                m["S3_all"] = unit((cross[-4:, :, p - 1, 1:].mean((0, 1)) - cross[-4:, :, :, 1:].mean((0, 1, 2))).detach())
                raw4 = cross[-4:, :, p - 1, 1:].mean((0, 1)).detach()
                m["A3_ratio"] = unit(raw4 / capmean[-4:].mean((0, 1)).detach().clamp_min(1e-8))
                pp = self.phrase_positions(ids, p)
                phr = torch.stack([cross[-4:, :, q - 1, 1:].mean((0, 1)) for q in pp]).mean(0).detach()
                m["A2_phrase"] = unit(phr - capmean[-4:].mean((0, 1)).detach())
                # gradients of the first word-piece log-probability w.r.t. all attention maps
                target = lp_all[p - 1, ids[0, p]]
                grads = self.grad(target, list(cross_l) + list(self_l))
                gc = torch.stack([g[0] for g in grads[:12]]); gs = torch.stack([g[0] for g in grads[12:]])
                local = cross[-4:, :, p - 1, 1:].detach()
                m["S2_gradcam"] = unit((gc[-4:, :, p - 1, 1:] * local).clamp_min(0).mean((0, 1)))
                alts = self.class_tokens.get(c["blip_alts"][0], []) if c["blip_alts"] else []
                if alts:
                    alt = alts[int(out.logits[0, p - 1, alts].argmax())]
                    ga = self.grad(lp_all[p - 1, alt], list(cross_l))
                    gca = torch.stack([g[0] for g in ga])
                    m["S4_contrast"] = unit(((gc - gca)[-4:, :, p - 1, 1:] * local).clamp_min(0).mean((0, 1)))
                else:
                    m["S4_contrast"] = m["S3"]
                m["A4_chefer"] = unit(self.chefer(cross.detach(), selfa.detach(), gc, gs, p - 1, valid_t + 1))
                maps.append(m)
                info.append(dict(phrase_len=len(pp), contrast=bool(alts)))
        self.salience_dropout(ids, image.detach(), maps, valid_t)
        return np.stack(heads), maps, info

    @staticmethod
    def grad(target, tensors):
        """d target / d tensors; zeros for maps outside the graph (layer-0 self-attention sees no image)."""
        live = [i for i, t in enumerate(tensors) if t.requires_grad]
        got = torch.autograd.grad(target, [tensors[i] for i in live], retain_graph=True, allow_unused=True)
        out = [torch.zeros_like(t) for t in tensors]
        for i, g in zip(live, got):
            if g is not None:
                out[i] = g
        return out

    @staticmethod
    def chefer(cross, selfa, gc, gs, t: int, T: int) -> torch.Tensor:
        """Encoder-decoder relevance (Chefer et al. ICCV'21), text queries x image keys, R_ii = I."""
        r_tt = torch.eye(T)
        r_ti = torch.zeros(T, cross.shape[-1])
        for layer in range(cross.shape[0]):
            a_s = (gs[layer] * selfa[layer]).clamp_min(0).mean(0)[:T, :T]
            r_ti = r_ti + a_s @ r_ti
            r_tt = r_tt + a_s @ r_tt
            a_c = (gc[layer] * cross[layer]).clamp_min(0).mean(0)[:T]
            r_ti = r_ti + chefer_normalise(r_tt).T @ a_c
        return r_ti[t, 1:]

    @torch.no_grad()
    def salience_dropout(self, ids, image, maps, valid_t: int):
        """One PnP-OVSS dropout round per claim and fraction: mask top-q patches, recompute S3, add."""
        jobs = []
        for ci, m in enumerate(maps):
            order = np.argsort(-m["S3"].ravel())
            for name, frac in DROPOUT_FRACTIONS.items():
                mask = torch.ones(577, dtype=torch.long)
                mask[1 + torch.from_numpy(order[:int(round(frac * 576))].copy())] = 0
                jobs.append((ci, name, mask))
        if not jobs:
            return
        masks = torch.stack([j[2] for j in jobs])
        n = len(jobs)
        with self.capture() as attention:
            self.forward(ids.expand(n, -1), image.expand(n, -1, -1), masks)
        cross = torch.stack(attention, 1)                      # n,L,H,T,577
        capmean = cross[:, -4:, :, :valid_t, 1:].mean((1, 2, 3))
        for k, (ci, name, mask) in enumerate(jobs):
            p = self._pos[ci]
            s3m = (cross[k, -4:, :, p - 1, 1:].mean((0, 1)) - capmean[k]).clamp_min(0) * mask[1:]
            second = unit(s3m)
            maps[ci][name] = unit(torch.from_numpy(maps[ci]["S3"].ravel() + second.ravel()))

    def run_row(self, row, image_features):
        if not row["claims"]:
            return np.zeros((0, 12, 12, 576), np.float16), [], []
        self._pos = [c["pos"] for c in row["claims"]]
        return self.extract_e3(row, image_features)
