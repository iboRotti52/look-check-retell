"""Retell: a zero-initialised gated evidence pathway on top of the frozen BLIP decoder.

Design (after Flamingo's GATED XATTN-DENSE block):

    h <- h + tanh(a_attn) * CrossAttn(LN(h), evidence tokens)
    h <- h + tanh(a_ff)   * FF(LN(h))
    h <- frozen BLIP decoder layer(h)

Both gates start at 0, so tanh(gate) = 0 and the wrapped decoder is *exactly*
the base captioner before training (verified in tests/test_retell.py).
Only the evidence encoder and the gated blocks are trainable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .check import VERDICTS, Record
from .vocab import CLASSES, CLASS_INDEX

N_SCALARS = 9                      # s_claim, s_alt, s_image, excess, cx, cy, w, h, has_region
TYPE_IDS = {**{v: i for i, v in enumerate(VERDICTS)}, "tag": len(VERDICTS)}
REGION_DIM = 512


@dataclass
class EvidenceBatch:
    """Padded evidence for a batch of images. Slot 0 is reserved for the null token."""
    claim: torch.Tensor      # (B, R) long, class index, -1 = padding
    alt: torch.Tensor        # (B, R) long, class index, -1 = none
    region: torch.Tensor     # (B, R, 512)
    scalars: torch.Tensor    # (B, R, N_SCALARS)
    type_id: torch.Tensor    # (B, R) long
    valid: torch.Tensor      # (B, R) bool

    def to(self, dev):
        return EvidenceBatch(*(getattr(self, f).to(dev) for f in self.__dataclass_fields__))


def record_features(rec: Record | dict, region_embed: np.ndarray | None):
    r = rec if isinstance(rec, dict) else rec.to_dict()
    has = r["region"] >= 0
    if has:
        x0, y0, x1, y1 = r["box"]
        geom = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    else:
        geom = [0.0, 0.0, 0.0, 0.0]
    # Historical v4 stored the new scores separately; never pair its new alternative with old scores.
    score_keys = ("s_claim_cal", "s_alt_cal", "s_image_cal") if r.get("check") == "v4" else ("s_claim", "s_alt", "s_image")
    scal = [float(r.get(k) or 0.) for k in score_keys] + [r["excess"], *geom, float(has)]
    reg = region_embed if (has and region_embed is not None) else np.zeros(REGION_DIM, np.float32)
    return (CLASS_INDEX[r["claim"]], CLASS_INDEX.get(r["alt"], -1),
            np.asarray(reg, np.float32), np.asarray(scal, np.float32), TYPE_IDS[r["verdict"]])


def optimiser_groups(pathway, weight_decay=.01):
    """Keep zero-initialised gates, biases and normalisation scales out of decay."""
    decay, no_decay = [], []
    for name, p in pathway.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "gate_" in name else decay).append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.)]


def collate(per_image: list[list[tuple]]) -> EvidenceBatch:
    """per_image[b] is a list of `record_features` tuples (possibly empty)."""
    b = len(per_image)
    r = max(1, max(len(x) for x in per_image))
    claim = torch.full((b, r), -1, dtype=torch.long)
    alt = torch.full((b, r), -1, dtype=torch.long)
    region = torch.zeros(b, r, REGION_DIM)
    scal = torch.zeros(b, r, N_SCALARS)
    typ = torch.zeros(b, r, dtype=torch.long)
    valid = torch.zeros(b, r, dtype=torch.bool)
    for i, recs in enumerate(per_image):
        for j, (c, a, reg, s, t) in enumerate(recs):
            claim[i, j], alt[i, j], typ[i, j], valid[i, j] = c, a, t, True
            region[i, j] = torch.from_numpy(reg)
            scal[i, j] = torch.from_numpy(s)
    return EvidenceBatch(claim, alt, region, scal, typ, valid)


class EvidenceEncoder(nn.Module):
    """Record -> one 768-d token. Label text enters through BLIP's own (frozen) word embeddings."""

    def __init__(self, class_table: torch.Tensor, d: int = 768):
        super().__init__()
        self.register_buffer("class_table", class_table, persistent=False)   # (|V|, d), frozen
        self.claim_proj = nn.Linear(d, d)
        self.alt_proj = nn.Linear(d, d)
        self.visual = nn.Sequential(nn.Linear(REGION_DIM + N_SCALARS, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Embedding(len(TYPE_IDS), d)
        self.norm = nn.LayerNorm(d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.null, std=0.02)

    def forward(self, ev: EvidenceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns tokens (B, 1+R, d) and key_padding_mask (B, 1+R), True = ignore."""
        claim = self.class_table[ev.claim.clamp_min(0)]
        alt = self.class_table[ev.alt.clamp_min(0)] * (ev.alt >= 0).unsqueeze(-1)
        tok = (self.claim_proj(claim) + self.alt_proj(alt)
               + self.visual(torch.cat([ev.region, ev.scalars], dim=-1))
               + self.type_emb(ev.type_id))
        tok = self.norm(tok)
        null = self.null.expand(tok.shape[0], -1, -1)
        tokens = torch.cat([null, tok], dim=1)
        pad = torch.cat([torch.zeros_like(ev.valid[:, :1]), ~ev.valid], dim=1)
        return tokens, pad


class GatedBlock(nn.Module):
    def __init__(self, d: int = 768, heads: int = 12, ff_mult: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.norm_ff = nn.LayerNorm(d)
        hidden = int(d * ff_mult)
        self.ff = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.gate_ff = nn.Parameter(torch.zeros(1))

    def forward(self, h, tokens, pad):
        a, _ = self.attn(self.norm_attn(h), tokens, tokens, key_padding_mask=pad, need_weights=False)
        h = h + torch.tanh(self.gate_attn) * a
        h = h + torch.tanh(self.gate_ff) * self.ff(self.norm_ff(h))
        return h


class GatedLayer(nn.Module):
    """Wraps one frozen BLIP decoder layer; the gated block runs first."""

    def __init__(self, base: nn.Module, block: GatedBlock, pathway: "RetellPathway"):
        super().__init__()
        self.base = base
        # keep trainable parts OUT of the decoder's state_dict
        object.__setattr__(self, "_block", block)
        object.__setattr__(self, "_pathway", pathway)

    def forward(self, hidden_states, *args, **kwargs):
        ctx = self._pathway.context
        if ctx is not None:
            tokens, pad = ctx
            rep = hidden_states.shape[0] // tokens.shape[0]
            if rep > 1:                                   # beam search expands the batch
                tokens = tokens.repeat_interleave(rep, dim=0)
                pad = pad.repeat_interleave(rep, dim=0)
            hidden_states = self._block(hidden_states, tokens, pad)
        return self.base(hidden_states, *args, **kwargs)


class RetellPathway(nn.Module):
    """Evidence encoder + gated blocks. `install` wires it into a BLIP text decoder."""

    def __init__(self, blip_model, layers=(6, 7, 8, 9, 10, 11), heads: int = 12, ff_mult: float = 1.0):
        super().__init__()
        tok_emb = blip_model.text_decoder.get_input_embeddings().weight.detach()
        self.layers = tuple(layers)
        self.encoder = EvidenceEncoder(self._class_table(tok_emb), d=tok_emb.shape[1])
        self.blocks = nn.ModuleList(GatedBlock(tok_emb.shape[1], heads, ff_mult) for _ in self.layers)
        self.context = None
        self._installed = False

    @staticmethod
    def _class_table(tok_emb: torch.Tensor) -> torch.Tensor:
        from transformers import BlipProcessor
        from . import config
        tok = BlipProcessor.from_pretrained(config.BLIP_NAME).tokenizer
        rows = []
        for name in CLASSES:
            ids = tok(name, add_special_tokens=False)["input_ids"]
            rows.append(tok_emb[ids].mean(dim=0))
        return torch.stack(rows).clone()

    def install(self, text_decoder) -> "RetellPathway":
        assert not self._installed
        layer_list = text_decoder.bert.encoder.layer
        for block, idx in zip(self.blocks, self.layers):
            layer_list[idx] = GatedLayer(layer_list[idx], block, self)
        self._installed = True
        return self

    def uninstall(self, text_decoder):
        layer_list = text_decoder.bert.encoder.layer
        for idx in self.layers:
            if isinstance(layer_list[idx], GatedLayer):
                layer_list[idx] = layer_list[idx].base
        self._installed = False

    def set_evidence(self, ev: EvidenceBatch | None):
        """Encode and expose evidence to the gated layers; None disables the pathway."""
        self.context = None if ev is None else self.encoder(ev)

    def gates(self) -> dict[str, float]:
        out = {}
        for b, idx in zip(self.blocks, self.layers):
            out[f"L{idx}.attn"] = float(torch.tanh(b.gate_attn.detach()))
            out[f"L{idx}.ff"] = float(torch.tanh(b.gate_ff.detach()))
        return out

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def null_only(batch_size: int) -> EvidenceBatch:
    """Evidence batch with no records: the pathway sees only the learned null token."""
    return collate([[] for _ in range(batch_size)])
