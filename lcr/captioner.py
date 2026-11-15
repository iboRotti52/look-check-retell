"""BLIP captioner wrapper: image encoding, drafting, and decoder cross-attention.

The vision encoder is always frozen. Its patch embeddings (1 CLS + 24x24
patches, 768-d) are computed once per image and reused by every system, which
is what makes decoder fine-tuning and Retell training cheap.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from . import config


class Captioner:
    def __init__(self, decoder_weights: str | None = None, dev: torch.device | None = None):
        self.dev = dev or config.device()
        self.processor = BlipProcessor.from_pretrained(config.BLIP_NAME)
        self.tok = self.processor.tokenizer
        self.model = BlipForConditionalGeneration.from_pretrained(config.BLIP_NAME).to(self.dev).eval()
        if decoder_weights:
            state = torch.load(decoder_weights, map_location="cpu")
            self.model.text_decoder.load_state_dict(state)
        for p in self.model.parameters():
            p.requires_grad_(False)
        tc = self.model.config.text_config
        self.bos, self.sep, self.pad = tc.bos_token_id, tc.sep_token_id, tc.pad_token_id

    # ------------------------------------------------------------------ vision
    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor:
        px = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.dev)
        return self.model.vision_model(pixel_values=px)[0]          # (B, 577, 768)

    # ------------------------------------------------------------------- text
    def encode_captions(self, captions: list[str], max_len: int = 40):
        """[BOS] w1 ... wn [SEP], padded. Returns (input_ids, attention_mask)."""
        enc = self.tok(captions, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].clone()
        ids[:, 0] = self.bos
        return ids, enc["attention_mask"]

    def decoder_forward(self, input_ids, attention_mask, image_embeds, labels: bool = False, img_mask=None):
        if img_mask is None:
            img_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
        lab = None
        if labels:
            lab = input_ids.masked_fill(attention_mask == 0, -100)
        return self.model.text_decoder(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=img_mask,
            labels=lab, reduction="mean",
        )

    @torch.no_grad()
    def generate(self, image_embeds: torch.Tensor, **overrides) -> torch.Tensor:
        """Beam-search drafts. Returns token ids including the leading BOS."""
        kw = {**config.GEN_KWARGS, **overrides}
        b = image_embeds.shape[0]
        start = torch.full((b, 1), self.bos, dtype=torch.long, device=image_embeds.device)
        img_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
        return self.model.text_decoder.generate(
            input_ids=start, eos_token_id=self.sep, pad_token_id=self.pad,
            encoder_hidden_states=image_embeds, encoder_attention_mask=img_mask, **kw,
        )

    def decode(self, ids: torch.Tensor) -> list[str]:
        return [t.strip() for t in self.tok.batch_decode(ids, skip_special_tokens=True)]

    # --------------------------------------------------------------- attention
    @contextmanager
    def _capture_cross_attention(self):
        store: list[torch.Tensor] = []
        hooks = []
        for layer in self.model.text_decoder.bert.encoder.layer:
            base = getattr(layer, "base", layer)       # unwrap a gated layer if present
            hooks.append(base.crossattention.self.register_forward_hook(
                lambda _m, _i, out, s=store: s.append(out[1].detach())))
        try:
            yield store
        finally:
            for h in hooks:
                h.remove()

    @torch.no_grad()
    def cross_attention(self, image_embeds: torch.Tensor, ids: torch.Tensor):
        """Teacher-forced pass over finished captions.

        Returns (attn, logits): attn has shape (B, L, H, T, 577), where
        attn[b, l, h, t] is the attention used at decoder position t, i.e. when
        predicting token t+1. logits has shape (B, T, vocab).
        """
        mask = (ids != self.pad).long()
        with self._capture_cross_attention() as store:
            out = self.decoder_forward(ids, mask, image_embeds)
        return torch.stack(store, dim=1), out.logits

    # ----------------------------------------------------------- word alignment
    def word_alignment(self, ids_row: torch.Tensor) -> tuple[list[str], list[int]]:
        """Merge word pieces. Returns (words, first_piece_position) for one caption.

        Positions index into `ids_row` (BOS is position 0). Non-alphabetic
        pieces are dropped so that word indices agree with `claims.words`.
        """
        toks = self.tok.convert_ids_to_tokens(ids_row.tolist())
        words: list[str] = []
        pos: list[int] = []
        special = set(self.tok.all_special_tokens)
        prev_was_hyphen_t = False
        for p, t in enumerate(toks):
            if t is None or t in special:
                continue
            if t.startswith("##") and words:
                words[-1] += t[2:]
                continue
            if not t.isalpha():
                prev_was_hyphen_t = (t == "-" and bool(words) and words[-1] == "t")
                continue
            if prev_was_hyphen_t and t == "shirt":      # "t - shirt" -> "tshirt"
                words[-1] = "tshirt"
                prev_was_hyphen_t = False
                continue
            prev_was_hyphen_t = False
            words.append(t)
            pos.append(p)
        return words, pos
