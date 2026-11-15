"""The properties the proposal promises to 'verify numerically' for the gated pathway."""
import copy

import numpy as np
import pytest
import torch

from lcr import retell
from lcr.vocab import CLASS_INDEX


@pytest.fixture(scope="module")
def setup(captioner, sample_images):
    emb = captioner.encode_images(sample_images)
    ids = captioner.generate(emb)
    mask = (ids != captioner.pad).long()
    base_logits = captioner.decoder_forward(ids, mask, emb).logits.clone()
    pathway = retell.RetellPathway(captioner.model).to(captioner.dev)
    pathway.install(captioner.model.text_decoder)
    recs = []
    for i in range(len(sample_images)):
        feats = (CLASS_INDEX["ball"], CLASS_INDEX["frisbee"], np.random.default_rng(i).normal(size=512).astype(np.float32),
                 np.array([.02, .7, .4, .3, .5, .5, .2, .2, 1.], np.float32), retell.TYPE_IDS["contradicted"])
        recs.append([feats] * (i % 3))            # 0, 1, 2, 0 records -> exercises padding
    ev = retell.collate(recs).to(captioner.dev)
    yield captioner, pathway, emb, ids, mask, base_logits, ev
    pathway.uninstall(captioner.model.text_decoder)
    pathway.set_evidence(None)


def test_zero_init_reproduces_base_logits_exactly(setup):
    cap, pathway, emb, ids, mask, base_logits, ev = setup
    pathway.eval(); pathway.set_evidence(ev)
    with torch.no_grad():
        logits = cap.decoder_forward(ids, mask, emb).logits
    assert torch.equal(logits, base_logits)       # bit-identical, not just close


def test_zero_init_reproduces_base_beam_search(setup):
    cap, pathway, emb, ids, *_ , ev = setup
    pathway.eval(); pathway.set_evidence(ev)
    assert torch.equal(cap.generate(emb), ids)


def test_only_pathway_is_trainable_and_gates_get_gradient(setup):
    cap, pathway, emb, ids, mask, _, ev = setup
    pathway.train(); pathway.set_evidence(ev)
    loss = cap.decoder_forward(ids, mask, emb, labels=True).loss
    loss.backward()
    assert all(not p.requires_grad for p in cap.model.parameters())
    for b in pathway.blocks:
        assert b.gate_attn.grad is not None and b.gate_attn.grad.abs().item() > 0
    pathway.zero_grad()
    # trainable parameters must not leak into the frozen decoder's state dict
    assert not any("_block" in k or "gate" in k for k in cap.model.text_decoder.state_dict())


def test_open_gates_make_evidence_matter(setup):
    cap, pathway, emb, ids, mask, base_logits, ev = setup
    saved = copy.deepcopy(pathway.state_dict())
    try:
        with torch.no_grad():
            for b in pathway.blocks:
                b.gate_attn.fill_(0.5)
        pathway.eval()
        with torch.no_grad():
            pathway.set_evidence(ev)
            with_ev = cap.decoder_forward(ids, mask, emb).logits
            pathway.set_evidence(retell.null_only(len(ids)).to(cap.dev))
            null = cap.decoder_forward(ids, mask, emb).logits
        assert not torch.allclose(with_ev, base_logits)
        # image 0 and 3 carry no records: evidence == null-only for them, but not for 1 and 2
        assert torch.allclose(with_ev[0], null[0], atol=1e-4) and torch.allclose(with_ev[3], null[3], atol=1e-4)
        assert not torch.allclose(with_ev[1], null[1], atol=1e-4)
    finally:
        pathway.load_state_dict(saved)


def test_cached_decoding_matches_teacher_forcing_with_open_gates(setup):
    """Beam search uses key/value caching (one token at a time); the gated block must
    give the same result as a full teacher-forced pass, otherwise training and
    inference would disagree."""
    cap, pathway, emb, ids, mask, _, ev = setup
    saved = copy.deepcopy(pathway.state_dict())
    try:
        with torch.no_grad():
            for b in pathway.blocks:
                b.gate_attn.fill_(0.5); b.gate_ff.fill_(0.3)
        pathway.eval(); 
        with torch.no_grad():
            pathway.set_evidence(ev)
            out = cap.generate(emb, num_beams=1, min_new_tokens=0)
            m = (out != cap.pad).long()
            logits = cap.decoder_forward(out, m, emb).logits
        greedy = logits.argmax(-1)[:, :-1]
        tgt = out[:, 1:]
        valid = m[:, 1:].bool()
        assert torch.equal(greedy[valid], tgt[valid])
    finally:
        pathway.load_state_dict(saved)
