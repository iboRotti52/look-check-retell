"""Retell sees Check's decision: S4 action and chosen substitute reach the evidence encoder (CPU, no BLIP)."""
import copy
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from lcr import evidence, pipeline, retell
from lcr import region_evidence as re
from lcr.vocab import CLASSES, CLASS_INDEX


def _checker():
    q = np.zeros((len(CLASSES), 3), np.float32)
    q[CLASS_INDEX["dog"]] = [2, 0, 0]; q[CLASS_INDEX["cat"]] = [0, 2, 0]; q[CLASS_INDEX["frisbee"]] = [0, 0, 2]
    bundle = dict(classes=CLASSES, query_sha256=hashlib.sha256(q.tobytes()).hexdigest(), decision="rival_np",
                  mu=[0.] * len(CLASSES), sd=[1.] * len(CLASSES), look=dict(min_patches=0, alpha=.5), pool="max",
                  alternatives="all", tau=[1.] * len(CLASSES), centre=[0.] * len(CLASSES), rival_weight=2.,
                  confusion=re.CONFUSION_CLEAN)
    bundle["id"] = re.fingerprint(bundle)
    return re.RegionChecker(bundle, q)


def _prop(embeds):
    n = len(embeds)
    return dict(boxes=np.array([[.05 + .3 * i, .1, .3 + .3 * i, .5] for i in range(n)], np.float32),
                objectness=np.ones(n, np.float32), embeds=np.asarray(embeds, np.float32),
                shift=np.zeros(n, np.float32), scale=np.ones(n, np.float32))


def _att():
    a = np.zeros((24, 24), np.float32); a[3:12, 2:7] = 1
    return a / a.sum()


def _tokens(rec, prop):
    row = dict(claims=[dict(cls=rec["claim"], rec_att=rec, blip_alts=[])])
    feats, _ = evidence.features(row, "ours", pipeline.DEFAULT_THRESHOLDS, prop, None)
    pad = lambda r: np.pad(r, (0, retell.REGION_DIM - len(r)))           # toy 3-d embeddings -> real 512-d slot
    return [t._replace(region=pad(t.region)) for t in feats]


@pytest.fixture
def cases():
    ch = _checker()
    p_rep, p_rem = _prop([[.6, 1, 0], [0, 0, 0]]), _prop([[0, 0, 1], [0, 0, 0]])
    replace, remove = ch.record("dog", _att(), p_rep), ch.record("dog", _att(), p_rem)
    assert replace["action"] == "replace" and replace["substitute"] == "cat" and remove["action"] == "remove"
    assert replace["verdict"] == remove["verdict"] == "contradicted"        # what Retell used to see
    return (replace, p_rep), (remove, p_rem)


def _encoder(d=32):
    torch.manual_seed(0)
    return retell.EvidenceEncoder(torch.randn(len(CLASSES), d), d=d)


def test_every_box_token_carries_the_action_and_substitute(cases):
    (rep, p_rep), (rem, p_rem) = cases
    t_rep, t_rem = _tokens(rep, p_rep), _tokens(rem, p_rem)
    assert {t.action for t in t_rep} == {retell.ACTION_IDS["replace"]}
    assert {t.substitute for t in t_rep} == {CLASS_INDEX["cat"]}
    assert {t.action for t in t_rem} == {retell.ACTION_IDS["remove"]} and {t.substitute for t in t_rem} == {-1}
    assert {t.type_id for t in t_rep} == {t.type_id for t in t_rem}       # the verdict alone cannot tell them apart


def test_remove_and_replace_are_distinguishable_once_trained(cases):
    (rep, p_rep), (rem, p_rem) = cases
    enc = _encoder()
    batch = retell.collate([_tokens(rep, p_rep)[:1], _tokens(rem, p_rep)[:1]])  # same box features, different decision
    batch.region[1] = batch.region[0]; batch.scalars[1] = batch.scalars[0]; batch.alt[1] = batch.alt[0]
    out, _ = enc(batch)
    assert torch.allclose(out[0, 1], out[1, 1])                          # zero-init: no effect before training
    # a random readout, as a downstream layer would apply (sum() of a LayerNorm output has zero gradient)
    loss = (out[:, 1] * torch.randn(out.shape[-1], generator=torch.Generator().manual_seed(1))).sum()
    loss.backward()
    assert enc.action_emb.weight.grad[retell.ACTION_IDS["replace"]].abs().sum() > 0
    assert enc.action_emb.weight.grad[retell.ACTION_IDS["remove"]].abs().sum() > 0
    assert enc.action_emb.weight.grad[0].abs().sum() == 0                  # "no action" stays exactly zero
    assert enc.sub_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        enc.action_emb.weight[1:].normal_(); enc.sub_proj.weight.normal_()
    out, _ = enc(batch)
    assert not torch.allclose(out[0, 1], out[1, 1])                       # the encoder can now tell them apart


def test_records_without_an_action_encode_exactly_as_before(cases):
    """Historical records and B2 tags carry no action: a trained action embedding must not touch them."""
    (rep, p_rep), _ = cases
    enc = _encoder()
    with torch.no_grad():
        enc.action_emb.weight[1:].normal_(); enc.sub_proj.weight.normal_()
    tok = _tokens(rep, p_rep)[0]
    plain = tok._replace(action=0, substitute=-1)
    old = _encoder()                                                      # same seed: same other weights
    a, _ = enc(retell.collate([[plain]]))
    b, _ = old(retell.collate([[plain]]))
    assert torch.allclose(a, b)
    tag = evidence.tag_feature("dog", .9)
    assert tag.action == 0 and tag.substitute == -1


def test_old_checkpoint_loads_with_new_inputs_at_zero(cases):
    (rep, p_rep), _ = cases

    class Pathway(nn.Module):                                             # the encoder part of RetellPathway
        def __init__(self):
            super().__init__()
            self.encoder = _encoder()
    new = Pathway()
    old_state = {k: v for k, v in new.state_dict().items() if k not in retell.NEW_ENCODER_KEYS}
    with torch.no_grad():
        new.encoder.action_emb.weight.normal_(); new.encoder.sub_proj.weight.normal_()
    retell.load_pathway_state(new, old_state)
    assert new.encoder.action_emb.weight.abs().sum() == 0 and new.encoder.sub_proj.weight.abs().sum() == 0
    tok = _tokens(rep, p_rep)[0]
    a, _ = new.encoder(retell.collate([[tok]]))
    b, _ = new.encoder(retell.collate([[tok._replace(action=0, substitute=-1)]]))
    assert torch.allclose(a, b)                                           # behaves as the pre-action encoder did
    with pytest.raises(RuntimeError):
        retell.load_pathway_state(new, {k: v for k, v in old_state.items() if k != "encoder.null"})


def test_false_evidence_also_swaps_the_substitute(cases):
    (rep, p_rep), _ = cases
    saved = copy.deepcopy(rep)
    fake = evidence.false_evidence([rep], ["frisbee"], np.random.default_rng(0))[0]
    assert rep == saved
    assert fake["substitute"] == fake["alt"] == "frisbee" and fake["action"] == "replace"
    assert {t.substitute for t in _tokens(fake, p_rep)} == {CLASS_INDEX["frisbee"]}


def test_collate_still_accepts_plain_five_field_tokens():
    """tests/test_retell.py (needs BLIP + Flickr8k) builds 5-tuples; they mean 'no action, no substitute'."""
    old = (CLASS_INDEX["ball"], CLASS_INDEX["frisbee"], np.zeros(512, np.float32), np.zeros(9, np.float32),
           retell.TYPE_IDS["contradicted"])
    ev = retell.collate([[old], []])
    assert ev.action.tolist() == [[0], [0]] and ev.substitute.tolist() == [[-1], [-1]]
    assert ev.valid.tolist() == [[True], [False]] and ev.claim[0, 0] == CLASS_INDEX["ball"]
