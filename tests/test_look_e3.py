import numpy as np
import torch

from lcr.look_e3 import E3Decoder, chefer_normalise, unit
from lcr.look_e3_eval import component_entropy, rank_maps, unit_maps


def test_chefer_normalise_rows_sum_and_identity():
    r = torch.eye(4) + torch.tensor([[0, 0, 0, 0], [2., 0, 0, 0], [1., 1., 0, 0], [0, 0, 3., 0]])
    out = chefer_normalise(r)
    off = out - torch.eye(4)
    assert torch.allclose(off[1:].sum(-1), torch.ones(3))
    assert torch.allclose(off[0], torch.zeros(4))      # empty row stays empty


def test_chefer_single_layer_matches_hand_rule():
    # one layer, T=2 text tokens, 1 CLS + 2 image tokens; self attention gradients zero
    cross = torch.tensor([[[[0., .25, .75], [0., .5, .5]]]])     # L=1,H=1,T=2,3
    gc = torch.ones_like(cross)
    selfa = torch.zeros(1, 1, 2, 2); gs = torch.zeros_like(selfa)
    rel = E3Decoder.chefer(cross, selfa, gc, gs, t=1, T=2)
    assert torch.allclose(rel, torch.tensor([.5, .5]))           # R_tt = I -> relevance = A_c row


def test_unit_and_rank_maps():
    a = unit(torch.tensor([-1.] + [0.] * 575))
    assert np.isclose(a.sum(), 1) and np.allclose(a, 1 / 576)    # empty -> uniform
    r = rank_maps(np.arange(576, dtype=np.float32)[None])
    assert r[0, 0] == 0 and r[0, -1] == 1
    u = unit_maps(np.array([[1., 3.]]))
    assert np.allclose(u, [[.25, .75]])


def test_component_entropy_one_blob_vs_two():
    one = np.zeros((24, 24)); one[5:9, 5:9] = 1
    two = one.copy(); two[15:19, 15:19] = 1
    assert component_entropy(one) == 0
    assert np.isclose(component_entropy(two), np.log(2))


def test_v4_selected_heads_match_signed_cache_precision():
    from lcr import look_v4
    rng = np.random.default_rng(415)
    a = torch.from_numpy(rng.random((12, 12, 4, 577), dtype=np.float32))
    heads = [102, 114, 115, 97]
    signed = (a[:, :, 1, 1:] - a[:, :, :3, 1:].mean(2)).numpy().astype(np.float16)
    assert np.array_equal(look_v4.attention_map(a, 2, 3, heads), look_v4.selected_map(signed, heads))


def test_v4_sam_updates_geometry_and_keeps_score_provenance():
    from lcr import pipeline, look, look_v4
    from lcr.vocab import CLASSES, CLASS_INDEX
    boxes = np.array([[0, 0, .5, .5], [.5, .5, 1, 1]], np.float32)
    prop = dict(boxes=boxes, objectness=np.ones(2))
    scores = np.zeros((2, len(CLASSES)), np.float32)
    scores[:, CLASS_INDEX['person']] = [.9, .1]
    att = np.zeros((24, 24), np.float32); att[:12, :12] = 1/144
    claim = dict(cls='person', kind='person', blip_alts=[], rec_att=dict(look='v3'), rec_det=dict(region=0))
    sam = np.array([[0, 0, .6, .6], [0, 0, .1, .1], [.5, .5, 1, 1]], np.float32)
    out = look_v4.rewrite_claim(claim, att, prop, scores, 'B2_a0.25', sam)
    rec = out['rec_att']
    assert np.allclose(rec['box'], sam[0])
    assert rec['score_box'] == boxes[0].tolist()
    assert rec['score_source'] == 'owl_nearest_proposal_proxy'
    assert rec['s_claim'] == float(scores[0, CLASS_INDEX['person']])
    assert out['excess_by_k']['4'] == rec['excess']
    assert np.isclose(rec['excess'], look.excess_mass(att, sam[:1])[0])
    corrupted = pipeline.relabel(rec, 'dog', prop, scores, [])
    assert corrupted['box'] == rec['box']
    assert corrupted['score_box'] == rec['score_box']
    assert corrupted['s_claim'] == 0.


def test_v4_sam_empty_mask_falls_back_without_inventing_a_box():
    from lcr import look_v4
    from lcr.vocab import CLASSES
    prop = dict(boxes=np.array([[0, 0, 1, 1]], np.float32), objectness=np.ones(1))
    c = dict(cls='person', kind='person', blip_alts=[], rec_att={}, rec_det=dict(region=0))
    out = look_v4.rewrite_claim(c, np.ones((24, 24), np.float32)/576, prop,
                                np.zeros((1, len(CLASSES))), 'B2_a0.25', np.zeros((3, 4)))
    assert out['rec_att']['box'] == [0., 0., 1., 1.]
    assert 'sam_fallback' in out['rec_att']


def test_v4_rejects_raw_map_and_only_advertises_available_heads():
    import pytest
    from lcr import pipeline
    with pytest.raises(ValueError, match='sink'):
        pipeline.output_stem('A', 'B', False, 'v4')
    meta = pipeline.look_metadata(True, 'v4')
    assert meta['available_last_k'] == [4]
    assert meta['head_pooling'] == 'selected_heads'


def test_rank_maps_preserves_ties_in_zero_background():
    ranks = rank_maps(np.array([[0., 0., 1., 1.]], np.float32))
    assert ranks[0,0] == ranks[0,1]
    assert ranks[0,2] == ranks[0,3]
    assert ranks[0,0] < ranks[0,2]
