from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lcr import config, data, grounding, look_eval
from lcr.look_sources import LookDecoder, install_visual_mask_routing


def test_smoothing_preserves_probability_mass_without_moving_a_symmetric_peak():
    a = np.zeros((24, 24), np.float32); a[10, 10] = 1
    assert np.array_equal(look_eval.transformed(a, 0), a)
    smoothed = look_eval.transformed(a, 1)
    assert np.isclose(smoothed.sum(), 1)
    assert np.unravel_index(smoothed.argmax(), smoothed.shape) == (10, 10)
    assert smoothed[10, 10] < 1 and smoothed[10, 11] > 0


def test_feature_intervention_matches_random_count_and_retains_cls():
    decoder = LookDecoder.__new__(LookDecoder)
    observed = {}
    def forward(ids, image, image_mask):
        observed["mask"] = image_mask.clone()
        observed["prefix"] = ids.clone()
        return SimpleNamespace(logits=torch.zeros(4, ids.shape[1], 10))
    decoder.forward = forward
    ids = torch.tensor([[1, 2, 3, 4]])
    claim = dict(pos=2, rec_att=dict(box=[0, 0, .5, .5]), rec_det=dict(box=[0, 0, 1, 1]))
    r = decoder.intervention(ids, claim, np.zeros((577, 24), np.float32), seed=415)
    mask = observed["mask"]
    assert torch.all(mask[:, 0] == 1)
    assert int((mask[1] == 0).sum()) == int((mask[2] == 0).sum()) == 144
    assert int((mask[3] == 0).sum()) == 576
    assert torch.equal(observed["prefix"], ids[:, :2].expand(4, -1))
    assert r["drop_look"] == r["drop_random"] == 0


def test_photo_id_reader_keeps_parquet_order_and_does_not_decode_images(tmp_path, monkeypatch):
    folder = tmp_path / "flickr8k" / "data"; folder.mkdir(parents=True)
    for part, path in ((1, "456_b.jpg"), (0, "123_a.jpg")):
        pq.write_table(pa.table({"image": [{"bytes": b"invalid image", "path": path}]}),
                       folder / f"train-{part}.parquet")
    monkeypatch.setattr(config, "DATA", tmp_path)
    def forbidden_frame(*args):
        raise AssertionError("Full image frame should not be loaded")
    monkeypatch.setattr(data, "_frame", forbidden_frame)
    grounding._photo_ids.cache_clear()
    try:
        assert grounding._photo_ids("train") == ("123", "456")
    finally:
        grounding._photo_ids.cache_clear()


def test_visual_mask_reaches_actual_blip_attention_probabilities():
    from transformers import BlipTextConfig
    from transformers.models.blip.modeling_blip_text import BlipTextLMHeadModel
    torch.manual_seed(415)
    conf = BlipTextConfig(vocab_size=32, hidden_size=24, encoder_hidden_size=24,
                         intermediate_size=48, num_hidden_layers=1, num_attention_heads=4,
                         is_decoder=True, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    model = BlipTextLMHeadModel(conf).eval()
    hooks = install_visual_mask_routing(model)
    captured = []
    hook = model.bert.encoder.layer[0].crossattention.self.register_forward_hook(
        lambda _m, _i, out: captured.append(out[1].detach()))
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[1, 0, 1, 0]])
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids), encoder_hidden_states=torch.randn(1, 4, 24),
              encoder_attention_mask=mask, use_cache=False)
    assert torch.all(captured[0][..., 1::2] == 0)
    assert torch.allclose(captured[0].sum(-1), torch.ones_like(captured[0].sum(-1)))
    hook.remove()
    for h in hooks:
        h.remove()
