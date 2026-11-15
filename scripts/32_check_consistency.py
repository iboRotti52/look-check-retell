"""Check v3 'cons' signals (docs/CHECK_V3_PROTOKOL.md §1): does the captioner keep saying the object?

    python scripts/32_check_consistency.py --stem A_B_look4
SelfCheckGPT (Manakul et al., EMNLP'23) adapted to object claims: from the cached ViT features and the frozen
decoder_A, 10 nucleus samples (p=0.9, T=1.0, fixed seed) and a 5-best beam list per image. For every class, the
fraction of samples / beams whose extracted claims contain it. Writes data/cache/pipeline/<stem>_cons.json.
"""
import argparse
import time

import numpy as np
import torch

import _common  # noqa: F401
from lcr import cache, claims, config, data
from lcr.look_sources import LookDecoder

P = config.CACHE / "pipeline"
N_SAMPLES, N_BEAMS = 10, 5


def classes_of(texts):
    return [{c.cls for c in claims.extract(t)} for t in texts]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stem", required=True); ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    assert "test" not in a.stem
    torch.set_num_threads(a.threads); BS = a.bs; dev = config.device()
    rows = cache.load_json(P / f"{a.stem}.json")
    split = data.parse_id(rows[0]["id"])[0]
    feats = np.load(config.CACHE / "blip" / f"{split}.f16.npy", mmap_mode="r")
    dec = LookDecoder(); model, tok = dec.model.to(dev), dec.tok
    sep, pad = model.config.sep_token_id, model.config.pad_token_id
    out_path = P / (f"{a.stem}_cons.json" if not a.limit else f"{a.stem}_cons_limit.json")
    out = cache.load_json(out_path) if out_path.exists() else {}
    todo = [r for r in rows if r["id"] not in out][:a.limit or None]
    t0 = time.monotonic()
    for s in range(0, len(todo), BS):
        batch = todo[s:s + BS]
        img = torch.tensor(np.stack([feats[data.parse_id(r["id"])[1]] for r in batch]), dtype=torch.float32, device=dev)
        torch.manual_seed(config.SEED + s); torch.cuda.manual_seed_all(config.SEED + s)
        with torch.no_grad():
            start = torch.full((len(batch) * N_SAMPLES, 1), dec.bos, dtype=torch.long, device=dev)
            enc = img.repeat_interleave(N_SAMPLES, 0)
            samp = model.generate(input_ids=start, encoder_hidden_states=enc,
                                  encoder_attention_mask=torch.ones(enc.shape[:2], dtype=torch.long, device=dev),
                                  do_sample=True, top_p=0.9, temperature=1.0, max_new_tokens=30, min_new_tokens=5,
                                  eos_token_id=sep, pad_token_id=pad)
            start = torch.full((len(batch), 1), dec.bos, dtype=torch.long, device=dev)
            beams = model.generate(input_ids=start, encoder_hidden_states=img,
                                   encoder_attention_mask=torch.ones(img.shape[:2], dtype=torch.long, device=dev),
                                   num_beams=N_BEAMS, num_return_sequences=N_BEAMS, max_new_tokens=30, min_new_tokens=5,
                                   eos_token_id=sep, pad_token_id=pad)
        st = [t.strip() for t in tok.batch_decode(samp, skip_special_tokens=True)]
        bt = [t.strip() for t in tok.batch_decode(beams, skip_special_tokens=True)]
        sc, bc = classes_of(st), classes_of(bt)
        for i, r in enumerate(batch):
            ss, bb = sc[i * N_SAMPLES:(i + 1) * N_SAMPLES], bc[i * N_BEAMS:(i + 1) * N_BEAMS]
            allc = set().union(*ss, *bb) | {c["cls"] for c in r["claims"]}
            out[r["id"]] = dict(sample={c: sum(c in x for x in ss) / N_SAMPLES for c in allc},
                                beam={c: sum(c in x for x in bb) / N_BEAMS for c in allc},
                                examples=st[i * N_SAMPLES:i * N_SAMPLES + 2])
        if (s // BS) % 25 == 0:
            cache.save_json(out, out_path)
            print(f"{a.stem} {s+len(batch)}/{len(todo)} {(time.monotonic()-t0)/60:.1f} min", flush=True)
    cache.save_json(out, out_path)
    print("done", a.stem, len(out), flush=True)


if __name__ == "__main__":
    main()
