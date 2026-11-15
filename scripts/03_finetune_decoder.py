"""Fine-tune the BLIP text decoder on one half of the Flickr8k training images.

    python scripts/03_finetune_decoder.py --half A --epochs 4

The vision encoder stays frozen (its patch embeddings are read from the cache),
so this runs on a laptop GPU. The checkpoint with the best validation CIDEr is
kept as data/cache/ckpt/decoder_<half>.pt and the log goes to
results/finetune_<half>.json.
"""
import argparse
import math
import time

import numpy as np
import torch

import _common  # noqa: F401
from lcr import cache, config, data, metrics
from lcr.captioner import Captioner


def generate_split(cap: Captioner, split_ids: list[str], bs: int = 32, pathway_hook=None) -> dict[str, str]:
    out = {}
    mm = {}
    for s in range(0, len(split_ids), bs):
        chunk = split_ids[s:s + bs]
        rows = []
        for iid in chunk:
            sp, idx = data.parse_id(iid)
            if sp not in mm:
                mm[sp] = cache.open_blip(sp)
            rows.append(np.asarray(mm[sp][idx]))
        emb = torch.from_numpy(np.stack(rows)).float().to(cap.dev)
        if pathway_hook is not None:
            pathway_hook(chunk)
        for iid, text in zip(chunk, cap.decode(cap.generate(emb))):
            out[iid] = text
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="A", choices=["A", "B"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N images")
    a = ap.parse_args()

    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    cap = Captioner()
    dec = cap.model.text_decoder
    for p in dec.parameters():
        p.requires_grad_(True)
    train_ids = data.ids(a.half)[: a.limit or None]
    val_ids = data.ids("validation")[: a.limit or None]
    val_refs = {i: data.references(i) for i in val_ids}
    pairs = [(i, k) for i in train_ids for k in range(5)]
    refs = {i: data.references(i) for i in train_ids}
    mm = cache.open_blip("train")

    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_total = a.epochs * math.ceil(len(pairs) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps_total, pct_start=0.05)

    log = dict(args=vars(a), n_train_images=len(train_ids), n_pairs=len(pairs), epochs=[], loss_curve=[])
    cap.model.eval()
    zs = generate_split(cap, val_ids)
    log["zero_shot"] = {**metrics.caption_scores(zs, val_refs), **metrics.ruor(zs, val_refs)}
    print("zero-shot", log["zero_shot"], flush=True)
    best, step, t0 = -1.0, 0, time.time()
    ckpt = config.CACHE / "ckpt" / f"decoder_{a.half}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(a.epochs):
        dec.train()
        order = rng.permutation(len(pairs))
        run = []
        for s in range(0, len(order), a.bs):
            batch = [pairs[j] for j in order[s:s + a.bs]]
            emb = torch.from_numpy(np.stack([np.asarray(mm[data.parse_id(i)[1]]) for i, _ in batch])).float().to(cap.dev)
            ids, mask = cap.encode_captions([refs[i][k] for i, k in batch])
            loss = cap.decoder_forward(ids.to(cap.dev), mask.to(cap.dev), emb, labels=True).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run.append(loss.item())
            if step % 25 == 0:
                log["loss_curve"].append([step, float(np.mean(run[-25:]))])
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{steps_total} loss {np.mean(run[-100:]):.3f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        dec.eval()
        gens = generate_split(cap, val_ids)
        sc = {**metrics.caption_scores(gens, val_refs), **metrics.ruor(gens, val_refs)}
        log["epochs"].append(dict(epoch=ep + 1, train_loss=float(np.mean(run)), **sc))
        print("epoch", ep + 1, log["epochs"][-1], flush=True)
        if sc["CIDEr"] > best:
            best = sc["CIDEr"]
            torch.save(dec.state_dict(), ckpt)
            log["best_epoch"] = ep + 1
        log["minutes"] = (time.time() - t0) / 60
        log["device"] = str(cap.dev)
        cache.save_json(log, config.RESULTS / f"finetune_{a.half}.json")


if __name__ == "__main__":
    main()
