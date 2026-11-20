"""Train the gated Retell pathway (or one of its matched controls) and evaluate on validation.

    python scripts/06_train_retell.py --mode ours   --rho 0.15
    python scripts/06_train_retell.py --mode b0plus            # no-evidence control
    python scripts/06_train_retell.py --mode b3     --rho 0.15 # detector-selected regions
    python scripts/06_train_retell.py --mode b2                # image-level tags

All modes share the frozen captioner (decoder fine-tuned on --decoder), the
training images (the other half), the pathway architecture, the optimiser and
the number of updates. Only the evidence differs.
"""
import argparse
import math
import time

import numpy as np
import torch

import _common  # noqa: F401
from lcr import cache, check, claims, config, data, evidence, metrics, pipeline, retell
from lcr.captioner import Captioner
from lcr.vocab import CLASSES, KIND


class Bank:
    """Cached per-image inputs for one split."""

    def __init__(self, rows):
        self.rows = {r["id"]: r for r in rows}
        self.q = pipeline.class_queries()
        self.blip, self.owl = {}, {}

    def _open(self, sp):
        if sp not in self.blip:
            self.blip[sp], self.owl[sp] = cache.open_blip(sp), cache.OwlCache(sp)

    def embeds(self, ids, dev):
        out = []
        for iid in ids:
            sp, idx = data.parse_id(iid); self._open(sp)
            out.append(np.asarray(self.blip[sp][idx]))
        return torch.from_numpy(np.stack(out)).float().to(dev)

    def prop_scores(self, iid):
        sp, idx = data.parse_id(iid); self._open(sp)
        prop = self.owl[sp][idx]
        return prop, check.score(prop, self.q)


def evaluate(cap, pathway, bank, ids, mode, th, refs, condition="real", seed=0, bs=32):
    """Generate on `ids` under an evidence condition: real / masked / false."""
    rng = np.random.default_rng(seed)
    gens, used = {}, {}
    for s in range(0, len(ids), bs):
        chunk = ids[s:s + bs]
        feats = []
        for iid in chunk:
            prop, scores = bank.prop_scores(iid)
            row = bank.rows[iid]
            if condition == "masked" or mode == "b0plus":
                f, recs = [], []
            elif condition == "false" and mode in ("ours", "b3"):
                recs = evidence.records_for(row, mode, th)
                present = claims.reference_classes(refs[iid])
                best = scores.max(axis=0) if len(scores) else np.zeros(len(CLASSES))
                absent = [c for j, c in enumerate(CLASSES)
                          if c not in present and best[j] < 0.05 and KIND[c] == "thing"]
                recs = evidence.false_evidence(recs, absent, rng)
                f, _ = evidence.features(row, mode, th, prop, scores, override=recs)
            else:
                recs = evidence.records_for(row, mode, th) if mode in ("ours", "b3") else []
                f, _ = evidence.features(row, mode, th, prop, scores)
            feats.append(f); used[iid] = recs
        pathway.set_evidence(retell.collate(feats).to(cap.dev))
        for iid, text in zip(chunk, cap.decode(cap.generate(bank.embeds(chunk, cap.dev)))):
            gens[iid] = text
    return gens, used


def behaviour(gens, used, bank, refs):
    """Automatic (NOT human-verified) descriptions of what the outputs did to draft claims."""
    n_img = len(gens)
    changed = sum(gens[i] != bank.rows[i]["draft"] for i in gens)
    stat = dict(contra=0, contra_kept=0, contra_alt=0, supp=0, supp_kept=0,
                unsup_ref=0, unsup_ref_kept=0, sup_ref=0, sup_ref_kept=0)
    for iid, text in gens.items():
        out_cls = {c.cls for c in claims.extract(text)}
        recs = used.get(iid) or []
        for j, c in enumerate(bank.rows[iid]["claims"]):
            kept = c["cls"] in out_cls
            key = "sup_ref" if c["in_refs"] else "unsup_ref"
            stat[key] += 1; stat[key + "_kept"] += kept
            if j < len(recs):
                r = recs[j]
                if r["verdict"] == "contradicted":
                    stat["contra"] += 1; stat["contra_kept"] += kept
                    stat["contra_alt"] += r["alt"] in out_cls
                elif r["verdict"] == "supported":
                    stat["supp"] += 1; stat["supp_kept"] += kept
    pct = lambda a, b: 100 * a / b if b else float("nan")
    return dict(
        changed_captions=pct(changed, n_img),
        n_contradicted=stat["contra"],
        contradicted_claim_retained=pct(stat["contra_kept"], stat["contra"]),
        alternative_adopted=pct(stat["contra_alt"], stat["contra"]),
        supported_claim_retained=pct(stat["supp_kept"], stat["supp"]),
        ref_supported_retained=pct(stat["sup_ref_kept"], stat["sup_ref"]),
        ref_unsupported_retained=pct(stat["unsup_ref_kept"], stat["unsup_ref"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=evidence.MODES)
    ap.add_argument("--decoder", default="A")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--p-drop", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--attn", default="lcr", help="lcr (main) or an explicit historical pipeline suffix")
    ap.add_argument("--report-validation", action="store_true",
                    help="after fixed-epoch training only, explicitly report on prepared validation records")
    ap.add_argument("--img-drop", type=float, default=0.0,
                    help="training only: probability that an example sees only a random subset of image patches")
    ap.add_argument("--img-keep", type=float, default=0.1, help="fraction of patches kept when --img-drop fires")
    a = ap.parse_args()
    if a.attn == "lcr":
        ap.error("Unified LCR is currently Look/Check only. Retell is deferred until their review is closed.")
    if a.attn == "lcr" and a.decoder != "A":
        ap.error("The unified bundle is calibrated for decoder A")
    if a.epochs < 1 or a.bs < 1:
        ap.error("epochs and bs must be positive")
    other = "B" if a.decoder == "A" else "A"
    sfx = "" if a.attn == "raw" else f"_{a.attn}"
    name = a.mode + (f"_{a.tag}" if a.tag else "") + sfx

    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    th = pipeline.thresholds()
    cap = Captioner(decoder_weights=str(config.CACHE / "ckpt" / f"decoder_{a.decoder}.pt"))
    pathway = retell.RetellPathway(cap.model).to(cap.dev).install(cap.model.text_decoder)

    rows = cache.load_json(config.CACHE / "pipeline" / f"{a.decoder}_{other}{sfx}.json")
    if a.attn == "lcr":
        from lcr.region_evidence import RegionChecker
        checker = RegionChecker.load()
        if any(r.get("bundle_id") != checker.bundle["id"] for r in rows):
            raise ValueError("Training records do not match the frozen LCR bundle")
        train = Bank([r for r in rows if r["evidence_partition"] == "fit"])
        val = Bank([r for r in rows if r["evidence_partition"] == "cal"])
    else:
        # Historical features still use a B-only held-out calibration subset for epoch diagnostics.
        from lcr.region_evidence import partition
        train = Bank([r for r in rows if partition(r["id"]) == "fit"])
        val = Bank([r for r in rows if partition(r["id"]) == "cal"])
    train_ids = list(train.rows)[: a.limit or None]
    val_ids = list(val.rows)[: a.limit or None]
    refs = {i: data.references(i) for i in train_ids}
    val_refs = {i: data.references(i) for i in val_ids}
    present = {i: claims.reference_classes(refs[i]) for i in train_ids}
    table = cache.load_json(config.RESULTS / "confusion_table.json")
    pairs = [(i, k) for i in train_ids for k in range(5)]

    opt = torch.optim.AdamW(retell.optimiser_groups(pathway), lr=a.lr)
    total = a.epochs * math.ceil(len(pairs) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    log = dict(args=vars(a), thresholds=th, trainable_params=pathway.num_trainable(),
               selection="fixed final epoch; B_cal diagnostics; no validation epoch selection",
               bundle_id=checker.bundle["id"] if a.attn == "lcr" else None,
               n_train_images=len(train_ids), loss_curve=[], gate_curve=[], epochs=[],
               evidence_stats=dict(tokens=0, corrupted=0, verdicts={}))
    # sanity: before any update the system must reproduce the frozen captioner
    pathway.eval()
    g0, _ = evaluate(cap, pathway, val, val_ids[:16], a.mode, th, val_refs)
    log["init_identical_to_draft"] = all(g0[i] == val.rows[i]["draft"] for i in g0)
    print("init identical to drafts:", log["init_identical_to_draft"], flush=True)

    step, t0 = 0, time.time()
    ckpt = config.CACHE / "ckpt" / f"retell_{name}_{a.decoder}.pt"
    if ckpt.exists() or (config.RESULTS / f"retell_{name}.json").exists():
        raise FileExistsError("Retell output exists; use a distinct --tag to keep this experiment separate")
    for ep in range(a.epochs):
        pathway.train()
        order = rng.permutation(len(pairs)); run = []
        for s in range(0, len(order), a.bs):
            batch = [pairs[j] for j in order[s:s + a.bs]]
            feats = []
            for iid, _ in batch:
                prop, scores = train.prop_scores(iid)
                f, info = evidence.features(train.rows[iid], a.mode, th, prop, scores, rng=rng, rho=a.rho,
                                            p_drop=a.p_drop, table=table, present=present[iid])
                feats.append(f)
                es = log["evidence_stats"]
                es["tokens"] += info["n"]; es["corrupted"] += info["corrupted"]
                for v in info["verdicts"]:
                    es["verdicts"][v] = es["verdicts"].get(v, 0) + 1
            pathway.set_evidence(retell.collate(feats).to(cap.dev))
            ids, mask = cap.encode_captions([refs[i][k] for i, k in batch])
            emb = train.embeds([i for i, _ in batch], cap.dev)
            img_mask = None
            if a.img_drop > 0:
                # patch dropout: the decoder must recover the objects from the evidence when the image is degraded
                img_mask = torch.ones(emb.shape[:2], dtype=torch.long, device=cap.dev)
                hit = torch.rand(emb.shape[0], device=cap.dev) < a.img_drop
                drop = torch.rand(emb.shape[0], emb.shape[1] - 1, device=cap.dev) >= a.img_keep
                img_mask[:, 1:] = torch.where(hit[:, None] & drop, 0, 1)
            loss = cap.decoder_forward(ids.to(cap.dev), mask.to(cap.dev), emb, labels=True, img_mask=img_mask).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pathway.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run.append(loss.item())
            if step % 25 == 0:
                log["loss_curve"].append([step, float(np.mean(run[-25:]))])
                log["gate_curve"].append([step, pathway.gates()])
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{total} loss {np.mean(run[-100:]):.3f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        pathway.eval()
        gens, used = evaluate(cap, pathway, val, val_ids, a.mode, th, val_refs)
        sc = {**metrics.caption_scores(gens, val_refs), **metrics.ruor(gens, val_refs)}
        log["epochs"].append(dict(epoch=ep + 1, train_loss=float(np.mean(run)), **sc))
        print("epoch", ep + 1, log["epochs"][-1], flush=True)
        # Fixed schedule: validation is never an epoch selector.
        torch.save(pathway.state_dict(), ckpt)
        log["final_epoch"] = ep + 1
        log["minutes"] = (time.time() - t0) / 60

    # ---- final B_cal diagnostic, or explicit one-time validation reporting after training
    retell.load_pathway_state(pathway, torch.load(ckpt, map_location=cap.dev)); pathway.eval()
    if a.report_validation:
        val = Bank(cache.load_json(config.CACHE / "pipeline" / f"{a.decoder}_validation{sfx}.json"))
        if a.attn == "lcr" and any(r.get("bundle_id") != checker.bundle["id"] for r in val.rows.values()):
            raise ValueError("Validation records do not match the training bundle")
        val_ids = list(val.rows)[: a.limit or None]
        val_refs = {i: data.references(i) for i in val_ids}
    log["report_split"] = "validation" if a.report_validation else "B_cal (development, not independent)"
    log["gates"] = pathway.gates()
    draft = {i: val.rows[i]["draft"] for i in val_ids}
    log["draft"] = {**metrics.caption_scores(draft, val_refs), **metrics.ruor(draft, val_refs)}
    log["conditions"], outputs = {}, {}
    conds = ["real", "masked"] + (["false"] if a.mode in ("ours", "b3") else [])
    real_used = None
    for cond in conds:
        gens, used = evaluate(cap, pathway, val, val_ids, a.mode, th, val_refs, condition=cond)
        if cond == "real":
            real_used = used
        # behaviour is always traced against the REAL records so that conditions are comparable
        beh = behaviour(gens, real_used if cond == "masked" else used, val, val_refs)
        log["conditions"][cond] = {**metrics.caption_scores(gens, val_refs), **metrics.ruor(gens, val_refs), **beh}
        outputs[cond] = gens
        print(cond, log["conditions"][cond], flush=True)
    # evidence sensitivity: share of captions that change when only the evidence changes
    log["sensitivity"] = {c: 100 * float(np.mean([outputs[c][i] != outputs["real"][i] for i in val_ids]))
                          for c in conds if c != "real"}
    print("sensitivity", log["sensitivity"], flush=True)
    cache.save_json(log, config.RESULTS / f"retell_{name}.json")
    suffix = "validation" if a.report_validation else "B_cal"
    cache.save_json(dict(draft=draft, **outputs), config.CACHE / "pipeline" / f"retell_{name}_{suffix}_outputs.json")
    cache.save_json(dict(bundle_id=log["bundle_id"], args=vars(a), report_split=log["report_split"],
                         evidence_schema="per-region scores, pooled claim verdict, no image maximum",
                         final_epoch=a.epochs), ckpt.with_suffix(".json"))


if __name__ == "__main__":
    main()
