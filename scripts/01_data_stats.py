"""Dataset and vocabulary statistics (no model involved).

Outputs results/data_stats.json, report/generated/data_macros.tex and the
claim-frequency figure. Uses only the training and validation splits.
"""
from collections import Counter

import numpy as np

from _common import C, COL_W, macros, plt, savefig
from lcr import cache, claims, config, corrupt, data
from lcr.vocab import CLASSES, KIND, SURFACE_TO_CLASS

stats = {}
for split in ("train", "validation", "test"):
    stats[f"n_{split}"] = data.num_images(split)

train_ids = data.ids("train")
lengths, per_caption_claims, per_image_classes = [], [], []
freq = Counter()
noun_tokens = covered_tokens = 0
import nltk
for iid in train_ids:
    refs = data.references(iid)
    union = set()
    for r in refs:
        ws = claims.words(r)
        lengths.append(len(ws))
        cl = claims.extract_from_words(ws)
        per_caption_claims.append(len(cl))
        for c in cl:
            freq[c.cls] += 1
            union.add(c.cls)
        for w, t in nltk.pos_tag(ws):
            if t in ("NN", "NNS"):
                noun_tokens += 1
                covered_tokens += claims.word_class(w) is not None
    per_image_classes.append(len(union))

kind_share = Counter()
for c, n in freq.items():
    kind_share[KIND[c]] += n
total_claims = sum(freq.values())
A, B = data.ids("A"), data.ids("B")
table = corrupt.build_confusions([data.references(i) for i in train_ids])

stats.update(
    n_A=len(A), n_B=len(B), overlap_AB=len(set(A) & set(B)),
    mean_caption_len=float(np.mean(lengths)),
    n_classes=len(CLASSES), n_surface=len(SURFACE_TO_CLASS),
    n_person=sum(k == "person" for k in KIND.values()),
    n_thing=sum(k == "thing" for k in KIND.values()),
    n_stuff=sum(k == "stuff" for k in KIND.values()),
    claims_per_caption=float(np.mean(per_caption_claims)),
    classes_per_image_union=float(np.mean(per_image_classes)),
    noun_token_coverage=100 * covered_tokens / noun_tokens,
    share_person=100 * kind_share["person"] / total_claims,
    share_thing=100 * kind_share["thing"] / total_claims,
    share_stuff=100 * kind_share["stuff"] / total_claims,
    classes_never_seen=[c for c in CLASSES if freq[c] == 0],
    top_classes=freq.most_common(25),
    example_confusions={k: table[k][:4] for k in ("ball", "frisbee", "bicycle", "snow", "beach", "guitar") if k in table},
)
cache.save_json(stats, config.RESULTS / "data_stats.json")
cache.save_json(table, config.RESULTS / "confusion_table.json")

macros("data_macros", dict(
    NumClasses=stats["n_classes"], NumSurface=stats["n_surface"],
    NumThing=stats["n_thing"], NumStuff=stats["n_stuff"],
    MeanCapLen=f"{stats['mean_caption_len']:.1f}",
    ClaimsPerCaption=f"{stats['claims_per_caption']:.2f}",
    ClassesPerImage=f"{stats['classes_per_image_union']:.2f}",
    NounCoverage=f"{stats['noun_token_coverage']:.1f}",
    SharePerson=f"{stats['share_person']:.1f}", ShareThing=f"{stats['share_thing']:.1f}",
    ShareStuff=f"{stats['share_stuff']:.1f}",
))

top = freq.most_common(25)
fig, ax = plt.subplots(figsize=(COL_W, 2.1))
col = {"person": C["grey"], "thing": C["blue"], "stuff": C["orange"]}
ax.bar(range(len(top)), [n for _, n in top], color=[col[KIND[c]] for c, _ in top])
ax.set_xticks(range(len(top)), [c for c, _ in top], rotation=60, ha="right")
ax.set_yscale("log"); ax.set_ylabel("claims in training captions")
ax.grid(axis="x", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in col.values()], labels=list(col), frameon=False, ncol=3)
savefig(fig, "claim_frequency")
print({k: v for k, v in stats.items() if not isinstance(v, (list, dict))})
print("never seen:", stats["classes_never_seen"]); print(stats["example_confusions"])
