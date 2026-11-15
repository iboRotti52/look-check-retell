"""S2 (current default five-way z rule) out-of-fold contradiction flags on true and swapped labels, via the project's own
scripts/43_check_fix.py functions (read-only import; nothing is written to the project)."""
import importlib
from common import *
s43 = importlib.import_module("43_check_fix")
rows, maps, owl, q, hist, IM, xs, pooled, top1 = s43.build()
cx = Ctx()
assert len(xs) == len(cx.xs) and all(a["id"] == b["id"] and a["cls"] == b["cls"] for a, b in zip(xs, cx.xs))
refs = cx.refs; sw = s43.swap_labels(xs, refs); assert (sw == cx.sw).all()
m = sw >= 0; xs_sw = [dict(x, cls=CLASSES[s]) if s >= 0 else x for x, s in zip(xs, sw)]
folds = np.array([x["fold"] for x in xs]); img_folds = cx.img_fold; nv = cx.nv
a2 = np.empty(len(xs), object); a2s = np.full(len(xs), "", object); sub2s = np.full(len(xs), "", object)
for f in range(5):
    tr, te = folds != f, folds == f
    v, a, s = s43.s2(IM, xs, xs, pooled, rows, img_folds != f, tr, te); a[te & nv] = "skip"; a2[te] = a[te]
    _, a, s = s43.s2(IM, xs, xs_sw, pooled, rows, img_folds != f, tr, te & m); a2s[te] = a[te]; sub2s[te] = s[te]
np.savez(config.CACHE / "review_fable2_B_s2.npz", a_true=a2.astype(str), a_sw=a2s.astype(str), sub_sw=sub2s.astype(str))
con = np.isin(a2, ["replace", "remove"]); cs = np.isin(a2s, ["replace", "remove"])
report("S2 five-way z (OOF)", con, cs, cx)
