"""Contact sheets for a visual look (Claude's observation, NOT a human label) at swap misses and false contradictions."""
from common import *
from PIL import Image, ImageDraw
SP = ROOT + "/tmp"
cx = Ctx(); L = cx.logits(); IM = L.max(1); names = np.array(CLASSES); m = cx.m
same = np.array([re.CONFUSION_OF.get(names[a]) is not None and re.CONFUSION_OF.get(names[a]) == re.CONFUSION_OF.get(names[b]) for a, b in zip(cx.k, cx.kk)])
ct = np.zeros(len(cx.n), bool); cs = np.zeros(len(cx.n), bool); ct5 = ct.copy()
for f in range(5):
    te = cx.fold == f; tr = cx.img_fold != f
    tau = pa_tau(IM[tr], cx.men[tr], 0.15); cs[te] = cx.has[te] & (IM[cx.ri[te], cx.kk[te]] < tau[cx.kk[te]])
    tau = pa_tau(IM[tr], cx.men[tr], 0.05); ct5[te] = cx.has[te] & (IM[cx.ri[te], cx.k[te]] < tau[cx.k[te]])
pos = cx.ok & ~cx.nv & (cx.kind == "thing") & cx.has
rng = np.random.default_rng(7)
def sheet(idx, fname, label):
    tiles = []
    for i in idx:
        x = cx.xs[i]; im = data.load_image(x["id"]); W, H = im.size
        d = ImageDraw.Draw(im); b = cx.BX[x["ri"], cx.top[i, 0]].astype(float) * [W, H, W, H]; d.rectangle(list(b), outline="red", width=3)
        im = im.resize((400, int(400 * H / W))); canvas = Image.new("RGB", (400, 360), "white"); canvas.paste(im.crop((0, 0, 400, min(im.size[1], 330))), (0, 0))
        ImageDraw.Draw(canvas).text((4, 334), label(i), fill="black"); tiles.append(canvas)
    S = Image.new("RGB", (1600, 360 * ((len(tiles) + 3) // 4)), "white")
    for t, c in enumerate(tiles): S.paste(c, (400 * (t % 4), 360 * (t // 4)))
    S.save(f"{SP}/{fname}.jpg", quality=80)
    for i in idx: print(fname, cx.xs[i]["id"], label(i), "| draft:", cx.xs[i]["draft"])
lab_sw = lambda i: f"{cx.xs[i]['id']} true={names[cx.k[i]]} SWAP={names[cx.kk[i]]}"
sheet(rng.permutation(np.where(m & ~cs & ~same)[0])[:16], "A_cross_missed", lab_sw)
sheet(rng.permutation(np.where(m & ~cs & same & cx.clo)[0])[:16], "B_cloth_missed", lab_sw)
sheet(rng.permutation(np.where(pos & ct5 & ~cx.clo)[0])[:16], "C_false_contra", lambda i: f"{cx.xs[i]['id']} claim={names[cx.k[i]]} (contradicted)")
sheet(rng.permutation(np.where(m & ~cs & same & ~cx.clo)[0])[:16], "D_sameset_missed", lab_sw)
