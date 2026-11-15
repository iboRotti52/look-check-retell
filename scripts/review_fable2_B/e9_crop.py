"""CPU pilot of crop re-detection with the SAME frozen OWLv2 (lcr.check.Detector): the Look top-1 box (+10% pad) is
re-encoded as an image. Small fixed sample of same-confusion-set swaps, label-blind crop geometry. B only."""
import os, sys, time, pickle
os.environ["LCR_DEVICE"] = "cpu"
from common import *
import torch; torch.set_num_threads(2)
from lcr import check
OUT = config.CACHE / "review_fable2_B_crop.pkl"
cx = Ctx(); names = np.array(CLASSES); m = cx.m
same = np.array([re.CONFUSION_OF.get(names[a]) is not None and re.CONFUSION_OF.get(names[a]) == re.CONFUSION_OF.get(names[b]) for a, b in zip(cx.k, cx.kk)])
rng = np.random.default_rng(1)
N_CLO, N_OTH = int(sys.argv[1]), int(sys.argv[2])
a = np.where(m & same & cx.clo & cx.has)[0]; b = np.where(m & same & ~cx.clo & cx.has)[0]
sel = np.r_[rng.permutation(a)[:N_CLO], rng.permutation(b)[:N_OTH]]
if len(sys.argv) > 3 and sys.argv[3] == "all":      # every swap-set claim (reference-confirmed thing claims), random order
    sel = np.r_[sel, rng.permutation(np.where(m & cx.has)[0])]
done = pickle.load(open(OUT, "rb")) if OUT.exists() else {}
det = check.Detector(torch.device("cpu")); q = cx.q
t0 = time.time()
for cnt_, i in enumerate(sel):
    i = int(i)
    if i in done: continue
    x = cx.xs[i]; im = data.load_image(x["id"]).convert("RGB"); W, H = im.size
    x0, y0, x1, y1 = cx.BX[x["ri"], cx.top[i, 0]].astype(np.float32) * np.array([W, H, W, H])
    pw, ph = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
    box = (max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph))
    crop = im.crop(tuple(int(round(v)) for v in box))
    if min(crop.size) < 8: continue
    prop = det.proposals([crop])[0]; lg = re.proposal_logits(prop, q)
    done[i] = dict(max=lg.max(0).astype(np.float32), crop=box, size=(W, H), nprop=len(lg),
                   # logits at the crop proposal that covers most of the crop (object-level view of the attended box)
                   big=lg[int(np.argmax((prop["boxes"][:, 2] - prop["boxes"][:, 0]).astype(np.float32) * (prop["boxes"][:, 3] - prop["boxes"][:, 1]).astype(np.float32)))].astype(np.float32))
    if (cnt_ + 1) % 10 == 0:
        pickle.dump(done, open(OUT, "wb")); print(cnt_ + 1, len(sel), f"{time.time() - t0:.0f}s", flush=True)
pickle.dump(done, open(OUT, "wb")); print("done", len(done))
