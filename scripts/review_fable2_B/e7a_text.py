"""Extra OWLv2 text queries (frozen text tower, CPU): clothing 'worn' prompts and sense-specific names."""
from common import *
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor
torch.set_num_threads(2)
proc = Owlv2Processor.from_pretrained(config.OWL_NAME); model = Owlv2ForObjectDetection.from_pretrained(config.OWL_NAME).eval()
@torch.no_grad()
def embed(prompts):
    out = []
    for s in range(0, len(prompts), 32):
        enc = proc(text=prompts[s:s + 32], return_tensors="pt", padding="max_length")
        o = model.owlv2.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        q = model.owlv2.text_projection(o.pooler_output); out.append((q / (q.norm(dim=-1, keepdim=True) + 1e-6)).numpy())
    return np.concatenate(out)
base = np.load(config.CACHE / "owl" / "class_queries_single.npy").astype(np.float32)
chk = embed([f"a photo of a {c}" for c in CLASSES[:5]]); print("reproduces cached single queries:", float(np.abs(chk - base[:5]).max()))
MASS = {c for c in CLASSES if KIND[c] == "stuff"} | {"pants", "shorts", "glasses", "goggles", "stairs"}
art = lambda n: n if n in MASS else ("an " if n[0] in "aeiou" else "a ") + n
SENSE = dict(bat="baseball bat", slide="playground slide", swing="playground swing", glasses="eyeglasses", pool="swimming pool",
             net="sports net", tube="inflatable tube", cone="traffic cone", bone="dog bone", sign="signboard", stick="wooden stick",
             ramp="skateboard ramp", suit="business suit", uniform="sports uniform", shirt="t-shirt", ski="skis", log="tree log",
             hose="garden hose", mask="face mask", collar="dog collar", harness="dog harness", muzzle="dog muzzle", glove="gloves", boot="boots", shoe="shoes")
out = {}
out["sense"] = embed([f"a photo of {art(SENSE.get(c, c))}" for c in CLASSES])
clo = [c for c in CLASSES if re.GROUP[c] == "clothing"]
worn = base.copy(); e = embed([f"a person wearing {art(c)}" for c in clo])
for c, v in zip(clo, e): worn[CLASS_INDEX[c]] = v
out["worn"] = worn
temps = [lambda n: f"a photo of {art(n)}", lambda n: f"{art(n)}", lambda n: f"a close-up photo of {art(n)}", lambda n: f"a cropped photo of {art(n)}",
         lambda n: f"a person wearing {art(n)}" if n in clo else f"there is {art(n)} in the scene"]
E = np.mean([embed([t(SENSE.get(c, c)) for c in CLASSES]) for t in temps], 0); out["sense_ens"] = E / np.linalg.norm(E, axis=1, keepdims=True)
np.savez(config.CACHE / "review_fable2_B_queries.npz", **out); print({k: v.shape for k, v in out.items()})
