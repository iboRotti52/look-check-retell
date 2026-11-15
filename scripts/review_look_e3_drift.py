"""Bounded numerical probe on the first 16 B drafts; no selection/validation/test.

Distinguishes attention/mean quantisation from full half-precision decoder
arithmetic and the SEP convention. It does not identify historical GPU kernels.
"""
import numpy as np
import torch

import _common  # noqa: F401
from lcr import cache, config, data, look
from lcr.look_sources import LookDecoder


def main():
    rows=cache.load_json(config.CACHE/'pipeline/A_B_look2.json')[:16]
    archived=np.load(config.CACHE/'pipeline/A_B_look2.maps.npz')['maps']
    feats=np.load(config.CACHE/'blip/train.f16.npy',mmap_mode='r');dec=LookDecoder()
    half_dec=LookDecoder();half_dec.model.half()
    errors={n:[] for n in ('exclude_sep','include_sep','half_attention','half_means','half_subtract','half_model')}
    k=0
    for row in rows:
        ids=dec.tokens(row);image=torch.tensor(np.array(feats[data.parse_id(row['id'])[1]]),dtype=torch.float32)[None]
        with torch.no_grad(),dec.capture() as att:dec.forward(ids,image)
        a=torch.stack([v[0] for v in att]);t=ids.shape[1]-1
        with torch.no_grad(),half_dec.capture() as att:half_dec.forward(ids,image.half())
        ah=torch.stack([v[0] for v in att])
        for c in row['claims']:
            orig=archived[k].astype(np.float32);orig/=orig.sum();k+=1;p=c['pos']
            raw=a[-4:,:,p-1,1:].mean((0,1));mean=a[-4:,:,:t,1:].mean((0,1,2))
            maps=dict(exclude_sep=look.attention_map(a,p,sink=True,valid_t=t),
                      include_sep=look.attention_map(a,p,sink=True,valid_t=t+1),
                      half_attention=look.attention_map(a.half(),p,sink=True,valid_t=t),
                      half_model=look.attention_map(ah,p,sink=True,valid_t=t),
                      half_means=(raw.half()-mean.half()).float().clamp_min(0).numpy().reshape(24,24),
                      half_subtract=(raw-mean).half().float().clamp_min(0).numpy().reshape(24,24))
            for name,m in maps.items():
                m=m/m.sum() if m.sum()>0 else np.ones_like(m)/m.size
                errors[name].append(float(np.abs(m-orig).max()))
    out=dict(protocol='first 16 B drafts; CPU; maximum patch error per claim after normalisation',n_claims=k,
             variants={n:dict(mean_max_error=float(np.mean(x)),max_error=max(x)) for n,x in errors.items()})
    cache.save_json(out,config.RESULTS/'look_e3_drift_probe.json');print(out)


if __name__=='__main__':main()
