"""Audit frozen E3; never select or promote a method from validation.

Writes results/look_e3_review.json and a deterministic B-only map gallery.
Run 29 --suffix look4_review first. Optional --live checks new decoder/SAM
inference on fixed B drafts, independently of the E3 extraction caches.
"""
import argparse
import importlib

import numpy as np
from scipy.stats import spearmanr

import _common
from _common import plt
from lcr import cache, check, config, data, look, look_eval, look_v4, pipeline
from lcr.look_e3_eval import Bank, sanity, record_list

P = config.CACHE / 'pipeline'


def parity(stem):
    archive = P / 'archive/pre_e3_review' / f'{stem}.json'
    old = cache.load_json(archive if archive.exists() else P / f'{stem}.json')
    new = cache.load_json(P / f'{stem}_review.json')
    a = np.load(P / f'{stem}.maps.npz'); b = np.load(P / f'{stem}_review.maps.npz')
    assert np.array_equal(a['owner'], b['owner'])
    changed, stale, score_drift, box_drift = 0, 0, 0., 0.
    for r, s in zip(old, new, strict=True):
        assert r['id'] == s['id'] and r['draft'] == s['draft']
        for c, d in zip(r['claims'], s['claims'], strict=True):
            ar, br = c['rec_att'], d['rec_att']
            changed += ar['region'] != br['region']
            box_drift = max(box_drift, float(np.max(np.abs(np.array(ar['box'])-br['box']))) if ar['box'] else 0.)
            score_drift = max(score_drift, *(abs(ar[k]-br[k]) for k in ('s_claim','s_alt','s_image')))
            stale += c['excess_by_k']['4'] != ar['excess']
            assert d['excess_by_k']['4'] == br['excess']
    result = dict(claims=sum(len(r['claims']) for r in old), changed_regions=changed,
                  max_box_error=box_drift, max_score_error=score_drift,
                  max_map_error=float(np.abs(a['maps']-b['maps']).max()), old_stale_excess=stale)
    assert changed == 0 and box_drift == 0 and score_drift == 0 and result['max_map_error'] < 1e-7
    return result


def b_diagnostics():
    s26 = importlib.import_module('26_look_e3_select')
    bank = Bank('A_B_look2'); _, heads, maps = s26.load('A_B_look2')
    sel, rules = look_v4.selection(); m = s26.head_mean(heads, sel['heads']).reshape(-1,24,24)
    stats = cache.load_json(config.RESULTS/'look_e3_selection.json')
    rank = np.argsort(-np.asarray(stats['head_label_free']['label_free_score']))
    audit = dict(head_score_pointing_spearman=float(spearmanr(stats['head_label_free']['label_free_score'],stats['head_pointing_B']).statistic),
                 sanity=sanity(bank,m), smaller_m_exploratory_only={str(n):bank.summary(s26.head_mean(heads,rank[:n])) for n in (1,2,3,4)},
                 matched_recomputed_baseline=bank.summary(maps['S3']), archived_baseline=bank.summary(bank.archived))
    audit['selected_heads'] = {f'L{h//12+1}H{h%12+1}':bank.summary(s26.head_mean(heads,[h])) for h in sel['heads']}
    frozen=cache.load_json(P/'A_B_look4.json'); prepared=bank.prepare(m)
    sub={}; overlaps=[]
    for j,e in enumerate(bank.e):
        if e['kind'] != 'person' or e['gt'] is None: continue
        c=frozen[e['ri']]['claims'][e['ci']]; box=np.array(c['rec_att']['box'],np.float32)
        key='multiple_person_boxes' if len(e['gt'])>1 else 'single_person_box'
        hit=float(look.iou(box,e['gt']).max()) >= .5
        group=sub.setdefault(key,dict(n=0,hits=0));group['n']+=1;group['hits']+=int(hit)
        overlaps.append(c['rec_att']['sam_proposal_iou'])
    audit['person_groups']={k:dict(**v,region=100*v['hits']/v['n']) for k,v in sub.items()}
    audit['sam_score_proxy'] = dict(n=len(overlaps), iou_quantiles=np.quantile(overlaps,[0,.1,.5,.9,1]).tolist(),
                                   below_half=int((np.array(overlaps)<.5).sum()))
    # First two eligible claims of each kind in B order, plus three multi-person failures.
    chosen=[]
    for kind in ('person','thing','stuff'):
        chosen += [j for j,e in enumerate(bank.e) if e['kind']==kind and e['gt'] is not None][:2]
    chosen += [j for j,e in enumerate(bank.e) if e['kind']=='person' and e['gt'] is not None and len(e['gt'])>1
               and max(look.iou(np.array(frozen[e['ri']]['claims'][e['ci']]['rec_att']['box']),e['gt']))<.5][:3]
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(len(chosen),5,figsize=(12,2.2*len(chosen)))
    examples=[]
    for row,j in enumerate(chosen):
        e=bank.e[j]; image=data.load_image(e['iid']); w,h=image.size
        for col,head in enumerate([None,*sel['heads']]):
            ax=axes[row,col]; ax.imshow(image)
            hm=prepared[j] if head is None else look.smooth_map(s26.head_mean(heads[j:j+1],[head])[0].reshape(24,24),0)
            ax.imshow(hm,extent=(0,w,h,0),alpha=.55,cmap='magma',interpolation='nearest')
            for box in e['gt']:
                ax.add_patch(Rectangle((box[0]*w,box[1]*h),(box[2]-box[0])*w,(box[3]-box[1])*h,fill=False,edgecolor='lime',linewidth=1))
            if head is None:
                box=frozen[e['ri']]['claims'][e['ci']]['rec_att']['box']
                ax.add_patch(Rectangle((box[0]*w,box[1]*h),(box[2]-box[0])*w,(box[3]-box[1])*h,fill=False,edgecolor='cyan',linewidth=1.5))
            ax.set_title(f"{e['iid']} {e['cls']} | "+('v4; cyan box' if head is None else f'L{head//12+1}H{head%12+1}'));ax.axis('off')
        examples.append(dict(iid=e['iid'],kind=e['kind'],claim=e['cls'],owner=[e['ri'],e['ci']],n_gt_boxes=len(e['gt'])))
    fig.tight_layout();fig.savefig(config.RESULTS/'look_e3_review_B.png');plt.close(fig)
    cache.save_json(examples,config.RESULTS/'look_e3_review_B_examples.json')
    return audit


def live():
    """Replay fixed drafts through pipeline.run; compute attention and SAM afresh."""
    import torch
    from lcr.captioner import Captioner
    from lcr.look_sources import LookDecoder
    dec=LookDecoder(); rows=cache.load_json(P/'A_B_look2.json'); frozen=cache.load_json(P/'A_B_look4_review.json')
    saved=np.load(P/'A_B_look4_review.maps.npz'); idx={tuple(o):j for j,o in enumerate(saved['owner'])}
    class ReplayCap:
        dev=torch.device('cpu');tok=dec.tok;bos=dec.bos;pad=0
        encode_captions=Captioner.encode_captions
        word_alignment=Captioner.word_alignment
        decode=Captioner.decode
        def generate(self,emb): return dec.tokens(self.row)
        def cross_attention(self,emb,ids):
            with torch.no_grad(),dec.capture() as att:
                out=dec.forward(ids,emb)
            return torch.stack(att,1),out.logits
    cap=ReplayCap(); sam=look_v4.SamBoxes(); results=[]
    for ri in (0,1,2,3):
        cap.row=rows[ri]
        row,mm=next(pipeline.run(cap,[cap.row['id']],bs=1,sink=True,region_rule='v4',sam=sam))
        error=0.; boxerr=0.; changed=0
        assert row['draft']==frozen[ri]['draft']
        for ci,(c,d,m) in enumerate(zip(row['claims'],frozen[ri]['claims'],mm,strict=True)):
            assert (c['cls'],c['pos'])==(d['cls'],d['pos'])
            error=max(error,float(np.abs(m-saved['maps'][idx[ri,ci]]).max()))
            boxerr=max(boxerr,float(np.max(np.abs(np.array(c['rec_att']['box'])-d['rec_att']['box']))))
            changed+=c['rec_att']['region']!=d['rec_att']['region']
        results.append(dict(id=row['id'],n_claims=len(mm),max_map_error=error,max_box_error=boxerr,changed_regions=changed))
        assert error<1e-6 and boxerr<1e-6 and changed==0
    return dict(protocol='first four B drafts replayed; fresh CPU decoder + fresh SAM; no generation comparison',images=results)


def verify_migrated():
    result = {}
    for stem in ('A_B_look4', 'A_validation_look4'):
        for suffix in ('', '_check2', '_check2vis'):
            name = stem + suffix
            old = cache.load_json(P / 'archive/pre_e3_review' / f'{name}.json')
            new = cache.load_json(P / f'{name}.json')
            count = 0
            for a, b in zip(old, new, strict=True):
                for c, d in zip(a['claims'], b['claims'], strict=True):
                    for key in ('rec_att', 'rec_det'):
                        assert all(d[key][k] == v for k, v in c[key].items())
                    assert d['excess_by_k']['4'] == d['rec_att']['excess']
                    count += 1
            result[name] = dict(claims=count, old_record_values_unchanged=True, auxiliary_excess_consistent=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--live',action='store_true');ap.add_argument('--verify-only',action='store_true');a=ap.parse_args()
    if a.verify_only:
        out=cache.load_json(config.RESULTS/'look_e3_review.json')
        out['after_migration']=verify_migrated()
        cache.save_json(out,config.RESULTS/'look_e3_review.json')
        print(out['after_migration']);return
    out=dict(protocol='frozen selection unchanged; B-only diagnostics; validation artifact reproduction only',
             parity={s:parity(s) for s in ('A_B_look4','A_validation_look4')},B=b_diagnostics())
    if a.live: out['live']=live()
    cache.save_json(out,config.RESULTS/'look_e3_review.json')
    print(out['parity']);print('B diagnostics',out['B']['person_groups']);print(out.get('live','live not requested'))


if __name__=='__main__':main()
