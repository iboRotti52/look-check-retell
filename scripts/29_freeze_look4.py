"""Freeze E3 caches through the same geometry code as pipeline.run(v4).

Existing outputs require --overwrite. --suffix allows an independent replay.
Caption-only signals may be copied because the drafts and token positions agree.
"""
import argparse
import shutil

import numpy as np

import _common  # noqa: F401
from lcr import cache, check, config, data, look, pipeline, look_v4

P = config.CACHE / 'pipeline'
E3 = config.CACHE / 'look_e3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--suffix', default='look4')
    ap.add_argument('--sync-check-metadata', action='store_true')
    a = ap.parse_args()
    if '/' in a.suffix or not a.suffix.startswith('look4'):
        ap.error('suffix must start with look4 and contain no path separator')
    sources = ('A_B_look2', 'A_validation_look2')
    # Check every destination before writing any split.
    for stem in sources:
        dest = stem.replace('look2', a.suffix)
        for ext in ('.json', '.maps.npz', '_capsig.json'):
            if (P / f'{dest}{ext}').exists() and not a.overwrite:
                ap.error(f'Output exists: {dest}{ext}; use --overwrite')
    selected, box_rules = look_v4.selection()
    for stem in sources:
        out_stem = stem.replace('look2', a.suffix)
        rows = cache.load_json(P / f'{stem}.json')
        info = cache.load_json(E3 / f'{stem}_info.json')
        heads = np.load(E3 / f'{stem}_heads.f16.npy', mmap_mode='r')
        sam = np.load(E3 / f'{stem}_sam.npz')
        owners = [[ri, ci] for ri, r in enumerate(rows) for ci in range(len(r['claims']))]
        if owners != info['owner'] or not np.array_equal(sam['owner'], owners) or len(heads) != len(owners):
            raise ValueError('E3 cache ownership mismatch')
        if str(sam['source']) != selected['family'] + ':' + selected['config']:
            raise ValueError('SAM cache belongs to a different head selection')
        owl = cache.OwlCache(data.parse_id(rows[0]['id'])[0]); q = pipeline.class_queries()
        out, maps_out, k = [], [], 0
        for r in rows:
            prop = owl[data.parse_id(r['id'])[1]]; scores = check.score(prop, q)
            r2 = dict(r, claims=[], look_metadata=look_v4.metadata())
            for c in r['claims']:
                m = look_v4.selected_map(heads[k], selected['heads'])
                att = look.smooth_map(m, config.LOOK_RULES_BY_KIND[c['kind']]['sigma'])
                c2 = look_v4.rewrite_claim(c, att, prop, scores, box_rules[c['kind']], sam['boxes'][k])
                maps_out.append(att); r2['claims'].append(c2); k += 1
            out.append(r2)
        cache.save_json(out, P / f'{out_stem}.json')
        np.savez_compressed(P / f'{out_stem}.maps.npz', maps=np.asarray(maps_out, np.float32), owner=sam['owner'])
        shutil.copyfile(P / f'{stem}_capsig.json', P / f'{out_stem}_capsig.json')
        if a.sync_check_metadata:
            sync_check_metadata(out_stem, out)
        print(f'{out_stem}: {k} claims')


def sync_check_metadata(stem, source_rows):
    """Refresh descriptive fields only; refuse any change to verifier inputs."""
    for suffix in ('_check2', '_check2vis'):
        path = P / f'{stem}{suffix}.json'
        if not path.exists():
            continue
        rows = cache.load_json(path)
        for source, dest in zip(source_rows, rows, strict=True):
            if (source['id'], source['draft']) != (dest['id'], dest['draft']):
                raise ValueError('Check draft mismatch; recompute Check explicitly')
            dest['look_metadata'] = source['look_metadata']
            for c, d in zip(source['claims'], dest['claims'], strict=True):
                for key in ('cls', 'pos', 'att_entropy', 'iou_att_det'):
                    if c[key] != d[key]:
                        raise ValueError(f'Verifier input changed: {key}')
                for key in ('rec_att', 'rec_det'):
                    for field in ('box', 'region', 's_claim', 's_image', 'excess'):
                        if c[key][field] != d[key][field]:
                            raise ValueError(f'Verifier input changed: {key}.{field}')
                    for field in ('box_source', 'score_source', 'score_box', 'sam_proposal_iou'):
                        if field in c[key]:
                            d[key][field] = c[key][field]
                d['excess_by_k'] = c['excess_by_k']
        cache.save_json(rows, path)


if __name__ == '__main__':
    main()
