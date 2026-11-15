"""Generate the review's numerical appendix from saved audit outputs."""
import _common  # noqa: F401
from lcr import cache, config


def main():
    r=cache.load_json(config.RESULTS/'look_e3_review.json')
    drift=cache.load_json(config.RESULTS/'look_e3_drift_probe.json')
    gen=cache.load_json(config.RESULTS/'look_e3_live_generation.json')
    final=cache.load_json(config.RESULTS/'look_benchmark_A_validation_look4.json')
    a=cache.load_json(config.RESULTS/'look_e3_stageA.json')
    sel=cache.load_json(config.RESULTS/'look_e3_selection.json')
    names=[f'L{h//12+1}H{h%12+1}' for h in sel['winner']['heads']]
    lines=['# Look E3 — inceleme ölçümleri','', 'Üreten: `scripts/look_e3_review_table.py`. Kaynaklar: `results/look_e3_review.json`,',
           '`look_e3_drift_probe.json`, `look_e3_live_generation.json`, `look_benchmark_A_validation_look4.json`.',
           '', 'Başlıklar (katman ve başlık 1’den başlar): '+', '.join(names)+'.', '',
           '| Yeniden üretim | İddia | Değişen bölge | En büyük harita farkı | Eski tutarsız excess |','|---|---:|---:|---:|---:|']
    for name,v in r['parity'].items():
        lines.append(f"| {name} | {v['claims']} | {v['changed_regions']} | {v['max_map_error']:.9f} | {v['old_stale_excess']} |")
    lines += ['',f"Canlı sabit taslak deneyi: {len(r['live']['images'])} B görüntüsü, "+
              f"{sum(x['n_claims'] for x in r['live']['images'])} iddia; en büyük harita/kutu farkı "+
              f"{max(x['max_map_error'] for x in r['live']['images']):.9f}/{max(x['max_box_error'] for x in r['live']['images']):.9f}.",
              f"Gerçek taslak üretimi: {gen['n_images']} B görüntüsü, {gen['claims']} iddia; taslak eşitliği {gen['drafts_equal']}, "+
              f"kutu eşitliği {gen['boxes_equal']}, en büyük harita farkı {gen['max_map_error']:.9f}.", '',
              '| Bağımsız validation yeniden ölçümü | Bölge | Pointing |','|---|---:|---:|']
    for kind in ('all','person','thing','stuff'):
        v=final[kind];lines.append(f"| {kind} | {v['region_att']:.2f} | {v['pointing']:.2f} |")
    lines += ['', '| Eşlenmiş yeniden hesap hattı (bölge) | S3 | Seçili başlıklar | Fark |','|---|---:|---:|---:|']
    for split,base,heads in [('B',r['B']['matched_recomputed_baseline']['all']['region'],r['B']['smaller_m_exploratory_only']['4']['all']['region']),
                            ('validation (önceden raporlanmış)',a['S3_recomputed']['all']['region'],a['families']['A1_G3_labelfree_heads']['summary']['all']['region'])]:
        lines.append(f'| {split} | {base:.2f} | {heads:.2f} | {heads-base:+.2f} |')
    lines += ['', 'Aşağıdaki küçük m ölçümleri sonradan yapılan **B tanısıdır**, yeni seçim değildir.', '',
              '| Başlık sayısı | B bölge |','|---|---:|']
    for k,v in r['B']['smaller_m_exploratory_only'].items():lines.append(f"| {k} | {v['all']['region']:.2f} |")
    lines += ['', '| Tek başlık (B) | person pointing | thing pointing | stuff pointing |','|---|---:|---:|---:|']
    for k,v in r['B']['selected_heads'].items():
        lines.append(f"| {k} | {v['person']['pointing']:.2f} | {v['thing']['pointing']:.2f} | {v['stuff']['pointing']:.2f} |")
    lines += ['', '| Kişi grubu (B, GT kutu adedi) | İddia | Bölge |','|---|---:|---:|']
    for k,v in r['B']['person_groups'].items():lines.append(f"| {k} | {v['n']} | {v['region']:.2f} |")
    proxy=r['B']['sam_score_proxy']
    lines += ['',f"SAM–skor kutusu proxy’si (B, GT’li kişi): {proxy['below_half']}/{proxy['n']} eşleşmede IoU < 0.5; "+
              f"medyan IoU {proxy['iou_quantiles'][2]:.4f}. Bu, SAM bölgesinin yeniden detektörle skorlanması değildir.",
              '', f"Sayısal sapma tanısı: ilk 16 B görüntüsündeki {drift['n_claims']} iddia.",
              '', '| Koşul | İddia başına en büyük yama hatasının ortalaması | En büyük hata |','|---|---:|---:|']
    for k,v in drift['variants'].items():lines.append(f"| {k} | {v['mean_max_error']:.6f} | {v['max_error']:.6f} |")
    lines += ['', 'Sonuç: SEP dahil etmek örnekte farkı azaltıyor; CPU yarım duyarlık deneyi arşivi yeniden üretmiyor.',
              'Tarihsel GPU hattındaki sapmanın kök nedeni henüz saptanmadı. V4 kendi yeniden hesap hattında birebir üretildi.', '',
              'Önceki 95% aralıklar tekil, görüntü bootstrap aralıklarıdır; aileler/türler için eşzamanlı güvence vermez.',
              'Seçim sonrası B ÇD sayıları tüm A→B seçicisinin iç içe dış-kat performansı değildir.']
    path=config.ROOT/'docs/generated/LOOK_E3_REVIEW_METRICS.md';path.write_text('\n'.join(lines)+'\n');print(path)


if __name__=='__main__':main()
