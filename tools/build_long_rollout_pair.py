#!/usr/bin/env python3
"""Isolated two-run comparison; reuse the tested seek/frame-control template.

Only symlinks and gallery artifacts are created. Source videos are unchanged.
"""
import json
from pathlib import Path
import build_long_method_picker as shared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'web_reports/long_rollout_pair'
RUNS = [
    ('rollout_continue500opt', 'Rollout 3+9 · continue500opt', Path('/mnt/cpfs/users/lzk/codes/swap_face/idvtrain/output/lzk_rollout_distillation/evaluation/model_eval/rollout3p9_continue500opt_long200_h3c9/configs/7_h3c9/generation/videos')),
    ('temporal_self_rollout_sup48', 'L2 temporal self-rollout · sup48', Path('/mnt/cpfs/users/lyw/idvtrain/.cpfs_runtime/humanvid/output/l2_temporal_self_rollout_sup48_20260912/benchmark_long200/configs/7_h3c9/generation/videos')),
]


def main():
    # These overrides affect this process only, not the original builder files.
    shared.OUT = OUT
    shared.LIVING = RUNS[0][2]
    shared.OURS = RUNS[1][2]
    shared.main()
    data = json.loads((OUT / 'data.json').read_text())
    for run, (run_id, label, folder) in zip(data['runs'], RUNS):
        run.update(run_id=run_id, label=label, group='Rollout comparison')
        videos = {}
        for case_id, old_url in run['videos'].items():
            link = OUT / 'media' / run_id / f'{case_id}.mp4'
            shared.ensure_link(link, (OUT / old_url).resolve())
            new_url = str(link.relative_to(OUT))
            videos[case_id] = new_url
            data['video_metadata'][new_url] = data['video_metadata'][old_url]
        run['videos'] = videos
    (OUT / 'data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    page = (OUT / 'index.html').read_text()
    page = page.replace('Long 200 · 方法对比与 Case 筛选', 'Long 200 · 两组 Rollout 对比')
    page = page.replace('long-method-picker', 'long-rollout-pair')
    page = page.replace("+media('Inswapper',c.gan_swapped_video,'video','benchmark baseline')", '')
    page = page.replace(' methods + Inswapper', ' methods')
    page = page.replace("['Inswapper','LivingSwap','Ours rollout3p9_continue500opt']",
                        json.dumps([r[1] for r in RUNS], ensure_ascii=False))
    assert "media('Inswapper'" not in page
    (OUT / 'index.html').write_text(page)
    # Expose time/frame differences instead of assuming perfect correspondence.
    differences = []
    for c in data['cases']:
        source = data['video_metadata'][c['origin_video']]
        for run in data['runs']:
            m = data['video_metadata'][run['videos'][c['case_id']]]
            if abs(m['fps'] - source['fps']) > 0.01 or m['frames'] != source['frames']:
                differences.append(dict(case_id=c['case_id'], run_id=run['run_id'], source=source, generated=m))
    (OUT / 'validation.json').write_text(json.dumps(dict(cases=len(data['cases']),
        videos_per_method=[len(r['videos']) for r in data['runs']], timeline_differences=differences), indent=2))
    print(f'Final gallery: {OUT}; 2 methods; timeline differences: {len(differences)}')


if __name__ == '__main__':
    main()
