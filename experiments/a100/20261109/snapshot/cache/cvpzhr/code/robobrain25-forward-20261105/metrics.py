"""Episode-equal Pearson and frame-weighted absolute error, retaining undefined cases."""
from collections import defaultdict
import math


def correlation(x, y):
    if len(x) < 2:
        return None
    a, b = math.fsum(x) / len(x), math.fsum(y) / len(y)
    dx, dy = [v-a for v in x], [v-b for v in y]
    xx, yy = math.fsum(v*v for v in dx), math.fsum(v*v for v in dy)
    return None if xx == 0 or yy == 0 else max(-1., min(1., math.fsum(u*v for u,v in zip(dx,dy)) / math.sqrt(xx*yy)))


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        assert math.isfinite(row['predicted_pt']) and math.isfinite(row['target_pt'])
        grouped[row['episode_id']].append(row)
    correlations = {eid: correlation([r['predicted_pt'] for r in rr], [r['target_pt'] for r in rr])
                    for eid, rr in grouped.items()}
    def report(suite=None):
        rr = [r for r in records if suite is None or r['episode_id'].split('/')[0] == suite]
        eids = {r['episode_id'] for r in rr}
        valid = [correlations[e] for e in eids if correlations[e] is not None]
        return dict(samples=len(rr), episodes=len(eids),
                    mae_pp=100*math.fsum(abs(r['predicted_pt']-r['target_pt']) for r in rr)/len(rr),
                    pearson_episode_mean=math.fsum(valid)/len(valid) if valid else None,
                    undefined_pearson_episodes=sorted(e for e in eids if correlations[e] is None))
    return dict(overall=report(), by_suite={s:report(s) for s in sorted({e.split('/')[0] for e in grouped})})
