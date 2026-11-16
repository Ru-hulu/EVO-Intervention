"""Spearman(prediction, GT) restricted to the original CC triplet sample set.

Two events are pooled within each trajectory, preserving every event-local
triplet occurrence (including overlapping events) and excluding turning points.
Then retain the original trajectory and seed aggregation. No inference/writes
to original predictions or original reports.
"""
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

SETUP = Path('/cache/cvpzhr/setup/soc-disrupted-eval-20261115')
SOURCE = Path('/cache/cvpzhr/evaluations/soc-disrupted-eval-20261115')
OUT = Path('/cache/cvpzhr/evaluations/soc-disrupted-cc-spearman-20261116')


def calculate(predictions, target, indices, correlation):
    pred = [predictions[i] for i in indices]
    truth = [target[i] for i in indices]
    _, value, reason = correlation(pred, truth)
    return dict(spearman=value, undefined_reason=reason, observations=len(indices),
        invalid_frames=sum(v is None or not math.isfinite(v) for v in pred),
        constant_target=bool(truth) and max(truth)==min(truth),
        prediction_indices=indices)


def summarize(records, expected):
    values = [v['spearman'] for v in records if v['spearman'] is not None]
    assert len(records) == expected
    return dict(expected=expected, present=len(records), defined=len(values), undefined=len(records)-len(values),
        mean=statistics.mean(values) if values else None,
        undefined_reasons=dict(Counter(v['undefined_reason'] for v in records if v['undefined_reason'])),
        invalid_frames=sum(v['invalid_frames'] for v in records),
        observations=sum(v['observations'] for v in records))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(SETUP/'code'))
    from experiments.soc_disrupted_eval_20261115.common import read, sha, save, code_hashes, identity, result_path
    from experiments.soc_disrupted_eval_20261115.protocol import validate_record
    from experiments.soc_disrupted_eval_20261115.metrics import cycle_triplets, score
    from experiments.prm_five_groups_20261114.eval_common import correlations
    from scipy.stats import spearmanr
    source = read(SETUP/'inputs.json'); old = read(SOURCE/'report.json')
    assert old['status'] == 'completed' and old['full_coverage']
    source_hash = sha(SETUP/'inputs.json'); report_hash = sha(SOURCE/'report.json')
    assert source_hash == old['inputs_sha256']
    assert code_hashes() == source['source_code_sha256']
    models = ('forest','robometer-4b','procvlm-2b'); suites = ('State','Order','Count'); seeds = (42,43,44)
    assert len(source['episodes']) == 84
    # The live dataset has since been updated. Use the original audited input
    # snapshot and manifest targets, never substitute current dataset labels.
    current_pt_sources = Counter()
    for row in source['episodes']:
        path = Path(row['progress_path'])
        state = 'missing' if not path.exists() else ('matching' if sha(path)==row['progress_sha256'] else 'changed')
        current_pt_sources[state] += 1
        assert len(row['target_pt']) == len(row['frame_indices'])
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in row['target_pt'])
    results, hashes = [], {}
    for seed in seeds:
        condition = f'rewind_only-seed{seed}'
        manifest_info = source['conditions'][condition]
        assert sha(manifest_info['path']) == manifest_info['sha256']
        entries = {e['key']:e for e in read(manifest_info['path'])['episodes']}
        for model in models:
            for row in source['episodes']:
                entry = entries[row['key']]; path = result_path(model,condition,row['key'])
                digest = sha(path); assert digest == old['prediction_file_sha256'][str(path)]
                hashes[str(path)] = digest
                result = read(path); assert result['status'] == 'completed'
                validate_record(result,row,entry,identity(source_hash,source['registry'][f'{model}-{row["suite"]}'],row,entry),complete=True)
                previous = score(result['predictions'],row,entry)
                assert previous == result['metrics']
                events = cycle_triplets(row,entry); indices = []; details = []
                for event in events:
                    event_indices = [i for point in event['points'] for i in point['prediction_indices']]
                    indices.extend(event_indices)
                    for point in event['points']:
                        assert len({entry['target_pt'][i] for i in point['prediction_indices']}) == 1
                    detail = calculate(result['predictions'],entry['target_pt'],event_indices,correlations)
                    details.append(dict(event_index=event['event_index'], **detail))
                metrics = calculate(result['predictions'],entry['target_pt'],indices,correlations)
                if metrics['spearman'] is not None:
                    reference = float(spearmanr([result['predictions'][i] for i in indices],
                                              [entry['target_pt'][i] for i in indices]).statistic)
                    assert abs(metrics['spearman']-reference) < 1e-12
                assert len(indices) == previous['expected_triplets']*3
                results.append(dict(model=model,suite=row['suite'],task_id=row['task_id'],key=row['key'],seed=seed,
                    **metrics, events=details, original_cc_error=previous['circle_consistency'],
                    whole_trajectory_spearman=previous['spearman'], prediction_file_sha256=digest))
        print('Audited seed',seed,flush=True)
    assert len(results) == 756
    per_seed, across = [], []
    for model in models:
        for suite in (*suites,'All'):
            expected = sum(suite=='All' or row['suite']==suite for row in source['episodes'])
            summaries = []
            for seed in seeds:
                group = [r for r in results if r['model']==model and r['seed']==seed and (suite=='All' or r['suite']==suite)]
                summary = dict(model=model,suite=suite,seed=seed,**summarize(group,expected))
                summaries.append(summary); per_seed.append(summary)
            values = [s['mean'] for s in summaries if s['mean'] is not None]
            across.append(dict(model=model,suite=suite,mean=statistics.mean(values) if len(values)==3 else None,
                sample_std=statistics.stdev(values) if len(values)==3 else None,
                per_seed_mean=[s['mean'] for s in summaries],per_seed_defined=[s['defined'] for s in summaries],
                per_seed_undefined=[s['undefined'] for s in summaries],invalid_frames=sum(s['invalid_frames'] for s in summaries)))
    assert sha(SOURCE/'report.json') == report_hash
    assert all(sha(p)==h for p,h in hashes.items())
    report = dict(status='completed',created=time.time(),expected_curves=756,present_curves=len(results),
        protocol=dict(metric='Spearman(predicted progress, corresponding GT) on original CC sample set',
            points='All three visits of each event-local interior source timestamp; both events pooled per trajectory',
            turning_endpoints='excluded exactly as original CC',overlapping_events='event-local multiplicity retained',
            aggregation='one correlation per trajectory, equal trajectory mean within suite, mean and sample SD across seeds42/43/44',
            all_suites='trajectory equal, not suite equal',undefined='Constant/invalid required curves -> null, denominators reported, no zero fill',
            model_inference=False,gpu_used=False,original_artifacts_modified=False,
            ground_truth='Original frozen inputs and playback manifests, bound to the completed report and prediction hashes; current mutable dataset not substituted'),
        current_pt_sources=dict(current_pt_sources),
        source_inputs_sha256=source_hash,source_report_sha256=report_hash,source_code_sha256=sha(__file__),
        suite_trajectory_counts=dict(Counter(r['suite'] for r in source['episodes'])),
        across_seeds=across,per_seed=per_seed,episodes=results,prediction_file_sha256=hashes)
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'report.json',report)
    save(OUT/'summary.json',{k:v for k,v in report.items() if k not in ('episodes','prediction_file_sha256')})
    flat=[{k:v for k,v in r.items() if k not in ('events','prediction_indices')} for r in results]
    with (OUT/'trajectory-spearman.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(across,indent=2))


if __name__=='__main__': main()
