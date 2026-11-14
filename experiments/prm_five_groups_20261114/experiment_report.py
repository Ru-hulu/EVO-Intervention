"""Audited follow-up results, explicit denominators, seed comparisons and plots."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
from functools import lru_cache
import html
import json
import math
from pathlib import Path
import statistics
import time

from .eval_common import EVAL, OUT, PLAN
from .experiment_inputs import MODELS, CONDITIONS, verified_condition, event_annotations
from .experiment_worker import check_prefix, metrics, required_issues
from .group1_report import summarize
from .official_baseline import OLD
from .official_stream import paths
from .prepare_cache import BASE, progress_values, save, sha
from .recover_metrics import score_recover, fpl_functions
from .stream_manifests import digest

EVENT_METRICS = ('successful_endpoint_score', 'fpl', 'abnormal_plateau_absolute_mae', 'abnormal_plateau_drift_mae')


def audit_identity(record, row, entry, model, group, condition, manifest_sha, plan_sha):
    identity = record['identity']
    expected = dict(model=model, group=group, condition=condition, suite=row['suite'],
        task_id=row['task_id'], key=row['key'], input_fps=entry['input_fps'], task_prompt=row['task_prompt'],
        plan_sha256=plan_sha, manifest_sha256=manifest_sha, map_sha256=entry['map_sha256'],
        progress_sha256=row['progress_sha256'], source_video_sha256=row['video_hashes'],
        memory_enabled=entry['memory_enabled'], batch=1,
        uses_future_endpoint_reference=model in ('robobrain25-4b', 'dopamine-3b'))
    assert all(identity.get(k) == v for k, v in expected.items())
    assert 0 < identity['memory_fraction'] <= .35
    check_prefix(record, identity, entry, complete=True)
    return identity


def verify_initialization(audit, model, suite, memory_enabled, plan, catalog):
    assert audit['model'] == model and audit['native_model_loaded'] is True
    assert audit['uses_future_endpoint_reference'] is (model in ('robobrain25-4b', 'dopamine-3b'))
    if model == 'forest':
        from .forest_adapter import check_final_proof
        proof_path = BASE/suite/'checkpoint-verification.json'
        proof = json.loads(proof_path.read_text())
        expected = Path(f'/cache/cvpzhr/outputs/forest-{suite.lower()}-qwen35-clip-depth-e1-20261114/epoch-01.pt')
        check_final_proof(proof, suite, expected)
        assert audit['checkpoint_sha256'] == proof['sha256'] and audit['checkpoint'] == str(expected)
        assert audit['checkpoint_verification_sha256'] == sha(proof_path)
        assert audit['suite'] == suite and audit['final_epoch'] == 1 and not audit['trial_checkpoint_used']
        assert audit['strict_trainable_state_verified'] and audit['memory_enabled'] is memory_enabled
    else:
        ref = next(m for m in plan['models'] if m['id'] == model)['prior_provenance_reference']
        assert audit['checkpoint'] == ref['path'] and audit['official_revision'] == ref['revision'] == catalog[model]['revision']
        assert audit['adapter_loaded'] is False
        hashes = dict(ref['small_file_sha256'])
        hashes.update({f['path']: f['lfs']['sha256'] for f in catalog[model]['files'] if f['path'].endswith(('.safetensors', '.bin'))})
        assert all(audit['official_file_sha256'].get(k) == v for k, v in hashes.items())


def event_summary(rows, coverage_complete):
    result = {}
    for metric in EVENT_METRICS:
        values = [r['recover'][metric] for r in rows if r['recover'][metric] is not None]
        reasons = Counter(r['recover']['metric_reasons'].get(metric, 'undefined')
                          for r in rows if r['recover'][metric] is None)
        result[metric] = dict(defined_episodes=len(values), undefined_or_ineligible_episodes=len(rows)-len(values),
                              reasons=dict(reasons), mean=(math.fsum(values)/len(values) if coverage_complete and values else None))
    return result


def write_curve(path, record):
    xs = record['playback_timestamps']; series = [record['target_pt'], record['predictions']]
    valid = [v for values in series for v in values if v is not None]
    low, high = min([0.]+valid), max([1.]+valid)
    span = max(xs[-1]-xs[0], 1e-12)
    def segments(values):
        pieces, current = [], []
        for t, value in zip(xs, values):
            if value is None:
                if current: pieces.append(current); current = []
            else:
                current.append(f'{55+850*(t-xs[0])/span:.2f},{325-275*(value-low)/(high-low):.2f}')
        if current: pieces.append(current)
        return pieces
    title = html.escape(record['identity']['key'])
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="380" viewBox="0 0 960 380"><rect width="960" height="380" fill="white"/><g font-family="sans-serif" font-size="14" fill="#222"><text x="55" y="24">{title}</text><text x="55" y="365">Presentation time (s), 0–{xs[-1]:.2f}</text><text x="4" y="55">{high:.2f}</text><text x="4" y="325">{low:.2f}</text><text x="650" y="45" fill="#2368b0">GT</text><text x="720" y="45" fill="#c64735">Prediction; gaps = invalid</text></g>']
    for values, color in zip(series, ('#2368b0', '#c64735')):
        for piece in segments(values):
            body.append(f'<polyline points="{" ".join(piece)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    body.append('</svg>'); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(body))


def aggregate_condition(group, condition, model, *, evaluation=EVAL, output=OUT, plan_path=PLAN, plots=True):
    manifest, sources, manifest_sha = verified_condition(group, condition, evaluation, plan_path)
    plan = json.loads(Path(plan_path).read_text()); plan_sha = sha(plan_path)
    catalog = json.loads((OLD/'model-download-catalog.json').read_text()) if model != 'forest' else {}
    rows, missing, blockers, source_results, annotation_proofs = [], [], [], {}, {}
    @lru_cache(None)
    def audited(path, suite, memory):
        value = json.loads(Path(path).read_text())
        verify_initialization(value, model, suite, memory, plan, catalog)
        assert value['plan_sha256'] == plan_sha and value['manifest_sha256'] == manifest_sha
        assert value['group'] == group and value['condition'] == condition
        return value
    @lru_cache(None)
    def verify_source(path, expected):
        assert sha(path) == expected
    directory = Path(output)/f'group{group}'/condition/model
    for entry in manifest['episodes']:
        row = sources[entry['key']]; result_path, _ = paths(output, group, condition, model, row['key'])
        if required_issues(row, model):
            blockers.append(dict(key=row['key'], issues=required_issues(row, model)))
        if not result_path.exists():
            missing.append(row['key']); continue
        record = json.loads(result_path.read_text())
        identity = audit_identity(record, row, entry, model, group, condition, manifest_sha, plan_sha)
        for proof in record['initialization_audits']:
            audit = audited(proof['path'], row['suite'], entry['memory_enabled'])
            # The backend signature excludes the worker's appended audit fields.
            excluded = {'model_signature', 'plan_sha256', 'manifest_sha256', 'group', 'condition', 'worker_code_sha256'}
            backend = {k: v for k, v in audit.items() if k not in excluded}
            # Official backends do not have this field before the audit is wrapped.
            if model != 'forest': backend.pop('memory_enabled', None)
            assert digest(backend) == identity['model_signature']
        assert sha(row['progress_path']) == row['progress_sha256']
        for path, expected in zip(row['videos'],row['video_hashes']):
            if path: verify_source(path, expected)
        values = metrics(record['predictions'], entry)
        assert values == record['metrics']
        item = dict(key=row['key'], suite=row['suite'], task_id=row['task_id'], **values)
        if group == 3:
            events, proof = event_annotations(row, evaluation)
            annotation_proofs[row['key']] = proof
            if events and events.get('failure_onset_frame') is not None:
                native, _ = fpl_functions()
                episode = type('Episode', (), {'fps': float(Fraction(row['source_rate'])), 'num_frames': row['frames']})()
                assert native['_anchor_frames'](episode) == entry['source_frame_indices'], 'FPL reference scoring anchors differ'
            item['recover'] = score_recover(record['predictions'], entry['target_pt'], entry['source_frame_indices'],
                                            entry['playback_timestamps'], row['frames'], events)
        rows.append(item); source_results[str(result_path)] = sha(result_path)
        if plots:
            write_curve(directory/'curves'/Path(row['key']).with_name(Path(row['key']).name+'.svg'), record)
    suites = {}
    for suite in sorted({e['suite'] for e in manifest['episodes']}):
        expected = [e for e in manifest['episodes'] if e['suite'] == suite]
        subset = [r for r in rows if r['suite'] == suite]
        summary = summarize([r['full_stream'] for r in subset], len(expected), sum(len(e['target_pt']) for e in expected))
        if group == 5:
            summary['common_integer_seconds'] = summarize([r['common_integer_seconds'] for r in subset], len(expected),
                                                           sum(len(e['common_scoring_indices']) for e in expected))
            summary['common_grid_is_not_full_stream_coverage'] = True
        if group == 3:
            summary['recover_metrics'] = event_summary(subset, summary['coverage_complete'])
        suites[suite] = summary
    result = dict(schema='prm-followup-report-v1', group=group, condition=condition, model=model,
                  updated=time.time(), manifest_sha256=manifest_sha, plan_sha256=plan_sha,
                  coverage_complete=len(rows) == manifest['expected_episodes'],
                  expected_episodes=manifest['expected_episodes'], completed_episodes=len(rows),
                  by_suite=suites, by_episode=rows, missing_episodes=missing, input_blockers=blockers,
                  source_result_sha256=source_results, event_annotation_proofs=annotation_proofs,
                  uses_future_endpoint_reference=model in ('robobrain25-4b', 'dopamine-3b'),
                  aggregation='equal trajectory weight over defined values; incomplete suite means withheld')
    save(directory/'metrics.json', result)
    text = [f'# {model} — group {group}, {condition}', '', '| Suite | Coverage | Pearson | Spearman | VS | ER | C | Invalid frames |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for suite, summary in suites.items():
        values = summary['defined_trajectory_means']
        cells = ['N/A' if values[k] is None else f'{values[k]:.6f} ({summary["defined_episode_counts"][k]} defined)' for k in ('pearson','spearman','VS','ER','C')]
        text.append(f'| {suite} | {summary["completed_episodes"]}/{summary["expected_episodes"]} | '+ ' | '.join(cells)+f' | {summary["invalid_frames"]} |')
    text += ['', 'Full raw curves, undefined reasons, event eligibility and scoring-grid denominators are retained in metrics.json.',
             f'Uses true future endpoint reference: {result["uses_future_endpoint_reference"]}.']
    (directory/'metrics.md').write_text('\n'.join(text)+'\n')
    with (directory/'episode-metrics.csv').open('w') as f:
        fields = ['key','suite','task_id','pearson','spearman','VS','ER','C','invalid_predictions','correlation_undefined_reason']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, row['full_stream'].get(k)) for k in fields})
    return result


def comparison(output=OUT):
    output = Path(output); reports = {}
    for group, conditions in CONDITIONS.items():
        for condition in conditions:
            for model in MODELS:
                path = output/f'group{group}'/condition/model/'metrics.json'
                if path.exists():
                    reports[f'group{group}/{condition}/{model}'] = json.loads(path.read_text())
    seeds = {}
    for variant in ('static_only','rewind_only'):
        for model in ('robometer-4b','procvlm-2b','r2vlm-alfred-rl','forest'):
            inputs = [reports.get(f'group2/{variant}-seed{s}/{model}') for s in (42,43,44)]
            for suite in ('State','Order','Count'):
                summaries = [r['by_suite'].get(suite) if r else None for r in inputs]
                data = {}
                for metric in ('pearson','spearman'):
                    values = [s['defined_trajectory_means'][metric] if s and s['coverage_complete'] else None for s in summaries]
                    valid = all(v is not None for v in values)
                    data[metric] = dict(seed_values=dict(zip((42,43,44), values)),
                        mean=statistics.mean(values) if valid else None,
                        sample_standard_deviation=statistics.stdev(values) if valid else None,
                        complete_three_seed_coverage=valid)
                seeds[f'{variant}/{model}/{suite}'] = data
    result = dict(updated=time.time(), reports={k: {'by_suite': v['by_suite'], 'coverage_complete': v['coverage_complete'],
        'uses_future_endpoint_reference':v['uses_future_endpoint_reference']} for k,v in reports.items()}, disturbance_seed_summary=seeds)
    save(output/'followup-comparison.json', result)
    lines=['# Follow-up experiments', '', 'P/S and VS use equal trajectory weights. Undefined counts are retained; incomplete suite means are withheld.', '',
           '| Condition / model | Suite | Coverage | Pearson | Spearman | VS | Future endpoint |',
           '|---|---|---:|---:|---:|---:|---|']
    for name,record in sorted(reports.items()):
        for suite,summary in record['by_suite'].items():
            cells=['N/A' if summary['defined_trajectory_means'][m] is None else
                   f'{summary["defined_trajectory_means"][m]:.6f} (n={summary["defined_episode_counts"][m]})'
                   for m in ('pearson','spearman','VS')]
            lines.append(f'| {name} | {suite} | {summary["completed_episodes"]}/{summary["expected_episodes"]} | '+' | '.join(cells)+f' | {record["uses_future_endpoint_reference"]} |')
    lines+=['','Detailed full-stream/common-grid scores, three-seed mean/sample SD, event metrics and all denominators are in followup-comparison.json and each condition’s metrics.json.']
    (output/'followup-comparison.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--group', type=int, required=True)
    p.add_argument('--condition', required=True); p.add_argument('--model', choices=MODELS, required=True)
    a = p.parse_args(); report = aggregate_condition(a.group, a.condition, a.model); comparison()
    print(json.dumps({'model':a.model,'group':a.group,'condition':a.condition,'coverage_complete':report['coverage_complete']}))
