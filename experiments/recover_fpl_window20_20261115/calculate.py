"""GT-drop-onset FPL with an inclusive +/-20%-of-duration preferred window.

This is a custom, ground-truth-conditioned metric, not the original official FPL.
The pinned ZigZag descent extraction/ties and no-descent penalty are unchanged.
Only saved predictions are read; no model/GPU libraries are imported.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Sequence

SOURCE_SHA = '524c7a5310a44ea145048257f9dbb78e3efc3de89755acff4f327ff793fc1e57'
MODELS = ('robometer-4b', 'procvlm-2b', 'robobrain25-4b', 'dopamine-3b', 'forest')


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native_functions(path):
    assert sha(path) == SOURCE_SHA, 'Pinned descent algorithm changed'
    names = {'_maximum_drawdown', '_zigzag_descents', '_select_descent'}
    nodes = [n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = dict(Sequence=Sequence, VMBMKError=ValueError)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def extract(values, frames, native):
    curve = list(zip(frames, values))
    drop = native['_maximum_drawdown'](curve)
    return native['_zigzag_descents'](curve, .2*drop) if drop > 0 else [], drop


def choose(descents, onset, duration, native):
    # Membership is based on descent START, not its trough or any overlap.
    candidates = [d for d in descents if abs(d[0]-onset) <= .2*duration]
    return native['_select_descent'](candidates or descents), candidates


def error(selected, onset, duration):
    return (abs(selected[0]-onset) if selected else max(onset, duration-onset))/duration


def score(pred, truth, frames, total_frames, native):
    assert len(pred) == len(truth) == len(frames) >= 2
    assert frames[0] == 0 and frames[-1] == total_frames-1
    assert all(a < b for a,b in zip(frames, frames[1:]))
    assert all(v is not None and math.isfinite(v) for v in truth)
    duration = total_frames-1
    gt_descents, gt_drop = extract(truth, frames, native)
    gt = native['_select_descent'](gt_descents)
    out = dict(value=None, gt_global_fpl=None, reason=None, gt_descent=gt,
               gt_descents=gt_descents, gt_maximum_drawdown=gt_drop, duration_frames=duration,
               predicted_descent=None, predicted_global_descent=None, window_candidates=[],
               selection_mode=None, invalid_frames=sum(v is None or not math.isfinite(v) for v in pred))
    if gt is None:
        out['reason'] = 'no_ground_truth_descent'; return out
    onset = gt[0]
    out['window_frames'] = [max(0,onset-.2*duration), min(duration,onset+.2*duration)]
    if out['invalid_frames']:
        out['reason'] = 'invalid_predictions'; return out
    descents, drop = extract(pred, frames, native)
    global_choice = native['_select_descent'](descents)
    selected, candidates = choose(descents, onset, duration, native)
    out.update(value=error(selected,onset,duration), gt_global_fpl=error(global_choice,onset,duration),
        predicted_descent=selected, predicted_global_descent=global_choice, predicted_descents=descents,
        predicted_maximum_drawdown=drop, window_candidates=candidates,
        selection_mode='within_window' if candidates else 'global_fallback' if selected else 'no_descent_penalty')
    return out


def main(setup, original, output):
    started = time.time()
    assert not output.exists(), 'Never overwrite prior metric outputs'
    frozen = read(setup/'inputs.json'); old = read(original/'report.json')
    assert old['status'] == 'completed' and old['inputs_sha256'] == sha(setup/'inputs.json')
    assert frozen['selection_sha256'] == sha(setup/'selection.json')
    assert frozen['sample_fps'] == 5 and frozen['expected_samples'] == 2132
    source = setup/'code/experiments/prm_five_groups_20261114/metric_reference/fpl.py'
    native = native_functions(source)
    hashes = {str(p):sha(p) for p in [setup/'inputs.json',setup/'selection.json',source,
                                    original/'report.json',original/'comparison.md']}
    def checked(path, expected=None):
        path = Path(path); h = sha(path)
        if expected is not None: assert h == expected, str(path)
        hashes[str(path)] = h
        return read(path)
    for row in frozen['records']:
        path = Path(row['progress_path']); hashes[str(path)] = sha(path)
        assert hashes[str(path)] == row['progress_sha256']
        if path.suffix == '.csv':
            with path.open() as f: labels = list(csv.DictReader(f))
        else:
            label = read(path); assert label['video_sha256'] == row['video_hashes'][0]
            labels = label['frames']
        assert [int(v['frame_index']) for v in labels] == list(range(row['frames']))
        assert [float(labels[i]['pt']) for i in row['frame_indices']] == row['target_pt']
    results = {}; flat = []
    for model in MODELS:
        info = frozen['model_checkpoints'][model]
        checked(info['proof'], info['proof_sha256'])
        old_eps = {r['key']:r for r in old['models'][model]['episodes']}
        episodes = []
        for row in frozen['records']:
            prev = old_eps[row['key']]
            path = original/model/'episodes'/(row['key'].replace('/','__')+'.json')
            r = checked(path,prev['result_sha256'])
            assert r['key'] == row['key'] and r['model'] == model
            assert r['inputs_sha256'] == old['inputs_sha256'] and r['checkpoint_sha256'] == info['checkpoint_sha256']
            assert r['frame_indices'] == row['frame_indices'] and r['target_pt'] == row['target_pt']
            assert r['timestamps'] == row['timestamps'] and r['progress_sha256'] == row['progress_sha256']
            assert r['source_video_sha256'] == row['video_hashes']
            assert len(r['predictions']) == len(r['raw_responses']) == len(row['frame_indices'])
            init = checked(r['initialization_audit'])
            assert init['strict_final_state'] and init['checkpoint_sha256'] == info['checkpoint_sha256']
            assert init['inputs_sha256'] == old['inputs_sha256']
            event = checked(setup/'events'/(row['key'].replace('/','__')+'.json'),prev['event_annotation_sha256'])
            scored = score(r['predictions'], row['target_pt'], row['frame_indices'], row['frames'], native)
            # Independently reproduce original FPL to establish an apples-to-apples prediction source.
            if prev['fpl'] is not None:
                assert scored['predicted_global_descent'] is not None or scored['selection_mode']=='no_descent_penalty'
                reproduced = error(scored['predicted_global_descent'], event['failure_onset_frame'], row['frames']-1)
                assert math.isclose(reproduced, prev['fpl'], rel_tol=0,abs_tol=1e-12)
            item = dict(key=row['key'],task_id=row['task_id'],samples=len(r['predictions']),
                original_fpl=prev['fpl'],original_failure_onset_frame=event['failure_onset_frame'],
                fps=row['fps'],result_sha256=hashes[str(path)],**scored)
            episodes.append(item)
            flat.append(dict(model=model,key=row['key'],original_fpl=prev['fpl'],
                gt_global_fpl=scored['gt_global_fpl'],window20_fpl=scored['value'],reason=scored['reason'],
                gt_onset_frame=scored['gt_descent'][0] if scored['gt_descent'] else None,
                predicted_onset_frame=scored['predicted_descent'][0] if scored['predicted_descent'] else None,
                selection_mode=scored['selection_mode'],window_candidates=len(scored['window_candidates']),
                duration_frames=row['frames']-1,invalid_frames=scored['invalid_frames']))
        valid = [e for e in episodes if e['value'] is not None]
        results[model] = dict(original_fpl=old['models'][model]['summary']['metrics']['fpl']['mean'],
            gt_global_fpl=math.fsum(e['gt_global_fpl'] for e in valid)/len(valid) if valid else None,
            window20_fpl=math.fsum(e['value'] for e in valid)/len(valid) if valid else None,
            eligible_episodes=sum(e['gt_descent'] is not None for e in episodes),defined_episodes=len(valid),
            invalid_eligible_episodes=sum(e['reason']=='invalid_predictions' for e in episodes),
            no_gt_descent_episodes=sum(e['reason']=='no_ground_truth_descent' for e in episodes),
            invalid_frames=sum(e['invalid_frames'] for e in episodes),
            window_matched_episodes=sum(e['selection_mode']=='within_window' for e in valid),
            global_fallback_episodes=sum(e['selection_mode']=='global_fallback' for e in valid),
            no_descent_penalties=sum(e['selection_mode']=='no_descent_penalty' for e in valid),episodes=episodes)
        assert sum(e['samples'] for e in episodes) == 2132
    assert all(sha(p)==h for p,h in hashes.items()), 'Original input/result changed during rescoring'
    output.mkdir(parents=True)
    protocol = dict(name='GT-onset window-prioritized FPL-20',official_fpl=False,sample_fps=5,
        ground_truth_anchor='start of largest ZigZag descent in saved 5 FPS GT',
        window_half_width='0.20 * full trajectory duration; inclusive boundaries; clipped to trajectory',
        prediction_selection='largest descent whose onset lies in window; else global largest',
        descent_detection='unchanged pinned ZigZag, threshold 0.2 * each full curve maximum drawdown',
        tie_break='largest drop, shortest duration, earliest onset',no_descent='original farthest-boundary penalty',
        no_gt_descent='not applicable',invalid_predictions='undefined, never interpolate or replace',
        normalization='absolute onset-frame error / (original total_frames - 1)',
        aggregation='equal weight per defined trajectory; explicit missing/invalid denominators',
        ground_truth_used_for_prediction_event_selection=True)
    report = dict(status='completed',protocol=protocol,models=results,source_sha256=hashes,
        code_sha256=sha(__file__),old_results_unchanged=True,model_loaded=False,gpu_used=False,
        rescoring_seconds=time.time()-started,completed_unix=time.time())
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (output/'per-trajectory.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    summary=[]
    for model,v in results.items():
        summary.append(dict(model=model,**{k:x for k,x in v.items() if k!='episodes'}))
    with (output/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(dict(status='completed',seconds=report['rescoring_seconds'],summary=summary),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--setup',type=Path,default=Path('/cache/cvpzhr/setup/recover-five-validation-20261114'))
    p.add_argument('--original',type=Path,default=Path('/cache/cvpzhr/evaluations/recover-five-validation-20261114'))
    p.add_argument('--output',type=Path,default=Path('/cache/cvpzhr/evaluations/recover-five-validation-20261114/fpl-window20-20261115'))
    a=p.parse_args();main(a.setup,a.original,a.output)
