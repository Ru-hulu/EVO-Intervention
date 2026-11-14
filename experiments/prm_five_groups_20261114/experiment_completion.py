"""Completion requires all conditions, audited coverage and explicit event labels."""
import json
import time

from .eval_common import EVAL, OUT
from .experiment_inputs import CONDITIONS, MODELS, MEMORY_MODELS
from .prepare_cache import BASE, save


def assess(states, training, readiness, reports, baseline_complete):
    failures = [k for k,v in states.items() if v.get('status') == 'failed']
    unfinished = [k for k,v in states.items() if v.get('status') not in ('completed','completed_with_input_blockers')]
    missing = []; partial = []; annotation_blockers = []
    for group, conditions in CONDITIONS.items():
        for condition in conditions:
            for model in (MEMORY_MODELS if group == 2 else MODELS):
                key = f'group{group}/{condition}/{model}'; report = reports.get(key)
                if report is None:
                    missing.append(key); continue
                if not report['coverage_complete']:
                    partial.append(key)
                if group == 3:
                    for episode in report['by_episode']:
                        reasons = episode['recover']['metric_reasons']
                        if any('missing_' in reason for reason in reasons.values()):
                            annotation_blockers.append(dict(model=model,key=episode['key'],reasons=reasons))
    input_blockers = {k:v for k,v in readiness.get('conditions',{}).items() if v['status'] != 'ready'}
    training_done = len(training.get('training_completed',{})) == 4 and not training.get('training_failed')
    verified = training_done and baseline_complete and not (failures or unfinished or missing or partial or annotation_blockers or input_blockers)
    return dict(all_experiments_verified=verified, training_complete=training_done,
                first_group_coverage_complete=baseline_complete, failed_jobs=failures,
                unfinished_jobs=unfinished, missing_reports=missing, incomplete_coverage_reports=partial,
                input_blockers=input_blockers, event_annotation_blockers=annotation_blockers,
                status='completed_verified' if verified else 'incomplete_with_explicit_dependencies')


def build(states):
    reports = {}
    for group, conditions in CONDITIONS.items():
        for condition in conditions:
            for model in (MEMORY_MODELS if group == 2 else MODELS):
                path = OUT/f'group{group}'/condition/model/'metrics.json'
                if path.exists():reports[f'group{group}/{condition}/{model}']=json.loads(path.read_text())
    baseline = OUT/'group1/progress-report.json'
    first = json.loads(baseline.read_text()) if baseline.exists() else {}
    readiness_path = EVAL/'conditions/readiness.json'
    readiness = json.loads(readiness_path.read_text()) if readiness_path.exists() else {}
    comparison_complete=first.get('all_seven_models_comparison_complete',first.get('all_seven_models_complete',False))
    result = assess(states,json.loads((BASE/'status.json').read_text()),readiness,reports,comparison_complete)
    save(OUT/'completion-audit.json',dict(result,updated=time.time()))
    return result
