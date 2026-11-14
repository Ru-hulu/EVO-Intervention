"""Follow-up job definitions and dependency gates; no direct GPU launching."""
import json
from pathlib import Path
import time

from .eval_common import EVAL, OUT
from .experiment_inputs import MODELS, MEMORY_MODELS, CONDITIONS
from .experiment_worker import name_for
from .prepare_cache import BASE, save, sha
from .model_cancellation import cancelled_models

DONE = {'completed', 'completed_with_input_blockers'}
OFFICIAL = tuple(m for m in MODELS if m != 'forest')
COUNT_CHECKPOINT = BASE/'count-priority-stop/1794625292452731696/count-resume.pt'
COUNT_SHA = 'd6c6b2a6f01bef1f76bbf91f344e258f965969a87de4018f09b3637402e3a1e3'


def planned_followups():
    jobs = []
    for group, conditions in CONDITIONS.items():
        for condition in conditions:
            for model in (MEMORY_MODELS if group == 2 else MODELS):
                suites = (('Recover',) if group == 3 else ('State','Order') if group == 4 else ('State','Order','Count')) if model == 'forest' else (None,)
                for suite in suites:
                    for shard in range(4):
                        args = [model, '--group', str(group), '--condition', condition, '--shard', str(shard)]
                        if suite: args += ['--suite', suite]
                        python = ('/cache/robot-baselines/envs/vlac/bin/python' if model == 'vlac-2b' else
                                  '/cache/cvpzhr/envs/forest-labels/bin/python' if model == 'forest' else
                                  '/cache/cvpzhr/envs/robometer/bin/python' if model == 'robometer-4b' else
                                  '/cache/robot-baselines/envs/procvlm/bin/python')
                        jobs.append(dict(name=name_for(model, group, condition, shard, suite), model=model,
                            group=group, condition=condition, shard=shard, suite=suite, python=python,
                            module='experiments.prm_five_groups_20261114.experiment_worker', args=args,
                            memory_fraction=.30, batch=1,
                            canary_name=name_for(model, group, condition, 0, suite)))
    assert len(jobs) == len({j['name'] for j in jobs}) == 440
    return jobs


def followup_ready(job, readiness, priority_cleared, evaluation=EVAL):
    if not priority_cleared:
        return False, 'first_group_official_priority'
    state = readiness.get('conditions', {}).get(f"group{job['group']}/{job['condition']}", {})
    if state.get('status') != 'ready':
        return False, state.get('status', 'waiting_for_input_preparation')
    if job['shard']:
        path = Path(evaluation)/'canaries'/f"{job['canary_name']}.json"
        if not path.exists():
            return False, 'waiting_for_real_episode_canary'
        proof = json.loads(path.read_text())
        assert proof['model'] == job['model'] and proof['group'] == job['group']
        assert proof['condition'] == job['condition'] and proof['suite'] == job['suite']
        assert proof['manifest_sha256'] == state['manifest_sha256']
        assert proof['status'] == 'passed' and proof['complete_real_episode']
        assert sha(proof['result_path']) == proof['result_sha256']
    return True, None


def official_jobs_terminal(states):
    cancelled = cancelled_models()
    return all(states.get(f'{model}-group1-{s}', {}).get('status') in
               ({'cancelled_by_user'} if model in cancelled else DONE)
               for model in OFFICIAL for s in range(4))


def verify_official_handoff(report, states):
    assert official_jobs_terminal(states)
    cancelled = cancelled_models()
    for model in OFFICIAL:
        if model in cancelled:continue
        result = report['models'][model]
        blocked = {b['key'] for s in range(4) for b in states[f'{model}-group1-{s}'].get('input_blockers', [])}
        assert set(result['missing_episodes']) == blocked, f'Unexplained missing predictions for {model}'
        assert result['overall']['completed_episodes'] + len(blocked) == 385
    return {model: dict(completed=report['models'][model]['overall']['completed_episodes'],
                       missing=report['models'][model]['missing_episodes'],
                       invalid_frames=report['models'][model]['overall']['invalid_frames']) for model in OFFICIAL if model not in cancelled}


def finalize_official_handoff(states):
    from .group1_report import build
    report = build(); models = verify_official_handoff(report, states)
    receipt = dict(status='passed', at=time.time(), scope='all executable group-1 official predictions audited',
                   cancelled_models=sorted(cancelled_models()),
                   models=models, report_sha256=sha(OUT/'group1/progress-report.json'),
                   report_path=str(OUT/'group1/progress-report.json'), plan_sha256=report['plan_sha256'])
    # Preserve this exact audit independently of later group-1 FoREST refreshes.
    save(EVAL/'group1-official-final-report.json', report)
    receipt['report_path'] = str(EVAL/'group1-official-final-report.json')
    receipt['report_sha256'] = sha(receipt['report_path'])
    save(EVAL/'group1-official-handoff.json', receipt)
    return receipt


def release_count_reservation(training):
    policy_path = BASE/'evaluation-priority.json'
    policy = json.loads(policy_path.read_text())
    if policy.get('status') != 'active':
        return True
    proof_path = EVAL/'group1-official-handoff.json'
    if not proof_path.exists():
        return False
    proof = json.loads(proof_path.read_text())
    assert proof['status'] == 'passed' and sha(proof['report_path']) == proof['report_sha256']
    capability = BASE/'controller-capabilities.json'
    if not capability.exists(): return False
    caps = json.loads(capability.read_text())
    if caps.get('pid') != training.get('controller_pid') or not caps.get('verified_count_resume'):
        return False
    assert policy['user_authorized'] and policy['reserved_gpus'] == [2,3] and policy['paused_suites'] == ['Count']
    assert sha(COUNT_CHECKPOINT) == COUNT_SHA
    config = BASE/'Count/training.config.json'
    launch = json.loads((BASE/'Count/launch.json').read_text())
    assert sha(config) == launch['config_sha256']
    save(BASE/'Count/resume-authorization.json', dict(status='authorized_after_group1_official_audit',
        checkpoint=str(COUNT_CHECKPOINT), checkpoint_sha256=COUNT_SHA, step=1200, gpus=[2,3],
        config_sha256=sha(config), handoff_sha256=sha(proof_path), authorized_by='existing user instruction', at=time.time()))
    save(BASE/'evaluation-priority-history'/f'{time.time_ns()}.json', policy)
    save(policy_path, dict(policy, status='released_after_group1_official_audit', released=time.time(),
                          handoff_sha256=sha(proof_path)))
    return True
