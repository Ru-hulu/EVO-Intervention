"""Continue terminal-only DSRL through fixed training and evaluation stages."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/cache/robot-baselines/eval-results/dsrl-B-terminal-formal-20261107-v1')
SOURCE = Path(__file__).resolve().parent
PRIOR = Path('/cache/robot-baselines/eval-results/dsrl-B-terminal-20261107-v1')
STAGES = [
    dict(total_train=100, train_seed=331000, eval_seed=350000, eval_episodes=20, evaluation='validation'),
    dict(total_train=150, train_seed=332000, eval_seed=350000, eval_episodes=20, evaluation='validation'),
    dict(total_train=200, train_seed=333000, eval_seed=360000, eval_episodes=50, evaluation='final_test'),
]


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    prior_result = json.loads((PRIOR/'result.json').read_text())
    assert prior_result['status'] == 'complete' and len(prior_result['train']) == 50
    assert prior_result['reward_mode'] == 'terminal'
    ROOT.mkdir()  # Never overwrite an existing campaign.
    shutil.copytree(SOURCE, ROOT/'code', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    resume = PRIOR/'learner/last.pt'
    manifest = dict(prior_run=str(PRIOR), prior_train_episodes=50, new_train_episodes=150,
                    total_train_episodes=200, stages=STAGES, reward_mode='terminal',
                    policy_gpu=0, sim_gpu=1, progress=False, checkpoint_selection='final, no validation selection')
    dump(ROOT/'manifest.json', manifest)
    completed = []
    try:
        for stage in STAGES:
            output = ROOT/('stage-'+str(stage['total_train']))
            dump(ROOT/'status.json', dict(status='running', current=stage, completed=completed))
            args = ['bash', str(ROOT/'code/launch.sh'), '--reward-mode', 'terminal',
                    '--policy-gpu', '0', '--sim-gpu', '1', '--port', '18540',
                    '--batch-size', '64', '--learning-starts', '64', '--updates-per-step', '2',
                    '--train-episodes', '50', '--eval-episodes', str(stage['eval_episodes']),
                    '--train-seed', str(stage['train_seed']), '--eval-seed', str(stage['eval_seed']),
                    '--resume', str(resume), '--output', str(output)]
            with (ROOT/('stage-'+str(stage['total_train'])+'-controller.log')).open('w') as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output/'result.json').read_text())
            assert result['status'] == 'complete' and result['checkpoint_reload_verified']
            assert len(result['train']) == 50 and result['eval_episodes'] == stage['eval_episodes']
            completed.append(dict(stage, train_successes=sum(e['success'] for e in result['train']),
                                  eval_successes=result['eval_successes'], run=str(output)))
            resume = output/'learner/last.pt'
        dump(ROOT/'status.json', dict(status='complete', completed=completed, final_checkpoint=str(resume)))
    except BaseException as error:
        dump(ROOT/'status.json', dict(status='failed', completed=completed, error=repr(error)))
        raise


if __name__ == '__main__':
    main()
