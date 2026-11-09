
"""Detached one-shot evaluation and aggregation; no scheduled monitor."""
import fcntl, json, math, os, subprocess, sys, traceback
from pathlib import Path
from eval_common import write, now, sha, summarize
ROOT=Path(__file__).resolve().parent
def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]
def main():
    run=json.loads((ROOT/'run.json').read_text());output=Path(run['output_dir'])
    state=dict(status='starting',started_at=now(),pid=os.getpid(),output_dir=str(output),
               checkpoint=run['checkpoint'],epoch=1,step=9967,samples=78554,episodes=417,gpu_indices=[0,1,2,3])
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(ROOT/'status.json',state)
    handles=[]
    try:
        for path,expected in run['protected_hashes'].items():
            assert sha(path)==expected,'Input changed: '+path
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError('A GPU compute task appeared before launch: '+active)
        output.mkdir(parents=True,exist_ok=False)
        jobs=[]
        for rank in range(4):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
            log=(ROOT/f'worker-{rank}.log').open('x');handles.append(log)
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py'),'--rank',str(rank)],
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            jobs.append(p)
        state.update(status='running',worker_pids=[p.pid for p in jobs]);write(ROOT/'status.json',state)
        codes=[p.wait() for p in jobs]
        state['worker_returncodes']=codes
        if any(codes):raise RuntimeError('Evaluation worker failed; see worker status/logs; all predictions preserved')
        state.update(status='aggregating');write(ROOT/'status.json',state)
        for path,expected in run['protected_hashes'].items():
            assert sha(path)==expected,'Input changed during evaluation: '+path
        expected=rows(run['prepared_rows'])
        expected_by_key={(r['episode_id'],r['sample_index']):r for r in expected}
        assert len(expected_by_key)==len(expected)==78554
        def validate(records):
            seen=set()
            for r in records:
                key=(r['episode_id'],r['sample_index'])
                assert key in expected_by_key and key not in seen,(key,'missing/duplicate')
                seen.add(key);e=expected_by_key[key]
                assert math.isfinite(r['predicted_pt'])
                assert r['target_pt']==e['pt'],(key,'PT mismatch')
                for k in ['kind','source_sample_index','anchor_sample_index','frame_index']:
                    assert r[k]==e[k],(key,k)
            assert len(records)==78554 and seen==set(expected_by_key)
            return records
        records=validate([r for rank in range(4) for r in rows(output/f'predictions-{rank}.jsonl')])
        bykey={(r['episode_id'],r['sample_index']):r for r in records}
        records=[bykey[(r['episode_id'],r['sample_index'])] for r in expected]
        with (output/'predictions.jsonl').open('x') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        result=summarize(records)
        assert result['overall']['episodes']==417 and len(result['by_task'])==47
        result.update(status='completed',completed_at=now(),checkpoint=run['checkpoint'],epoch=1,step=9967,
            plan_sha256=run['plan_sha256'],split='augmented_validation',samples=78554,episodes=417,tasks=47,
            input_contract='grm-forward-same-episode-goal-8images-v1',uses_future_goal_reference=True,
            metric='Episode-equal Pearson(predicted PT, target PT); frame-weighted MAE in percentage points',
            invalid_generations=0,augmentation=run['augmentation'],coverage_verified=True)
        baseline=json.loads(Path(run['baseline_metrics']).read_text())
        assert baseline['plan_sha256']==run['plan_sha256']
        models={}
        for name in ['forest','robometer']:
            validate(rows(Path(run['baseline_root'])/name/'predictions.jsonl'))
            models[name]=baseline['models'][name]
        models['robobrain25']=result
        comparison=dict(status='completed',completed_at=now(),plan_sha256=run['plan_sha256'],
            episodes=417,tasks=47,samples=78554,all_prediction_coverage_and_targets_match=True,
            models=models,checkpoints=run['checkpoints'],
            input_note='RoboBrain uses the true same-episode final frame as a reference; Forest and Robometer do not.')
        write(output/'metrics.json',result)
        write(output/'comparison.json',comparison)
        lines=['# Forest / Robometer / RoboBrain2.5: epoch 1, shared augmented validation','',
            '417 episodes, 47 tasks, 78,554 observations; seed 42; inserted static and jitter frames retain anchor PT.',
            'MAE is frame weighted in percentage points. Pearson is computed per episode, then averaged equally.',
            'RoboBrain uses the true same-episode final frame as a goal reference; the other two models do not.','',
            '| Suite | Forest MAE | Robometer MAE | RoboBrain MAE | Forest Pearson | Robometer Pearson | RoboBrain Pearson |',
            '|---|---:|---:|---:|---:|---:|---:|']
        def fmt(x):return 'undefined' if x is None else f'{x:.6f}'
        for suite in ['Count','Order','State','overall']:
            vv=[v['overall'] if suite=='overall' else v['by_suite'][suite] for v in models.values()]
            lines.append('| '+suite+' | '+' | '.join([fmt(v['mae_pp']) for v in vv]+[fmt(v['pearson_episode_mean']) for v in vv])+' |')
        lines+=['','Undefined Pearson episodes: '+', '.join(n+'='+str(len(v['overall']['undefined_pearson_episodes'])) for n,v in models.items())+'.',
            '', 'Checkpoint paths and all per-task/per-episode results are in comparison.json.','']
        (output/'comparison.md').write_text('\n'.join(lines))
        write(output/'provenance.json',run)
        state.update(status='completed',finished_at=now(),metrics_path=str(output/'metrics.json'),
                     comparison_path=str(output/'comparison.md'),overall=result['overall'])
        write(ROOT/'status.json',state)
        print(json.dumps({k:state[k] for k in ['status','finished_at','overall']}),flush=True)
    except BaseException:
        state.update(status='failed',finished_at=now(),error=traceback.format_exc())
        write(ROOT/'status.json',state);raise
    finally:
        for f in handles:f.close()
if __name__=='__main__':main()
