
"""Wait for the authorized predecessor; run four GPU lanes and verify complete paired predictions."""
import concurrent.futures,fcntl,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
from common import read,rows,write,now,sha,summarize
ROOT=Path(__file__).resolve().parent
def main():
    run=read(ROOT/'run.json');out=Path(run['output_dir'])
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='checking_inputs',pid=os.getpid(),started_at=now(),output_dir=str(out),
               gpu_policy='Wait for existing training, then use all four GPUs',samples=66891,episodes=417,tasks=47)
    write(ROOT/'status.json',state)
    try:
        for p,h in run['protected_hashes'].items():assert sha(p)==h,'Changed input: '+p
        while True:
            active=[]
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                except OSError:continue
                if run['wait_for_training_config'] in cmd and 'mem_lite.train_stage3' in cmd:active.append(int(p.name))
            gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            memory=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
            if not active and not gpu_processes and all(x<1024 for x in memory):break
            state.update(status='waiting_for_gpus',updated_at=now(),predecessor_pids=active,gpu_memory_mib=memory,
                         reason='Existing training or other GPU compute jobs still active')
            write(ROOT/'status.json',state);time.sleep(30)
        for p,h in run['protected_hashes'].items():assert sha(p)==h,'Changed input while waiting: '+p
        out.mkdir(parents=True,exist_ok=False)
        for kind in run['model_order']:(out/kind).mkdir()
        state.update(status='running',evaluation_started_at=now());write(ROOT/'status.json',state)
        def lane(rank):
            results={}
            for kind in run['model_order']:
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
                with (ROOT/f'{kind}-{rank}.log').open('x') as log:
                    p=subprocess.Popen([run['python'][kind],'-u',str(ROOT/'worker.py'),kind,'--rank',str(rank)],
                        cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    write(ROOT/f'lane-{rank}.json',dict(status='running',kind=kind,pid=p.pid,started_at=now(),completed=results))
                    code=p.wait();results[kind]=code
                    if code:raise RuntimeError(f'{kind} rank {rank} failed ({code}); predictions preserved')
            write(ROOT/f'lane-{rank}.json',dict(status='completed',finished_at=now(),completed=results))
            return results
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(lane,i) for i in range(4)]
            results=[f.result() for f in futures]
        state.update(status='aggregating',lane_results=results);write(ROOT/'status.json',state)
        expected=[r for rank in range(4) for r in rows(ROOT/f'meta-{rank}.jsonl')]
        bykey={(r['episode_id'],r['sample_index']):r for r in expected}
        assert len(bykey)==len(expected)==66891
        report=dict(status='completed',completed_at=now(),samples=66891,episodes=417,tasks=47,
            split='clean_validation',augmentation='none',fresh_inference=True,
            metric='Per-episode Pearson then equal mean; frame-weighted MAE in percentage points',
            checkpoints=run['checkpoints'],models={},all_prediction_coverage_and_targets_match=True,
            input_note='RoboBrain uses the true same-episode final frame as a goal reference; Forest and Robometer do not.')
        order={e['episode_id']:i for i,e in enumerate(run['episodes'])}
        for kind in run['model_order']:
            records=[r for rank in range(4) for r in rows(out/kind/f'predictions-{rank}.jsonl')]
            seen=set()
            for r in records:
                k=r['episode_id'],r['sample_index'];assert k in bykey and k not in seen;seen.add(k)
                m=bykey[k];assert r['target_pt']==m['pt'] and math.isfinite(r['predicted_pt'])
                for field in ['frame_index','source_frame_index','source_sample_index','anchor_sample_index','kind','task','suite']:
                    assert r[field]==m[field]
            assert len(records)==66891 and seen==set(bykey)
            records.sort(key=lambda r:(order[r['episode_id']],r['sample_index']))
            with (out/kind/'predictions.jsonl').open('x') as f:
                for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            result=summarize(records)
            assert result['overall']['episodes']==417 and len(result['by_task'])==47
            result.update(epoch=1,checkpoint=run['checkpoints'][kind]['path'],fresh_inference=True)
            write(out/kind/'metrics.json',result);report['models'][kind]=result
        for p,h in run['protected_hashes'].items():assert sha(p)==h,'Changed input during evaluation: '+p
        write(out/'metrics.json',report)
        write(out/'provenance.json',run)
        models=['forest','robometer','robobrain25']
        lines=['# Three-model epoch-1 clean validation','',
            'Fresh inference: 417 episodes, 47 tasks, 66,891 original observations; no static or jitter insertion.',
            'Pearson is episode-equal. MAE is frame-weighted in percentage points.',
            report['input_note'],'',
            '| Suite | Forest MAE | Robometer MAE | RoboBrain MAE | Forest Pearson | Robometer Pearson | RoboBrain Pearson |',
            '|---|---:|---:|---:|---:|---:|---:|']
        def fmt(x):return 'undefined' if x is None else f'{x:.6f}'
        for suite in ['State','Count','Order','overall']:
            vv=[report['models'][m]['overall'] if suite=='overall' else report['models'][m]['by_suite'][suite] for m in models]
            lines.append('| '+suite+' | '+' | '.join([fmt(v['mae_pp']) for v in vv]+[fmt(v['pearson_episode_mean']) for v in vv])+' |')
        lines+=['','Undefined Pearson episodes: '+', '.join(m+'='+str(len(report['models'][m]['overall']['undefined_pearson_episodes'])) for m in models)+'.','']
        (out/'comparison.md').write_text('\n'.join(lines))
        state.update(status='completed',finished_at=now(),metrics_path=str(out/'metrics.json'),comparison_path=str(out/'comparison.md'))
        write(ROOT/'status.json',state)
        print(json.dumps(state),flush=True)
    except BaseException:
        state.update(status='failed',finished_at=now(),error=traceback.format_exc());write(ROOT/'status.json',state);raise
if __name__=='__main__':main()
