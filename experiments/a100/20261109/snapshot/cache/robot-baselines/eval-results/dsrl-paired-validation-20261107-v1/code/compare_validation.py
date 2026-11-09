"""Run missing paired evaluations and produce an auditable comparison."""
import json
import shutil
import subprocess
import time
from pathlib import Path

BASE=Path('/cache/robot-baselines/eval-results')
FORMAL=BASE/'dsrl-B-terminal-formal-20261107-v1'
PILOT=BASE/'dsrl-B-terminal-20261107-v1'
ROOT=BASE/'dsrl-paired-validation-20261107-v1'


def dump(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def records(path):
    rows=[json.loads(x) for x in (path/'learner/events.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['event']=='episode_complete' and r['mode']=='eval']
    assert len(rows)==20 and len({r['seed'] for r in rows})==20
    assert len({r['updates'] for r in rows})==1
    return {r['seed']:r for r in rows}


def layouts(path):
    out={}
    for file in (path/'eval/evidence').glob('*/evidence.json'):
        evidence=json.loads(file.read_text())
        with (file.parent/'states.jsonl').open() as f: state=json.loads(next(f))
        assert state['action_step']==0
        out[evidence['seed']]={k:state[k] for k in ['poses','open_order']}
    assert len(out)==20
    return out


def close(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)):return abs(a-b)<1e-6
    return a==b


def main():
    ROOT.mkdir()
    shutil.copytree(Path(__file__).resolve().parent,ROOT/'code',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    paths={'original':ROOT/'original','dsrl50':ROOT/'dsrl50',
           'dsrl100':FORMAL/'stage-100','dsrl150':FORMAL/'stage-150','dsrl200':ROOT/'dsrl200'}
    try:
        dump(ROOT/'status.json',{'status':'waiting_for_final_test'})
        while True:
            status=json.loads((FORMAL/'status.json').read_text())
            if status['status']=='complete':break
            if status['status']=='failed':raise RuntimeError('Existing campaign failed; inspect before evaluation')
            time.sleep(30)
        # Ensure the training launcher's final cleanup has released the GPU services.
        time.sleep(30)
        reference=records(paths['dsrl100'])
        assert set(reference)==set(range(350000,350020))
        for name,checkpoint in [('original',None),('dsrl50',PILOT/'learner/last.pt'),
                                ('dsrl200',FORMAL/'stage-200/learner/last.pt')]:
            dump(ROOT/'status.json',{'status':'running','current':name})
            args=['bash',str(ROOT/'code/launch.sh'),'--reward-mode','terminal',
                  '--policy-gpu','0','--sim-gpu','1','--port','18640',
                  '--batch-size','64','--learning-starts','64','--updates-per-step','2',
                  '--eval-episodes','20','--eval-seed','350000',
                  '--evaluation-kind','trained' if checkpoint else 'original',
                  '--output',str(paths[name])]
            if checkpoint:args+=['--resume',str(checkpoint)]
            with (ROOT/(name+'-controller.log')).open('w') as log:
                subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
        allrows={k:records(p) for k,p in paths.items()}
        initial={k:layouts(p) for k,p in paths.items()}
        baseline=allrows['original']
        for name,rows in allrows.items():
            assert rows.keys()==baseline.keys(),name
            for seed,row in rows.items():
                assert row['prompt']==baseline[seed]['prompt'],(name,seed,'prompt')
                assert close(initial[name][seed],initial['original'][seed]),(name,seed,'layout')
        summary=[]
        for name,rows in allrows.items():
            shared=[s for s in rows if rows[s]['success'] and baseline[s]['success']]
            summary.append(dict(model=name,n=20,successes=sum(r['success'] for r in rows.values()),
                mean_steps=sum(r['steps'] for r in rows.values())/20,
                wins=sum(r['success'] and not baseline[s]['success'] for s,r in rows.items()),
                losses=sum(not r['success'] and baseline[s]['success'] for s,r in rows.items()),
                jointly_successful_n=len(shared),
                steps_delta_on_joint_success=sum(rows[s]['steps']-baseline[s]['steps'] for s in shared)/len(shared) if shared else None))
        dump(ROOT/'comparison.json',dict(summary=summary,episodes=allrows,
            alignment='seed, prompt, initial object poses and open_order verified',
            limitation='20 scenes, one rollout per model per scene; not a repeated-trial or recovery-start evaluation'))
        lines=['# 同场景验证结果','', '| 模型 | 成功数 | 平均控制步 | 对原始模型胜/负 |','|---|---:|---:|---:|']
        lines += [f"| {r['model']} | {r['successes']}/20 | {r['mean_steps']:.1f} | {r['wins']}/{r['losses']} |" for r in summary]
        lines += ['', '已核对实际 seed、指令、初始物体位姿及颜色揭盖顺序。原始模型使用默认噪声，DSRL 使用确定性控制器。',
                  '每个模型每场景仅执行一次；这不是多随机种子或异常起点 recovery 结论。平均步数包含失败超时，请同时参考 JSON 中共同成功场景的步数差。']
        (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
        dump(ROOT/'status.json',{'status':'complete','summary':summary})
    except BaseException as error:
        dump(ROOT/'status.json',{'status':'failed','error':repr(error)})
        raise


if __name__=='__main__':main()
