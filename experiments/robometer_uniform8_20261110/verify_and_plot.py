"""Gate the next task on all epoch archives, final metrics and complete plots."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from safetensors.torch import load_file
import torch


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(config):
    output=Path(config['output_dir']);setup=Path(config['setup_dir']);archives=[]
    for epoch in [1,2,3]:
        step=epoch*config['steps_per_epoch'];p=output/f'checkpoint-{step}'
        receipt=read(p/'complete.json');state=read(p/'trainer_state.json')
        assert receipt['complete'] and receipt['saved_before_validation'] and receipt['step']==step
        assert state['global_step']==step and abs(state['epoch']-epoch)<1e-8
        assert read(p/'run_config.json')==config
        hashes={name:sha(p/name) for name in receipt['files']}
        for name in ['adapter_model.safetensors','progress_head.safetensors']:
            assert all(torch.isfinite(v).all() for v in load_file(p/name).values())
        archives.append(dict(epoch=epoch,checkpoint=str(p),files_sha256=hashes))
    for rank in [0,1]:
        a=read(setup/f'initialization-rank{rank}.json')
        assert a['strict_original_weight_load'] and a['official_progress_head_exactly_preserved'] and a['fresh_lora_B_all_zero']
        assert a['initial_checkpoint'] is None and a['official_revision']==config['initial_model_revision']
    report=read(output/'report.json');checkpoint=str(output/f"checkpoint-{config['expected_steps']}")
    assert report['status']=='completed' and report['final_checkpoint']==checkpoint
    preds=[json.loads(x) for x in (output/'evaluation/predictions.jsonl').read_text().splitlines()]
    expected=[json.loads(x) for x in Path(config['validation_data']).read_text().splitlines()]
    assert len(preds)==len(expected)==config['expected_validation_samples']
    grouped=defaultdict(list)
    for p,r in zip(preds,expected,strict=True):
        assert p['id']==r['id'] and p['source_frame_index']==r['source_frame_index']
        assert abs(p['target_pt']-r['pt'])<1e-6 and np.isfinite(p['predicted_pt'])
        grouped[p['episode_id']].append(p)
    assert len(grouped)==config['expected_validation_episodes']
    episodes=[];correlations=[]
    for eid,rows in sorted(grouped.items()):
        a=np.array([r['predicted_pt'] for r in rows]);b=np.array([r['target_pt'] for r in rows])
        corr=None if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0 else float(np.corrcoef(a,b)[0,1])
        if corr is not None:correlations.append(corr)
        episodes.append(dict(episode_id=eid,samples=len(rows),mae_pp=float(np.abs(a-b).mean()*100),pearson=corr))
    mae=float(np.mean([abs(r['predicted_pt']-r['target_pt']) for r in preds])*100)
    corr=float(np.mean(correlations)) if correlations else None
    metrics=read(output/'evaluation/metrics.json')
    assert abs(metrics['eval_mae_pp']-mae)<1e-5
    assert metrics['eval_pearson_undefined_episodes']==len(episodes)-len(correlations)
    if corr is not None:assert abs(metrics['eval_episode_mean_pearson']-corr)<1e-8
    life=[json.loads(x) for x in (output/'lifecycle.jsonl').read_text().splitlines()]
    validations=[r for r in life if r['event']=='validation_started']
    assert validations and all(r['step']==config['expected_steps'] for r in validations)
    assert any(r['event']=='checkpoint_saved' and r['step']==config['expected_steps'] and r['time']<=validations[0]['time'] for r in life)
    verified=dict(checkpoint=checkpoint,coverage_complete=True,temporal_sampling=config['temporal_sampling'],
                  overall=dict(episodes=len(episodes),frames=len(preds),mae_pp=mae,pearson=corr,
                               pearson_undefined_episodes=len(episodes)-len(correlations)),episode_metrics=episodes,
                  protocol=dict(mae='frame equal',pearson='episode equal; constant sequences undefined'))
    write(output/'evaluation/verified-metrics.json',verified)
    write(setup/'checkpoint-verification.json',dict(status='passed',archives=archives))
    return verified,grouped


def plot(config,verified,grouped):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    from matplotlib.ticker import PercentFormatter
    from PIL import Image
    font=Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    output=Path(config['output_dir'])/'evaluation/curves';output.mkdir(exist_ok=True)
    sources=read(Path(config['cache_setup_dir'])/'source/validation.sample_files.json');panels=[]
    for folder in sources:
        source=read(Path(folder)/'summary.json')['source'];eid=source['episode_id']
        assert source['task']==config['task'] and eid in grouped
        text=Path(source['progress_csv']).read_bytes();assert hashlib.sha256(text).hexdigest()==source['progress_sha256']
        dense=list(csv.DictReader(io.StringIO(text.decode())));gt=np.array([float(r['pt']) for r in dense])
        assert len(dense)==source['frames'] and source['fps']==30
        rows=sorted(grouped[eid],key=lambda r:r['source_frame_index'])
        frames=np.array([r['source_frame_index'] for r in rows]);pred=np.array([r['predicted_pt'] for r in rows])
        expected=list(range(0,len(gt),6))
        if expected[-1]!=len(gt)-1:expected.append(len(gt)-1)
        assert frames.tolist()==expected and np.allclose(gt[frames],[r['target_pt'] for r in rows],atol=1e-6,rtol=0)
        panels.append((eid,gt,frames,pred))
    assert len(panels)==config['expected_validation_episodes']
    def panel(ax,item):
        eid,gt,frames,pred=item;t=np.arange(len(gt))/30
        ax.plot(t,gt,color='#1678b5',lw=1.8,label='真实进度')
        ax.step(frames/30,pred,where='post',color='#e87a24',lw=1.4,label='预测进度')
        m=next(r for r in verified['episode_metrics'] if r['episode_id']==eid)
        ax.set_title(f"{eid.split('/')[-1]} · MAE {m['mae_pp']:.2f} pp",loc='left')
        ax.set(xlabel='视频时间 / 秒',ylabel='任务进度',ylim=(-.035,1.045),xlim=(0,t[-1]))
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.18)
        ax.legend(loc='upper left',fontsize=9)
    fig,axes=plt.subplots(math.ceil((len(panels)+1)/2),2,figsize=(14,3.2*math.ceil((len(panels)+1)/2)))
    for ax,item in zip(axes.flat,panels):panel(ax,item)
    for ax in list(axes.flat)[len(panels):]:ax.axis('off')
    m=verified['overall'];corr='未定义' if m['pearson'] is None else f"{m['pearson']:.4f}"
    list(axes.flat)[len(panels)].text(.03,.85,f"{m['episodes']} 条轨迹 · {m['frames']} 个观测\nMAE {m['mae_pp']:.2f} pp\n逐轨迹平均 Pearson {corr}\n首帧至当前均匀采样 · 右侧末帧补齐\n5 FPS 预测，帧间保持，无平滑",va='top',linespacing=1.8)
    fig.suptitle(f"Robometer · {config['title']} · 均匀历史窗口",fontsize=20)
    fig.text(.5,.947,f"官方原始权重独立微调 3 epoch · {Path(verified['checkpoint']).name} · 无人工增强",ha='center')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(output/'validation_overview.png',dpi=150);plt.close(fig)
    for item in panels:
        fig,ax=plt.subplots(figsize=(11,4.5));panel(ax,item);fig.tight_layout()
        fig.savefig(output/(item[0].split('/')[-1]+'.prediction_curve.png'),dpi=150);plt.close(fig)
    paths=sorted(output.glob('*.png'));assert len(paths)==len(panels)+1
    for p in paths:
        with Image.open(p) as im:im.verify()
        assert p.stat().st_size>10000
    manifest=dict(status='passed',checkpoint=verified['checkpoint'],episodes=len(panels),samples=m['frames'],
                  labels_checked_against_original_dense_pt=True,smoothing=False,files={p.name:sha(p) for p in paths})
    write(output/'manifest.json',manifest)
    with zipfile.ZipFile(output/'validation_images.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.name)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=read(a.config);verified,grouped=verify(config);print(json.dumps(plot(config,verified,grouped),ensure_ascii=False))
