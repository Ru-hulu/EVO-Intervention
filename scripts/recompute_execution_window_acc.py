"""Rescore saved absolute XYZ chunks over only the next execution window.

No velocity/range normalization. Output unit is mm, using mean absolute XYZ
error, not Euclidean distance. Original paper ACC is left unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def execution_window_error_mm(previous, current, *, stride=6, window=6):
    old, new = np.asarray(previous, dtype=np.float64), np.asarray(current, dtype=np.float64)
    if old.shape != new.shape or old.ndim != 2 or old.shape[1] != 3:
        raise ValueError('Expected matching [H,3] absolute XYZ in meters')
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Non-finite XYZ')
    if not isinstance(stride, int) or not isinstance(window, int) or not (0 < stride < len(old)) or not (0 < window <= len(old)-stride):
        raise ValueError('Need 0 < stride and 0 < window <= H-stride')
    return float(np.abs(old[stride:stride+window] - new[:window]).mean() * 1000.0)


def smooth(raw, alpha):
    previous = 0.0
    values = []
    for value in raw:
        previous = alpha * previous + (1-alpha) * float(value)
        values.append(previous)
    return np.asarray(values)


def describe(values):
    return {'median':float(np.median(values)), 'p95':float(np.quantile(values,.95)), 'max':float(np.max(values))}


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def verify_definition():
    t=np.arange(50,dtype=float)[:,None]
    previous=np.concatenate([t,2*t,-t],axis=1)*.01
    current=previous+np.array([.06,.12,-.06])
    assert execution_window_error_mm(previous,current)<1e-12
    shifted=current+.003
    assert abs(execution_window_error_mm(previous,shifted)-3)<1e-12
    previous[12:]+=1000
    shifted[6:]-=1000
    assert abs(execution_window_error_mm(previous,shifted)-3)<1e-12
    assert abs(smooth([3],.9)[0]-.3)<1e-12


def run(predictions: Path, reference: Path, output: Path, stride=6, window=6, alpha=.9):
    if not (0 <= alpha < 1): raise ValueError('alpha must be in [0,1)')
    verify_definition()
    if (output/'RESULTS.json').exists(): raise FileExistsError('Preserve earlier results; choose a new output directory')
    identities=[('hf_success',10),('hf_success',13),('local_failure',2),('local_failure',3)]
    statistics, verification, curves = [], [], []
    for group, number in identities:
        key=f'{group}/episode_{number:06d}'
        source=predictions/(key+'.npz')
        old_summary=json.loads((reference/'reports'/key/'summary.json').read_text(encoding='utf-8-sig'))
        with (reference/'reports'/key/'scores.csv').open(encoding='utf-8-sig',newline='') as f: old_rows=list(csv.DictReader(f))
        with source.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        with np.load(source,allow_pickle=False) as z:
            frames=z['frame_indices'].copy()
            times=z['timestamps'].astype(np.float64)
            assert str(z['fingerprint'].item())==old_summary['fingerprint'], 'Prediction provenance mismatch'
            assert len(frames)==old_summary['observations']==len(old_rows)
            assert np.all(np.diff(frames)==stride), 'Saved predictions do not have the requested interval'
            fps=old_summary['provenance']['fps']
            np.testing.assert_allclose(times,frames/fps,rtol=0,atol=1e-3)
            horizon=z['positions_left'].shape[1]
            scores, whole, old_norm, checks = {}, {}, {}, {}
            for arm in ['left','right']:
                xyz=z['positions_'+arm].astype(np.float64)
                raw=np.asarray([execution_window_error_mm(a,b,stride=stride,window=window) for a,b in zip(xyz[:-1],xyz[1:])])
                # Independent vectorized implementation verifies all 3860 pairs.
                expected=np.abs(xyz[:-1,stride:stride+window,:]-xyz[1:,:window,:]).sum(axis=(1,2))*1000/(window*3)
                np.testing.assert_allclose(raw,expected,rtol=1e-12,atol=1e-12)
                ema=smooth(raw,alpha)
                full_diff=np.abs(xyz[:-1,stride:,:]-xyz[1:,:horizon-stride,:])
                full_raw=full_diff.mean(axis=(1,2))*1000
                old_raw=(full_diff.mean(axis=1)/np.maximum(np.ptp(xyz[1:,:horizon-stride,:],axis=1),.01)).mean(axis=1)
                old_ema=smooth(old_raw,alpha)
                np.testing.assert_allclose(old_raw,[float(r['acc_raw_'+arm]) for r in old_rows[1:]],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(old_ema,[float(r['acc_'+arm]) for r in old_rows[1:]],rtol=1e-10,atol=1e-10)
                scores[arm]=(raw,ema)
                whole[arm]=smooth(full_raw,alpha)
                old_norm[arm]=old_ema
                checks[arm]={'new_formula_max_error_mm':float(np.max(np.abs(raw-expected))), 'old_formula_max_error':float(np.max(np.abs(old_ema-np.asarray([float(r['acc_'+arm]) for r in old_rows[1:]]))))}
        rows=[]
        for i,(frame,stamp) in enumerate(zip(frames,times)):
            row={'episode_id':key,'frame_index':int(frame),'time_s':float(stamp)}
            for arm in ['left','right']:
                row['acc_raw_'+arm+'_mm']=float(scores[arm][0][i-1]) if i else None
                row['acc_'+arm+'_mm']=float(scores[arm][1][i-1]) if i else None
                row['full_window_'+arm+'_mm']=float(whole[arm][i-1]) if i else None
            row['acc_max_mm']=max(row['acc_left_mm'],row['acc_right_mm']) if i else None
            rows.append(row)
        dest=output/'reports'/key
        dest.mkdir(parents=True,exist_ok=True)
        with (dest/'scores.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        maximum=np.maximum(scores['left'][1],scores['right'][1])
        raw_max=np.maximum(scores['left'][0],scores['right'][0])
        full_max=np.maximum(whole['left'],whole['right'])
        normalized_max=np.maximum(old_norm['left'],old_norm['right'])
        peak=int(np.argmax(maximum))+1
        item={'episode_id':key,'group':group,'duration_s':old_summary['duration_s'],'valid_samples':len(maximum),
              **{'ema_'+k+'_mm':v for k,v in describe(maximum).items()},
              'raw_p95_mm':float(np.quantile(raw_max,.95)), 'raw_max_mm':float(raw_max.max()),
              'ema_peak_time_s':float(times[peak]),'ema_peak_arm':'left' if scores['left'][1][peak-1]>=scores['right'][1][peak-1] else 'right',
              'full_window_ema_p95_mm':float(np.quantile(full_max,.95)),
              'original_normalized_ema_p95':float(np.quantile(normalized_max,.95))}
        statistics.append(item)
        metadata={'metric':'execution_window_xyz_mae','unit':'mm','normalized':False,'action_horizon':horizon,'query_stride':stride,
                  'comparison_steps':window,'previous_slice':[stride,stride+window],'current_slice':[0,window],
                  'query_hz':fps/stride,'video_fps':fps,'ema_alpha':alpha,'ema_initialization':0,
                  'first_sample':'missing','coordinate_type':'absolute end-effector XYZ in meters, converted from model joint predictions by FK',
                  'distance':'mean absolute error over XYZ and time, not Euclidean norm',
                  'reference_prediction_sha256':sha,'reference_prediction_fingerprint':old_summary['fingerprint'],
                  'prediction_reuse':True,'new_model_inference':False,'terminal_window':'fixed predicted window; not an assertion that all future steps were actually executed',
                  'source_outcome_label':old_summary['provenance'].get('source_outcome_label','unknown'),**item}
        json_file(dest/'summary.json',metadata)
        verification.append({'episode_id':key,'observations':len(frames),'valid_pairs':len(maximum),'sha256':sha,'arms':checks})
        curves.append({'episode_id':key,'times':times,'scores':scores,'whole':whole})
        print(json.dumps(item,ensure_ascii=False),flush=True)
    json_file(output/'RESULTS.json',{'method':'execution-window unnormalized ACC adaptation','units':'mm','stride':stride,'window':window,'alpha':alpha,'episodes':statistics})
    json_file(output/'VERIFICATION.json',{'passed':True,'synthetic_alignment_and_tail_invariance':True,'all_source_scores_reproduced':True,'total_valid_pairs':sum(x['valid_pairs'] for x in verification),'episodes':verification})
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(statistics[0]));writer.writeheader();writer.writerows(statistics)
    plot(output,curves,statistics)


def plot(output, curves, statistics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#078575','#d57721']
    fig, axes=plt.subplots(4,2,figsize=(15,11),sharey='col',layout='constrained')
    for i,item in enumerate(curves):
        t=item['times'][1:]
        for side,arm in enumerate(['left','right']):
            for col,index in [(0,0),(1,1)]:
                axes[i,col].step(t,item['scores'][arm][index],where='post',color=colors[side],lw=.8,label=arm)
        for col in range(2):
            axes[i,col].set_title(item['episode_id']+(' | raw' if col==0 else ' | EMA alpha=0.9'),loc='left',fontsize=10)
            axes[i,col].set_ylabel('XYZ MAE (mm)');axes[i,col].set_ylim(bottom=0);axes[i,col].grid(alpha=.2)
            axes[i,col].set_xlabel('Recorded episode time (s)')
    # Shared axes must be sized after every episode has been added. Calling
    # set_ylim(bottom=0) on the first row disables later automatic expansion.
    for col in [0,1]:
        upper=max(float(item['scores'][arm][col].max()) for item in curves for arm in ['left','right'])*1.05
        for ax in axes[:,col]:ax.set_ylim(0,upper)
    axes[0,0].legend(ncol=2);axes[0,1].legend(ncol=2)
    fig.suptitle('Next-execution-window consistency: old[6:12] vs new[0:6], no normalization, 5 Hz',fontsize=14)
    fig.savefig(output/'curves.png',dpi=160);fig.savefig(output/'curves.svg');plt.close(fig)
    svg=output/'curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    x=np.arange(4);w=.34
    ax.bar(x-w/2,[s['full_window_ema_p95_mm'] for s in statistics],w,label='Full overlap (44 steps), no normalization',color='#9aa8b8')
    ax.bar(x+w/2,[s['ema_p95_mm'] for s in statistics],w,label='Next execution window (6 steps), no normalization',color='#078575')
    ax.set_xticks(x,['Demo 10','Demo 13','Failure 2','Failure 3'])
    ax.set_ylabel('P95 of max(left EMA, right EMA) [mm]');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.legend(fontsize=9);ax.set_title('Window-length comparison using identical predictions and physical units')
    fig.savefig(output/'window_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=6)
    p.add_argument('--window',type=int,default=6)
    p.add_argument('--alpha',type=float,default=.9)
    p.add_argument('--plots-only',action='store_true')
    args=p.parse_args()
    if args.plots_only:
        output=args.output.resolve()
        statistics=json.loads((output/'RESULTS.json').read_text(encoding='utf-8'))['episodes']
        curves=[]
        for item in statistics:
            with (output/'reports'/item['episode_id']/'scores.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
            curves.append({'episode_id':item['episode_id'],'times':np.asarray([float(r['time_s']) for r in rows]),'scores':{arm:tuple(np.asarray([float(r[column]) for r in rows[1:]]) for column in ['acc_raw_'+arm+'_mm','acc_'+arm+'_mm']) for arm in ['left','right']}})
        plot(output,curves,statistics)
    else:
        if args.predictions is None or args.reference is None:p.error('--predictions and --reference are required for scoring')
        run(args.predictions.resolve(),args.reference.resolve(),args.output.resolve(),args.stride,args.window,args.alpha)
