"""Audit every saved prediction and render the unmodified five-model curves."""
import math
import time
from .common import *

DISPLAY = {'robometer-4b':'RoboMeter','procvlm-2b':'ProcVLM','robobrain25-4b':'RoboBrain',
           'dopamine-3b':'Robo-Dopamine','forest':'FoREST'}
ORDER = ['robometer-4b','procvlm-2b','robobrain25-4b','dopamine-3b','forest']
COLORS = {'robometer-4b':'#377eb8','procvlm-2b':'#39a477','robobrain25-4b':'#e69f00',
          'dopamine-3b':'#9467bd','forest':'#d94440'}


def main():
    from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics
    from experiments.prm_five_groups_20261114.prepare_cache import progress_values
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=read(ROOT/'inputs.json');digest=sha(ROOT/'inputs.json');row=manifest['records'][0]
    assert manifest['selection_sha256']==sha(ROOT/'selection.json')
    assert manifest['training_validation_disjoint'] and manifest['selected_without_model_predictions']
    for p,h in manifest['sources'].items():assert sha(p)==h,p
    assert sha(row['progress_path'])==row['progress_sha256']
    truth=progress_values(row);assert [truth[i] for i in row['frame_indices']]==row['target_pt']
    decoded_root=IMAGES/row['task_id']/row['sample_id'];decoded=read(decoded_root/'verification.json')
    assert sha(decoded_root/'verification.json')==row['image_receipt_sha256']
    for p,h in decoded['png_sha256'].items():assert sha(decoded_root/p)==h
    predictions={};metrics={};files={}
    for name in ORDER:
        info=manifest['model_checkpoints'][name];proof=read(info['proof']);validate_proof(name,proof,info['config'])
        assert sha(info['proof'])==info['proof_sha256']
        p=Path(info['checkpoint']);assert sha(p if name=='forest' else p/'trainable-state.pt')==info['checkpoint_sha256']
        result_file=result_path(name,row['key']);result=read(result_file);files[name]=dict(path=str(result_file),sha256=sha(result_file))
        assert result['inputs_sha256']==digest and result['checkpoint_sha256']==info['checkpoint_sha256']
        assert result['key']==row['key'] and result['frame_indices']==row['frame_indices'] and result['target_pt']==row['target_pt']
        assert result['timestamps']==row['timestamps'] and result['source_video_sha256']==row['video_hashes']
        assert len(result['predictions'])==len(result['raw_responses'])==len(row['frame_indices'])
        audit=read(result['initialization_audit']);assert audit['strict_final_state'] and audit['checkpoint_sha256']==info['checkpoint_sha256']
        assert audit['regime']=='Recover-R03-20-train-trajectories-1-epoch'
        if name!='forest':assert audit['loaded_tensor_digest']==read(p/'trainable-state.json')['sha256']
        vals=result['predictions'];predictions[name]=[np.nan if v is None else v for v in vals]
        assert all(v is None or math.isfinite(v) for v in vals)
        metrics[name]=trajectory_metrics(vals,row['target_pt'],row['timestamps'])
        metrics[name]['global_mae']=float(np.mean(np.abs(np.array(vals)-row['target_pt']))) if all(v is not None for v in vals) else None
        metrics[name]['invalid_frames']=sum(v is None for v in vals)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.2,2.8))
    for name in ORDER:
        ax.plot(row['timestamps'],predictions[name],label=DISPLAY[name],color=COLORS[name],
            linewidth=1.9 if name=='forest' else 1.25,alpha=.95,zorder=4 if name=='forest' else 2)
    ax.plot(row['timestamps'],row['target_pt'],color='#242424',linestyle='--',linewidth=1.65,label='Ground truth',zorder=5)
    ax.set_xlabel('Time (s)');ax.set_ylabel('Progress');ax.set_xlim(0,row['timestamps'][-1])
    ax.grid(axis='y',alpha=.18,linewidth=.6)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=6,frameon=False,
        fontsize=8,columnspacing=1.15,handlelength=2)
    fig.subplots_adjust(left=.085,right=.99,top=.96,bottom=.28)
    figures={}
    for extension in ('png','pdf'):
        p=OUT/('spray34-five-model-progress.'+extension);fig.savefig(p,dpi=300,facecolor='white');figures[extension]=str(p)
    plt.close(fig)
    save(OUT/'report.json',dict(status='completed',key=row['key'],models=ORDER,samples_per_model=len(row['frame_indices']),
        source_video_duration_s=row['frames']/row['fps'],sample_fps=5,epochs=1,training_episodes=20,
        inputs_sha256=digest,source_and_checkpoint_hashes_verified=True,selected_without_model_predictions=True,
        no_smoothing_clipping_or_filling=True,metrics=metrics,episode_files=files,figures=figures,
        future_endpoint_reference_models=['robobrain25-4b','dopamine-3b'],completed=time.time()))
    print('All five results audited; figure saved',figures,flush=True)


if __name__=='__main__':main()
