"""Audit every saved prediction and render the unmodified five-model curves."""
import math
import time
from .common import *

DISPLAY = {'robometer-4b':'RoboMeter','procvlm-2b':'ProcVLM','robobrain25-4b':'RoboBrain2.5',
           'dopamine-3b':'Robo-Dopamine','forest':'FoREST'}
ORDER = ['forest','robometer-4b','procvlm-2b','robobrain25-4b','dopamine-3b']
COLORS = {'robometer-4b':'#198bcb','procvlm-2b':'#bc55b1','robobrain25-4b':'#eaaa10',
          'dopamine-3b':'#19b990','forest':'#d55e00'}


def main():
    from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics
    from experiments.prm_five_groups_20261114.prepare_cache import progress_values
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
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
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.5,'axes.labelsize':13,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,4.15));ax=fig.add_axes([.11,.14,.825,.465])
    selected=np.rint(np.linspace(0,len(row['timestamps'])-1,8)).astype(int)
    thumbnail_times=[row['timestamps'][i] for i in selected]
    thumbnail_frames=[row['frame_indices'][i] for i in selected]
    gap=.004;tile_width=(.934-7*gap)/8
    for j,(frame,timestamp) in enumerate(zip(thumbnail_frames,thumbnail_times)):
        path=decoded_root/f'{frame:08d}-0.png'
        with Image.open(path) as im:
            height=tile_width*12*im.height/im.width/4.15
            thumb=fig.add_axes([.055+j*(tile_width+gap),.79,tile_width,height])
            thumb.imshow(im);thumb.set_axis_off()
        fig.text(.055+j*(tile_width+gap)+tile_width/2,.776,f'{timestamp:.1f} s',
            ha='center',va='top',fontsize=10.5,color='#444444')
    ax.plot(row['timestamps'],row['target_pt'],color='#202020',linestyle='--',linewidth=2.5,label='Ground truth',zorder=6)
    for name in ORDER:
        style='--' if name=='robobrain25-4b' else ((0,(2,1.5)) if name=='dopamine-3b' else '-')
        ax.plot(row['timestamps'],predictions[name],label=DISPLAY[name],color=COLORS[name],
            linestyle=style,linewidth=2.5 if name=='forest' else 1.5,alpha=.98,zorder=5 if name=='forest' else 2)
    ax.set_xlabel('Time (s)');ax.set_ylabel('Progress');ax.set_xlim(0,row['timestamps'][-1])
    ax.set_xticks(thumbnail_times,labels=[f'{t:.1f}' for t in thumbnail_times])
    values=np.concatenate([np.array(row['target_pt']),*[np.array(v) for v in predictions.values()]])
    low,high=np.nanmin(values),np.nanmax(values)
    ax.set_ylim(min(-.035,low-.035),max(1.035,high+.035))
    if low>=-.035 and high<=1.035:ax.set_yticks([0,.25,.5,.75,1],labels=['0','0.25','0.50','0.75','1.00'])
    ax.grid(axis='y',alpha=.25,linewidth=.7);ax.grid(axis='x',alpha=.13,linewidth=.7,linestyle='--')
    fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.52,.695),ncol=6,frameon=False,
        fontsize=11.2,columnspacing=1.25,handlelength=2.4,handletextpad=.5)
    figures={}
    for extension in ('png','pdf'):
        p=OUT/('spray34-five-model-progress.'+extension);fig.savefig(p,dpi=300,facecolor='white');figures[extension]=str(p)
    plt.close(fig)
    save(OUT/'report.json',dict(status='completed',key=row['key'],models=ORDER,samples_per_model=len(row['frame_indices']),
        source_video_duration_s=row['frames']/row['fps'],sample_fps=5,epochs=1,training_episodes=20,
        inputs_sha256=digest,source_and_checkpoint_hashes_verified=True,selected_without_model_predictions=True,
        no_smoothing_clipping_or_filling=True,metrics=metrics,episode_files=files,figures=figures,
        thumbnail_frame_indices=thumbnail_frames,thumbnail_timestamps=thumbnail_times,
        future_endpoint_reference_models=['robobrain25-4b','dopamine-3b'],completed=time.time()))
    print('All five results audited; figure saved',figures,flush=True)


if __name__=='__main__':main()
