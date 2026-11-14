"""Bind immutable validation observations and independently trained suite checkpoints."""
import collections
import csv
import math
import time
from .common import *

def main():
    assert not (ROOT/'inputs.json').exists()
    original=read(SOURCE);selection=read(SELECTION);registry={};train_sha=sha(SELECTION)
    assert original['expected_episodes']==84 and original['input_fps']==5
    assert not {r['key'] for r in original['episodes']}&{r['episode_id'] for r in selection['episodes']}
    image_count=0
    for row in original['episodes']:
        assert row['split']=='val' and not row['input_issues']
        assert sha(row['progress_path'])==row['progress_sha256']
        with open(row['progress_path']) as stream:labels=list(csv.DictReader(stream))
        assert len(labels)==row['frames'] and [float(labels[i]['pt']) for i in row['frame_indices']]==row['target_pt']
        proof=read(Path(row['image_root'])/'verification.json')
        assert proof['input']==dict(video_hashes=row['video_hashes'],frame_indices=row['frame_indices'])
        for name,h in proof['png_sha256'].items():
            assert sha(Path(row['image_root'])/name)==h
        image_count+=len(proof['png_sha256'])
    for suite in SUITES:
        tc=collections.Counter(r['task_id'] for r in selection['episodes'] if r['suite']==suite)
        vc=collections.Counter(r['task_id'] for r in original['episodes'] if r['suite']==suite)
        assert sum(tc.values())==35 and min(tc.values())>=2 and set(vc)==set(tc) and set(vc.values())=={2}
        ref=selection['suites'][suite];proof_path=ref['checkpoint_verification_source'];proof=read(proof_path)
        assert proof['status']=='completed' and proof['final_checkpoint_verified'] and proof['epochs']==1
        assert sha(ref['forest_checkpoint'])==ref['forest_checkpoint_sha256']
        registry[job_name('forest',suite)]=dict(model='forest',suite=suite,python=PYTHON,
            checkpoint=ref['forest_checkpoint'],checkpoint_sha256=ref['forest_checkpoint_sha256'],
            proof=proof_path,proof_sha256=sha(proof_path),config=read(ref['forest_config']))
        for model in ('procvlm-2b','robometer-4b'):
            p=TRAIN/model/suite/'formal/verification.json';proof=read(p);ckpt=Path(proof['checkpoint']);cfg=read(ckpt/'run_config.json')
            assert proof['status']=='passed' and proof['epoch']==1 and proof['selection_sha256']==train_sha
            assert cfg['suite']==suite and cfg['episodes']==35 and cfg['selection_sha256']==train_sha
            assert sha(ckpt/'trainable-state.pt')==proof['trainable_state_sha256']
            registry[job_name(model,suite)]=dict(model=model,suite=suite,python=cfg['python'],
                checkpoint=str(ckpt),checkpoint_sha256=proof['trainable_state_sha256'],
                proof=str(p),proof_sha256=sha(p),config=cfg)
    policy=dict(memory_enabled=False,input_fps=5,augmentation='none',conditions=['no_memory'],
        procvlm='one actual current main-view image; SOC progress-only prompt; no previous text',
        robometer='three current view slots and one current progress token; no history images/tokens',
        forest='current view slots; history masks zero; no StreamMemory instance or memory writes',
        metrics=['pearson','spearman','VS','ER','C','SA'],SA='Mean prediction over last ceil(0.02*N) sampled frames of each successful trajectory; equal episode weight; continuous endpoint score, not binary accuracy',
        prediction_unit='0..1',smoothing=False,per_curve_rescaling=False,invalid_predictions='null with coverage and denominator counts',
        aggregation='per-trajectory metrics, equally averaged within tasks and suites',
        success_eligibility='separate explicit success-policy.json; never infer success from PT=1')
    save(ROOT/'inputs.json',dict(schema='group2-soc35-no-memory-v1',episodes=original['episodes'],
        expected_episodes=84,expected_predictions_per_model=15096,registry=registry,protocol=policy,
        source_inputs=str(SOURCE),source_inputs_sha256=sha(SOURCE),training_selection_sha256=train_sha,
        metric_provenance=read(ROOT/'code/experiments/prm_five_groups_20261114/metric_reference/provenance.json')))
    save(ROOT/'preflight.json',dict(status='passed',checkpoints=9,episodes=84,points_per_model=15096,
        labels_verified=True,png_files_verified=image_count,training_started=False,evaluation_started=False,finished=time.time()))

if __name__=='__main__':main()
