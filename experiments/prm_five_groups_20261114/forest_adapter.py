"""Strict final-suite FoREST loading and native online/no-memory inference."""
import hashlib
import json
import math
from pathlib import Path

from .eval_common import PLAN
from .prepare_cache import BASE,sha

SUITES=('State','Order','Count','Recover')
BASE_RECEIPT=Path('/cache/cvpzhr/setup/qwen35-original-download-20261113/verification.json')


def check_final_proof(proof,suite,expected_path):
    if suite not in SUITES:raise ValueError('Unknown suite')
    assert proof['status']=='passed' and proof['suite']==suite
    assert proof['trial_only'] is False and proof['epoch']==1 and proof['step']>0
    assert proof['optimizer_scheduler_rng_verified'] and proof['fresh_initialization_verified_both_ranks']
    assert Path(proof['checkpoint']).resolve()==Path(expected_path).resolve()


def check_metadata(checkpoint,suite,plan,training_config):
    c=checkpoint['config'];p=c['run_protocol'];fixed=plan['forest_training']
    assert checkpoint['format']=='mem-lite-streamforest-v1'
    assert checkpoint['epoch']==1 and checkpoint['next_batch']==0
    assert checkpoint['step']==c['total_optimizer_steps']==checkpoint['scheduler']['last_epoch']
    assert checkpoint['optimizer']['state'] and len(checkpoint['rng_by_rank'])==2
    assert p['suite']==suite and not p['trial_only'] and not p['trial_weights_reused']
    assert not p['artificial_augmentation'] and p['epochs']==1 and p['world_size']==2
    assert p['effective_batch']==18 and p['fps']==5
    assert c['base_revision']==fixed['base_revision'] and c['model_path']==fixed['base_path']
    assert c['seed']==42 and c['epochs']==1 and c['learning_rate']==1e-4
    assert c['lr_schedule']=='cosine' and c['warmup_ratio']==.03
    assert c['data_signature']==training_config['data_signature']
    assert c['history_source']==training_config['history_source']
    assert c['history_source']['memory_config']==fixed['memory_config']
    assert c['event_split_method']=='max_depth' and c['history_capacity']==training_config['history_capacity']
    assert c['model']==training_config['model'] and c['max_pixels']==training_config['max_pixels']
    assert c['max_sequence_length']==training_config['max_sequence_length']


def state_digest(state):
    import torch
    digest=hashlib.sha256()
    for group,items in sorted(state.items()):
        for name,tensor in sorted(items.items()):
            value=tensor.detach().cpu().contiguous()
            digest.update(f'{group}/{name}'.encode());digest.update(str((value.dtype,tuple(value.shape))).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ForestRuntime:
    """No labels enter this interface; frame positions must follow playback time.

    Prefix replay updates only the native vision/memory stream, preserving the
    already recorded predictions instead of re-running or overwriting them.
    """
    def __init__(self,model,processor,config,memory_enabled=True):
        from mem_lite.runtime import StreamingProgressPredictor
        from mem_lite.datasets.m3_inputs import M3Collator
        self.model=model.eval();self.config=config;self.memory_enabled=memory_enabled
        self.device=next(model.parameters()).device
        self.runner=(StreamingProgressPredictor(model,processor,config['history_source'],
            config['max_sequence_length'],event_split_method='max_depth',
            clip_model_path=config['history_source']['memory_config']['clip_model_path']) if memory_enabled else None)
        self.collator=(None if memory_enabled else M3Collator(processor,config['model']['history_dim'],
            config['max_sequence_length'],config['history_capacity']))
        self.reset()

    def reset(self):
        if self.runner is not None:self.runner.reset()
        self.session=None;self.last_position=None

    def _position(self,frame_position,episode_id,prompt):
        session=(episode_id,prompt)
        if session!=self.session:self.reset();self.session=session
        if self.last_position is not None and frame_position<=self.last_position:
            raise ValueError('FoREST positions must increase in playback order, including rewind')
        self.last_position=frame_position

    def advance_memory(self,images,frame_position,episode_id,prompt):
        import torch
        from mem_lite.datasets.views import fill_views
        self._position(frame_position,episode_id,prompt)
        if self.runner is not None:
            with torch.no_grad():self.runner.stream.step(fill_views(images)[0],frame_position,episode_id,prompt)

    def predict(self,images,frame_position,episode_id,prompt):
        import torch
        from mem_lite.runtime import to_device
        self._position(frame_position,episode_id,prompt)
        with torch.no_grad():
            if self.runner is not None:
                result=self.runner(images,frame_position,episode_id,prompt)
            else:
                inputs=self.collator([{'images':images,'task_prompt':prompt,'history_tokens':None}])
                assert not inputs['history_valid_mask'].any()
                result=self.model(to_device(inputs,self.device));result['history_length']=0
        value=float(result['pt'][0])
        return {'prediction':value if math.isfinite(value) else None,
                'raw_nonfinite':None if math.isfinite(value) else repr(value),
                'history_length':int(result['history_length'])}


def load_verified_forest(suite,device='cuda',memory_enabled=True):
    import torch
    from mem_lite.runtime import load_checkpoint
    from mem_lite.train_stage3 import trainable_state
    expected=Path(f'/cache/cvpzhr/outputs/forest-{suite.lower()}-qwen35-clip-depth-e1-20261114/epoch-01.pt')
    proof_path=BASE/suite/'checkpoint-verification.json';proof=json.loads(proof_path.read_text())
    check_final_proof(proof,suite,expected);assert sha(expected)==proof['sha256']
    plan=json.loads(PLAN.read_text());training=json.loads((BASE/suite/'training.config.json').read_text())
    receipt=json.loads(BASE_RECEIPT.read_text());fixed=plan['forest_training']
    assert receipt['all_official_hashes_verified'] and receipt['revision']==fixed['base_revision']
    assert receipt['destination']==fixed['base_path']
    files={name:item['sha256'] for name,item in receipt['files'].items()}
    for name,digest in files.items():assert sha(Path(fixed['base_path'])/name)==digest,name
    checkpoint=torch.load(expected,map_location='cpu',weights_only=True)
    check_metadata(checkpoint,suite,plan,training)
    expected_state=state_digest(checkpoint['model']);del checkpoint
    ranks=[json.loads((expected.parent/f'initialization-rank{r}-fresh.json').read_text()) for r in range(2)]
    assert all(r['fresh_language_lora'] and not r['resume'] and r['world_size']==2 for r in ranks)
    assert ranks[0]['initial_trainable_sha256']==ranks[1]['initial_trainable_sha256']
    assert all(r['base_revision']==fixed['base_revision'] for r in ranks)
    model,processor,config=load_checkpoint(expected,device=device,event_split_method='max_depth')
    loaded_state=state_digest(trainable_state(model));assert loaded_state==expected_state
    runtime=ForestRuntime(model,processor,config,memory_enabled)
    audit={'suite':suite,'regime':'suite_fine_tuned','checkpoint':str(expected),
        'checkpoint_sha256':proof['sha256'],'checkpoint_verification_sha256':sha(proof_path),
        'base_revision':fixed['base_revision'],'official_base_file_sha256':files,
        'loaded_trainable_state_sha256':loaded_state,'strict_trainable_state_verified':True,
        'final_epoch':1,'memory_enabled':memory_enabled,'history_source':config['history_source'],
        'max_sequence_length':config['max_sequence_length'],'max_pixels':config['max_pixels'],
        'adapter_code_sha256':sha(__file__),'trial_checkpoint_used':False,
        'uses_future_endpoint_reference':False,'device':str(device)}
    return runtime,audit
