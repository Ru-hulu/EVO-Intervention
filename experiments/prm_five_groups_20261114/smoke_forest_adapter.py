"""CPU-only disposable runtime smoke; never emits formal evaluation predictions."""
import json
import os
import time

from .prepare_cache import BASE,save,sha
from .eval_common import EVAL
from .overlap_guard import resources
from .forest_adapter import ForestRuntime,state_digest


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('This disposable smoke must be explicitly CPU-only')
    sample=resources();reserve=32*2**30
    assert sample['resident_and_shmem']+reserve<.65*sample['memory_limit']
    assert sample['memory_working_set']+reserve<.85*sample['memory_limit']
    assert sample['oom_kill']==0 and not sample['under_oom']
    import torch
    from mem_lite.runtime import load_checkpoint
    from mem_lite.train_stage3 import trainable_state
    from mem_lite.datasets.m3_dataset import M3Dataset
    torch.set_num_threads(2);torch.manual_seed(42)
    proof=json.loads((BASE/'smoke/checkpoint-verification.json').read_text())
    assert proof['status']=='passed' and proof['trial_only']
    path=BASE/'training-smoke/last.pt';assert str(path)==proof['checkpoint'] and sha(path)==proof['sha256']
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['config']['run_protocol']['trial_only']
    expected=state_digest(checkpoint['model']);del checkpoint
    started=time.time();model,processor,config=load_checkpoint(path,device='cpu',event_split_method='max_depth')
    assert state_digest(trainable_state(model))==expected
    dataset=M3Dataset(config['sample_files'],load_history=False)
    first,second=dataset[0],dataset[1];assert first['episode_id']==second['episode_id']
    runtime=ForestRuntime(model,processor,config)
    def predict(sample):return runtime.predict(sample['images'],sample['frame_index'],sample['episode_id'],sample['task_prompt'])
    one=predict(first);two=predict(second)
    assert one['history_length']==0 and two['history_length']>0
    runtime.reset();runtime.advance_memory(first['images'],first['frame_index'],first['episode_id'],first['task_prompt'])
    replay=predict(second);assert replay==two
    empty=ForestRuntime(model,processor,config,memory_enabled=False)
    no_memory=empty.predict(first['images'],first['frame_index'],first['episode_id'],first['task_prompt'])
    assert no_memory==one and all(r['prediction'] is not None for r in [one,two,replay,no_memory])
    receipt={'status':'passed','cpu_only':True,'gpu_tested':False,'trial_checkpoint_only':True,
        'trial_checkpoint_sha256':proof['sha256'],'loaded_trainable_state_sha256':expected,
        'first_history_length':one['history_length'],'second_history_length':two['history_length'],
        'prefix_rebuild_prediction_exact':True,'no_memory_first_prediction_exact':True,
        'native_vs_online_cpu_precision':'float32 only; not a GPU BF16 equivalence claim',
        'formal_evaluation_predictions_written':False,'final_checkpoint_verified_for_evaluation':False,
        'elapsed_seconds':time.time()-started,'code_sha256':sha(__file__)}
    save(EVAL/'forest-adapter-smoke.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':run()
