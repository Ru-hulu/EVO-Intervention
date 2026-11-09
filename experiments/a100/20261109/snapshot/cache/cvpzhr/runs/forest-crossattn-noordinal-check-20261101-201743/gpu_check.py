"""Real BF16 model consistency check, without backward or optimizer updates."""
import json
from pathlib import Path
import sys
import traceback

import torch
from transformers import AutoProcessor
from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.runtime import StreamingProgressPredictor, to_device
from mem_lite.memory.vision import verify_vision_source
from mem_lite.train_stage3 import CHECKPOINT_FORMAT

out=Path(sys.argv[1])
try:
    torch.manual_seed(42)
    config=json.loads(Path('mem_lite/configs/a100_crossattn_train.json').read_text())
    verify_vision_source(config['model_path'],config['history_source'])
    processor=AutoProcessor.from_pretrained(config['model_path'],max_pixels=config['max_pixels'],local_files_only=True)
    model=M3ProgressPredictor(config['model_path'],**{**config['model'],'lora_dropout':0}).to('cuda')
    # Exercise a nonzero history contribution without training any parameter.
    with torch.no_grad():
        for module in model.memory_adapters.values():module.gate.fill_(0.1)
    root=Path('/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030/cache/validation')
    episodes={2:root/'Count/07-pick_place_bowl_3_times/episode_000307',
              3:root/'Count/01-press_buttons_by_number/episode_000904'}
    checks=[]
    runner=StreamingProgressPredictor(model,processor,config['history_source'],config['max_sequence_length'])
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for views,directory in episodes.items():
            dataset=M3Dataset([str(directory)])
            assert dataset.history_source==config['history_source']
            batcher=Stage3Collator(processor,dataset.history_dim,config['max_sequence_length'],dataset.history_capacity)
            differences=[]
            mode_differences=[]
            runner.reset()
            for i in range(16):
                row=dataset[i]
                batch=batcher([row])
                inputs=to_device(batch['inputs'],'cuda')
                model.train()
                training=model(inputs,batch['target_pt'].cuda())
                assert torch.isfinite(training['loss']) and torch.isfinite(training['pt']).all()
                model.eval()
                cached=model(inputs)['pt']
                online=runner(row['images'],row['frame_index'],row['episode_id'],row['task_prompt'])['pt']
                differences.append(float((cached-online).abs().max()))
                mode_differences.append(float((training['pt']-cached).abs().max()))
            assert max(differences)==0 and max(mode_differences)==0
            first=dataset[0]
            reset=runner(first['images'],first['frame_index'],first['episode_id']+'-reset',first['task_prompt'])
            expected=model(to_device(batcher([first])['inputs'],'cuda'))['pt']
            assert reset['history_length']==0
            torch.testing.assert_close(reset['pt'],expected,atol=0,rtol=0)
            checks.append({'views':views,'episode':str(directory),'samples':16,
                           'cache_online_max_difference':max(differences),
                           'train_eval_max_difference_dropout_disabled':max(mode_differences),'episode_reset':True})
    report={'status':'passed','format':CHECKPOINT_FORMAT,'ordinal_encoding':False,'original_spatiotemporal_encoding':True,
            'training_forward_dropout_disabled':True,'test_gate_value':0.1,'parameter_updates':0,
            'checks':checks,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
except Exception:
    (out/'failure.txt').write_text(traceback.format_exc())
    raise
