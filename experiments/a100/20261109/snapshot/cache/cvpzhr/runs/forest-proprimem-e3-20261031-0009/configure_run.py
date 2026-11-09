"""将已通过实测的配置固定为本次三轮训练配置。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

RUN=Path('/cache/cvpzhr/runs/forest-proprimem-e3-20261031-0009')
CODE=Path('/cache/cvpzhr/code/forest-train3-20261031-0009')

p=argparse.ArgumentParser();p.add_argument('benchmark');args=p.parse_args()
assert not (RUN/'started.json').exists(),'Training already started'
chosen=json.loads((RUN/'benchmarks'/f'{args.benchmark}.json').read_text())
assert chosen['status']=='passed' and chosen['world_size']==4 and chosen['shape_replay']
assert max(chosen['peak_reserved_gib'])<70,'Need GPU memory headroom'
config=json.loads(Path('/cache/cvpzhr/setup/proprimem-5fps-20261030/train-4gpu.template.json').read_text())
config.update(output_dir='/cache/cvpzhr/outputs/forest-proprimem-e3-20261031-0009',epochs=3,max_steps=None,
              batch_size=chosen['batch_per_gpu'],gradient_accumulation_steps=1,num_workers=chosen['workers'],
              persistent_workers=True,fused_optimizer=True,learning_rate=1e-4,log_every_steps=10,
              save_every_steps=None,save_every_epochs=3,keep_only_latest=True,
              sequence_padding_multiple=64,sequence_extra_padding_side='right')
config['model']['gradient_checkpointing']=chosen['gradient_checkpointing']
(RUN/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
comparison=[dict(name=path.stem,**json.loads(path.read_text())) for path in sorted((RUN/'benchmarks').glob('ddp-pad64-*.json'))]
summary=dict(selected=args.benchmark,configuration=chosen,comparison=comparison,
             steps_per_epoch=math.ceil(math.ceil(126441/4)/config['batch_size']),
             train_samples=126441,validation_samples=32164,epochs=3,effective_batch=4*config['batch_size'],
             forecast_training_hours=3*126441/chosen['samples_per_second']/3600,
             note='Shape-replay throughput, four GPUs; LR=0 independent benchmark, formal training reloads original base. Actual full-dataset time may vary.')
(RUN/'batch-selection.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
manifest={str(path.relative_to(CODE)):hashlib.sha256(path.read_bytes()).hexdigest()
          for path in sorted(CODE.rglob('*')) if path.is_file() and path.suffix in ('.py','.json','.toml')}
(RUN/'code-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('comparison','configuration')},ensure_ascii=False))
