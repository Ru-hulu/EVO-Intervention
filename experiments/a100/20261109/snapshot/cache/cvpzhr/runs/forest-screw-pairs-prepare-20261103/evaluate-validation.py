"""Evaluate a fine-tuned checkpoint on the 16 held-out paired sequences."""
import argparse, json, pathlib
from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.runtime import load_checkpoint
from mem_lite.evaluate import evaluate
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',default='/cache/cvpzhr/outputs/forest-screw-pairs-finetune-20261103/last.pt')
p.add_argument('--output-dir',default='/cache/cvpzhr/outputs/forest-screw-pairs-finetune-20261103/validation-last')
a=p.parse_args()
out=pathlib.Path(a.output_dir)
if out.exists():raise FileExistsError(out)
files=json.loads(pathlib.Path('/cache/cvpzhr/datasets/forest-screw-pairs-5fps-20261103/validation.sample_files.json').read_text())
data=M3Dataset(files)
assert len(data.episodes)==16 and all(s['source']['split']=='validation' and s['source']['label_status']=='ready' for _,s in data.episodes)
model,processor,cfg=load_checkpoint(a.checkpoint)
assert data.history_source==cfg['history_source'] and data.history_capacity==cfg['history_capacity']
metrics,records=evaluate(model,processor,data,cfg['max_sequence_length'])
out.mkdir(parents=True)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
(out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
print(json.dumps(metrics,indent=2))
